"""Federated training loop for FedDAC, baselines and ablations.

Every method is expressed as   (grouping, intra-group schedule, inter-group weighting):

  method            grouping                  intra     inter-group weights   label stats used
  ----------------  ------------------------  --------  --------------------  ----------------
  fedavg            singletons                parallel  n_i (samples)         no
  fedprox           singletons (+prox term)   parallel  n_i                   no
  scaffold          singletons (control var.) parallel  n_i                   no   (Karimireddy et al. 2020, option II)
  feddyn            singletons (dyn. reg.)    parallel  n_i                   no   (Acar et al. 2021)
  ppcfl             fuzzy c-means on RR       parallel  membership, mean      RR presence bits
  fedse             label-balanced (KL)       sequential n_g                  histograms
  fedseq            FedSeq greedy superclient sequential n_g                  histograms
  feddac            DAC                       sequential |C_l| (Eq. 12)       histograms
  -- ablations --
  rand_seq          random, L groups          sequential |C_l|                no
  feddac_order      DAC                       seq., coverage order |C_l|      histograms (docs/ORDER_PLAN.md)
  rand_order        random, L groups          seq., coverage order |C_l|      histograms (order only)
  dac_par           DAC                       parallel  |C_l|                 histograms
  dac_uniform       DAC                       sequential 1/L                  histograms
  dac_samplew       DAC                       sequential n_l                  histograms
  dac_noisyhist     DAC on Laplace(eps_h=1)   sequential |C_l|                noisy histograms
  -- multi-model baselines (feddac/multimodel.py) --
  ifca              loss-based, k models      parallel  per cluster, n_i      no
  cfl               cosine bipartitioning     parallel  per cluster, n_i      no
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from . import clustering as CL
from .dp import ClientDP
from .models import build_model
from .multimodel import MULTI_MODEL, per_class_accuracy, personal_accuracy, run_multimodel
from .train import (evaluate, local_train, num_params, state_clone, weighted_average)

USES_HIST = {"feddac", "fedse", "fedseq", "ppcfl", "dac_par", "dac_uniform", "dac_samplew",
             "dac_noisyhist", "feddac_order", "rand_order"}
SEQUENTIAL = {"feddac", "fedse", "fedseq", "rand_seq", "dac_uniform", "dac_samplew", "dac_noisyhist",
              "feddac_order", "rand_order"}
COVERAGE_ORDER = {"feddac_order", "rand_order"}
CORRECTED = {"scaffold", "feddyn"}
METHODS = ["scaffold", "feddyn", "fedavg", "fedprox", "ppcfl", "fedse", "fedseq", "feddac", "rand_seq", "dac_par",
           "dac_uniform", "dac_samplew", "dac_noisyhist", "ifca", "cfl", "feddac_order", "rand_order"]


def make_groups(method, hist_true, L, rng, cfg):
    """Returns (groups, hist_seen_by_server, extra_info)."""
    N = len(hist_true)
    info = {}
    hist = hist_true
    if cfg.get("dp_eps") and method in USES_HIST and method != "ppcfl":
        hist = CL.laplace_histograms(hist_true, cfg["eps_hist"], rng)
    if method == "dac_noisyhist":
        hist = CL.laplace_histograms(hist_true, cfg.get("noisyhist_eps", 1.0), rng)
    if method in ("fedavg", "fedprox", "scaffold", "feddyn"):
        groups = [[i] for i in range(N)]
    elif method in ("feddac", "dac_par", "dac_uniform", "dac_samplew", "dac_noisyhist", "feddac_order"):
        groups = CL.dac_cluster(hist, L)
    elif method in ("rand_seq", "rand_order"):
        groups = CL.random_groups(N, L, rng)
    elif method == "fedse":
        groups = CL.label_balanced(hist, L)
    elif method == "fedseq":
        groups = CL.fedseq_superclients(hist, rng, k_max=cfg.get("fedseq_kmax", 11),
                                        min_samples=cfg.get("fedseq_min_samples", 800))
    elif method == "ppcfl":
        rr_eps = cfg["eps_hist"] if cfg.get("dp_eps") else cfg.get("ppcfl_rr_eps", 0.8)
        B = CL.randomized_response_presence(hist_true, rr_eps, rng)
        S = CL.jaccard_matrix(B)
        U = CL.fuzzy_cmeans(S, L, rng)
        info["membership"] = U
        groups = [np.where(U.argmax(1) == k)[0].tolist() for k in range(L)]
        groups = [g for g in groups if g]
    else:
        raise ValueError(method)
    info["cluster_stats"] = CL.cluster_stats(groups, hist_true)
    info["hist_seen"] = hist
    return groups, info


def run_federated(ds, part, cfg, device, log=print, capture_rounds=(), save_units=False):
    """Main loop.  Returns a result dict (JSON-serialisable) and the final global model."""
    method = cfg["method"]
    if method in MULTI_MODEL:
        return run_multimodel(ds, part, cfg, device, log=log)
    seed = cfg["seed"]
    rng = np.random.default_rng(seed + 1000)
    torch.manual_seed(seed)
    N = len(part.client_indices)
    L = CL.num_clusters(N)
    C = ds.num_classes
    hist = part.histograms
    n_i = hist.sum(1).astype(float)
    # declared public capacity: with DP and cfg["capacity"], q/T/sigma and the sample weights use S_i
    cap = cfg.get("capacity")
    if cap == "auto":
        cap = part.capacity
    use_cap = bool(cfg.get("dp_eps")) and cap is not None
    n_w = np.full(N, float(cap)) if use_cap else n_i  # sample weights seen by the server
    idx = [torch.as_tensor(ix, device=device) for ix in part.client_indices]

    groups, ginfo = make_groups(method, hist, L, rng, cfg)
    hist_order = ginfo.get("hist_seen", hist)  # histograms the server holds (noisy under DP)
    G = len(groups)
    group_sizes = [len(g) for g in groups]
    group_samples = [float(n_w[g].sum()) for g in groups]
    seq = method in SEQUENTIAL
    if method == "feddac" or method in ("rand_seq", "dac_par", "dac_noisyhist", "feddac_order", "rand_order"):
        inter_w = group_sizes
    elif method == "dac_uniform" or method in CORRECTED:  # SCAFFOLD/FedDyn: uniform mean as published
        inter_w = [1.0] * G
    else:  # fedavg/fedprox/fedse/fedseq/dac_samplew
        inter_w = group_samples

    global_model = build_model(ds.name, C, cfg.get("model")).to(device)
    P = num_params(global_model)
    amp = (ds.name in ("cifar100", "tinyimagenet") or cfg.get("model") == "resnet18gn") and device.type == "cuda"
    train_cfg = {k: cfg[k] for k in ("lr", "batch_size", "local_epochs", "optimizer", "momentum") if k in cfg}
    if method == "fedprox":
        train_cfg["mu"] = cfg.get("mu", 0.01)
    # ---- SCAFFOLD / FedDyn state (full participation; parameters only) ----
    if method in CORRECTED:
        if cfg.get("dp_eps"):
            raise ValueError(f"{method} is not run under DP in this study")
        zeros = lambda: [torch.zeros_like(p) for p in global_model.parameters()]  # noqa: E731
        c_glob, c_loc = zeros(), [zeros() for _ in range(N)]  # SCAFFOLD control variates
        h_glob, h_loc = zeros(), [zeros() for _ in range(N)]  # FedDyn: server state and client gradients
        alpha = float(cfg.get("feddyn_alpha", 0.01))
        if method == "feddyn":
            train_cfg["mu"] = alpha
        B_loc = cfg.get("batch_size", 10)

    # ---- differential privacy set-up (record-level DP-SGD) ----
    dps, dp_report = None, None
    if cfg.get("dp_eps"):
        eps_sgd = cfg["dp_eps"] - (cfg["eps_hist"] if method in USES_HIST else 0.0)
        dps = [ClientDP(int(n_i[i]), cfg["dp_batch_size"], cfg["rounds"], cfg.get("local_epochs", 1),
                        eps_sgd, cfg["dp_delta"], cfg.get("dp_clip", 1.0),
                        capacity=int(cap) if use_cap else None) for i in range(N)]
        reps = [d.report() for d in dps]
        dp_report = {"eps_total_target": cfg["dp_eps"], "eps_sgd": eps_sgd,
                     "eps_hist": cfg["eps_hist"] if method in USES_HIST else 0.0,
                     "delta": cfg["dp_delta"], "clip": cfg.get("dp_clip", 1.0),
                     "sigma_min": min(r["sigma"] for r in reps), "sigma_max": max(r["sigma"] for r in reps),
                     "eps_sgd_spent_max": max(r["eps"] for r in reps),
                     "capacity": int(cap) if use_cap else None,
                     "q_min": min(r["q"] for r in reps), "q_max": max(r["q"] for r in reps),
                     "steps": max(r["steps"] for r in reps)}

    history, captures = [], []
    comm_transfers = 0
    t_start = time.perf_counter()
    sim_latency = 0.0
    client_time_total = 0.0
    eval_every = cfg.get("eval_every", 1)
    for r in range(cfg["rounds"]):
        gstate = state_clone(global_model)
        if method in CORRECTED:
            delta_c = [torch.zeros_like(p) for p in global_model.parameters()]
            dyn_sum = [torch.zeros_like(p) for p in global_model.parameters()]
        group_states, group_times = [], []
        capture = r + 1 in capture_rounds
        cap = {"round": r + 1, "units": []} if capture else None
        if capture:
            cap["global_out"] = _out_params(global_model)
            cap["global_state"] = {k: v.cpu() for k, v in gstate.items()}
        want_full = cfg.get("mia") and r + 1 == cfg["rounds"]
        full_units = []
        for g in groups:
            if seq:
                model = copy.deepcopy(global_model)
                if method in COVERAGE_ORDER:
                    order = CL.coverage_order(g, hist_order, rng)
                else:
                    order = list(rng.permutation(g)) if cfg.get("seq_order", "random") == "random" else list(g)
                t_g = 0.0
                prefix = []
                for c in order:
                    t_g += local_train(model, ds, idx[c], train_cfg, dp=dps[c] if dps else None, amp=amp)
                    if capture:
                        prefix.append({"clients": [int(x) for x in prefix[-1]["clients"] + [int(c)]] if prefix else [int(c)],
                                       "out": _out_params(model)})
                group_states.append(state_clone(model))
                if want_full:
                    full_units.append({"clients": [int(x) for x in g], "state": {k: v.cpu() for k, v in group_states[-1].items()}})
                group_times.append(t_g)
                client_time_total += t_g
                comm_transfers += len(g) + 1  # server->c1, hand-overs, c_last->server
                if capture:
                    # server view: final cluster model; peer views: prefixes (seen by next client)
                    cap["units"].append({"view": "server", "clients": [int(x) for x in order],
                                         "out": prefix[-1]["out"]})
                    for p in prefix[:-1]:
                        cap["units"].append({"view": "peer", **p})
            else:
                states, ts, ws = [], [], []
                for c in g:
                    model = copy.deepcopy(global_model)
                    gcorr = None
                    if method == "scaffold":
                        gcorr = [cg - cl for cg, cl in zip(c_glob, c_loc[c])]
                    elif method == "feddyn":
                        gcorr = [-hl for hl in h_loc[c]]
                    ts.append(local_train(model, ds, idx[c], train_cfg, prox_ref=global_model,
                                          dp=dps[c] if dps else None, amp=amp, gcorr=gcorr))
                    if method in CORRECTED:
                        with torch.no_grad():
                            d = [p.detach() - q.detach() for p, q in zip(model.parameters(), global_model.parameters())]
                            if method == "scaffold":
                                K = cfg.get("local_epochs", 1) * int(np.ceil(len(part.client_indices[c]) / B_loc))
                                lr_ = train_cfg.get("lr", 1e-3)
                                new_c = [cl - cg - di / (K * lr_) for cl, cg, di in zip(c_loc[c], c_glob, d)]
                                for j in range(len(new_c)):
                                    delta_c[j] += (new_c[j] - c_loc[c][j]) / N
                                c_loc[c] = new_c
                            else:
                                h_loc[c] = [hl - alpha * di for hl, di in zip(h_loc[c], d)]
                                for j in range(len(d)):
                                    dyn_sum[j] += d[j] / N
                    states.append(state_clone(model))
                    ws.append(n_w[c])
                    if want_full:
                        full_units.append({"clients": [int(c)], "state": {k: v.cpu() for k, v in states[-1].items()}})
                    if capture:
                        cap["units"].append({"view": "server", "clients": [int(c)], "out": _out_params(model)})
                client_time_total += sum(ts)
                comm_transfers += 2 * len(g)
                if method == "ppcfl":
                    group_states.extend(states)  # aggregated below with memberships
                else:
                    group_states.append(weighted_average(states, ws))
                group_times.append(max(ts))
        # ---- aggregation ----
        if method == "ppcfl":
            members = [c for g in groups for c in g]
            U = ginfo["membership"][members]
            cluster_models = []
            for k in range(U.shape[1]):
                if U[:, k].sum() > 0:
                    cluster_models.append(weighted_average(group_states, U[:, k].tolist()))
            new_state = weighted_average(cluster_models, [1.0] * len(cluster_models))
        else:
            new_state = weighted_average(group_states, inter_w)
        global_model.load_state_dict(new_state)
        if method == "scaffold":
            c_glob = [cg + dc for cg, dc in zip(c_glob, delta_c)]
        elif method == "feddyn":
            with torch.no_grad():
                h_glob = [hg - alpha * ds_ for hg, ds_ in zip(h_glob, dyn_sum)]
                for p, hg in zip(global_model.parameters(), h_glob):
                    p.sub_(hg / alpha)
        if want_full:
            captures.append({"round": r + 1, "final_units": full_units})
        sim_latency += max(group_times)
        if capture:
            captures.append(cap)
        if (r + 1) % eval_every == 0 or r + 1 == cfg["rounds"]:
            ev = evaluate(global_model, ds)
            if ds.y_val is not None:
                ev.update({f"val_{k}": v for k, v in evaluate(global_model, ds, split="val").items()})
            ev["round"] = r + 1
            history.append(ev)
            if (r + 1) % max(1, cfg["rounds"] // 10) == 0 or r == 0:
                log(f"[{method}] round {r + 1}/{cfg['rounds']} acc={ev['acc']:.2f} f1={ev['f1']:.2f}")
    wall = time.perf_counter() - t_start
    final = history[-1]
    last_k = [h for h in history if h["round"] > cfg["rounds"] - 5]
    res = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "final_acc": final["acc"], "final_f1": final["f1"], "final_loss": final["loss"],
        "final_personal_acc": personal_accuracy([per_class_accuracy(global_model, ds)] * N, hist),
        "last5_acc": float(np.mean([h["acc"] for h in last_k])),
        "last5_f1": float(np.mean([h["f1"] for h in last_k])),
        "best_acc": max(h["acc"] for h in history),
        **({"final_val_acc": final["val_acc"], "final_val_f1": final["val_f1"],
            "last5_val_acc": float(np.mean([h["val_acc"] for h in last_k]))} if "val_acc" in final else {}),
        "history": [{k: h[k] for k in ("round", "acc", "f1", "loss", "val_acc", "val_f1", "val_loss") if k in h}
                    for h in history],
        "num_groups": G, "L": L, "group_sizes": group_sizes, "group_samples": group_samples,
        "cluster_stats": ginfo["cluster_stats"],
        "client_types": part.client_type, "histograms": part.histograms.tolist(),
        "reused_samples": part.reused_samples,
        "dropped_samples": part.dropped_samples, "replacement": part.replacement,
        "capacity": part.capacity, "client_sizes": n_i.astype(int).tolist(),
        "cost": {"wall_clock_s": wall, "client_compute_s": client_time_total,
                 "simulated_latency_s": sim_latency, "model_transfers": comm_transfers,
                 "params": P, "comm_MB": comm_transfers * P * 4 / 2 ** 20,
                 "sequential": seq},
        "dp": dp_report,
        "groups": groups,
    }
    return res, global_model, captures


def _out_params(model):
    o = model.out
    return torch.cat([o.weight.detach().reshape(-1), o.bias.detach()]).float().cpu()
