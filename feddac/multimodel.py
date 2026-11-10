"""Clustered federated learning baselines that keep several models: IFCA and CFL.

IFCA (Ghosh et al., NeurIPS 2020; model-averaging variant, Alg. 2)
    k cluster models with independent initialisations. In every round each client evaluates all
    k models on its local training data, joins the cluster with the lowest loss, trains a copy of
    that model, and the server averages the returned models per cluster (weights n_i). A cluster
    that receives no client keeps its model.

CFL (Sattler et al., IEEE TNNLS 2021)
    Starts from one cluster and runs FedAvg inside each cluster. A cluster c is bipartitioned when
    the norm of its mean update is below eps1 (stationary point of the cluster objective), the
    largest client update norm exceeds eps2 (some clients are far from stationarity), the round is
    at least `cfl_warmup` and the cluster has more than two clients. The split is the complete-
    linkage bipartition of the cosine similarities of the client updates.

Evaluation with several models (the other methods produce one global model):
    final_acc         sum over clients of (n_i / n) x accuracy of the client's assigned model on
                      the global test set, i.e. the expected global-test accuracy of the model held
                      by a randomly drawn training example's client. Directly comparable with the
                      global-model accuracy of the single-model methods.
    final_personal_acc  sum over clients of (n_i / n) x sum_c p_i(c) acc_k(i)(c), the per-class
                      test accuracy of the assigned model weighted by the client's own label
                      distribution. Reported for every method (algorithms.py), so the advantage of
                      personalised models under label skew is visible rather than hidden.

DP-SGD, LDI captures and MIA are not supported for these two methods (not in L3-L5 of the plan).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import clustering as CL
from .models import build_model
from .train import evaluate, local_train, num_params, state_clone, weighted_average

MULTI_MODEL = {"ifca", "cfl"}


@torch.no_grad()
def per_class_accuracy(model, ds, batch_size=2000):
    """Per-class accuracy (%) on the test set; classes absent from the test set get NaN."""
    model.eval()
    preds = []
    for s in range(0, ds.y_test.numel(), batch_size):
        preds.append(model(ds.prep(ds.x_test[s:s + batch_size])).float().argmax(1))
    p = torch.cat(preds).cpu().numpy()
    y = ds.y_test.cpu().numpy()
    out = np.full(ds.num_classes, np.nan)
    for c in range(ds.num_classes):
        m = y == c
        if m.any():
            out[c] = (p[m] == c).mean() * 100
    return out


def personal_accuracy(class_acc_per_client, hist):
    """sum_i (n_i/n) sum_c p_i(c) acc_i(c); class_acc_per_client is (N, C)."""
    hist = np.asarray(hist, dtype=float)
    n_i = hist.sum(1)
    p = hist / np.maximum(n_i[:, None], 1.0)
    acc = np.nan_to_num(np.asarray(class_acc_per_client, dtype=float))
    per_client = (p * acc).sum(1)
    return float((n_i * per_client).sum() / n_i.sum())


@torch.no_grad()
def _local_loss(model, ds, idx, max_n=None, batch_size=2000):
    model.eval()
    if max_n is not None and idx.numel() > max_n:
        idx = idx[:max_n]
    tot = 0.0
    for s in range(0, idx.numel(), batch_size):
        b = idx[s:s + batch_size]
        tot += F.cross_entropy(model(ds.prep(ds.x_train[b])).float(), ds.y_train[b], reduction="sum").item()
    return tot / max(idx.numel(), 1)


def _flat_update(state, ref):
    return torch.cat([(state[k].float() - ref[k].float()).reshape(-1) for k in ref
                      if ref[k].dtype.is_floating_point])


def _bipartition(updates):
    """Complete-linkage bipartition on cosine similarity (Sattler et al., Alg. 1)."""
    from sklearn.cluster import AgglomerativeClustering

    U = torch.stack(updates)
    U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-12)
    S = (U @ U.T).cpu().numpy()
    lab = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="complete").fit(1.0 - S).labels_
    return [np.where(lab == 0)[0].tolist(), np.where(lab == 1)[0].tolist()], S


def _evaluate_assignment(models, assign, ds, hist, n_w):
    """Weighted global-test accuracy, F1, loss and personalised accuracy of a model assignment."""
    used = sorted(set(assign))
    evs = {k: evaluate(models[k], ds) for k in used}
    w = np.array([n_w[i] for i in range(len(assign))], dtype=float)
    w = w / w.sum()
    out = {m: float(sum(w[i] * evs[assign[i]][m] for i in range(len(assign)))) for m in ("acc", "f1", "loss")}
    if ds.y_val is not None:
        vev = {k: evaluate(models[k], ds, split="val") for k in used}
        out.update({f"val_{m}": float(sum(w[i] * vev[assign[i]][m] for i in range(len(assign))))
                    for m in ("acc", "f1", "loss")})
    return out


def run_multimodel(ds, part, cfg, device, log=print):
    method = cfg["method"]
    if method not in MULTI_MODEL:
        raise ValueError(method)
    if cfg.get("dp_eps") or cfg.get("mia"):
        raise ValueError(f"{method}: DP-SGD and MIA are not implemented for multi-model baselines")
    seed = cfg["seed"]
    rng = np.random.default_rng(seed + 1000)
    torch.manual_seed(seed)
    N = len(part.client_indices)
    C = ds.num_classes
    hist = part.histograms
    n_i = hist.sum(1).astype(float)
    n_w = n_i
    idx = [torch.as_tensor(ix, device=device) for ix in part.client_indices]
    amp = (ds.name in ("cifar100", "tinyimagenet") or cfg.get("model") == "resnet18gn") and device.type == "cuda"
    train_cfg = {k: cfg[k] for k in ("lr", "batch_size", "local_epochs", "optimizer", "momentum") if k in cfg}

    if method == "ifca":
        k = int(cfg.get("ifca_k", 3))
        models = []
        for j in range(k):
            torch.manual_seed(seed * 1000 + j)  # independent initialisations
            models.append(build_model(ds.name, C, cfg.get("model")).to(device))
        torch.manual_seed(seed)
        clusters = None
    else:
        k = 1
        models = [build_model(ds.name, C, cfg.get("model")).to(device)]
        clusters = [list(range(N))]
        eps1, eps2 = float(cfg.get("cfl_eps1", 0.4)), float(cfg.get("cfl_eps2", 1.6))
        warmup = int(cfg.get("cfl_warmup", 20))
        max_clusters = int(cfg.get("cfl_max_clusters", N))
    P = num_params(models[0])

    history, splits = [], []
    assign = [0] * N
    comm_transfers = 0
    t_start = time.perf_counter()
    sim_latency = client_time_total = 0.0
    eval_every = cfg.get("eval_every", 1)
    for r in range(cfg["rounds"]):
        if method == "ifca":
            for i in range(N):
                losses = [_local_loss(m, ds, idx[i], cfg.get("ifca_eval_n")) for m in models]
                assign[i] = int(np.argmin(losses))
            comm_transfers += N * (k + 1)  # k models down, one up
        else:
            for c, members in enumerate(clusters):
                for i in members:
                    assign[i] = c
            comm_transfers += 2 * N
        refs = [state_clone(m) for m in models]
        new_states = {}
        ts_all = []
        for c in range(len(models)):
            members = [i for i in range(N) if assign[i] == c]
            if not members:
                continue
            states, ws, ups = [], [], []
            for i in members:
                m = copy.deepcopy(models[c])
                ts_all.append(local_train(m, ds, idx[i], train_cfg, amp=amp))
                states.append(state_clone(m))
                ws.append(n_w[i])
                if method == "cfl":
                    ups.append(_flat_update(states[-1], refs[c]))
            new_states[c] = weighted_average(states, ws)
            if method == "cfl" and len(members) > 2 and r + 1 >= warmup and len(clusters) < max_clusters:
                wt = torch.tensor(ws, dtype=torch.float32, device=ups[0].device)
                mean_norm = (torch.stack(ups) * (wt / wt.sum())[:, None]).sum(0).norm().item()
                max_norm = max(u.norm().item() for u in ups)
                if mean_norm < eps1 and max_norm > eps2:
                    halves, _ = _bipartition(ups)
                    if halves[0] and halves[1]:
                        splits.append({"round": r + 1, "cluster": c, "mean_norm": mean_norm, "max_norm": max_norm,
                                       "sizes": [len(halves[0]), len(halves[1])]})
                        new_states[("split", c)] = [[members[j] for j in h] for h in halves]
        client_time_total += sum(ts_all)
        sim_latency += max(ts_all) if ts_all else 0.0
        for c, st in new_states.items():
            if not isinstance(c, tuple):
                models[c].load_state_dict(st)
        if method == "cfl":
            for key in [c for c in new_states if isinstance(c, tuple)]:
                c = key[1]
                a_, b_ = new_states[key]
                clusters[c] = a_
                clusters.append(b_)
                models.append(copy.deepcopy(models[c]))
            for c, members in enumerate(clusters):
                for i in members:
                    assign[i] = c
        if (r + 1) % eval_every == 0 or r + 1 == cfg["rounds"]:
            ev = _evaluate_assignment(models, assign, ds, hist, n_w)
            ev["round"] = r + 1
            ev["num_models_used"] = len(set(assign))
            history.append(ev)
            if (r + 1) % max(1, cfg["rounds"] // 10) == 0 or r == 0:
                log(f"[{method}] round {r + 1}/{cfg['rounds']} acc={ev['acc']:.2f} models={ev['num_models_used']}")
    wall = time.perf_counter() - t_start
    class_acc = {c: per_class_accuracy(models[c], ds) for c in set(assign)}
    personal = personal_accuracy([class_acc[assign[i]] for i in range(N)], hist)
    groups = [[i for i in range(N) if assign[i] == c] for c in sorted(set(assign))]
    final = history[-1]
    last_k = [h for h in history if h["round"] > cfg["rounds"] - 5]
    res = {
        "config": {k_: v for k_, v in cfg.items() if not k_.startswith("_")},
        "final_acc": final["acc"], "final_f1": final["f1"], "final_loss": final["loss"],
        "final_personal_acc": personal,
        "last5_acc": float(np.mean([h["acc"] for h in last_k])),
        "last5_f1": float(np.mean([h["f1"] for h in last_k])),
        "best_acc": max(h["acc"] for h in history),
        **({"final_val_acc": final["val_acc"], "final_val_f1": final["val_f1"],
            "last5_val_acc": float(np.mean([h["val_acc"] for h in last_k]))} if "val_acc" in final else {}),
        "history": [{k_: h[k_] for k_ in ("round", "acc", "f1", "loss", "val_acc", "val_f1", "val_loss",
                                          "num_models_used") if k_ in h} for h in history],
        "num_groups": len(groups), "L": len(models), "group_sizes": [len(g) for g in groups],
        "group_samples": [float(n_w[g].sum()) for g in groups],
        "cluster_stats": CL.cluster_stats(groups, hist),
        "client_types": part.client_type, "histograms": part.histograms.tolist(),
        "reused_samples": part.reused_samples,
        "dropped_samples": part.dropped_samples, "replacement": part.replacement,
        "capacity": part.capacity, "client_sizes": n_i.astype(int).tolist(),
        "cost": {"wall_clock_s": wall, "client_compute_s": client_time_total,
                 "simulated_latency_s": sim_latency, "model_transfers": comm_transfers,
                 "params": P, "comm_MB": comm_transfers * P * 4 / 2 ** 20, "sequential": False},
        "dp": None,
        "groups": groups,
        "multi_model": {"method": method, "num_models": len(models), "assignment": list(map(int, assign)),
                        "splits": splits,
                        "params": ({"ifca_k": k} if method == "ifca" else
                                   {"cfl_eps1": eps1, "cfl_eps2": eps2, "cfl_warmup": warmup})},
    }
    largest = max(set(assign), key=lambda c: sum(n_w[i] for i in range(N) if assign[i] == c))
    return res, models[largest], []
