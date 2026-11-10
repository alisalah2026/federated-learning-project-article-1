#!/usr/bin/env python
"""Run a single federated experiment and write results/<exp>/<run_id>.json (+ optional model).

Example:
  python scripts/run.py --dataset mnist --method feddac --clients 20 --alpha 0.2 \
      --ratios 0.1 0.3 0.6 --seed 0 --exp smoke --rounds 5
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feddac import attacks as A  # noqa: E402
from feddac.algorithms import run_federated  # noqa: E402
from feddac.data import load_dataset, partition, validation_split  # noqa: E402

DEFAULTS = {  # Table 1 of the manuscript (unchanged unless stated)
    "mnist": dict(rounds=50, batch_size=10, lr=1e-3),
    "cifar10": dict(rounds=100, batch_size=10, lr=1e-3),
    "cifar100": dict(rounds=100, batch_size=32, lr=1e-3, eval_every=5),
    "tinyimagenet": dict(rounds=60, batch_size=32, lr=1e-3, eval_every=5),
    "synthetic": dict(rounds=5, batch_size=10, lr=1e-3),
    "femnist": dict(rounds=100, batch_size=10, lr=1e-3),
}


# resubmission-protocol options; hashed only when set, so historical run ids are unchanged
PROTOCOL_KEYS = ["model", "no_replacement", "capacity", "val_frac", "deterministic", "subset_frac",
                 "ifca_k", "cfl_eps1", "cfl_eps2", "cfl_warmup", "lt_rho", "size_sigma"]


def run_id(cfg):
    keys = ["dataset", "method", "clients", "alpha", "ratios", "seed", "dp_eps", "rounds", "tag"]
    d = {k: cfg.get(k) for k in keys}
    d.update({k: cfg[k] for k in PROTOCOL_KEYS if cfg.get(k) not in (None, False, 0, 0.0, "")})
    s = json.dumps(d, sort_keys=True)
    h = hashlib.sha1(s.encode()).hexdigest()[:10]
    r = "-".join(f"{x:g}" for x in cfg["ratios"])
    dp = f"_dp{cfg['dp_eps']:g}" if cfg.get("dp_eps") else ""
    return f"{cfg['dataset']}_{cfg['method']}_N{cfg['clients']}_a{cfg['alpha']:g}_r{r}{dp}_s{cfg['seed']}_{h}"


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--method", default="feddac")
    p.add_argument("--clients", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.1, 0.3, 0.6])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lt_rho", type=float, default=0.0,
                   help="long-tailed training labels: class c keeps n_max * rho^(-c/(C-1)) examples (0 = off)")
    p.add_argument("--size_sigma", type=float, default=0.0,
                   help="quantity skew: log-normal(0, sigma) planned client sizes (0 = equal sizes)")
    p.add_argument("--rounds", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--feddyn_alpha", type=float, default=0.01)
    p.add_argument("--eval_every", type=int)
    p.add_argument("--dp_eps", type=float, default=None, help="total record-level epsilon")
    p.add_argument("--dp_delta", type=float, default=1e-5)
    p.add_argument("--dp_clip", type=float, default=1.0)
    p.add_argument("--dp_batch_size", type=int, default=64)
    p.add_argument("--eps_hist", type=float, default=0.5)
    p.add_argument("--ldi_rounds", type=int, nargs="*", default=[])
    p.add_argument("--mia", action="store_true")
    p.add_argument("--save_model", action="store_true")
    # resubmission protocol (docs/PLAN.md section 5); all off by default = submitted protocol
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic CUDA kernels (no cudnn.benchmark); repeat runs are bitwise identical")
    p.add_argument("--no_replacement", action="store_true",
                   help="partition without storing any training example twice (n_i <= S)")
    p.add_argument("--capacity", default=None,
                   help="declared public client capacity for DP-SGD accounting and sample weights: int or 'auto' (= floor(n/N))")
    p.add_argument("--val_frac", type=float, default=0.0,
                   help="stratified hold-out of the training set for model selection (reported as val_*)")
    p.add_argument("--val_seed", type=int, default=0,
                   help="seed of the validation split; fixed across run seeds so that example indices align across runs")
    p.add_argument("--subset_frac", type=float, default=None,
                   help="shadow federations: train on a random subset of this fraction of the training split")
    p.add_argument("--save_conf", action="store_true",
                   help="save the logit-scaled confidence of every training-split and test example (offline LiRA shadows)")
    p.add_argument("--lira_stats", default=None, help="npz with per-example OUT statistics for offline LiRA (MIA)")
    p.add_argument("--model", default=None, choices=[None, "default", "resnet18gn"])
    # multi-model baselines (feddac/multimodel.py); unset = defaults of the original papers
    p.add_argument("--ifca_k", type=int, default=None, help="IFCA: number of cluster models (default 3)")
    p.add_argument("--cfl_eps1", type=float, default=None, help="CFL: mean-update-norm threshold (default 0.4)")
    p.add_argument("--cfl_eps2", type=float, default=None, help="CFL: max-update-norm threshold (default 1.6)")
    p.add_argument("--cfl_warmup", type=int, default=None, help="CFL: first round at which a split may occur (default 20)")
    p.add_argument("--exp", default="default")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=os.environ.get("FEDDAC_RESULTS", "results"))
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    cfg = dict(DEFAULTS.get(a.dataset, {}))
    for k, v in vars(a).items():
        if v is not None and k not in ("out",):
            cfg[k] = v
    cfg["ratios"] = list(cfg["ratios"])
    if a.no_replacement and a.dp_eps and cfg.get("capacity") is None:
        cfg["capacity"] = "auto"  # unequal n_i: the DP mechanism must not depend on them
    if cfg.get("capacity") not in (None, "auto"):
        cfg["capacity"] = int(cfg["capacity"])
    rid = run_id(cfg)
    out_dir = os.path.join(a.out, a.exp)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, rid + ".json")
    if os.path.exists(out_json):
        print("exists, skipping", out_json)
        return
    done_list = os.path.join(os.path.dirname(__file__), "..", "configs", "done.txt")
    if os.path.exists(done_list) and f"results/{a.exp}/{rid}.json" in set(open(done_list).read().split()):
        print("already finished on another worker, skipping", rid)
        return
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before the first CUDA call
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.dataset == "femnist":  # natural partition by writer (feddac.data.load_femnist_federation)
        from feddac.data import load_femnist_federation
        ds, part, writers = load_femnist_federation(a.clients, a.seed, device)
        cfg["femnist_writers"] = [int(x) for x in writers[:5]] + ["..."]
    else:
        ds = load_dataset(a.dataset, device)
    labels = ds.y_train.cpu().numpy()
    if a.val_frac > 0 and a.dataset != "femnist":  # FEMNIST has per-writer validation splits
        tr, va = validation_split(labels, a.val_frac, seed=a.val_seed)
        tr_t, va_t = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
        ds.x_val, ds.y_val = ds.x_train[va_t], ds.y_train[va_t]
        ds.x_train, ds.y_train = ds.x_train[tr_t], ds.y_train[tr_t]
        labels = labels[tr]
    subset = None
    if a.subset_frac:
        # shadow federation: partition a random subset of the training split; indices refer to the full split
        subset = np.sort(np.random.default_rng(a.seed + 999).permutation(len(labels))[: int(a.subset_frac * len(labels))])
    if a.lt_rho and a.lt_rho > 1:
        assert subset is None, "--lt_rho and --subset_frac are exclusive"
        C_ = ds.num_classes
        n_max = int(np.bincount(labels, minlength=C_).max())
        lt_rng = np.random.default_rng(80_000)  # fixed across seeds: same long-tailed pool for every run
        keep = [lt_rng.permutation(np.where(labels == c)[0])[: int(round(n_max * a.lt_rho ** (-c / (C_ - 1))))]
                for c in range(C_)]
        subset = np.sort(np.concatenate(keep))
    if a.dataset != "femnist":
        part = partition(labels if subset is None else labels[subset], a.clients, tuple(a.ratios), a.alpha, ds.num_classes,
                         seed=a.seed, replacement=not a.no_replacement, size_sigma=a.size_sigma)
    if subset is not None:
        part.client_indices = [subset[np.asarray(ix)] for ix in part.client_indices]
    t0 = time.time()
    res, model, caps = run_federated(ds, part, cfg, device, capture_rounds=set(a.ldi_rounds))
    rng = np.random.default_rng(a.seed + 7)
    if a.ldi_rounds:
        tcfg = {k: cfg[k] for k in ("lr", "batch_size", "local_epochs") if k in cfg}
        res["ldi"] = []
        for cap in caps:
            if "units" in cap:
                recs = A.ldi_attack(ds, cap, part, tcfg, device, rng)
                for r_ in recs:
                    r_["round"] = cap["round"]
                res["ldi"].extend(recs)
    lira = None
    if a.lira_stats:
        z = np.load(a.lira_stats)
        lira = {k: torch.as_tensor(z[k], device=device) for k in z.files}
    if a.mia:
        res["mia"] = {"global": [], "unit": []}
        final_units = next((c["final_units"] for c in caps if "final_units" in c), [])
        for c in range(len(part.client_indices)):
            m = A.mia(model, ds, part.client_indices[c], device, rng, lira=lira)
            m.update(client=c, type=part.client_type[c])
            res["mia"]["global"].append(m)
        for u in final_units:
            member = np.concatenate([part.client_indices[c] for c in u["clients"]])
            for c in u["clients"]:
                m = A.mia(u["state"], ds, part.client_indices[c], device, rng, lira=lira)
                m.update(client=c, unit_size=len(u["clients"]), type=part.client_type[c])
                res["mia"]["unit"].append(m)
    if a.save_conf:
        cdir = os.path.join(a.out, "conf", a.exp)
        os.makedirs(cdir, exist_ok=True)
        np.savez_compressed(os.path.join(cdir, rid + ".npz"),
                            train=A.logit_conf(model, ds, ds.x_train, ds.y_train).cpu().numpy().astype(np.float32),
                            test=A.logit_conf(model, ds, ds.x_test, ds.y_test).cpu().numpy().astype(np.float32),
                            subset=subset if subset is not None else np.arange(ds.y_train.numel()))
        res["conf_file"] = f"conf/{a.exp}/{rid}.npz"
    if subset is not None:
        res["subset_size"] = int(len(subset))
    res["run_id"] = rid
    res["total_time_s"] = time.time() - t0
    res["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    res["torch"] = torch.__version__
    res["deterministic"] = bool(a.deterministic)
    if a.save_model:
        mdir = os.path.join(a.out, "models", a.exp)
        os.makedirs(mdir, exist_ok=True)
        sd = {k: (v.half() if ds.name in ("cifar100", "tinyimagenet") else v).cpu()
              for k, v in model.state_dict().items()}
        torch.save(sd, os.path.join(mdir, rid + ".pt"))
        res["model_file"] = f"models/{a.exp}/{rid}.pt"
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, out_json)
    print(f"done {rid}: acc={res['final_acc']:.2f} f1={res['final_f1']:.2f} ({res['total_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
