#!/usr/bin/env python
"""Job grids of the reduced A-core plan (docs/PLAN.md section 7), one JSONL file per block.

Common protocol for every block: no-replacement partitions, 10 % stratified validation split,
deterministic CUDA kernels. CIFAR-10 uses plain SGD; its learning rate is selected per method on
the validation split in L2 and passed to L1 via --lr_file (JSON {method: lr}).

  L2  learning-rate selection: CIFAR-10 CNN, default setting, 3 rates x 9 methods, seed 0; plus two
      repeat runs (FedAvg, FedDAC at lr 0.02, tag 'rep') to check bitwise reproducibility on the GPU.
  L1  main comparison: CIFAR-10 CNN, 4 settings x seeds 0-4 x 9 methods; seeds 5-9 for FedDAC and
      Random+Seq (confirmatory contrast, docs/SEED_PROTOCOL.md); IID FedAvg reference at N = 20/50/100.
  L3  DP-SGD (declared capacity, eps_h = 0.5): CIFAR-10 and MNIST x eps {1, 4} x 5 seeds x 7 methods.
  L4  attacks (LDI at rounds 1, R/2, R; loss MIA): CIFAR-10 {non-DP, DP eps 4} and MNIST non-DP,
      3 seeds x 7 methods.
  L5  offline-LiRA shadows: 8 FedAvg federations on random halves of the CIFAR-10 training split
      (seeds 100-107), saving per-example confidences; scripts/lira_stats.py -> configs/lira_stats.npz,
      which L4 needs (CIFAR-10 runs; MNIST runs use the loss attack only).

  L6  coverage GPU check (docs/COVERAGE_PLAN.md section 4): FedDAC and Random+Seq, seeds 0-9, in the
      scenario selected by scripts/select_coverage_scenario.py (configs/coverage_selection.json).

  L7  learning-rate re-selection in the L6 scenario: FedDAC and Random+Seq, seed 0, 4 rates (docs/COVERAGE_PLAN.md s. 5).
  L8  FedDAC and Random+Seq at the L7 rates, seeds 0-9; (method, rate) pairs already run in L6 are reused.

  L9  FedDAC-Order, Random-Order and FedDAC at lr 0.02, N = 20 default and 90 % extreme, seeds 0-9 (docs/ORDER_PLAN.md).

Usage: python scripts/make_alite.py L2|L1|L3|L4|L5|L6|L7|L8|L9 [--lr_file f.json]
"""
import argparse
import json
import os

M7 = ["fedavg", "fedprox", "ppcfl", "fedse", "fedseq", "feddac", "rand_seq"]
M9 = M7 + ["ifca", "cfl"]
DEF = dict(clients=20, alpha=0.2, ratios=[0.1, 0.3, 0.6])
PROTO = ["--no_replacement", "--val_frac", "0.1", "--deterministic"]
SETTINGS = {  # the four settings of the confirmatory family (fixed before any result)
    "N20": dict(clients=20, alpha=0.2, ratios=[0.1, 0.3, 0.6]),
    "N50": dict(clients=50, alpha=0.2, ratios=[0.1, 0.3, 0.6]),
    "N100": dict(clients=100, alpha=0.2, ratios=[0.1, 0.3, 0.6]),
    "N20x90": dict(clients=20, alpha=0.2, ratios=[0.1, 0.0, 0.9]),
}
IID = dict(alpha=0.2, ratios=[1.0, 0.0, 0.0])  # all clients homogeneous (balanced class counts)


def job(exp, dataset, method, seed, extra, **kw):
    j = dict(DEF)
    j.update(kw)
    j.update(exp=exp, dataset=dataset, method=method, seed=seed, extra=list(PROTO) + list(extra))
    return j


def sgd(lr, tag=""):
    return ["--optimizer", "sgd", "--lr", str(lr), "--tag", f"sgd{lr}{tag}"]


def L2():
    jobs = [job("L2", "cifar10", m, 0, sgd(lr)) for lr in (0.01, 0.02, 0.05) for m in M9]
    jobs += [job("L2", "cifar10", m, 0, sgd(0.02, "rep")) for m in ("fedavg", "feddac")]
    return jobs


def L1(lr):
    jobs = []
    for name, s in SETTINGS.items():
        for seed in range(5):
            for m in M9:
                jobs.append(job("L1", "cifar10", m, seed, sgd(lr[m]), **s))
        for seed in range(5, 10):
            for m in ("feddac", "rand_seq"):
                jobs.append(job("L1", "cifar10", m, seed, sgd(lr[m]), **s))
    for n in (20, 50, 100):
        for seed in range(5):
            jobs.append(job("L1", "cifar10", "fedavg", seed, sgd(lr["fedavg"], "iid"), clients=n, **IID))
    return jobs


def L3():
    dp = ["--optimizer", "adam", "--lr", "0.001"]
    return [job("L3", d, m, seed, dp, dp_eps=eps)
            for d in ("cifar10", "mnist") for eps in (1.0, 4.0) for seed in range(5) for m in M7]


LIRA = ["--lira_stats", "configs/lira_stats.npz"]


def L4(lr):
    if not os.path.exists("configs/lira_stats.npz"):
        raise SystemExit("L4 needs configs/lira_stats.npz (scripts/lira_stats.py on the L5 shadow files)")
    jobs = []
    for seed in range(3):
        for m in M7:
            jobs.append(job("L4", "cifar10", m, seed, sgd(lr[m]) + LIRA, ldi_rounds=[1, 50, 100], mia=True))
            jobs.append(job("L4", "cifar10", m, seed, ["--optimizer", "adam", "--lr", "0.001"] + LIRA,
                            dp_eps=4.0, ldi_rounds=[1, 50, 100], mia=True))
            jobs.append(job("L4", "mnist", m, seed, [], ldi_rounds=[1, 25, 50], mia=True))
    return jobs


def L5(lr):
    return [job("L5", "cifar10", "fedavg", seed, sgd(lr["fedavg"], "shadow") + ["--subset_frac", "0.5", "--save_conf"])
            for seed in range(100, 108)]


def L6(lr):
    sel = json.load(open("configs/coverage_selection.json"))
    if not sel.get("run"):
        raise SystemExit("coverage selection: no scenario qualifies; L6 is not run")
    c = sel["config"]
    extra = []
    if c.get("rho", 1) > 1:
        extra += ["--lt_rho", str(c["rho"])]
    if c.get("sigma", 0) > 0:
        extra += ["--size_sigma", str(c["sigma"])]
    return [job("L6", "cifar10", m, seed, sgd(lr[m]) + extra, clients=c["N"], ratios=c["ratios"])
            for seed in range(10) for m in ("feddac", "rand_seq")]


EXT = dict(clients=50, alpha=0.2, ratios=[0.0, 0.0, 1.0])  # the L6 scenario (configs/coverage_selection.json)


def L7():
    return [job("L7", "cifar10", m, 0, sgd(lr), **EXT) for lr in (0.005, 0.01, 0.02, 0.05) for m in ("feddac", "rand_seq")]


def L8(lr):
    l6 = json.load(open("configs/alite_lr.json"))  # rates used in L6; those runs are reused
    return [job("L8", "cifar10", m, seed, sgd(lr[m]), **EXT)
            for seed in range(10) for m in ("feddac", "rand_seq") if lr[m] != l6[m]]


def L9():
    """FedDAC-Order experiment (docs/ORDER_PLAN.md): 3 new arms x 2 settings x seeds 0-9 at the common rate 0.02;
    Random+Seq at 0.02 is reused from L1."""
    return [job("L9", "cifar10", m, seed, sgd(0.02), **SETTINGS[st])
            for st in ("N20", "N20x90") for seed in range(10) for m in ("feddac_order", "rand_order", "feddac")]


def L10():
    """FedDAC-Order at N = 50 and N = 100 (docs/ORDER_PLAN.md section 6); Random+Seq at 0.02 reused from L1."""
    return [job("L10", "cifar10", m, seed, sgd(0.02), **SETTINGS[st])
            for st in ("N50", "N100") for seed in range(10) for m in ("feddac_order", "rand_order", "feddac")]


def L11():
    """FedDAC-Order and Random-Order under DP with the L3 protocol; FedDAC and Random+Seq reused from L3."""
    dp = ["--optimizer", "adam", "--lr", "0.001"]
    return [job("L11", d, m, seed, dp, dp_eps=eps)
            for d in ("cifar10", "mnist") for eps in (1.0, 4.0) for seed in range(5) for m in ("feddac_order", "rand_order")]


EH = (0.05, 0.1, 0.25)


def L14():
    """Histogram budget under DP (docs/ORDER_PLAN.md section 7): eps_total = 1, L3/L11 protocol.
    FedDAC-Order at eps_h in {0.05, 0.1, 0.25} (seeds 0-9) and at 0.5 (seeds 5-9; 0-4 in L11);
    Random+Seq seeds 5-9 (0-4 in L3); Random-Order at eps_h in {0.05, 0.1, 0.25}, seeds 0-4 (descriptive)."""
    dp = ["--optimizer", "adam", "--lr", "0.001"]
    eh = lambda e: ["--eps_hist", str(e), "--tag", f"eh{e:g}"]  # noqa: E731
    jobs = []
    for d in ("cifar10", "mnist"):
        jobs += [job("L14", d, "feddac_order", s, dp + eh(e), dp_eps=1.0) for e in EH for s in range(10)]
        jobs += [job("L14", d, "feddac_order", s, dp, dp_eps=1.0) for s in range(5, 10)]
        jobs += [job("L14", d, "rand_seq", s, dp, dp_eps=1.0) for s in range(5, 10)]
        jobs += [job("L14", d, "rand_order", s, dp + eh(e), dp_eps=1.0) for e in EH for s in range(5)]
    return jobs


MOD = ("scaffold", "feddyn")


def L15():
    """Modern baselines, stage 1 (docs/BASELINE_PLAN.md): learning rate of SCAFFOLD and FedDyn on the validation split,
    N = 20 default, seed 0, rates {0.01, 0.02, 0.05} (the L2 protocol)."""
    return [job("L15", "cifar10", m, 0, sgd(lr), **SETTINGS["N20"]) for lr in (0.01, 0.02, 0.05) for m in MOD]


def L16(lr):
    """Modern baselines, stage 2: SCAFFOLD and FedDyn at the rates selected in L15, four CIFAR-10 settings, seeds 0-9."""
    return [job("L16", "cifar10", m, seed, sgd(lr[m]), **SETTINGS[st])
            for st in ("N20", "N50", "N100", "N20x90") for seed in range(10) for m in MOD]


def L16F(lr):
    """Helper for L16 (owner decision 2026-10-28, ceiling USD 95): the FedDyn jobs at queue positions 13-20 of L16 shards
    1 and 3, which the original pods cannot finish within their time limit. Same configurations (exp L16), so the results
    are identical in kind to the original runs; the original FedDyn pods are stopped after their first 12 runs."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from worker import shard_jobs
    jobs = L16(lr)
    return [j for k in (1, 3) for j in shard_jobs(jobs, k, 4)[0][12:]]


REP = {"R1": dict(dataset="cifar100", clients=50, alpha=0.2, ratios=[0.1, 0.3, 0.6]),
       "R2": dict(dataset="femnist", clients=50, alpha=0.2, ratios=[0.1, 0.3, 0.6]),
       "R3": dict(dataset="femnist", clients=100, alpha=0.2, ratios=[0.1, 0.3, 0.6])}


def _rep_job(exp, st, m, seed, lr):
    d = dict(REP[st])
    ds = d.pop("dataset")
    return job(exp, ds, m, seed, sgd(lr), **d)


def L12():
    """Replication stage 1 (docs/REPLICATION_PLAN.md section 4): Random+Seq, seed 0, rates 0.01/0.02/0.05, R1 and R2."""
    return [_rep_job("L12", st, "rand_seq", 0, lr) for st in ("R1", "R2") for lr in (0.01, 0.02, 0.05)]


def L13(lr):
    """Replication stage 2: four arms x seeds 0-9 in R1-R3 at the stage-1 rate of each dataset
    (lr = {"cifar100": r, "femnist": r})."""
    return [_rep_job("L13", st, m, seed, lr[REP[st]["dataset"]])
            for st in ("R1", "R2", "R3") for seed in range(10) for m in ("feddac_order", "rand_order", "feddac", "rand_seq")]


def L12F():
    """Repeat of the two FEMNIST stage-1 runs that failed in L12 (concurrent-download race, fixed in feddac/data.py);
    repeated once with the same configuration, as the plan prescribes."""
    return [j for j in L12() if j["dataset"] == "femnist" and j["extra"][-1] != "sgd0.01"]


def L13C(lr):
    return [_rep_job("L13", "R1", m, seed, lr["cifar100"]) for seed in range(10)
            for m in ("feddac_order", "rand_order", "feddac", "rand_seq")]


def L13F(lr):
    return [_rep_job("L13", st, m, seed, lr["femnist"]) for st in ("R2", "R3") for seed in range(10)
            for m in ("feddac_order", "rand_order", "feddac", "rand_seq")]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("block", choices=["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L12F", "L13C", "L13F", "L14", "L15", "L16", "L16F"])
    p.add_argument("--lr_file")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    lr = json.load(open(a.lr_file)) if a.lr_file else None
    if a.block in ("L1", "L4", "L5", "L6", "L8", "L13", "L13C", "L13F", "L16", "L16F") and lr is None:
        raise SystemExit(f"{a.block} needs --lr_file (per-method learning rates selected in L2)")
    jobs = {"L1": lambda: L1(lr), "L2": L2, "L3": L3, "L4": lambda: L4(lr), "L5": lambda: L5(lr), "L6": lambda: L6(lr), "L7": L7, "L8": lambda: L8(lr), "L9": L9, "L10": L10, "L11": L11, "L12": L12, "L13": lambda: L13(lr), "L12F": L12F, "L13C": lambda: L13C(lr), "L13F": lambda: L13F(lr), "L14": L14, "L15": L15, "L16": lambda: L16(lr), "L16F": lambda: L16F(lr)}[a.block]()
    out = a.out or os.path.join("configs", f"alite_{a.block}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{a.block}: {len(jobs)} jobs -> {out}")
