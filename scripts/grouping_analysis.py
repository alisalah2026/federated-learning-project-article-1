#!/usr/bin/env python
"""Training-free analyses of the grouping step under the resubmission protocol (CPU only).

Protocol: no-replacement partitions (feddac.data.partition(replacement=False)), label pools of
the training split after the 10 % validation hold-out (CIFAR-10: 4,500 per class; CIFAR-100:
450 per class; 200 classes: 450 per class).

(1) Grouping quality per setting over 100 federations: share of groups with full true class
    coverage, mean and minimum class coverage, KL of the group label distribution to uniform,
    for DAC, private DAC (Laplace histograms, eps_h = 0.5; coverage measured on the true
    histograms), random groups, label-balanced groups (FedSe) and FedSeq superclients.
    Paired difference DAC - random and private DAC - random (same partition) with a percentile
    bootstrap 95 % interval over federations (10,000 resamples).
(2) Cluster-size side channel of DAC (2,000 federations, N in {20, 30, 40, 50}, extreme share
    0.3-0.9): mutual information between the size of a client's own group and its type, and
    between the sorted size vector and the number of extremely heterogeneous clients, both
    conditional on N, plus the Bayes accuracy of guessing the type from the own group size.

Usage: python scripts/grouping_analysis.py <out.json>
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feddac import clustering as CL  # noqa: E402
from feddac.data import partition  # noqa: E402

EPS_H = 0.5
SETTINGS = [  # (classes, per-class pool, N, ratios)
    (10, 4500, 20, (0.1, 0.3, 0.6)), (10, 4500, 50, (0.1, 0.3, 0.6)), (10, 4500, 100, (0.1, 0.3, 0.6)),
    (10, 4500, 20, (0.1, 0.0, 0.9)),
    (100, 450, 20, (0.1, 0.3, 0.6)), (100, 450, 50, (0.1, 0.3, 0.6)),
    (200, 450, 20, (0.1, 0.3, 0.6)), (200, 450, 50, (0.1, 0.3, 0.6)),
]
METHODS = ["dac", "dac_priv", "random", "fedse", "fedseq"]


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def stats(groups, hist):
    st = CL.cluster_stats(groups, hist)
    cov = [s["coverage"] for s in st]
    return {"full_cov": float(np.mean([c == 1.0 for c in cov])), "mean_cov": float(np.mean(cov)),
            "min_cov": float(np.min(cov)), "kl": float(np.mean([s["kl_uniform"] for s in st])),
            "groups": len(groups)}


def coverage_study(n_fed=100):
    out = {}
    for C, per, N, r in SETTINGS:
        labels = np.repeat(np.arange(C), per)
        rec = {m: [] for m in METHODS}
        for s in range(n_fed):
            part = partition(labels, N, r, 0.2, C, seed=s, replacement=False)
            h = part.histograms
            L = CL.num_clusters(N)
            rng = np.random.default_rng(s)
            G = {"dac": CL.dac_cluster(h, L),
                 "dac_priv": CL.dac_cluster(CL.laplace_histograms(h, EPS_H, np.random.default_rng(10_000 + s)), L),
                 "random": CL.random_groups(N, L, rng),
                 "fedse": CL.label_balanced(h, L),
                 "fedseq": CL.fedseq_superclients(h, rng)}
            for m, g in G.items():
                rec[m].append(stats(g, h))
        key = f"C{C}_N{N}_r{'-'.join(f'{x:g}' for x in r)}"
        res = {m: {k: float(np.mean([x[k] for x in v])) for k in v[0]} for m, v in rec.items()}
        for k in ("full_cov", "min_cov", "mean_cov"):
            res[f"diff_dac_random_{k}"] = boot_ci([a[k] - b[k] for a, b in zip(rec["dac"], rec["random"])])
            res[f"diff_priv_random_{k}"] = boot_ci([a[k] - b[k] for a, b in zip(rec["dac_priv"], rec["random"])])
        out[key] = res
        print(key, {m: round(res[m]["full_cov"], 3) for m in METHODS},
              "DAC-rand", [round(x, 3) for x in res["diff_dac_random_full_cov"]],
              "priv-rand", [round(x, 3) for x in res["diff_priv_random_full_cov"]], flush=True)
    return out


def mi_discrete(x, y):
    n = len(x)
    cx, cy, cxy = Counter(x), Counter(y), Counter(zip(x, y))
    mi = sum(c / n * np.log2((c / n) / ((cx[a] / n) * (cy[b] / n))) for (a, b), c in cxy.items())
    hy = -sum(c / n * np.log2(c / n) for c in cy.values())
    return float(mi), float(hy)


def size_channel(n_fed=2000):
    labels = np.repeat(np.arange(10), 4500)
    own_size, own_type, size_vec, n_ext = [], [], [], []
    rng = np.random.default_rng(0)
    for s in range(n_fed):
        N = int(rng.choice([20, 30, 40, 50]))
        e = float(rng.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        part = partition(labels, N, (0.1, round(0.9 - e, 1), e), 0.2, 10, seed=s, replacement=False)
        g = CL.dac_cluster(part.histograms, CL.num_clusters(N))
        ext, _, _ = CL.classify_clients(part.histograms)
        size_vec.append((N, tuple(sorted(len(x) for x in g))))
        n_ext.append((N, len(ext)))
        for grp in g:
            for c in grp:
                own_size.append((N, len(grp)))
                own_type.append((N, "extreme" if c in ext else "other"))
    mi1, h1 = mi_discrete(own_size, own_type)
    mi1b, _ = mi_discrete([x[0] for x in own_size], own_type)
    mi2, h2 = mi_discrete(size_vec, n_ext)
    mi2b, _ = mi_discrete([x[0] for x in size_vec], n_ext)

    def bayes(keys):
        best = {}
        for (k, t), c in Counter(zip(keys, own_type)).items():
            if c > best.get(k, (None, 0))[1]:
                best[k] = (t, c)
        return sum(c for _, c in best.values()) / len(own_type)

    return {"n_federations": n_fed,
            "type_given_own_size": {"mi_bits": mi1 - mi1b, "H_given_N_bits": h1 - mi1b,
                                    "bayes_acc": bayes(own_size), "prior_acc": bayes([x[0] for x in own_size])},
            "n_extreme_given_size_vector": {"mi_bits": mi2 - mi2b, "H_given_N_bits": h2 - mi2b}}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/analysis/grouping_analysis.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = {"protocol": "no-replacement partitions, 10 % validation hold-out", "eps_h": EPS_H,
           "coverage": coverage_study()}
    res["size_channel"] = size_channel()
    print(json.dumps(res["size_channel"], indent=1))
    json.dump(res, open(out, "w"), indent=1)
