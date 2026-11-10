#!/usr/bin/env python
"""Coverage measures M0-M7 over scenario families S1-S4 (plan: docs/COVERAGE_PLAN.md, CPU only).

For every scenario, 100 federations (partition seeds 0-99, no-replacement partitions) are grouped
by DAC, private DAC (eps_h = 0.5, measured on the true histograms), random groups, label-balanced
groups (FedSe) and FedSeq superclients. Paired differences DAC - random carry a percentile
bootstrap 95 % interval over federations (10,000 resamples).

Usage: python scripts/coverage_measures.py <out.json> [--n_fed 100]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feddac import clustering as CL  # noqa: E402
from feddac.data import partition  # noqa: E402

EPS_H, TAU, TAIL_FRAC, K_LAST = 0.5, 0.1, 0.3, 2
DROP_P, N_DROP, N_ORDER = (0.1, 0.2, 0.3), 50, 20
GROUP_SIZES = (2, 3, 4, 6, 8)
RULES = ["dac", "dac_priv", "random", "fedse", "fedseq"]
DEF_R = (0.1, 0.3, 0.6)


# ------------------------------------------------------------------------------------------ pools
def pool_labels(C, rho=1.0):
    per = 450 if C == 200 else 45_000 // C
    n = [int(round(per * rho ** (-c / (C - 1)))) if rho > 1 else per for c in range(C)]
    return np.repeat(np.arange(C), n), np.asarray(n)


def ratios_for(e):
    h = min(0.1, 1.0 - e)
    return (round(h, 2), round(1.0 - h - e, 2), round(e, 2))


# --------------------------------------------------------------------------------------- measures
def _H(p):
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum())


def _js2(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def measures(groups, hist, tail, rng, extended=False):
    """Returns the measures of one grouping on the true histograms."""
    hist = np.asarray(hist, dtype=np.float64)
    C = hist.shape[1]
    glob = hist.sum(0) / hist.sum()
    T = np.stack([hist[g].sum(0) for g in groups])            # (G, C)
    n = T.sum(1, keepdims=True)
    P = T / np.maximum(n, 1)
    pres = T > 0
    thr = T >= TAU * n / C
    out = {
        "m0_full": float(pres.all(1).mean()), "m0_min": float(pres.mean(1).min()),
        "m1_full": float(thr.all(1).mean()), "m1_mean": float(thr.mean(1).mean()),
        "m2_mean": float(np.mean([np.exp(_H(p)) / C for p in P])),
        "m2_min": float(np.min([np.exp(_H(p)) / C for p in P])),
        "m3_min": float(pres[:, tail].mean(0).min()), "m3_mean": float(pres[:, tail].mean(0).mean()),
        "m4_mean": float(np.mean([_js2(p, glob) for p in P])),
        "m4_max": float(np.max([_js2(p, glob) for p in P])),
        "groups": len(groups), "max_group": int(max(len(g) for g in groups)),
    }
    if extended:
        B = hist > 0
        for p in DROP_P:  # M5
            acc = []
            for _ in range(N_DROP):
                keep = rng.random(len(hist)) >= p
                acc.append(np.mean([bool(B[[c for c in g if keep[c]]].any(0).all()) if keep[g].any() else False
                                    for g in groups]))
            out[f"m5_p{p:g}"] = float(np.mean(acc))
        rec = []  # M7
        for g in groups:
            if len(g) == 1:
                rec.append(B[g[0]].mean())
                continue
            v = [B[rng.permutation(g)[-K_LAST:]].any(0).mean() for _ in range(N_ORDER)]
            rec.append(float(np.mean(v)))
        out["m7"] = float(np.mean(rec))
    return out


def tail_classes(pool_n):
    C = len(pool_n)
    if pool_n.min() == pool_n.max():
        return np.arange(C)
    k = max(1, int(round(TAIL_FRAC * C)))
    return np.argsort(pool_n, kind="stable")[:k]


def groupings(h, L, s, include_fedseq=True):
    rng = np.random.default_rng(s)
    G = {"dac": CL.dac_cluster(h, L),
         "dac_priv": CL.dac_cluster(CL.laplace_histograms(h, EPS_H, np.random.default_rng(10_000 + s)), L),
         "random": CL.random_groups(len(h), L, rng),
         "fedse": CL.label_balanced(h, L)}
    if include_fedseq:
        G["fedseq"] = CL.fedseq_superclients(h, rng)
    return G


# --------------------------------------------------------------------------------------- summary
def boot_ci(x, n=10_000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarise(rec):
    rules = list(rec)
    keys = [k for k in rec[rules[0]][0] if k not in ("groups", "max_group")]
    res = {r: {k: float(np.mean([x[k] for x in rec[r]])) for k in rec[r][0]} for r in rules}
    res["diff_dac_random"] = {k: boot_ci([a[k] - b[k] for a, b in zip(rec["dac"], rec["random"])]) for k in keys}
    res["diff_priv_random"] = {k: boot_ci([a[k] - b[k] for a, b in zip(rec["dac_priv"], rec["random"])])
                               for k in keys}
    return res


def scenario(C=10, N=50, ratios=DEF_R, rho=1.0, sigma=0.0, L=None, n_fed=100, extended=False,
             include_fedseq=True):
    labels, pool_n = pool_labels(C, rho)
    tail = tail_classes(pool_n)
    rules = RULES if include_fedseq else [r for r in RULES if r != "fedseq"]
    rec = {r: [] for r in rules}
    Lg = L or CL.num_clusters(N)
    for s in range(n_fed):
        part = partition(labels, N, ratios, 0.2, C, seed=s, replacement=False, size_sigma=sigma)
        h = part.histograms
        G = groupings(h, Lg, s, include_fedseq)
        mrng = np.random.default_rng(20_000 + s)
        for r, g in G.items():
            rec[r].append(measures(g, h, tail, np.random.default_rng(mrng.integers(1 << 31)), extended))
    res = summarise(rec)
    res["config"] = {"C": C, "N": N, "ratios": list(ratios), "rho": rho, "sigma": sigma, "L": Lg,
                     "n_fed": n_fed, "tail_classes": len(tail)}
    return res


def show(key, res):
    d = res["diff_dac_random"]
    print(f"{key:28s} m0 {d['m0_full'][0]:+.3f} m1 {d['m1_full'][0]:+.3f} [{d['m1_full'][1]:+.3f},{d['m1_full'][2]:+.3f}]"
          f" m2 {d['m2_mean'][0]:+.3f} m3 {d['m3_min'][0]:+.3f} m4 {d['m4_mean'][0]:+.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n_fed", type=int, default=100)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t0 = time.time()
    out = {"plan": "docs/COVERAGE_PLAN.md", "tau": TAU, "eps_h": EPS_H, "tail_frac": TAIL_FRAC,
           "k_last": K_LAST, "drop_p": list(DROP_P), "n_fed": a.n_fed}
    fams = a.only.split(",") if a.only else ["S1", "S2", "S3", "S4", "EXT"]
    if "S1" in fams:
        out["S1"] = {}
        for N in (20, 50):
            for e in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
                k = f"N{N}_e{e:g}"
                out["S1"][k] = scenario(N=N, ratios=ratios_for(e), n_fed=a.n_fed)
                show("S1 " + k, out["S1"][k])
    if "S2" in fams:
        out["S2"] = {}
        for N in (20, 50):
            for C in (10, 20, 50, 100, 200):
                k = f"N{N}_C{C}"
                out["S2"][k] = scenario(C=C, N=N, n_fed=a.n_fed)
                show("S2 " + k, out["S2"][k])
    if "S3" in fams:
        out["S3"] = {}
        for C in (10, 100):
            for rho in (1, 10, 100):
                k = f"C{C}_rho{rho}"
                out["S3"][k] = scenario(C=C, rho=float(rho), n_fed=a.n_fed)
                show("S3 " + k, out["S3"][k])
    if "S4" in fams:
        out["S4"] = {}
        for sg in (0.0, 0.5, 1.0):
            k = f"sigma{sg:g}"
            out["S4"][k] = scenario(sigma=sg, n_fed=a.n_fed)
            show("S4 " + k, out["S4"][k])
    if "EXT" in fams:  # M5, M7 at defaults and 90 % extreme; M6 group-size sweep
        out["M5M7"], out["M6"] = {}, {}
        for N in (20, 50, 100):
            for name, r in (("default", DEF_R), ("ext90", (0.1, 0.0, 0.9))):
                k = f"N{N}_{name}"
                out["M5M7"][k] = scenario(N=N, ratios=r, n_fed=a.n_fed, extended=True)
                show("M5M7 " + k, out["M5M7"][k])
                for s in GROUP_SIZES:
                    L = max(2, int(round(N / s)))
                    if L >= N:
                        continue
                    kk = f"{k}_s{s}"
                    out["M6"][kk] = scenario(N=N, ratios=r, L=L, n_fed=a.n_fed, include_fedseq=False)
                    show("M6 " + kk, out["M6"][kk])
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
