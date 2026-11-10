#!/usr/bin/env python
"""Histogram budget of FedDAC-Order under record-level DP (docs/ORDER_PLAN.md section 7, block L14).

eps_total = 1, L3/L11 protocol (N = 20, default composition, Adam 1e-3, declared capacity), CIFAR-10 and MNIST.
Run sources: L14 (new), L11 (FedDAC-Order and Random-Order at eps_h = 0.5, seeds 0-4) and L11_ref (copies of the
L3 Random+Seq and FedDAC runs, seeds 0-4).

Selection (per dataset): eps_h* = the value in {0.05, 0.1, 0.25, 0.5} with the highest mean last-five validation
accuracy of FedDAC-Order over seeds 0-9; ties go to the smaller value. The test split is not used.
H5 (primary): FedDAC-Order at eps_h* minus Random+Seq, final test accuracy, seeds 0-9; exact two-sided Wilcoxon,
Holm over the two datasets (m = 2), percentile bootstrap 95 % interval.
Pre-stated reading of the deficit D (mean difference) relative to the eps_h = 0.5 deficit D0 (same seeds):
  closed   - the 95 % interval contains 0 and |D| <= 1 point;
  reduced  - not closed and |D| <= |D0| / 2;
  not reduced - otherwise.
Superiority of FedDAC-Order is claimed only if Holm p < 0.05 and D > 0.
Secondary (descriptive): every eps_h vs Random+Seq (FedDAC-Order seeds 0-9; Random-Order seeds 0-4).

Usage: python scripts/analyse_L14.py <results dir> <tables dir> <summary.json>
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from seed_analysis import holm_fixed, paired_stats  # noqa: E402

res, td, sp = sys.argv[1:4]
EH = (0.05, 0.1, 0.25, 0.5)
DS = {"cifar10": "CIFAR-10", "mnist": "MNIST"}
fix = lambda r: re.sub(r"(?<![-\w$])-(?=\d)", "$-$", r)  # noqa: E731

runs = {}
for d in ("L14", "L11", "L11_ref"):
    for f in sorted(glob.glob(os.path.join(res, d, "*.json"))):
        r = json.load(open(f))
        c = r["config"]
        if c.get("dp_eps") != 1.0 or c["method"] not in ("feddac_order", "rand_order", "rand_seq"):
            continue
        if not np.isfinite(r["final_acc"]):
            continue
        eh = 0.0 if c["method"] == "rand_seq" else float(r["dp"]["eps_hist"])
        key = (c["dataset"], c["method"], round(eh, 4), c["seed"])
        runs.setdefault(key, {"final": r["final_acc"], "val5": r["last5_val_acc"], "last5": r["last5_acc"],
                              "eps_sgd": r["dp"]["eps_sgd"], "src": d})


def vals(ds, m, eh, seeds, k="final"):
    return [runs[(ds, m, eh, s)][k] for s in seeds]


def have(ds, m, eh, seeds):
    return [s for s in seeds if (ds, m, eh, s) in runs]


out = {"selection": {}, "H5": {}, "secondary": {}, "means": {}, "n_runs": len(runs)}
pvals = []
for ds in DS:
    v = {}
    for eh in EH:
        s = have(ds, "feddac_order", eh, range(10))
        if s:
            v[eh] = float(np.mean(vals(ds, "feddac_order", eh, s, "val5")))
    best = max(v.values())
    sel = min(e for e in v if v[e] == best)
    out["selection"][ds] = {"val5_mean": {str(k): x for k, x in v.items()}, "eps_h_star": sel}
    seeds = [s for s in have(ds, "feddac_order", sel, range(10)) if (ds, "rand_seq", 0.0, s) in runs]
    t = paired_stats(vals(ds, "feddac_order", sel, seeds), vals(ds, "rand_seq", 0.0, seeds))
    s0 = [s for s in have(ds, "feddac_order", 0.5, range(10)) if (ds, "rand_seq", 0.0, s) in runs]
    t0 = paired_stats(vals(ds, "feddac_order", 0.5, s0), vals(ds, "rand_seq", 0.0, s0))
    out["H5"][ds] = {"eps_h_star": sel, "seeds": seeds, "test": t, "ref_eps_h_0.5": t0}
    pvals.append(t["p"])
    for m, sd in (("feddac_order", range(10)), ("rand_order", range(5))):
        for eh in EH:
            s = [x for x in have(ds, m, eh, sd) if (ds, "rand_seq", 0.0, x) in runs]
            if len(s) >= 2:
                tt = paired_stats(vals(ds, m, eh, s), vals(ds, "rand_seq", 0.0, s))
                tt["n_pos"] = int(sum(a > b for a, b in zip(vals(ds, m, eh, s), vals(ds, "rand_seq", 0.0, s))))
                out["secondary"][f"{ds}|{m}|{eh:g}"] = tt | {"seeds": s}
    for m, eh, sd in [("rand_seq", 0.0, range(10))] + [(m, e, range(10)) for m in ("feddac_order", "rand_order") for e in EH]:
        s = have(ds, m, eh, sd)
        if len(s) >= 2:
            a = vals(ds, m, eh, s)
            out["means"][f"{ds}|{m}|{eh:g}"] = {"final": [float(np.mean(a)), float(np.std(a, ddof=1))], "n": len(s),
                                                "eps_sgd": runs[(ds, m, eh, s[0])]["eps_sgd"]}

for ds, p in zip(DS, holm_fixed(pvals, m=2)):
    h = out["H5"][ds]
    h["p_holm"] = float(p)
    D, D0, t = h["test"]["mean_diff"], h["ref_eps_h_0.5"]["mean_diff"], h["test"]
    if t["ci_lo"] <= 0 <= t["ci_hi"] and abs(D) <= 1.0:
        h["reading"] = "closed"
    elif abs(D) <= abs(D0) / 2:
        h["reading"] = "reduced"
    else:
        h["reading"] = "not reduced"
    h["superior"] = bool(p < 0.05 and D > 0)

json.dump(out, open(sp, "w"), indent=1)

# ---------------------------------------------------------------- table
NAME = {"feddac_order": "FedDAC-Order", "rand_order": "Random-Order", "rand_seq": "Random+Seq"}
rows = []
for ds, lab in DS.items():
    first = True
    for m, eh in [("rand_seq", 0.0)] + [(m, e) for m in ("feddac_order", "rand_order") for e in EH]:
        mm = out["means"].get(f"{ds}|{m}|{eh:g}")
        if mm is None:
            continue
        sec = out["secondary"].get(f"{ds}|{m}|{eh:g}")
        diff = "--" if sec is None else f"{sec['mean_diff']:+.2f} [{sec['ci_lo']:+.2f}, {sec['ci_hi']:+.2f}] ({sec['n_pos']}/{len(sec['seeds'])})"
        star = r"$^{*}$" if (m == "feddac_order" and eh == out["selection"][ds]["eps_h_star"]) else ""
        rows.append(fix(f"{lab if first else ''} & {NAME[m]}{star} & {'--' if m == 'rand_seq' else f'{eh:g}'} & "
                        f"{mm['eps_sgd']:g} & {mm['n']} & {mm['final'][0]:.2f}$\\pm${mm['final'][1]:.2f} & {diff} \\\\"))
        first = False
    h = out["H5"][ds]
    rows.append(fix(f" & \\multicolumn{{6}}{{l}}{{H5 ($\\varepsilon_h^*={h['eps_h_star']:g}$): $p={h['test']['p']:.3f}$, "
                    f"$p_{{\\mathrm{{Holm}}}}={h['p_holm']:.3f}$; reading: {h['reading']}}} \\\\"))
    rows.append(r"\midrule")
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Histogram budget of FedDAC-Order under record-level DP at $\varepsilon=1$ (prespecified, "
     r"Section~7 of the order plan). The total budget is split as $\varepsilon=\varepsilon_h+\varepsilon_{\mathrm{sgd}}$; "
     r"Random+Seq releases no histograms. Difference: paired difference to Random+Seq in final test accuracy with "
     r"percentile bootstrap 95\,\% interval and the number of seeds with a positive difference. $^{*}$: $\varepsilon_h$ "
     r"selected on the validation split.}",
     r"\label{tab:dp_eh}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llccccc}", r"\toprule",
     r"Dataset & Method & $\varepsilon_h$ & $\varepsilon_{\mathrm{sgd}}$ & Seeds & Final accuracy & Difference to Random+Seq \\",
     r"\midrule"] + rows[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(td, "dp_eh.tex"), "w").write("\n".join(L) + "\n")
for ds in DS:
    h = out["H5"][ds]
    print(ds, "eps_h*", h["eps_h_star"], "D", round(h["test"]["mean_diff"], 2), "D0", round(h["ref_eps_h_0.5"]["mean_diff"], 2),
          "p_holm", round(h["p_holm"], 4), h["reading"])
