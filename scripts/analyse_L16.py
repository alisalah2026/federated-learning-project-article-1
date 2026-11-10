#!/usr/bin/env python
"""Modern baselines (docs/BASELINE_PLAN.md): H6 FedDAC-Order vs SCAFFOLD, H7 FedDAC-Order vs FedDyn in the four CIFAR-10
settings, seeds 0-9, exact two-sided Wilcoxon with Holm over the eight tests; descriptive comparison with Random+Seq.
Run sources: L16 (SCAFFOLD, FedDyn), L9/L10 (FedDAC-Order, lr 0.02), L9_ref/L10_ref (Random+Seq, lr 0.02).

Usage: python scripts/analyse_L16.py <results dir> <tables dir> <summary.json>
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
SETS = {"N20": (20, 0.6), "N50": (50, 0.6), "N100": (100, 0.6), "N20x90": (20, 0.9)}
LAB = {"N20": "$N=20$", "N50": "$N=50$", "N100": "$N=100$", "N20x90": "\\makecell[l]{$N=20$,\\\\ 90\\,\\% extreme}"}
NAMES = {"feddac_order": "FedDAC-Order", "scaffold": "SCAFFOLD", "feddyn": "FedDyn", "rand_seq": "Random+Seq"}
fix = lambda r: re.sub(r"(?<![-\w$])-(?=\d)", "$-$", r)  # noqa: E731

runs = {}
for d, ok in (("L16", ("scaffold", "feddyn")), ("L9", ("feddac_order",)), ("L10", ("feddac_order",)),
              ("L9_ref", ("rand_seq",)), ("L10_ref", ("rand_seq",))):
    for f in sorted(glob.glob(os.path.join(res, d, "*.json"))):
        r = json.load(open(f))
        c = r["config"]
        if c["method"] not in ok or c.get("dp_eps") or not np.isfinite(r["final_acc"]):
            continue
        if c["method"] in ("feddac_order", "rand_seq") and c.get("lr") != 0.02:
            continue
        st = next((k for k, (n, e) in SETS.items() if c["clients"] == n and abs(c["ratios"][2] - e) < 1e-9), None)
        if st is None:
            continue
        h = [x["acc"] for x in r["history"]]
        runs.setdefault((st, c["method"], c["seed"]), {"final": r["final_acc"], "last5": float(np.mean(h[-5:])),
                                                        "osc": float(np.std(np.diff(h[-20:]))), "lr": c.get("lr")})

out = {"means": {}, "tests": {}, "n_runs": len(runs)}
fam = []
for st in SETS:
    for m in NAMES:
        v = [runs[(st, m, s)] for s in range(10) if (st, m, s) in runs]
        if len(v) > 1:
            out["means"][f"{st}|{m}"] = {k: [float(np.mean([x[k] for x in v])), float(np.std([x[k] for x in v], ddof=1))]
                                         for k in ("final", "last5", "osc")} | {"n": len(v), "lr": v[0]["lr"]}
    for hid, a, b in (("H6", "feddac_order", "scaffold"), ("H7", "feddac_order", "feddyn"),
                      ("D1", "scaffold", "rand_seq"), ("D2", "feddyn", "rand_seq")):
        s = [x for x in range(10) if (st, a, x) in runs and (st, b, x) in runs]
        t = paired_stats([runs[(st, a, x)]["final"] for x in s], [runs[(st, b, x)]["final"] for x in s])
        t["n_pos"] = int(sum(runs[(st, a, x)]["final"] > runs[(st, b, x)]["final"] for x in s))
        out["tests"][f"{st}|{hid}"] = t | {"seeds": s}
        if hid in ("H6", "H7"):
            fam.append(f"{st}|{hid}")
for k, p in zip(fam, holm_fixed([out["tests"][k]["p"] for k in fam], m=8)):
    t = out["tests"][k]
    t["p_holm"] = float(p)
    t["decision"] = ("superior" if t["mean_diff"] > 0 else "inferior") if p < 0.05 else "no detected difference"
json.dump(out, open(sp, "w"), indent=1)

rows = []
for st in SETS:
    first = True
    for m in NAMES:
        d = out["means"].get(f"{st}|{m}")
        if d is None:
            continue
        rows.append(fix(f"{LAB[st] if first else ''} & {NAMES[m]} & {d['lr']:g} & {d['final'][0]:.2f}$\\pm${d['final'][1]:.2f} & "
                        f"{d['last5'][0]:.2f}$\\pm${d['last5'][1]:.2f} & {d['osc'][0]:.2f} \\\\"))
        first = False
    rows.append(r"\midrule")
crows = []
for st in SETS:
    first = True
    for hid, lab in (("H6", "F-Order $-$ SCAFFOLD"), ("H7", "F-Order $-$ FedDyn")):
        t = out["tests"][f"{st}|{hid}"]
        pf = lambda p: "$<$0.001" if p < 1e-3 else f"{p:.3f}"  # noqa: E731
        crows.append(fix(f"{LAB[st] if first else ''} & {hid}: {lab} & {t['mean_diff']:+.2f} [{t['ci_lo']:+.2f}, {t['ci_hi']:+.2f}] & "
                         f"{t['n_pos']}/{len(t['seeds'])} & {pf(t['p'])} & {pf(t['p_holm'])} \\\\"))
        first = False
    crows.append(r"\midrule")
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{FedDAC-Order and the drift-correcting parallel methods SCAFFOLD and FedDyn (prespecified). CIFAR-10 CNN, "
     r"100 rounds, SGD, seeds 0--9; learning rates of SCAFFOLD and FedDyn selected on the validation split, FedDAC-Order and "
     r"Random+Seq at 0.02. Differences: paired final test accuracy with percentile bootstrap 95\,\% interval and the number of "
     r"seeds with a positive difference; exact two-sided Wilcoxon test with Holm correction over the eight tests. "
     r"F-Order: FedDAC-Order.}",
     r"\label{tab:modern}", r"\footnotesize", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lllccc}", r"\toprule",
     r"Setting & Method & Rate & Final accuracy & Last five & Oscillation \\", r"\midrule"] + rows[:-1] + [
     r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", "", r"\medskip", r"\scriptsize",
     r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{llcccc}", r"\toprule",
     r"Setting & Comparison & Difference & Pos. & $p$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"] + crows[:-1] + [
     r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(os.path.join(td, "modern.tex"), "w").write("\n".join(L) + "\n")
for k in fam:
    t = out["tests"][k]
    print(k, f"{t['mean_diff']:+.2f} [{t['ci_lo']:+.2f},{t['ci_hi']:+.2f}] pos={t['n_pos']}/{len(t['seeds'])} holm={t['p_holm']:.4f} {t['decision']}")
