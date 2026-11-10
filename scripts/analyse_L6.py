#!/usr/bin/env python
"""Secondary GPU checks of docs/COVERAGE_PLAN.md sections 4-5: FedDAC vs Random+Seq in the selected scenario
(CIFAR-10 CNN, N = 50, every client holding two classes), seeds 0-9.

(a) L6: learning rates selected on the default setting (configs/alite_lr.json).
(b) L8: learning rates re-selected in this scenario on the validation split (L7, configs/alite_lr_ext.json);
    a (method, rate) pair already run in L6 is reused from L6.
For each condition: paired mean difference with percentile bootstrap 95 % interval and exact two-sided Wilcoxon
signed-rank test (scripts/seed_analysis.paired_stats) for final test accuracy and, descriptively, for the mean
test accuracy of the last five rounds. Coverage M0/M1 of the groups used in the runs is recomputed from the
stored groups and histograms.

Usage: python scripts/analyse_L6.py <results dir containing L6/ [L8/]> <tables dir> <summary.json>
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from seed_analysis import paired_stats  # noqa: E402

res, td, sp = sys.argv[1:4]


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if "final_acc" in r and np.isfinite(r["final_acc"]):
            out[(r["config"]["method"], r["config"]["lr"], r["config"]["seed"])] = r
    return out


def cov(r, tau=0.1):
    h = np.asarray(r["histograms"], float)
    T = np.stack([h[g].sum(0) for g in r["groups"]])
    n = T.sum(1, keepdims=True)
    C = h.shape[1]
    return float((T > 0).all(1).mean()), float((T >= tau * n / C).all(1).mean())


runs = load(os.path.join(res, "L6"))
runs.update(load(os.path.join(res, "L8")))
conds = [("a", "Learning rates selected on the default setting", json.load(open("configs/alite_lr.json")))]
if os.path.exists("configs/alite_lr_ext.json") and glob.glob(os.path.join(res, "L8", "*.json")):
    conds.append(("b", "Learning rates selected in this scenario", json.load(open("configs/alite_lr_ext.json"))))

out, rows = {}, []
for cid, title, lr in conds:
    fa, fr = lr["feddac"], lr["rand_seq"]
    seeds = [s for s in range(10) if ("feddac", fa, s) in runs and ("rand_seq", fr, s) in runs]
    o = {"lr": {"feddac": fa, "rand_seq": fr}, "seeds": seeds}
    rows.append(f"\\multicolumn{{6}}{{l}}{{({cid}) {title}: FedDAC {fa:g}, Random+Seq {fr:g}}} \\\\")
    for metric, name in (("final_acc", "Final test accuracy"), ("last5_acc", "Mean test accuracy, last five rounds")):
        a = [runs[("feddac", fa, s)][metric] for s in seeds]
        b = [runs[("rand_seq", fr, s)][metric] for s in seeds]
        st = paired_stats(a, b)
        o[metric] = {"feddac_mean": float(np.mean(a)), "feddac_sd": float(np.std(a, ddof=1)),
                     "rand_mean": float(np.mean(b)), "rand_sd": float(np.std(b, ddof=1)),
                     "n_negative": int(sum(x < y for x, y in zip(a, b))), **st}
        p = st["p"]
        pf = "$<$0.001" if p < 1e-3 else f"{p:.3f}"
        rows.append(f"\\quad {name} & {np.mean(a):.2f}$\\pm${np.std(a, ddof=1):.2f} & {np.mean(b):.2f}$\\pm${np.std(b, ddof=1):.2f}"
                    f" & {st['mean_diff']:+.2f} [{st['ci_lo']:+.2f}, {st['ci_hi']:+.2f}] & {st['n_eff']} & {pf} \\\\")
    out[cid] = o
    rows.append(r"\midrule")
# coverage of the groups: the grouping does not depend on the learning rate
seeds = out["a"]["seeds"]
lr = conds[0][2]
cf = [cov(runs[("feddac", lr["feddac"], s)]) for s in seeds]
cr = [cov(runs[("rand_seq", lr["rand_seq"], s)]) for s in seeds]
c = {"feddac_m0": float(np.mean([x[0] for x in cf])), "feddac_m1": float(np.mean([x[1] for x in cf])),
     "rand_m0": float(np.mean([x[0] for x in cr])), "rand_m1": float(np.mean([x[1] for x in cr]))}
out["coverage_runs"] = c
rows.append(f"Full coverage M0 of the groups & {c['feddac_m0']:.3f} & {c['rand_m0']:.3f} & {c['feddac_m0'] - c['rand_m0']:+.3f} & -- & -- \\\\")
rows.append(f"Thresholded coverage M1 of the groups & {c['feddac_m1']:.3f} & {c['rand_m1']:.3f} & {c['feddac_m1'] - c['rand_m1']:+.3f} & -- & -- \\\\")
rows = [re.sub(r"(?<![-\w])-(?=\d)", "$-$", r) for r in rows]
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Training check in the scenario with the largest coverage advantage of DAC (selected by the rule fixed "
     r"before the coverage analysis): CIFAR-10 CNN, $N=50$, every client holds two classes, 100 rounds. (a) Learning "
     r"rates selected on the default setting (Table~\ref{tab:lrsel}); (b) learning rates re-selected in this scenario on "
     r"the validation split, seed 0, rates $\{0.005, 0.01, 0.02, 0.05\}$. Mean $\pm$ standard deviation over seeds 0--9; "
     r"paired difference FedDAC $-$ Random+Seq with percentile bootstrap 95\,\% interval, number of non-zero differences "
     r"$n_{\mathrm{eff}}$ and exact two-sided Wilcoxon signed-rank $p$ (secondary tests, outside the confirmatory family). "
     r"Lower rows: coverage of the groups used in the runs, mean over seeds.}",
     r"\label{tab:l6}", r"\footnotesize", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lccccc}",
     r"\toprule", r"Quantity & FedDAC & Random+Seq & Difference & $n_{\mathrm{eff}}$ & $p$ \\", r"\midrule"]
L += rows + [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(os.path.join(td, "l6.tex"), "w").write("\n".join(L) + "\n")
json.dump(out, open(sp, "w"), indent=1)
for k in out:
    if k in ("a", "b"):
        print(k, out[k]["lr"], len(out[k]["seeds"]), {m: [round(out[k][m][x], 3) for x in ("mean_diff", "ci_lo", "ci_hi", "p")]
                                                      + [out[k][m]["n_negative"]] for m in ("final_acc", "last5_acc")})
print(out["coverage_runs"])
