#!/usr/bin/env python
"""FedDAC-Order experiment (docs/ORDER_PLAN.md): hypotheses H1-H4, all arms at learning rate 0.02.

H1 (primary): FedDAC-Order vs Random+Seq, exact two-sided Wilcoxon, Holm over the two settings.
H2: FedDAC-Order vs Random-Order; H3: FedDAC-Order vs FedDAC; H4: FedDAC vs Random+Seq (unadjusted p).
Descriptive: last-five-round mean and round-to-round oscillation (std of round-to-round changes, last 20 rounds).

Usage: python scripts/analyse_L9.py <results dir with L9/ and L9_ref/> <tables dir> <summary.json>
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
runs = {}
for f in glob.glob(os.path.join(res, "L9", "*.json")) + glob.glob(os.path.join(res, "L9_ref", "*.json")):
    r = json.load(open(f))
    c = r["config"]
    if c["lr"] != 0.02 or not np.isfinite(r["final_acc"]):
        continue
    st = "N20" if tuple(c["ratios"]) == (0.1, 0.3, 0.6) else "N20x90"
    h = [x["acc"] for x in r["history"]][-20:]
    runs[(st, c["method"], c["seed"])] = {"final": r["final_acc"], "last5": r["last5_acc"], "osc": float(np.std(np.diff(h)))}

NAMES = {"feddac_order": "FedDAC-Order", "rand_order": "Random-Order", "feddac": "FedDAC", "rand_seq": "Random+Seq"}
SHORT = {"feddac_order": "F-Order", "rand_order": "R-Order", "feddac": "FedDAC", "rand_seq": "R+Seq"}
SET = {"N20": "$N=20$, default", "N20x90": r"$N=20$, 90\,\% ext."}
COMP = [("H1", "feddac_order", "rand_seq"), ("H2", "feddac_order", "rand_order"), ("H3", "feddac_order", "feddac"),
        ("H4", "feddac", "rand_seq")]
out = {"means": {}, "tests": {}}
for st in SET:
    for m in NAMES:
        v = [runs[(st, m, s)] for s in range(10) if (st, m, s) in runs]
        out["means"][f"{st}|{m}"] = {k: [float(np.mean([x[k] for x in v])), float(np.std([x[k] for x in v], ddof=1))] if len(v) > 1 else None
                                     for k in ("final", "last5", "osc")} | {"n": len(v)}
    for hid, a, b in COMP:
        seeds = [s for s in range(10) if (st, a, s) in runs and (st, b, s) in runs]
        t = paired_stats([runs[(st, a, s)]["final"] for s in seeds], [runs[(st, b, s)]["final"] for s in seeds])
        t5 = paired_stats([runs[(st, a, s)]["last5"] for s in seeds], [runs[(st, b, s)]["last5"] for s in seeds])
        t["n_pos"] = int(sum(runs[(st, a, s)]["final"] > runs[(st, b, s)]["final"] for s in seeds))
        out["tests"][f"{st}|{hid}"] = {"final": t, "last5": t5, "seeds": seeds}
ph = holm_fixed([out["tests"][f"{st}|H1"]["final"]["p"] for st in SET], m=2)
for st, p in zip(SET, ph):
    out["tests"][f"{st}|H1"]["p_holm"] = float(p)


def fmt_p(p):
    return "--" if p != p else ("$<$0.001" if p < 1e-3 else f"{p:.3f}")


# table 1: accuracy of the four arms
rows = []
for st, lab in SET.items():
    first = True
    for m, name in NAMES.items():
        d = out["means"][f"{st}|{m}"]
        if d["final"] is None:
            continue
        rows.append(f"{lab if first else ''} & {name} & {d['final'][0]:.2f}$\\pm${d['final'][1]:.2f} & "
                    f"{d['last5'][0]:.2f}$\\pm${d['last5'][1]:.2f} & {d['osc'][0]:.2f} \\\\")
        first = False
    rows.append(r"\midrule")
rows = rows[:-1]
# table rows: comparisons
crow = []
SET_SHORT = {"N20": "Default", "N20x90": r"90\,\% ext."}
for st, lab in SET_SHORT.items():
    first = True
    for hid, a, b in COMP:
        t = out["tests"][f"{st}|{hid}"]
        f, l5 = t["final"], t["last5"]
        ph_ = fmt_p(t.get("p_holm", float("nan")))
        crow.append(f"{lab if first else ''} & {hid}: {SHORT[a]} $-$ {SHORT[b]} & {f['mean_diff']:+.2f} [{f['ci_lo']:+.2f}, {f['ci_hi']:+.2f}] & "
                    f"{f['n_pos']}/{len(t['seeds'])} & {fmt_p(f['p'])} & {ph_} & {l5['mean_diff']:+.2f} [{l5['ci_lo']:+.2f}, {l5['ci_hi']:+.2f}] \\\\")
        first = False
    crow.append(r"\midrule")
crow = crow[:-1]
fix = lambda r: re.sub(r"(?<![-\w$])-(?=\d)", "$-$", r)  # noqa: E731
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Coverage-aware chain ordering (pre-registered experiment). CIFAR-10 CNN, 100 rounds, SGD with learning "
     r"rate 0.02 for every arm, seeds 0--9. Upper part: final test accuracy and mean accuracy of the last five rounds (\%), "
     r"mean $\pm$ standard deviation over seeds, and round-to-round oscillation (standard deviation of the change in test "
     r"accuracy between consecutive rounds over the last 20 rounds, mean over seeds). Lower part: paired differences in "
     r"final test accuracy with percentile bootstrap 95\,\% interval, number of seeds with a positive difference, exact "
     r"two-sided Wilcoxon $p$ and, for the primary hypothesis H1, Holm-adjusted $p$ over the two settings; last column: "
     r"paired difference in the last-five-round mean. F-Order: FedDAC-Order; R-Order: Random-Order; R+Seq: Random+Seq (runs of the main grid).}",
     r"\label{tab:order}", r"\footnotesize",
     r"\begin{tabular}{llccc}", r"\toprule", r"Setting & Method & Final accuracy & Last five rounds & Oscillation \\", r"\midrule"]
L += [fix(r) for r in rows] + [r"\bottomrule", r"\end{tabular}", "", r"\medskip", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
      r"\begin{tabular}{llccccc}", r"\toprule",
      r"Setting & Comparison & Difference & Pos. & $p$ & $p_{\mathrm{Holm}}$ & Last five rounds \\", r"\midrule"]
L += [fix(r) for r in crow] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(td, "order.tex"), "w").write("\n".join(L) + "\n")
json.dump(out, open(sp, "w"), indent=1)
for k, v in out["means"].items():
    print(k, v)
for k, v in out["tests"].items():
    f = v["final"]
    print(k, f"{f['mean_diff']:+.2f} [{f['ci_lo']:+.2f},{f['ci_hi']:+.2f}] p={f['p']:.3f}", "holm", v.get("p_holm"), "pos", f.get("n_pos", v['final'].get('n_pos')))
