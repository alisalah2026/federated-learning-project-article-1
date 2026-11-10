#!/usr/bin/env python
"""Pre-registered replication (docs/REPLICATION_PLAN.md): H1 FedDAC-Order vs Random+Seq with exact two-sided Wilcoxon and
Holm over R1-R3 (m = 3); H2-H4 secondary; descriptive last-five and oscillation. Fixed before the replication runs.

Usage: python scripts/analyse_rep.py <results dir with L13/> <tables dir> <summary.json>
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
NAMES = {"feddac_order": "FedDAC-Order", "rand_order": "Random-Order", "feddac": "FedDAC", "rand_seq": "Random+Seq"}
SHORT = {"feddac_order": "F-Order", "rand_order": "R-Order", "feddac": "FedDAC", "rand_seq": "R+Seq"}
SETS = {"R1": ("cifar100", 50, r"\makecell[l]{CIFAR-100,\\ ResNet-18, $N=50$}"), "R2": ("femnist", 50, r"\makecell[l]{FEMNIST,\\ $N=50$ writers}"),
        "R3": ("femnist", 100, r"\makecell[l]{FEMNIST,\\ $N=100$ writers}")}
COMP = [("H1", "feddac_order", "rand_seq"), ("H2", "feddac_order", "rand_order"), ("H3", "feddac_order", "feddac"),
        ("H4", "feddac", "rand_seq")]
fix = lambda r: re.sub(r"(?<![-\w$])-(?=\d)", "$-$", r)  # noqa: E731
runs = {}
for f in glob.glob(os.path.join(res, "L13", "*.json")):
    r = json.load(open(f))
    if not np.isfinite(r["final_acc"]):
        continue
    c = r["config"]
    st = next((k for k, (d, n, _) in SETS.items() if d == c["dataset"] and n == c["clients"]), None)
    if st is None:
        continue
    h = [x["acc"] for x in r["history"]]
    k = 4 if c["dataset"] == "cifar100" else 20
    # last five evaluations as stated in REPLICATION_PLAN.md section 6 (correction 2026-10-28: the run field last5_acc
    # averages the evaluations within the last five rounds, which for CIFAR-100, evaluated every 5 rounds, is the final one)
    runs[(st, c["method"], c["seed"])] = {"final": r["final_acc"], "last5": float(np.mean(h[-5:])),
                                          "osc": float(np.std(np.diff(h[-k:]))), "lr": c["lr"]}
out = {"means": {}, "tests": {}, "lr": {}}
for st in SETS:
    for m in NAMES:
        v = [runs[(st, m, s)] for s in range(10) if (st, m, s) in runs]
        out["means"][f"{st}|{m}"] = ({k: [float(np.mean([x[k] for x in v])), float(np.std([x[k] for x in v], ddof=1))]
                                      for k in ("final", "last5", "osc")} | {"n": len(v)}) if len(v) > 1 else None
        if v:
            out["lr"][st] = v[0]["lr"]
    for hid, a, b in COMP:
        s_ = [s for s in range(10) if (st, a, s) in runs and (st, b, s) in runs]
        t = paired_stats([runs[(st, a, s)]["final"] for s in s_], [runs[(st, b, s)]["final"] for s in s_])
        t5 = paired_stats([runs[(st, a, s)]["last5"] for s in s_], [runs[(st, b, s)]["last5"] for s in s_])
        t["n_pos"] = int(sum(runs[(st, a, s)]["final"] > runs[(st, b, s)]["final"] for s in s_))
        out["tests"][f"{st}|{hid}"] = {"final": t, "last5": t5, "seeds": s_}
for st, p in zip(SETS, holm_fixed([out["tests"][f"{st}|H1"]["final"]["p"] for st in SETS], m=3)):
    out["tests"][f"{st}|H1"]["p_holm"] = float(p)


def fmt_p(p):
    return "--" if p != p else ("$<$0.001" if p < 1e-3 else f"{p:.3f}")


arows, crows = [], []
for st, (_, _, lab) in SETS.items():
    first = True
    for m, name in NAMES.items():
        d = out["means"][f"{st}|{m}"]
        if d is None:
            continue
        arows.append(f"{lab if first else ''} & {name} & {d['final'][0]:.2f}$\\pm${d['final'][1]:.2f} & "
                     f"{d['last5'][0]:.2f}$\\pm${d['last5'][1]:.2f} & {d['osc'][0]:.2f} \\\\")
        first = False
    arows.append(r"\midrule")
    first = True
    for hid, a, b in COMP:
        t = out["tests"][f"{st}|{hid}"]
        f, l5 = t["final"], t["last5"]
        crows.append(" & ".join([lab if first else "", f"{hid}: {SHORT[a]} $-$ {SHORT[b]}",
                                 f"{f['mean_diff']:+.2f} [{f['ci_lo']:+.2f}, {f['ci_hi']:+.2f}]", f"{f['n_pos']}/{len(t['seeds'])}",
                                 fmt_p(f["p"]), fmt_p(t.get("p_holm", float("nan"))),
                                 f"{l5['mean_diff']:+.2f} [{l5['ci_lo']:+.2f}, {l5['ci_hi']:+.2f}]"]) + r" \\")
        first = False
    crows.append(r"\midrule")
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Pre-registered replication of the coverage-aware chain order on CIFAR-100 with ResNet-18 (default "
     r"composition) and on FEMNIST with its natural partition by writer. SGD with the learning rate selected on the validation "
     r"split for Random+Seq (CIFAR-100: " + f"{out['lr'].get('R1', float('nan')):g}" + r"; FEMNIST: "
     + f"{out['lr'].get('R2', float('nan')):g}" + r"), common to all arms; seeds 0--9. Quantities as in Table~\ref{tab:order}; "
     r"Holm correction over the three settings for H1. CIFAR-100 is evaluated every five rounds, so its last-five mean "
     r"and oscillation refer to evaluations. F-Order: FedDAC-Order; R-Order: Random-Order; R+Seq: Random+Seq.}",
     r"\label{tab:rep}", r"\footnotesize", r"\begin{tabular}{llccc}", r"\toprule",
     r"Setting & Method & Final accuracy & Last five & Oscillation \\", r"\midrule"] + [fix(r) for r in arows[:-1]] + \
    [r"\bottomrule", r"\end{tabular}", "", r"\medskip", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
     r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{llccccc}", r"\toprule",
     r"Setting & Comparison & Difference & Pos. & $p$ & $p_{\mathrm{Holm}}$ & Last five \\", r"\midrule"] + \
    [fix(r) for r in crows[:-1]] + [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(os.path.join(td, "rep.tex"), "w").write("\n".join(L) + "\n")
json.dump(out, open(sp, "w"), indent=1)
for k, v in out["tests"].items():
    f = v["final"]
    print(k, f"{f['mean_diff']:+.2f} [{f['ci_lo']:+.2f},{f['ci_hi']:+.2f}] p={f['p']:.3f} holm={v.get('p_holm')} pos={f['n_pos']}/{len(v['seeds'])}")
