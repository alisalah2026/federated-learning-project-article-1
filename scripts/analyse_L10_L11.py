#!/usr/bin/env python
"""FedDAC-Order extension (docs/ORDER_PLAN.md section 6).

L10 (no DP; N = 50 and 100; seeds 0-9; all arms at lr 0.02): H1' FedDAC-Order vs Random+Seq with exact two-sided
Wilcoxon and Holm over the two settings; H2-H4 as in L9.
L11 (DP, L3 protocol; seeds 0-4): descriptive paired differences with percentile bootstrap 95 % intervals and the
number of positive seeds (no superiority claim: with five seeds the exact p cannot fall below 0.0625).

Usage: python scripts/analyse_L10_L11.py <results dir> <tables dir> <summary.json>
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
fix = lambda r: re.sub(r"(?<![-\w$])-(?=\d)", "$-$", r)  # noqa: E731


def fmt_p(p):
    return "--" if p != p else ("$<$0.001" if p < 1e-3 else f"{p:.3f}")


def load(dirs, key):
    runs = {}
    for d in dirs:
        for f in glob.glob(os.path.join(res, d, "*.json")):
            r = json.load(open(f))
            if not np.isfinite(r["final_acc"]):
                continue
            c = r["config"]
            k = key(c)
            if k is None:
                continue
            h = [x["acc"] for x in r["history"]][-20:]
            runs[(k, c["method"], c["seed"])] = {"final": r["final_acc"], "last5": r["last5_acc"],
                                                 "osc": float(np.std(np.diff(h)))}
    return runs


def compare(runs, st, a, b, seeds):
    s = [x for x in seeds if (st, a, x) in runs and (st, b, x) in runs]
    t = paired_stats([runs[(st, a, x)]["final"] for x in s], [runs[(st, b, x)]["final"] for x in s])
    t5 = paired_stats([runs[(st, a, x)]["last5"] for x in s], [runs[(st, b, x)]["last5"] for x in s])
    t["n_pos"] = int(sum(runs[(st, a, x)]["final"] > runs[(st, b, x)]["final"] for x in s))
    return {"final": t, "last5": t5, "seeds": s}


def means(runs, st, m, seeds):
    v = [runs[(st, m, x)] for x in seeds if (st, m, x) in runs]
    if len(v) < 2:
        return None
    return {k: [float(np.mean([x[k] for x in v])), float(np.std([x[k] for x in v], ddof=1))] for k in ("final", "last5", "osc")} | {"n": len(v)}


out = {"L10": {"means": {}, "tests": {}}, "L11": {"means": {}, "tests": {}}}
COMP = [("H1", "feddac_order", "rand_seq"), ("H2", "feddac_order", "rand_order"), ("H3", "feddac_order", "feddac"),
        ("H4", "feddac", "rand_seq")]

# ---------------------------------------------------------------- L10
r10 = load(["L10", "L10_ref"], lambda c: (f"N{c['clients']}" if (c["lr"] == 0.02 and not c.get("dp_eps")) else None))
S10 = {"N50": "$N=50$", "N100": "$N=100$"}
for st in S10:
    for m in NAMES:
        out["L10"]["means"][f"{st}|{m}"] = means(r10, st, m, range(10))
    for hid, a, b in COMP:
        out["L10"]["tests"][f"{st}|{hid}"] = compare(r10, st, a, b, range(10))
for st, p in zip(S10, holm_fixed([out["L10"]["tests"][f"{st}|H1"]["final"]["p"] for st in S10], m=2)):
    out["L10"]["tests"][f"{st}|H1"]["p_holm"] = float(p)

# ---------------------------------------------------------------- L11
r11 = load(["L11", "L11_ref"], lambda c: (f"{c['dataset']}|{c['dp_eps']:g}" if c.get("dp_eps") else None))
S11 = {"cifar10|1": r"CIFAR-10, $\varepsilon=1$", "cifar10|4": r"CIFAR-10, $\varepsilon=4$",
       "mnist|1": r"MNIST, $\varepsilon=1$", "mnist|4": r"MNIST, $\varepsilon=4$"}
C11 = COMP[:3]
for st in S11:
    for m in NAMES:
        out["L11"]["means"][f"{st}|{m}"] = means(r11, st, m, range(5))
    for hid, a, b in C11:
        out["L11"]["tests"][f"{st}|{hid}"] = compare(r11, st, a, b, range(5))


def acc_rows(blk, sets, short=False):
    rows = []
    for st, lab in sets.items():
        first = True
        for m, name in NAMES.items():
            d = out[blk]["means"].get(f"{st}|{m}")
            if d is None:
                continue
            rows.append(f"{lab if first else ''} & {name} & {d['final'][0]:.2f}$\\pm${d['final'][1]:.2f} & "
                        f"{d['last5'][0]:.2f}$\\pm${d['last5'][1]:.2f} & {d['osc'][0]:.2f} \\\\")
            first = False
        rows.append(r"\midrule")
    return [fix(r) for r in rows[:-1]]


def cmp_rows(blk, sets, comps, holm=True):
    rows = []
    for st, lab in sets.items():
        first = True
        for hid, a, b in comps:
            t = out[blk]["tests"][f"{st}|{hid}"]
            f, l5 = t["final"], t["last5"]
            cells = [lab if first else "", f"{hid}: {SHORT[a]} $-$ {SHORT[b]}",
                     f"{f['mean_diff']:+.2f} [{f['ci_lo']:+.2f}, {f['ci_hi']:+.2f}]", f"{f['n_pos']}/{len(t['seeds'])}"]
            if holm:
                cells += [fmt_p(f["p"]), fmt_p(t.get("p_holm", float("nan")))]
            cells += [f"{l5['mean_diff']:+.2f} [{l5['ci_lo']:+.2f}, {l5['ci_hi']:+.2f}]"]
            rows.append(" & ".join(cells) + r" \\")
            first = False
        rows.append(r"\midrule")
    return [fix(r) for r in rows[:-1]]


def write(name, caption, label, arows, crows, holm):
    head = (r"Setting & Comparison & Difference & Pos. & $p$ & $p_{\mathrm{Holm}}$ & Last five rounds \\" if holm
            else r"Setting & Comparison & Difference & Pos. & Last five rounds \\")
    L = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\footnotesize",
         r"\begin{tabular}{llccc}", r"\toprule", r"Setting & Method & Final accuracy & Last five rounds & Oscillation \\",
         r"\midrule"] + arows + [r"\bottomrule", r"\end{tabular}", "", r"\medskip", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                                  r"\begin{tabular}{ll" + ("ccccc" if holm else "ccc") + "}", r"\toprule", head, r"\midrule"]
    L += crows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(td, name), "w").write("\n".join(L) + "\n")


ABBR = "F-Order: FedDAC-Order; R-Order: Random-Order; R+Seq: Random+Seq."
write("order_scale.tex",
      r"Coverage-aware chain ordering with $N=50$ and $N=100$ clients (default composition, prespecified extension). "
      r"CIFAR-10 CNN, 100 rounds, SGD with learning rate 0.02 for every arm, seeds 0--9; Random+Seq runs are those of the "
      r"main grid. Quantities as in Table~\ref{tab:order}; Holm correction over the two settings for H1. " + ABBR,
      "tab:order_scale", acc_rows("L10", S10), cmp_rows("L10", S10, COMP), True)
write("order_dp.tex",
      r"Coverage-aware chain ordering under record-level DP (protocol of Table~\ref{tab:dp}: $N=20$, default composition, "
      r"Adam $10^{-3}$, declared capacity, $\varepsilon_h=0.5$ included in the budget). The order uses the noised "
      r"histograms held by the server. Seeds 0--4; FedDAC and Random+Seq runs are those of Table~\ref{tab:dp}. "
      r"Descriptive: paired difference in final test accuracy with percentile bootstrap 95\,\% interval and number of seeds "
      r"with a positive difference; with five seeds no test is reported. " + ABBR,
      "tab:order_dp", acc_rows("L11", S11), cmp_rows("L11", S11, C11, holm=False), False)
json.dump(out, open(sp, "w"), indent=1)
for blk in ("L10", "L11"):
    for k, v in out[blk]["tests"].items():
        f = v["final"]
        print(blk, k, f"{f['mean_diff']:+.2f} [{f['ci_lo']:+.2f},{f['ci_hi']:+.2f}] p={f['p']:.3f} holm={v.get('p_holm')} pos={f['n_pos']}/{len(v['seeds'])}")
    for k, v in out[blk]["means"].items():
        if v:
            print(blk, k, [round(x, 2) for x in v["final"]], "osc", round(v["osc"][0], 2))
