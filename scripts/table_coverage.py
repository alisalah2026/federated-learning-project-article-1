#!/usr/bin/env python
"""LaTeX coverage table from grouping_analysis.json. Usage: table_coverage.py <json> <out.tex>"""
import json
import sys

d = json.load(open(sys.argv[1]))
names = {"dac": "DAC", "dac_priv": r"DAC, $\varepsilon_h=0.5$", "random": "Random", "fedse": "Label-balanced",
         "fedseq": "FedSeq superclients"}
lab = {"C10_N20_r0.1-0.3-0.6": r"10, 20", "C10_N50_r0.1-0.3-0.6": r"10, 50", "C10_N100_r0.1-0.3-0.6": r"10, 100",
       "C10_N20_r0.1-0-0.9": r"10, 20, 90\,\% ext.", "C100_N20_r0.1-0.3-0.6": "100, 20", "C100_N50_r0.1-0.3-0.6": "100, 50",
       "C200_N20_r0.1-0.3-0.6": "200, 20", "C200_N50_r0.1-0.3-0.6": "200, 50"}
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Grouping quality over 100 simulated federations per setting (partitions without duplicated examples, "
     r"$\alpha=0.2$, 10/30/60\,\% homogeneous/heterogeneous/extremely heterogeneous clients unless stated). "
     r"Full cov.: share of groups that contain every class; Min cov.: smallest class coverage among the groups of a "
     r"federation; KL: divergence of the group label distribution from uniform (nats); all averaged over federations. "
     r"The private variant groups on Laplace-noised histograms and is evaluated on the true histograms. "
     r"$\Delta$: paired difference in full coverage with respect to random groups, mean and percentile bootstrap 95\,\% interval.}",
     r"\label{tab:coverage}", r"\footnotesize", r"\begin{adjustbox}{max width=\linewidth}",
     r"\begin{tabular}{llccccc}", r"\toprule",
     r"Classes, $N$ & Grouping & Groups & Full cov. & Min cov. & KL & $\Delta$ full cov. vs.\ random \\", r"\midrule"]
first = True
for k, v in d["coverage"].items():
    if not first:
        L.append(r"\midrule")
    first = False
    for i, m in enumerate(["dac", "dac_priv", "random", "fedse", "fedseq"]):
        r = v[m]
        if m == "dac":
            dd = v["diff_dac_random_full_cov"]
        elif m == "dac_priv":
            dd = v["diff_priv_random_full_cov"]
        else:
            dd = None
        delta = f"{dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]" if dd else "--"
        L.append(f"{lab[k] if i == 0 else ''} & {names[m]} & {r['groups']:.1f} & {r['full_cov']:.3f} & "
                 f"{r['min_cov']:.3f} & {r['kl']:.3f} & {delta} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n")
