#!/usr/bin/env python
"""Tables and figure of the coverage-measure analysis (docs/COVERAGE_PLAN.md) from coverage_measures.json.

Usage: table_coverage_measures.py <coverage_measures.json> <tables_dir> <figures_dir>
Writes tables/covmeasures.tex (main text), tables/covmeasures_full.tex, tables/covmeasures_m6.tex,
tables/covmeasures_m5m7.tex (appendix) and figures/coverage_sweeps.pdf.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = json.load(open(sys.argv[1]))
TD, FD = sys.argv[2], sys.argv[3]
MEAS = [("m0_full", r"M0 full"), ("m1_full", r"M1 full"), ("m2_mean", r"M2 mean"), ("m3_min", r"M3 min"),
        ("m4_mean", r"M4 mean")]


def _n(x, digits):
    return f"{x:+.{digits}f}".replace("-", "$-$")


def ci(d, k, digits=3):
    m, lo, hi = d[k]
    return f"{_n(m, digits)} [{_n(lo, digits)}, {_n(hi, digits)}]"


def short(d, k, digits=3):
    m, lo, hi = d[k]
    star = r"$^{*}$" if (lo > 0 or hi < 0) else ""
    return f"{_n(m, digits)}{star}"


def label(fam, key, c):
    r = c["ratios"]
    if fam == "S1":
        return f"S1 extreme share {r[2]:g}, $N={c['N']}$"
    if fam == "S2":
        return f"S2 {c['C']} classes, $N={c['N']}$"
    if fam == "S3":
        return f"S3 {c['C']} classes, $\\rho={c['rho']:g}$"
    return f"S4 $\\sigma={c['sigma']:g}$"


def dig(k):
    return 4 if k.startswith("m4") else 3


# ---------------------------------------------------------------- main-text table (selected rows)
SEL = [("S1", "N50_e0"), ("S1", "N50_e0.6"), ("S1", "N50_e0.9"), ("S1", "N50_e1"), ("S1", "N20_e1"),
       ("S2", "N20_C100"), ("S2", "N50_C200"), ("S3", "C10_rho10"), ("S3", "C10_rho100"), ("S4", "sigma1")]
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Coverage measures of DAC relative to random groups in selected scenarios (100 federations each; "
     r"defaults $N=50$, ten classes, 10/30/60\,\% homogeneous/heterogeneous/extremely heterogeneous clients, "
     r"$\alpha=0.2$). Entries are paired mean differences DAC $-$ random; $^{*}$: percentile bootstrap 95\,\% "
     r"interval excludes zero. M0: share of groups containing every class; M1: the same with a class counted only if "
     r"the group holds at least 10\,\% of its uniform share; M2: effective number of classes, $\exp(H)/C$; M3: smallest "
     r"share of groups reached by a tail class; M4: Jensen--Shannon divergence to the federation label distribution "
     r"(lower is better). S1: share of clients with two classes; S3: imbalance ratio $\rho$ of the labels; S4: "
     r"log-normal spread $\sigma$ of client sizes. Intervals for all scenarios are given in Tables~\ref{tab:covmeasures_full}--\ref{tab:covmeasures_m5m7}.}",
     r"\label{tab:covmeasures}", r"\footnotesize", r"\begin{tabular}{lccccc}", r"\toprule",
     r"Scenario & $\Delta$M0 & $\Delta$M1 & $\Delta$M2 & $\Delta$M3 & $\Delta$M4 \\", r"\midrule"]
for fam, key in SEL:
    v = R[fam][key]
    d = v["diff_dac_random"]
    L.append(label(fam, key, v["config"]) + " & " + " & ".join(short(d, k, dig(k)) for k, _ in MEAS) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(TD, "covmeasures.tex"), "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- appendix: all scenarios with intervals
rows = []
seen = set()
for fam in ("S1", "S2", "S3", "S4"):
    for key, v in R[fam].items():
        c = v["config"]
        sig = (c["C"], c["N"], tuple(c["ratios"]), c["rho"], c["sigma"])
        if sig in seen:  # the default scenario appears in several families
            continue
        seen.add(sig)
        rows.append((fam, key, v))
L = [r"\begin{sidewaystable}", r"\centering",
     r"\caption{Coverage measures, all scenarios S1--S4: paired difference DAC $-$ random, mean and percentile bootstrap "
     r"95\,\% interval over 100 federations. M4 is lower for groups closer to the federation label distribution.}",
     r"\label{tab:covmeasures_full}", r"\scriptsize", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lccccc}", r"\toprule",
     r"Scenario & $\Delta$M0 & $\Delta$M1 & $\Delta$M2 & $\Delta$M3 & $\Delta$M4 \\", r"\midrule"]
prev = None
for fam, key, v in rows:
    if prev and prev != fam:
        L.append(r"\midrule")
    prev = fam
    d = v["diff_dac_random"]
    L.append(label(fam, key, v["config"]) + " & " + " & ".join(ci(d, k, dig(k)) for k, _ in MEAS) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{sidewaystable}", "",
      r"\begin{table}[htbp]", r"\centering",
      r"\caption{M1 full-coverage share of each grouping rule in scenarios S1--S4 (mean over 100 federations). Private "
      r"DAC groups on Laplace histograms ($\varepsilon_h=0.5$) and is evaluated on the true histograms; FedSeq "
      r"superclients form a different number of groups.}",
      r"\label{tab:covmeasures_rules}", r"\scriptsize", r"\begin{tabular}{lccccc}", r"\toprule",
      r"Scenario & DAC & DAC, $\varepsilon_h=0.5$ & Random & Label-balanced & FedSeq \\", r"\midrule"]
prev = None
for fam, key, v in rows:
    if prev and prev != fam:
        L.append(r"\midrule")
    prev = fam
    L.append(label(fam, key, v["config"]) + " & "
             + " & ".join(f"{v[m]['m1_full']:.3f}" for m in ("dac", "dac_priv", "random", "fedse", "fedseq")) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(TD, "covmeasures_full.tex"), "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- appendix: M6 group-size sweep
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Small-group coverage (M6): the number of groups is set to $L=\mathrm{round}(N/s)$ for a target group "
     r"size $s$; paired difference DAC $-$ random with percentile bootstrap 95\,\% interval (100 federations). "
     r"Default: 10/30/60\,\% composition; ext.: 10\,\% homogeneous and 90\,\% extremely heterogeneous clients.}",
     r"\label{tab:covmeasures_m6}", r"\scriptsize", r"\begin{adjustbox}{max width=\linewidth}",
     r"\begin{tabular}{llcccc}", r"\toprule",
     r"Setting & $s$ & $\Delta$M0 & $\Delta$M1 & $\Delta$M2 & $\Delta$M3 \\", r"\midrule"]
prev = None
for key, v in R["M6"].items():
    base, s = key.rsplit("_s", 1)
    N, comp = base.split("_")
    name = f"$N={N[1:]}$, " + ("default" if comp == "default" else "90\\,\\% ext.")
    if prev and prev != base:
        L.append(r"\midrule")
    d = v["diff_dac_random"]
    L.append(f"{name if prev != base else ''} & {s} & " + " & ".join(ci(d, k) for k in ("m0_full", "m1_full", "m2_mean", "m3_min"))
             + r" \\")
    prev = base
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(os.path.join(TD, "covmeasures_m6.tex"), "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- appendix: M5 dropout and M7 chain recency
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Coverage under dropout (M5: share of groups whose surviving members cover every class, each client "
     r"dropping with probability $p$) and chain recency coverage (M7: share of classes held by the last two clients of "
     r"a chain in random order), with $L=\mathrm{round}(\sqrt{N/2})$. DAC and random values, and the paired difference "
     r"with percentile bootstrap 95\,\% interval (100 federations).}",
     r"\label{tab:covmeasures_m5m7}", r"\scriptsize", r"\begin{adjustbox}{max width=\linewidth}",
     r"\begin{tabular}{lccccc}", r"\toprule",
     r"Setting & M5 DAC / random, $p=0.2$ & $\Delta$M5, $p=0.1$ & $\Delta$M5, $p=0.3$ & M7 DAC / random & $\Delta$M7 \\",
     r"\midrule"]
for key, v in R["M5M7"].items():
    N, comp = key.split("_")
    name = f"$N={N[1:]}$, " + ("default" if comp == "default" else "90\\,\\% ext.")
    d = v["diff_dac_random"]
    L.append(f"{name} & {v['dac']['m5_p0.2']:.3f} / {v['random']['m5_p0.2']:.3f} & {ci(d, 'm5_p0.1')} & "
             f"{ci(d, 'm5_p0.3')} & {v['dac']['m7']:.3f} / {v['random']['m7']:.3f} & {ci(d, 'm7')} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(os.path.join(TD, "covmeasures_m5m7.tex"), "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- figure: S1 and M6 sweeps
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
for N, mk in ((20, "o"), (50, "s")):
    rows = [(v["config"]["ratios"][2], v["diff_dac_random"]) for k, v in R["S1"].items() if v["config"]["N"] == N]
    for meas, ls, col in (("m0_full", "--", "#7a7a7a"), ("m1_full", "-", "#1f4e79")):
        x = [r[0] for r in rows]
        y = [r[1][meas][0] for r in rows]
        lo = [r[1][meas][1] for r in rows]
        hi = [r[1][meas][2] for r in rows]
        ax[0].plot(x, y, ls=ls, marker=mk, ms=3, color=col, lw=1,
                   label=f"{'M1' if meas == 'm1_full' else 'M0'}, $N={N}$")
        ax[0].fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
ax[0].axhline(0, color="black", lw=0.5)
ax[0].set_xlabel("Share of clients with two classes")
ax[0].set_ylabel("Full coverage, DAC $-$ random")
ax[0].legend(frameon=False, fontsize=6.5)
ax[0].set_title("(a) Extreme-skew sweep (S1)", fontsize=8)
for key_base, mk, col in (("N50_default", "s", "#1f4e79"), ("N50_ext90", "^", "#c55a11"),
                          ("N100_default", "o", "#548235"), ("N100_ext90", "v", "#7030a0")):
    rows = [(int(k.rsplit("_s", 1)[1]), v["diff_dac_random"]["m1_full"]) for k, v in R["M6"].items()
            if k.rsplit("_s", 1)[0] == key_base]
    x = [r[0] for r in rows]
    ax[1].plot(x, [r[1][0] for r in rows], marker=mk, ms=3, lw=1, color=col,
               label=key_base.replace("_default", ", default").replace("_ext90", ", 90% ext.").replace("N", "$N=") .replace(",", "$,", 1))
    ax[1].fill_between(x, [r[1][1] for r in rows], [r[1][2] for r in rows], color=col, alpha=0.12, lw=0)
ax[1].axhline(0, color="black", lw=0.5)
ax[1].set_xlabel("Target group size $s$")
ax[1].set_ylabel("M1 full coverage, DAC $-$ random")
ax[1].legend(frameon=False, fontsize=6.5)
ax[1].set_title("(b) Group-size sweep (M6)", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FD, "coverage_sweeps.pdf"))
print("written")
