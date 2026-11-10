#!/usr/bin/env python
"""LaTeX tables of the manuscript from the A-lite result files (L1-L5). No number is typed by hand.

Usage: python scripts/make_tables.py --res <dir with L1/ L3/ L4/ subdirs> --lrsel <lr_report.json> --out <tables dir>
Tables are written only for blocks whose results exist; summary statistics go to <out>/summary.json.
Grouping key of a setting: dataset, N, client composition, alpha, DP epsilon (the learning rate is a
per-method choice and is not part of the setting).
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from seed_analysis import confirmatory, paired_stats  # noqa: E402

NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "ppcfl": "PP-CFL", "ifca": "IFCA", "cfl": "CFL",
         "fedse": r"FedSe$^\dagger$", "fedseq": r"FedSeq$^\dagger$", "rand_seq": "Random+Seq", "feddac": "FedDAC"}
ORDER = ["fedavg", "fedprox", "ppcfl", "ifca", "cfl", "fedse", "fedseq", "rand_seq", "feddac"]
FAMILY = ["N=20", "N=50", "N=100", "N=20, 90% ext."]


def key(c):
    r = tuple(c["ratios"])
    lab = f"N={c['clients']}" + (", 90% ext." if r == (0.1, 0.0, 0.9) else "") + (", IID" if r == (1.0, 0.0, 0.0) else "")
    return (c["dataset"], lab, c.get("dp_eps"))


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if "config" in r and "final_acc" in r and np.isfinite(r["final_acc"]):
            out.append(r)
    return out


def ms(v, digits=2):
    v = np.asarray(v, float)
    return f"{v.mean():.{digits}f}$\\pm${v.std(ddof=1):.{digits}f}" if len(v) > 1 else (f"{v.mean():.{digits}f}" if len(v) else "--")


def ci(st):
    return f"{st['mean_diff']:+.2f} [{st['ci_lo']:+.2f}, {st['ci_hi']:+.2f}]"


def fmt_p(p):
    return "--" if p != p else ("$<$0.001" if p < 1e-3 else f"{p:.3f}")


def table(caption, label, cols, header, rows, note=None):
    rows = [r.replace("90% ext.", r"90\,\% ext.") for r in rows]
    L = [r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\footnotesize",
         r"\begin{adjustbox}{max width=\linewidth}", f"\\begin{{tabular}}{{{cols}}}", r"\toprule", header + r" \\", r"\midrule"]
    L += rows
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        L.append(f"\\\\[2pt]{{\\scriptsize {note}}}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def lrsel_table(rep):
    rows = []
    by = defaultdict(dict)
    for r in rep["rows"]:
        by[r["method"]][r["lr"]] = r
    lrs = sorted({r["lr"] for r in rep["rows"]})
    for m in [x for x in ORDER if x in by]:
        cells = []
        for x in lrs:
            r = by[m].get(x)
            cells.append("--" if r is None else (f"\\textbf{{{r['last5_val_acc']:.2f}}}" if r["selected"] else f"{r['last5_val_acc']:.2f}"))
        rows.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
    n_id = sum(1 for x in rep["repeats"] if x.get("identical"))
    note = (f"Repeat check: {n_id} of {len(rep['repeats'])} repeated configurations reproduced every per-round metric "
            f"exactly on the same GPU model.")
    return table(r"Learning-rate selection on the validation split (CIFAR-10, $N=20$, default composition, seed 0): mean "
                 r"validation accuracy (\%) over the last five rounds. The selected rate per method is in bold.",
                 "tab:lrsel", "l" + "c" * len(lrs), "Method & " + " & ".join(f"$\\eta={x:g}$" for x in lrs), rows, note)


def main_tables(runs, out, summary):
    by = defaultdict(lambda: defaultdict(dict))
    pers = defaultdict(lambda: defaultdict(dict))
    iid = defaultdict(list)
    for r in runs:
        c = r["config"]
        ds, lab, _ = key(c)
        if lab.endswith("IID"):
            iid[lab.replace(", IID", "")].append(r["final_acc"])
            continue
        by[lab][c["method"]][c["seed"]] = r["final_acc"]
        pers[lab][c["method"]][c["seed"]] = r.get("final_personal_acc")
    rows, s_main = [], {}
    for s in [x for x in FAMILY if x in by]:
        first = True
        for m in [x for x in ORDER if x in by[s]]:
            seeds = [k for k in range(5) if k in by[s][m]]
            acc = [by[s][m][k] for k in seeds]
            pa = [pers[s][m][k] for k in seeds if pers[s][m].get(k) is not None]
            ref = by[s].get("fedavg", {})
            common = [k for k in seeds if k in ref]
            d = ci(paired_stats(acc and [by[s][m][k] for k in common], [ref[k] for k in common])) if m != "fedavg" and len(common) > 1 else "--"
            rows.append(f"{s if first else ''} & {NAMES[m]} & {ms(acc)} & {ms(pa)} & {d} \\\\")
            s_main[f"{s}|{m}"] = dict(n=len(acc), acc=float(np.mean(acc)) if acc else None)
            first = False
        if iid.get(s):
            rows.append(f" & FedAvg, IID reference & {ms(iid[s])} & -- & -- \\\\")
        rows.append(r"\midrule")
    if rows:
        rows = rows[:-1]
        open(os.path.join(out, "main.tex"), "w").write(table(
            r"CIFAR-10 test accuracy and personalized accuracy (\%), mean $\pm$ standard deviation over seeds 0--4, and paired "
            r"difference in test accuracy to FedAvg (mean and bootstrap 95\,\% interval, exploratory). Default composition "
            r"10/30/60\,\%; 90\,\% ext.: 10/0/90\,\%. $^\dagger$Re-implemented from the published description.",
            "tab:main", "llccc", r"Setting & Method & Accuracy & Personalized acc. & $\Delta$ to FedAvg", rows))
    summary["main"] = s_main
    # confirmatory contrast and exploratory grouping-rule comparisons
    if all(s in by for s in FAMILY):
        conf = confirmatory(by, FAMILY, "feddac", "rand_seq")
        rows = []
        for c in conf:
            s = c["setting"]
            ex = []
            for m in ("fedse", "fedseq"):
                common = [k for k in range(5) if k in by[s].get(m, {}) and k in by[s]["feddac"]]
                ex.append(ci(paired_stats([by[s]["feddac"][k] for k in common], [by[s][m][k] for k in common]))
                          if len(common) > 1 else "--")
            rows.append(f"{s} & {len(c['seeds'])} & {c['mean_diff']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}] & "
                        f"{c['n_eff']} & {fmt_p(c['p'])} & {fmt_p(c['p_holm'])} & {ex[0]} & {ex[1]} \\\\")
        open(os.path.join(out, "confirm.tex"), "w").write(table(
            r"Effect of the grouping rule with sequential training held fixed. Confirmatory contrast FedDAC minus Random+Seq "
            r"in test accuracy (percentage points): pairs of seeds 0--9, mean paired difference with bootstrap 95\,\% interval, "
            r"number of non-zero differences $n_{\mathrm{eff}}$, exact two-sided Wilcoxon signed-rank $p$ and Holm-adjusted $p$ "
            r"over the four settings. Exploratory: FedDAC minus FedSe and minus FedSeq over seeds 0--4.",
            "tab:confirm", "lccccccc",
            r"Setting & Pairs & FedDAC $-$ Random+Seq & $n_{\mathrm{eff}}$ & $p$ & $p_{\mathrm{Holm}}$ & FedDAC $-$ FedSe & FedDAC $-$ FedSeq",
            rows))
        summary["confirmatory"] = conf


def cost_table(runs, out, summary, target=50.0):
    agg = defaultdict(lambda: defaultdict(list))
    for r in runs:
        c = r["config"]
        _, lab, _ = key(c)
        if lab.endswith("IID") or c["seed"] > 4:
            continue
        hist = r["history"]
        hit = next((h["round"] for h in hist if h["acc"] >= target), None)
        R = len(hist)
        cp = r["cost"]["simulated_latency_s"] / max(R, 1)
        agg[(lab, c["method"])]["comp"].append(r["cost"]["client_compute_s"] / max(R, 1))
        agg[(lab, c["method"])]["cp"].append(cp)
        agg[(lab, c["method"])]["tx"].append(r["cost"]["model_transfers"] / max(R, 1))
        agg[(lab, c["method"])]["hit"].append(hit)
    rows, s = [], {}
    for lab in [x for x in FAMILY if any(k[0] == x for k in agg)]:
        first = True
        for m in [x for x in ORDER if (lab, x) in agg]:
            a = agg[(lab, m)]
            hits = [h for h in a["hit"] if h is not None]
            hit_s = f"{np.mean(hits):.0f} ({len(hits)}/{len(a['hit'])})" if hits else f"-- (0/{len(a['hit'])})"
            rows.append(f"{lab if first else ''} & {NAMES[m]} & {np.mean(a['comp']):.1f} & {np.mean(a['cp']):.1f} & "
                        f"{np.mean(a['tx']):.0f} & {hit_s} \\\\")
            s[f"{lab}|{m}"] = {k: float(np.mean(v)) for k, v in a.items() if k != "hit"}
            first = False
        rows.append(r"\midrule")
    if rows:
        open(os.path.join(out, "cost.tex"), "w").write(table(
            r"Cost per round on CIFAR-10 (mean over seeds 0--4): total client computation and critical path (seconds, from the "
            r"measured local training times; the critical path is the longest group chain, or the slowest client for parallel "
            f"methods), models transferred, and round at which the test accuracy first reaches {target:.0f}\\,\\% (mean over the "
            r"seeds that reach it; number of such seeds in parentheses). Measured times include the sharing of one GPU by four "
            r"concurrent runs and are comparable within a table row group only.",
            "tab:cost", "llcccc", r"Setting & Method & Computation (s) & Critical path (s) & Models sent & Round to target",
            rows[:-1]))
    summary["cost"] = s


def dp_table(runs, out, summary):
    by = defaultdict(lambda: defaultdict(list))
    sig = defaultdict(list)
    for r in runs:
        c = r["config"]
        by[(c["dataset"], c["dp_eps"])][c["method"]].append(r["final_acc"])
        if r.get("dp"):
            sig[(c["dataset"], c["dp_eps"])].append(r["dp"]["sigma_max"])
    rows, s = [], {}
    for k in sorted(by):
        first = True
        for m in [x for x in ORDER if x in by[k]]:
            rows.append(f"{(k[0].upper().replace('CIFAR10', 'CIFAR-10') + ', ' + f'$\\varepsilon={k[1]:g}$') if first else ''} & "
                        f"{NAMES[m]} & {ms(by[k][m])} \\\\")
            s[f"{k[0]}|{k[1]}|{m}"] = float(np.mean(by[k][m]))
            first = False
        rows.append(r"\midrule")
    if rows:
        open(os.path.join(out, "dp.tex"), "w").write(table(
            r"Test accuracy (\%) under record-level $(\varepsilon,10^{-5})$-DP, mean $\pm$ standard deviation over seeds 0--4. "
            r"Partitions without duplicated examples, declared capacity $S_i=\lfloor n/N\rfloor$, $N=20$, default composition; the "
            r"budget of methods that use label statistics includes their release ($\varepsilon_h=0.5$). "
            r"$^\dagger$Re-implemented from the published description.",
            "tab:dp", "llc", r"Dataset, budget & Method & Accuracy", rows[:-1]))
    summary["dp"] = s


def attack_tables(runs, out, summary):
    ldi = defaultdict(list)
    mia = defaultdict(lambda: defaultdict(list))
    for r in runs:
        c = r["config"]
        cond = f"{c['dataset'].upper().replace('CIFAR10', 'CIFAR-10')}{', DP $\\varepsilon=4$' if c.get('dp_eps') else ''}"
        R = c["rounds"]
        for rec in r.get("ldi", []):
            if rec.get("round") == R and "js_attack" in rec:
                # Eq. (9): advantage = JS(prior || p_i) - JS(attack estimate || p_i)
                ldi[(cond, c["method"], rec.get("view", "server"))].append(rec["js_prior"] - rec["js_attack"])
        m = r.get("mia") or {}
        units = m.get("unit") or m.get("global") or []
        for u in units:
            for k in ("auc", "tpr@1%fpr", "lira_auc", "lira_tpr@1%fpr"):
                if k in u:
                    mia[(cond, c["method"])][k].append(u[k])
    if ldi:
        rows = []
        for cond in sorted({k[0] for k in ldi}):
            first = True
            for m in ORDER:
                vals = [ldi.get((cond, m, v)) for v in ("server", "peer")]
                if not any(vals):
                    continue
                cells = [f"{np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]" if v else "--" for v in vals]
                rows.append(f"{cond if first else ''} & {NAMES[m]} & {cells[0]} & {cells[1]} \\\\")
                first = False
            rows.append(r"\midrule")
        open(os.path.join(out, "ldi.tex"), "w").write(table(
            r"Label-distribution inference at the final round: inference advantage (bits, Eq.~\eqref{eq:advantage}; positive values "
            r"mean leakage) of the server view (client models for parallel methods, group models for grouped methods) and the peer "
            r"view (models handed over inside a chain), mean over clients and seeds 0--2 with the 2.5th and 97.5th percentiles.",
            "tab:ldi", "llcc", r"Condition & Method & Server view & Peer view", rows[:-1]))
        summary["ldi"] = {f"{k[0]}|{k[1]}|{k[2]}": float(np.mean(v)) for k, v in ldi.items()}
    if mia:
        rows = []
        for cond in sorted({k[0] for k in mia}):
            first = True
            for m in [x for x in ORDER if (cond, x) in mia]:
                d = mia[(cond, m)]
                f = lambda k: f"{np.mean(d[k]):.3f}" if d.get(k) else "--"
                rows.append(f"{cond if first else ''} & {NAMES[m]} & {f('auc')} & {f('tpr@1%fpr')} & {f('lira_auc')} & {f('lira_tpr@1%fpr')} \\\\")
                first = False
            rows.append(r"\midrule")
        open(os.path.join(out, "mia.tex"), "w").write(table(
            r"Membership inference against the models received by the server (client models for parallel methods, group models for "
            r"grouped methods): ROC-AUC and true-positive rate at 1\,\% false-positive rate of the loss attack and of offline LiRA "
            r"(eight shadow federations; CIFAR-10 only), mean over clients and seeds 0--2.",
            "tab:mia", "llcccc", r"Condition & Method & Loss AUC & Loss TPR & LiRA AUC & LiRA TPR", rows[:-1]))
        summary["mia"] = {f"{k[0]}|{k[1]}": {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in mia.items()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--res", required=True)
    p.add_argument("--lrsel")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    if a.lrsel and os.path.exists(a.lrsel):
        open(os.path.join(a.out, "lrsel.tex"), "w").write(lrsel_table(json.load(open(a.lrsel))))
    l1 = load(os.path.join(a.res, "L1"))
    if l1:
        main_tables(l1, a.out, summary)
        cost_table(l1, a.out, summary)
    l3 = load(os.path.join(a.res, "L3"))
    if l3:
        dp_table(l3, a.out, summary)
    l4 = load(os.path.join(a.res, "L4"))
    if l4:
        attack_tables(l4, a.out, summary)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    print("tables written:", sorted(f for f in os.listdir(a.out) if f.endswith(".tex")))
