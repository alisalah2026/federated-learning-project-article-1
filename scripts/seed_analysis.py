#!/usr/bin/env python
"""Seed-level statistics, booktabs tables and figures for the resubmission runs.

Usage:
  python scripts/seed_analysis.py --results results/<exp> --out paper_assets --reference fedavg \
      [--metric final_acc] [--methods feddac fedavg ...]

Reads one run JSON per (setting, method, seed) written by scripts/run.py (fields: config.{dataset,
method,clients,ratios,alpha,seed,dp_eps,tag}, final_acc, final_f1, history[].acc) or a summary file
with a list of such records. Nothing is estimated: a cell without data is printed as "--".

Statistics per setting (dataset, N, ratios, alpha, dp_eps, tag):
  * mean and sample SD over seeds (ddof=1; n is reported);
  * paired two-sided Wilcoxon signed-rank test of every method against --reference on the seeds both
    share (exact permutation distribution, see wilcoxon_exact: zero differences are dropped, tied
    |differences| get mid-ranks, the effective sample size n_eff is reported); Holm correction over all
    comparisons of the table. With n_eff = 5 and no ties the exact two-sided p cannot fall below
    2/2^5 = 0.0625; the smallest attainable p of each comparison is reported (p_min);
  * paired mean difference (points) and its percentile bootstrap 95 % interval over seeds.
Input validation (see load/validate): every record keeps its source file, its full configuration and
its commit (field "commit" of the record, of its config or of its "env"; "unknown" if absent). Unreadable
files, records without config/method/seed/metric, duplicate (setting, method, seed) records and
configurations that differ between seeds of one (setting, method) stop the analysis; the problems are
written to input_report.json. --skip-invalid continues with the valid records only for unreadable/invalid
files (duplicates and conflicts are never skipped). Mixed commits inside a setting are reported.
Outputs (out dir): input_report.json, stats.csv, stats.json, table_<metric>.tex (booktabs), fig_<metric>.{png,pdf}
(300 dpi PNG and vector PDF, serif font, muted palette), fig_curves.{png,pdf} when histories exist.
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

NAMES = {"feddac": "FedDAC", "fedavg": "FedAvg", "fedprox": "FedProx", "ppcfl": "PP-CFL",
         "fedse": "FedSe", "fedseq": "FedSeq", "rand_seq": "Random groups + Seq.",
         "dac_par": "DAC groups + Parallel", "ifca": "IFCA", "cfl": "CFL"}
# muted, print-safe, distinguishable in greyscale by marker and line style as well
PALETTE = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD", "#8C8C8C", "#937860", "#DA8BC3"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]


class InputError(Exception):
    """Raised when the inputs contain duplicates or conflicting configurations."""


# config fields (besides method and seed) that identify a setting; a difference in any of them makes a
# different setting, so runs with different rounds, model, capacity etc. are never pooled
SETTING_KEYS = ("dataset", "clients", "ratios", "alpha", "dp_eps", "tag", "model", "rounds", "local_epochs",
                "no_replacement", "capacity", "val_frac", "deterministic")
# fields allowed to differ between runs of the same (setting, method) apart from the seed
VOLATILE_KEYS = ("seed", "out", "device", "workers", "procs", "commit")  # commit: provenance, reported separately


def _freeze(x):
    return tuple(_freeze(i) for i in x) if isinstance(x, (list, tuple)) else x


def _commit(r, c):
    for src in (r, c, r.get("env") if isinstance(r.get("env"), dict) else {}):
        if isinstance(src, dict) and src.get("commit"):
            return str(src["commit"])
    return "unknown"


def _finite(v):
    """True for a real, finite number (bool, NaN and +/-inf are rejected)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load(results, report=None):
    """Return flat records from run JSON files (or summary files holding a list of records).

    Each record keeps source file, full configuration and commit. Problems are appended to `report`
    (a dict with lists "unreadable", "invalid"); nothing is dropped silently."""
    report = report if report is not None else dict(unreadable=[], invalid=[])
    files = [results] if os.path.isfile(results) else sorted(glob.glob(os.path.join(results, "**", "*.json"), recursive=True))
    recs = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            report["unreadable"].append(dict(file=f, error=f"{type(e).__name__}: {e}"))
            continue
        for k, r in enumerate(d if isinstance(d, list) else [d]):
            where = f if not isinstance(d, list) else f"{f}[{k}]"
            if not isinstance(r, dict) or "config" not in r or "final_acc" not in r:
                report["invalid"].append(dict(file=where, error="not a run record (needs config and final_acc)"))
                continue
            c = r["config"]
            if not isinstance(c, dict):
                report["invalid"].append(dict(file=where, error=f"config is not a dictionary ({type(c).__name__})"))
                continue
            miss = [x for x in ("method", "seed") if c.get(x) is None]
            if miss:
                report["invalid"].append(dict(file=where, error=f"config lacks {miss}"))
                continue
            vals = [r["final_acc"], r.get("final_f1")]
            if any(v is not None and not _finite(v) for v in vals):
                report["invalid"].append(dict(file=where, error="metric is not a finite number"))
                continue
            full = {k2: v for k2, v in c.items()}
            recs.append(dict(
                setting=tuple(_freeze(c.get(k2, ())) if k2 == "ratios" else _freeze(c.get(k2, "" if k2 == "tag" else None))
                              for k2 in SETTING_KEYS),
                method=c["method"], seed=c["seed"], final_acc=r["final_acc"], final_f1=r.get("final_f1"),
                curve=[h["acc"] for h in r.get("history", []) if isinstance(h, dict) and "acc" in h],
                file=where, config=full, commit=_commit(r, c)))
    return recs


def validate(recs):
    """Raise InputError on duplicate (setting, method, seed) records or on configurations that differ
    between seeds of one (setting, method). Returns provenance notes (mixed commits per setting)."""
    seen, problems = {}, []
    for r in recs:
        k = (r["setting"], r["method"], r["seed"])
        if k in seen:
            problems.append(dict(kind="duplicate", setting=setting_label(r["setting"]), method=r["method"],
                                 seed=r["seed"], files=[seen[k]["file"], r["file"]]))
        else:
            seen[k] = r
    ref = {}
    for r in recs:
        k = (r["setting"], r["method"])
        cfg = {a: _freeze(b) for a, b in r["config"].items() if a not in VOLATILE_KEYS}
        if k not in ref:
            ref[k] = (cfg, r["file"])
        elif ref[k][0] != cfg:
            diff = sorted(a for a in set(cfg) | set(ref[k][0]) if cfg.get(a) != ref[k][0].get(a))
            problems.append(dict(kind="conflicting_config", setting=setting_label(r["setting"]), method=r["method"],
                                 fields=diff, files=[ref[k][1], r["file"]]))
    if problems:
        raise InputError(problems)
    commits = defaultdict(set)
    for r in recs:
        commits[r["setting"]].add(r["commit"])
    return [dict(setting=setting_label(s), commits=sorted(c)) for s, c in commits.items() if len(c) > 1 or "unknown" in c]


def setting_label(s):
    ds, n, ratios, alpha, eps, tag = s[:6]
    parts = [str(ds), f"N={n}"]
    if ratios:
        parts.append("ratios " + "/".join(f"{int(round(x * 100))}" for x in ratios))
    if alpha is not None:
        parts.append(f"alpha={alpha:g}")
    if eps:
        parts.append(f"eps={eps:g}")
    if tag:
        parts.append(f"[{tag}]")
    extra = [f"{k}={v}" for k, v in zip(SETTING_KEYS[6:], s[6:]) if v is not None]
    if extra:
        parts.append("{" + ", ".join(extra) + "}")
    return ", ".join(parts)


def holm(pvals):
    """Holm step-down adjusted p-values; NaN entries are left as NaN and not counted."""
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    m = len(idx)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * p[i])
        out[i] = min(1.0, run)
    return out


def holm_fixed(pvals, m=4):
    """Holm over a pre-specified family of exactly m tests (docs/SEED_PROTOCOL.md).
    An undefined p (NaN: setting with no complete pair) is set to 1.0 and stays in the family,
    so the family never shrinks; any other family size is an error."""
    p = np.asarray(pvals, float)
    if len(p) != m:
        raise ValueError(f"confirmatory family must have exactly {m} tests, got {len(p)}")
    return holm(np.where(np.isnan(p), 1.0, p))


def wilcoxon_exact(d):
    """Two-sided exact signed-rank test on differences d (exact permutation distribution).

    Zero differences are dropped (Wilcoxon's rule); n_eff is the number of nonzero differences. Tied
    |d| get mid-ranks. Under H0 every sign pattern of the n_eff nonzero differences is equally likely; the
    statistic is W+ (sum of ranks of positive differences) and the p-value is the share of the 2^n_eff
    sign patterns whose |W+ - E| is at least the observed one (E = total/2). Ranks are doubled to be
    integers, and the distribution is built by dynamic programming (no 2^n enumeration).
    Returns dict(p, n_eff, n_zero, ties, p_min); p_min is the p-value of the most extreme pattern for
    these |d| (all signs equal), i.e. the smallest p this sample could have produced. For n_eff > 60 the
    normal approximation (without tie correction) is used and flagged by method="normal"."""
    d = np.asarray(d, float)
    n_zero = int(np.sum(d == 0))
    nz = d[d != 0]
    n = len(nz)
    out = dict(p=1.0, n_eff=n, n_zero=n_zero, ties=False, p_min=1.0, method="exact")
    if n == 0:
        return out
    a = np.abs(nz)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n)
    sa_ = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa_[j + 1] == sa_[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    out["ties"] = bool(len(np.unique(a)) < n)
    r2 = np.rint(2 * ranks).astype(int)  # doubled ranks are integers
    total = int(r2.sum())
    obs = int(r2[nz > 0].sum())
    if n > 60:
        mu, sd = total / 4.0, np.sqrt(np.sum(ranks ** 2) / 4.0)
        from scipy import stats
        out["p"] = float(min(1.0, 2 * stats.norm.sf(abs(obs / 2 - mu) / sd)))
        out["p_min"] = float(min(1.0, 2 * stats.norm.sf(abs(total / 2 - mu) / sd)))
        out["method"] = "normal"
        return out
    cnt = np.zeros(total + 1)
    cnt[0] = 1.0
    for r in r2:
        cnt[r:] = cnt[r:] + cnt[:total + 1 - r].copy()
    cnt /= 2.0 ** n
    sums = np.arange(total + 1)
    dev = np.abs(2 * sums - total)  # |2*W2 - total| in doubled units
    out["p"] = float(min(1.0, cnt[dev >= abs(2 * obs - total)].sum()))
    out["p_min"] = float(min(1.0, cnt[dev >= total].sum()))
    return out


def paired_stats(a, b, n_boot=10000, rng=None):
    """a, b: arrays of equal length (same seeds). Returns dict with the exact Wilcoxon p (see
    wilcoxon_exact), the effective sample size, mean difference and percentile bootstrap interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = dict(n=len(d), mean_diff=float(d.mean()) if len(d) else float("nan"), p=float("nan"),
               ci_lo=float("nan"), ci_hi=float("nan"), n_eff=0, n_zero=0, ties=False, p_min=float("nan"))
    if len(d) >= 2:
        rng = rng or np.random.default_rng(0)
        bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        out["ci_lo"], out["ci_hi"] = (float(x) for x in np.percentile(bs, [2.5, 97.5]))
    if len(d) >= 1:
        w = wilcoxon_exact(d)
        out.update(p=w["p"], n_eff=w["n_eff"], n_zero=w["n_zero"], ties=w["ties"], p_min=w["p_min"], test=w["method"])
    return out


def analyse(recs, reference, metric="final_acc", methods=None):
    by = defaultdict(lambda: defaultdict(dict))  # setting -> method -> seed -> value
    for r in recs:
        if r[metric] is not None:
            if not _finite(r[metric]):
                raise InputError([dict(kind="non_finite_metric", method=r["method"], seed=r["seed"], files=[r["file"]])])
            if r["seed"] in by[r["setting"]][r["method"]]:
                raise InputError([dict(kind="duplicate", method=r["method"], seed=r["seed"], files=[r["file"]])])
            by[r["setting"]][r["method"]][r["seed"]] = r[metric]
    rows, comps = [], []
    for s in sorted(by, key=str):
        for m, sv in by[s].items():
            if methods and m not in methods:
                continue
            v = np.array([sv[k] for k in sorted(sv)], float)
            rows.append(dict(setting=s, method=m, n=len(v), mean=float(v.mean()),
                             sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan")))
            if m != reference and reference in by[s]:  # by[s][m] maps seed -> value
                common = sorted(set(sv) & set(by[s][reference]))
                if common:
                    st = paired_stats([sv[k] for k in common], [by[s][reference][k] for k in common])
                    comps.append(dict(setting=s, method=m, reference=reference, **st))
    adj = holm([c["p"] for c in comps])
    for c, a in zip(comps, adj):
        c["p_holm"] = float(a)
    return rows, comps, by


def holm_family(pvals, m):
    """Holm over a prespecified family of exactly m tests (docs/SEED_PROTOCOL.md). Unlike `holm`, a NaN or a
    wrong family size is an error, so the family can never shrink silently; callers supply p = 1.0 for
    settings whose test is undefined (no complete pair, all-zero differences)."""
    p = np.asarray(pvals, float)
    if len(p) != m:
        raise ValueError(f"Holm family must have exactly {m} p-values, got {len(p)}")
    if np.isnan(p).any():
        raise ValueError("NaN p-value in a fixed Holm family; supply 1.0 for undefined settings")
    return holm(p)


PRESPECIFIED_SEEDS = tuple(range(10))  # docs/SEED_PROTOCOL.md: seeds 0-4 from A1 plus 5-9, fixed before any result
FAMILY_SIZE = 4


def confirmatory(by, family, method, reference, n_boot=10000):
    """Confirmatory contrast of docs/SEED_PROTOCOL.md. `by` is the mapping returned by `analyse`
    (setting -> method -> seed -> value); `family` must be the four prespecified settings (exactly
    FAMILY_SIZE distinct labels; all are always analysed and always in the Holm family). Only the
    prespecified seeds 0-9 are used; other seed ids are listed as excluded, never analysed. A pair is
    complete when both methods have a finite value for the same prespecified seed id; incomplete seeds are
    listed with their reason (also seeds absent from both methods), never replaced or imputed. A setting
    with no complete pair gets p = 1.0 and stays in the family. Returns one dict per setting."""
    if len(family) != FAMILY_SIZE or len(set(family)) != FAMILY_SIZE:
        raise ValueError(f"the confirmatory family is exactly {FAMILY_SIZE} distinct settings, got {list(family)}")
    ok = lambda v: v is not None and _finite(v)
    out = []
    for s in family:
        a, b = by.get(s, {}).get(method, {}), by.get(s, {}).get(reference, {})
        common = [k for k in PRESPECIFIED_SEEDS if ok(a.get(k)) and ok(b.get(k))]
        missing = []
        for k in PRESPECIFIED_SEEDS:
            if k in common:
                continue
            gone = [f"no {n} run" for n, d in ((method, a), (reference, b)) if not ok(d.get(k))]
            missing.append(dict(seed=k, reason=" and ".join(gone)))
        excluded = sorted((set(a) | set(b)) - set(PRESPECIFIED_SEEDS), key=str)
        if common:
            st = paired_stats([a[k] for k in common], [b[k] for k in common], n_boot=n_boot)
        else:
            st = dict(n=0, mean_diff=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), n_eff=0, n_zero=0,
                      ties=False, p=1.0, p_min=1.0)
        out.append(dict(setting=s, method=method, reference=reference, seeds=common, missing=missing,
                        excluded_seeds=excluded, **st))
    for c, adj in zip(out, holm_family([c["p"] for c in out], FAMILY_SIZE)):
        c["p_holm"] = float(adj)
    return out


def fmt_p(p):
    if p != p:
        return "--"
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def latex_table(rows, comps, reference, metric, methods, caption, label):
    comp = {(c["setting"], c["method"]): c for c in comps}
    cell = {(r["setting"], r["method"]): r for r in rows}
    settings = sorted({r["setting"] for r in rows}, key=str)
    ms = [m for m in methods if any(r["method"] == m for r in rows)]
    effs = [c["n_eff"] for c in comps]
    min_p = max((c["p_min"] for c in comps if c["p_min"] == c["p_min"]), default=float("nan"))  # least favourable comparison
    min_n = min(effs, default=0)
    L = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\footnotesize",
         "\\begin{tabular}{llcccc}", "\\toprule",
         f"Setting & Method & Mean$\\pm$SD (\\%) & $n$ & $\\Delta$ vs.\\ {NAMES.get(reference, reference)} [95\\% CI] & $p_{{\\mathrm{{Holm}}}}$ \\\\",
         "\\midrule"]
    for s in settings:
        best = max((cell[(s, m)]["mean"] for m in ms if (s, m) in cell), default=None)
        first = True
        for m in ms:
            r = cell.get((s, m))
            if r is None:
                continue
            sd = "--" if r["sd"] != r["sd"] else f"{r['sd']:.2f}"
            txt = f"{r['mean']:.2f}$\\pm${sd}"
            if best is not None and r["mean"] == best:
                txt = f"\\textbf{{{txt}}}"
            c = comp.get((s, m))
            if m == reference:
                dtxt, ptxt = "reference", ""
            elif c:
                dtxt = f"{c['mean_diff']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]"
                ptxt = fmt_p(c["p_holm"])
            else:
                dtxt, ptxt = "--", "--"
            L.append(f"{setting_label(s).replace('_', ' ') if first else ''} & {NAMES.get(m, m)} & {txt} & {r['n']} & {dtxt} & {ptxt} \\\\")
            first = False
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    note = ("Mean$\\pm$SD over independent seeds (sample SD); $n$ is the number of seeds. $\\Delta$: paired mean difference "
            "to the reference in percentage points with a percentile bootstrap interval over seeds. $p_{\\mathrm{Holm}}$: "
            "paired two-sided Wilcoxon signed-rank test, Holm-adjusted over all comparisons in the table.")
    if comps:
        note += (" Test: exact permutation distribution of the signed-rank statistic; zero differences are dropped and tied "
                 "$|{\\Delta}|$ get mid-ranks. Effective paired sample size $n_{\\mathrm{eff}}$ (nonzero differences): "
                 f"{min(effs)}--{max(effs)}.")
        if min_p > 0.05:
            note += (f" For at least one comparison the smallest attainable two-sided $p$ for its observed ranks is "
                     f"{min_p:.4f} ($>0.05$), so it cannot reach significance before correction.")
    L += ["\\end{tabular}", f"\\\\[2pt]{{\\scriptsize {note}}}", "\\end{table}"]
    return "\n".join(L)


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
                         "mathtext.fontset": "dejavuserif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100,
                         "savefig.dpi": 300, "pdf.fonttype": 42})
    return plt


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")


def bar_figure(by, rows, methods, metric, out):
    plt = style()
    settings = sorted({r["setting"] for r in rows}, key=str)
    ms = [m for m in methods if any(r["method"] == m for r in rows)]
    w = 0.8 / max(len(ms), 1)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * len(settings) + 1.5), 3.2))
    for j, m in enumerate(ms):
        xs, mu, sd = [], [], []
        for i, s in enumerate(settings):
            if m in by[s]:
                v = np.array(list(by[s][m].values()), float)
                xs.append(i + (j - (len(ms) - 1) / 2) * w)
                mu.append(v.mean())
                sd.append(v.std(ddof=1) if len(v) > 1 else 0.0)
                ax.scatter([xs[-1]] * len(v), v, s=8, color="k", alpha=0.45, zorder=3, linewidths=0)
        ax.bar(xs, mu, w * 0.92, yerr=sd, capsize=2, color=PALETTE[j % len(PALETTE)], label=NAMES.get(m, m),
               edgecolor="k", linewidth=0.4)
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels([setting_label(s).replace(", ", "\n") for s in settings], fontsize=7)
    ax.set_ylabel("Test accuracy (%)" if metric == "final_acc" else "Macro-F1 (%)")
    lo = min((min(by[s][m].values()) for s in settings for m in ms if m in by[s]), default=0)
    ax.set_ylim(max(0, lo - 10), None)
    ax.legend(fontsize=7, ncol=min(len(ms), 4), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, out, f"fig_{metric}")
    plt.close(fig)


def curve_figure(recs, methods, out):
    plt = style()
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["curve"] and (not methods or r["method"] in methods):
            by[r["setting"]][r["method"]].append(r["curve"])
    if not by:
        return False
    settings = sorted(by, key=str)
    fig, axes = plt.subplots(1, len(settings), figsize=(3.6 * len(settings), 3.0), squeeze=False)
    for ax, s in zip(axes[0], settings):
        for j, (m, cs) in enumerate(sorted(by[s].items())):
            n = min(len(c) for c in cs)
            a = np.array([c[:n] for c in cs], float)
            x = np.arange(1, n + 1)
            ax.plot(x, a.mean(0), color=PALETTE[j % len(PALETTE)], marker=MARKERS[j % 9], markevery=max(1, n // 8),
                    ms=3, lw=1.2, label=NAMES.get(m, m))
            if len(a) > 1:
                ax.fill_between(x, a.mean(0) - a.std(0, ddof=1), a.mean(0) + a.std(0, ddof=1),
                                color=PALETTE[j % len(PALETTE)], alpha=0.15, lw=0)
        ax.set_title(setting_label(s), fontsize=8)
        ax.set_xlabel("Round")
        ax.set_ylabel("Test accuracy (%)")
    axes[0][0].legend(fontsize=7, frameon=False)
    save(fig, out, "fig_curves")
    plt.close(fig)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default="fedavg")
    ap.add_argument("--metric", default="final_acc", choices=["final_acc", "final_f1"])
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--caption", default="Test accuracy by setting and method.")
    ap.add_argument("--skip-invalid", action="store_true",
                    help="continue with valid records when files are unreadable or invalid (still listed in input_report.json)")
    ap.add_argument("--label", default="tab:seed_stats")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    report = dict(unreadable=[], invalid=[], problems=[], provenance=[], skipped_invalid=a.skip_invalid)
    recs = load(a.results, report)
    try:
        report["provenance"] = validate(recs)
    except InputError as e:
        report["problems"] = e.args[0]
    report["sources"] = [dict(file=r["file"], commit=r["commit"], config=r["config"]) for r in recs]
    json.dump(report, open(os.path.join(a.out, "input_report.json"), "w"), indent=1, default=str)
    if report["problems"]:
        raise SystemExit(f"{len(report['problems'])} duplicate/conflicting records; see {a.out}/input_report.json")
    if (report["unreadable"] or report["invalid"]) and not a.skip_invalid:
        raise SystemExit(f"{len(report['unreadable'])} unreadable and {len(report['invalid'])} invalid input(s); "
                         f"see {a.out}/input_report.json (or pass --skip-invalid)")
    if not recs:
        raise SystemExit(f"no run records under {a.results}")
    order = a.methods or [m for m in NAMES if any(r["method"] == m for r in recs)] + \
        sorted({r["method"] for r in recs} - set(NAMES))
    rows, comps, by = analyse(recs, a.reference, a.metric, a.methods)
    with open(os.path.join(a.out, "stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "method", "n", "mean", "sd", "reference", "n_paired", "n_eff", "n_zero", "ties", "mean_diff", "ci_lo", "ci_hi", "p", "p_min", "p_holm"])
        cm = {(c["setting"], c["method"]): c for c in comps}
        for r in rows:
            c = cm.get((r["setting"], r["method"]), {})
            w.writerow([setting_label(r["setting"]), r["method"], r["n"], r["mean"], r["sd"], a.reference if c else "",
                        c.get("n", ""), c.get("n_eff", ""), c.get("n_zero", ""), c.get("ties", ""), c.get("mean_diff", ""),
                        c.get("ci_lo", ""), c.get("ci_hi", ""), c.get("p", ""), c.get("p_min", ""), c.get("p_holm", "")])
    json.dump(dict(metric=a.metric, reference=a.reference,
                   rows=[dict(r, setting=setting_label(r["setting"])) for r in rows],
                   comparisons=[dict(c, setting=setting_label(c["setting"])) for c in comps]),
              open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    open(os.path.join(a.out, f"table_{a.metric}.tex"), "w").write(
        latex_table(rows, comps, a.reference, a.metric, order, a.caption, a.label) + "\n")
    bar_figure(by, rows, order, a.metric, a.out)
    curve_figure(recs, a.methods, a.out)
    print(f"{len(recs)} runs, {len(rows)} cells, {len(comps)} paired comparisons -> {a.out}")


if __name__ == "__main__":
    main()
