#!/usr/bin/env python
"""Booktabs table of the A0 pilot from a0_summary.json (written by scripts/a0_summary.py).

Usage: python scripts/a0_table.py <a0_summary.json> <out.tex>
Prints only values present in the summary; a missing, null or non-numeric value (accuracy, run time,
phase packing, rounds, repeat flag, GPU) is printed as "--".
"""
import json
import sys

d = json.load(open(sys.argv[1]))
s, pairs = d.get("summary") or {}, d.get("pairs") or []


def f(x, spec=".3f"):
    return format(x, spec) if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x else "--"


def first(xs):
    return xs[0] if isinstance(xs, list) and xs else None


def mean(xs):
    v = [x for x in (xs or []) if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return sum(v) / len(v) if v else None


def phase(name):
    return ((s.get("phases") or {}).get(name) or {}).get("packing")


L = ["\\begin{table}[htbp]", "\\centering",
     "\\caption{Pilot of the deterministic protocol: repeat-run identity and run time on one GPU.}", "\\label{tab:a0}",
     "\\footnotesize", "\\begin{tabular}{llrrrc}", "\\toprule",
     "Phase & Method & Rounds & Acc.\\ run 1 (\\%) & Run time (h) & Identical repeat \\\\", "\\midrule"]
for r in pairs:
    acc = f(first(r.get("final_acc")), ".2f")
    h = f(mean(r.get("run_h")), ".3f")
    ident = {True: "yes", False: "no"}.get(r.get("history_identical"), "--")
    L.append(f"{r.get('kind') or '--'} & {r.get('method') or '--'} & {f(r.get('rounds'), 'd') if isinstance(r.get('rounds'), int) else '--'} "
             f"& {acc} & {h} & {ident} \\\\")
L += ["\\bottomrule", "\\end{tabular}",
      f"\\\\[2pt]{{\\scriptsize GPU: {s.get('gpu') or '--'}; {f(s.get('pairs_identical'), 'd') if isinstance(s.get('pairs_identical'), int) else '--'} of "
      f"{f(s.get('pairs'), 'd') if isinstance(s.get('pairs'), int) else '--'} repeat pairs identical; "
      f"packing (resnet, cnn): {f(phase('resnet'), '.2f')}, {f(phase('cnn'), '.2f')}.}}",
      "\\end{table}"]
open(sys.argv[2], "w").write("\n".join(L).replace("_", "\\_") + "\n")
