#!/usr/bin/env python
"""Summarise the A0 pilot: repeat-run identity, run times and packing per phase.

Usage: python scripts/a0_summary.py <results_dir> <phase_times.json> <out.json>
  results_dir      results/ written by scripts/run.py (exp 'a0')
  phase_times.json {"resnet": {"wall_s": ...}, "cnn": {"wall_s": ...}, "gpu": "...", "procs": 4}
Repeats are runs whose tags differ only in the suffix _rep0/_rep1 (same seed, partition, groups).
Packing = phase wall-clock / summed single-run wall-clock (the cost model's packing factor).
"""
import glob
import json
import os
import sys
from collections import defaultdict

res_dir, times_path, out_path = sys.argv[1:4]
times = json.load(open(times_path))
runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(res_dir, "a0", "*.json")))]

pairs = defaultdict(list)
for r in runs:
    c = r["config"]
    tag = c.get("tag", "")
    key = (tag.rsplit("_rep", 1)[0], c["method"], c["clients"], tuple(c["ratios"]), c.get("dp_eps"))
    pairs[key].append(r)

rows = []
for (kind, m, N, ratios, dp), rs in sorted(pairs.items(), key=lambda kv: str(kv[0])):
    row = dict(kind=kind, method=m, N=N, ratios=list(ratios), dp_eps=dp, n_runs=len(rs),
               final_acc=[r["final_acc"] for r in rs], run_h=[r["total_time_s"] / 3600 for r in rs],
               device=sorted(set(r.get("device", "") for r in rs)),
               rounds=rs[0]["config"]["rounds"],
               reused_samples=rs[0].get("reused_samples"), dropped_samples=rs[0].get("dropped_samples"),
               client_sizes_min=min(rs[0].get("client_sizes", [0])), client_sizes_max=max(rs[0].get("client_sizes", [0])))
    if len(rs) == 2:
        a, b = rs
        row["history_identical"] = a["history"] == b["history"]
        row["groups_identical"] = a["groups"] == b["groups"]
        row["max_abs_acc_diff"] = max(abs(x["acc"] - y["acc"]) for x, y in zip(a["history"], b["history"]))
    rows.append(row)

phases = {}
for ph, kinds in (("resnet", ("resnet",)), ("cnn", ("cnn", "dp"))):
    rr = [r for r in rows if r["kind"] in kinds]
    run_h = sum(sum(r["run_h"]) for r in rr)
    wall_h = times.get(ph, {}).get("wall_s", 0) / 3600
    phases[ph] = dict(runs=sum(r["n_runs"] for r in rr), summed_run_h=run_h, wall_h=wall_h,
                      packing=(wall_h / run_h) if run_h else None)
rn = [r for r in rows if r["kind"] == "resnet"]
if rn:
    per_round = sum(sum(r["run_h"]) for r in rn) / sum(r["n_runs"] * r["rounds"] for r in rn)
    phases["resnet"]["single_run_h_per_round"] = per_round
    phases["resnet"]["single_run_h_100_rounds_extrapolated"] = per_round * 100
cn = [r for r in rows if r["kind"] == "cnn"]
if cn:
    phases["cnn"]["single_run_h_mean_cnn_100_rounds"] = sum(sum(r["run_h"]) for r in cn) / sum(r["n_runs"] for r in cn)

paired = [r for r in rows if "history_identical" in r]
summary = dict(
    gpu=times.get("gpu"), procs=times.get("procs"), runs=len(runs), pairs=len(paired),
    pairs_identical=sum(r["history_identical"] for r in paired),
    all_repeats_identical=bool(paired) and all(r["history_identical"] for r in paired),
    max_abs_acc_diff=max((r["max_abs_acc_diff"] for r in paired), default=None),
    all_reused_zero=all(r["reused_samples"] == 0 for r in rows),
    phases=phases)
json.dump(dict(summary=summary, pairs=rows), open(out_path, "w"), indent=1)
print(json.dumps(summary, indent=1))
