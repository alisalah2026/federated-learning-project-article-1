#!/usr/bin/env python
"""Per-method learning-rate selection from the L2 results, and the GPU repeat check.

Selection rule (fixed before the L2 results were read): for each method, the learning rate with the
highest mean validation accuracy over the last five rounds (last5_val_acc); ties go to the smaller
rate. The test split is not used.

Repeat check: the two runs tagged 'rep' are compared with their originals (same configuration, same
GPU model); they are identical if every recorded per-round metric is equal.

Usage: python scripts/select_lr.py <results/L2 dir> <out lr.json> <out report.json>
"""
import glob
import json
import os
import sys

src, out_lr, out_rep = sys.argv[1:4]
runs = [json.load(open(f)) for f in glob.glob(os.path.join(src, "*.json"))]
table = {}
for r in runs:
    c = r["config"]
    if c.get("tag", "").endswith("rep"):
        continue
    table.setdefault(c["method"], {})[c["lr"]] = r
lr, rows = {}, []
for m, by_lr in sorted(table.items()):
    best = sorted(by_lr, key=lambda x: (-by_lr[x]["last5_val_acc"], x))[0]
    lr[m] = best
    for x in sorted(by_lr):
        r = by_lr[x]
        rows.append({"method": m, "lr": x, "last5_val_acc": r["last5_val_acc"], "final_val_acc": r["final_val_acc"],
                     "final_acc": r["final_acc"], "selected": x == best, "time_s": r["total_time_s"],
                     "device": r.get("device")})
reps = []
for r in runs:
    c = r["config"]
    if not c.get("tag", "").endswith("rep"):
        continue
    orig = table.get(c["method"], {}).get(c["lr"])
    if orig is None:
        reps.append({"method": c["method"], "lr": c["lr"], "identical": None, "reason": "original missing"})
        continue
    same = orig["history"] == r["history"] and orig["final_acc"] == r["final_acc"]
    reps.append({"method": c["method"], "lr": c["lr"], "identical": bool(same),
                 "max_abs_diff_acc": max(abs(a["acc"] - b["acc"]) for a, b in zip(orig["history"], r["history"])),
                 "device": [orig.get("device"), r.get("device")]})
json.dump(lr, open(out_lr, "w"), indent=1)
json.dump({"rule": "argmax last5_val_acc per method, ties to smaller lr", "rows": rows, "repeats": reps},
          open(out_rep, "w"), indent=1)
print(json.dumps(lr), "\nrepeats:", reps)
