#!/usr/bin/env python
"""Selection rule of docs/COVERAGE_PLAN.md section 4 (fixed before the analysis was run).

Candidates: scenarios trainable with the CIFAR-10 CNN (C = 10): S1 (all), S3 with C = 10, S4.
Selected: the largest DAC - random difference in the M1 full-coverage share whose bootstrap 95 %
interval lies above zero; ties go to the scenario closest to the main-grid defaults
(N = 50, ratios (0.1, 0.3, 0.6), rho = 1, sigma = 0). Writes configs/coverage_selection.json.
"""
import json
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "results/analysis/coverage_measures.json"
R = json.load(open(src))
cands = [("S1", k, v) for k, v in R["S1"].items()]
cands += [("S3", k, v) for k, v in R["S3"].items() if v["config"]["C"] == 10]
cands += [("S4", k, v) for k, v in R["S4"].items()]


def dist(c):
    return (abs(c["N"] - 50) / 50 + sum(abs(a - b) for a, b in zip(c["ratios"], (0.1, 0.3, 0.6)))
            + abs(c["rho"] - 1) / 100 + c["sigma"])


ok = [(v["diff_dac_random"]["m1_full"][0], -dist(v["config"]), f, k, v) for f, k, v in cands
      if v["diff_dac_random"]["m1_full"][1] > 0]
table = [{"family": f, "key": k, "m1_full_diff": v["diff_dac_random"]["m1_full"]} for f, k, v in cands]
if not ok:
    out = {"run": False, "candidates": table}
else:
    d, _, f, k, v = max(ok, key=lambda x: (round(x[0], 6), x[1]))
    out = {"run": True, "family": f, "key": k, "config": v["config"], "m1_full_diff": v["diff_dac_random"]["m1_full"],
           "candidates": table}
json.dump(out, open("configs/coverage_selection.json", "w"), indent=1)
print({k: out[k] for k in out if k != "candidates"})
