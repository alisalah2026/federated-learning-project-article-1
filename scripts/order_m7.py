#!/usr/bin/env python
"""CPU check for docs/ORDER_PLAN.md: chain recency coverage M7 (share of classes held by the last k = 2
clients, presence and thresholded tau = 0.1) under random and coverage-aware order, for DAC and random groups.
100 federations, 20 orders per group. Usage: python scripts/order_m7.py <out.json>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feddac import clustering as CL  # noqa: E402
from feddac.data import partition  # noqa: E402

out = {}
labels = np.repeat(np.arange(10), 4500)
for name, N, r in (("S-A N20 default", 20, (0.1, 0.3, 0.6)), ("S-B N20 90% ext", 20, (0.1, 0.0, 0.9)),
                   ("N50 all extreme", 50, (0.0, 0.0, 1.0))):
    rec = {}
    for s in range(100):
        h = partition(labels, N, r, 0.2, 10, seed=s, replacement=False).histograms
        L = CL.num_clusters(N)
        rng = np.random.default_rng(s)
        G = {"dac": CL.dac_cluster(h, L), "random": CL.random_groups(N, L, rng)}
        B0 = h > 0
        B1 = h >= 0.1 * h.sum(1, keepdims=True) / 10
        for g_name, groups in G.items():
            for o_name in ("random", "coverage"):
                v0, v1 = [], []
                for g in groups:
                    for _ in range(20):
                        o = CL.coverage_order(g, h, rng) if o_name == "coverage" else list(rng.permutation(g))
                        last = o[-2:]
                        v0.append(B0[last].any(0).mean())
                        v1.append(B1[last].any(0).mean())
                rec.setdefault(f"{g_name}|{o_name}", []).append((np.mean(v0), np.mean(v1)))
    out[name] = {k: {"m7_presence": float(np.mean([x[0] for x in v])), "m7_thresholded": float(np.mean([x[1] for x in v]))}
                 for k, v in rec.items()}
    print(name, {k: (round(v["m7_presence"], 3), round(v["m7_thresholded"], 3)) for k, v in out[name].items()})
json.dump(out, open(sys.argv[1], "w"), indent=1)
