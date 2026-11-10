#!/usr/bin/env python
"""Descriptive M7 (chain-end coverage, thresholded tau = 0.1, last two clients) of the groups used in the replication runs (REPLICATION_PLAN.md section 6): random vs coverage-aware order, 20 orders per group. Usage: python scripts/rep_m7.py"""
import json, glob, sys, os, numpy as np
sys.path.insert(0, '.')
from feddac import clustering as CL
out = {}
for f in sorted(glob.glob('results/L13/*.json')):
    r = json.load(open(f)); c = r['config']
    if c['method'] not in ('feddac', 'rand_seq'):
        continue
    h = np.asarray(r['histograms']); C = h.shape[1]
    groups = r['groups']
    B1 = h >= 0.1 * h.sum(1, keepdims=True) / C
    key = f"{c['dataset']}|{c['clients']}|{'dac' if c['method']=='feddac' else 'random'}"
    rng = np.random.default_rng(c['seed'])
    for o in ('random', 'coverage'):
        v = []
        for g in groups:
            g = list(g)
            for _ in range(20):
                order = list(rng.permutation(g)) if o == 'random' else CL.coverage_order(g, h, rng)
                held = B1[g].any(0)
                v.append(B1[order[-2:]].any(0).sum() / max(1, held.sum()))
        out.setdefault(f"{key}|{o}", []).append(float(np.mean(v)))
    out.setdefault(f"{key}|classes_per_client", []).append(float(B1.sum(1).mean()))
json.dump({k: float(np.mean(v)) for k, v in out.items()}, open("results/analysis/rep_m7.json", "w"), indent=1)
for k, v in sorted(out.items()):
    print(k, round(np.mean(v), 3), len(v))
