#!/usr/bin/env python
"""Per-example OUT statistics for offline LiRA from shadow federations (Carlini et al., 2022).

Each shadow npz (run.py --subset_frac 0.5 --save_conf) holds the logit-scaled confidence of every
training-split example ('train'), of every test example ('test'), and the indices of the training
subset it was trained on ('subset'). For a training example the OUT shadows are those whose subset
excludes it; every shadow is OUT for test examples. Mean and standard deviation over the OUT shadows
are written; examples with fewer than two OUT shadows get the global OUT mean and standard deviation,
and standard deviations are floored at 1e-3.

Usage: python scripts/lira_stats.py <out.npz> <shadow1.npz> [<shadow2.npz> ...]
"""
import sys

import numpy as np

out, files = sys.argv[1], sys.argv[2:]
Z = [np.load(f) for f in files]
tr = np.stack([z["train"] for z in Z])
te = np.stack([z["test"] for z in Z])
inn = np.zeros_like(tr, dtype=bool)
for k, z in enumerate(Z):
    inn[k, z["subset"]] = True
out_mask = ~inn
cnt = out_mask.sum(0)
s1 = np.where(out_mask, tr, 0).sum(0)
s2 = np.where(out_mask, tr ** 2, 0).sum(0)
g_mu, g_sd = tr[out_mask].mean(), tr[out_mask].std()
with np.errstate(invalid="ignore", divide="ignore"):
    mu = np.where(cnt >= 2, s1 / np.maximum(cnt, 1), g_mu)
    var = np.where(cnt >= 2, s2 / np.maximum(cnt, 1) - mu ** 2, g_sd ** 2)
sd = np.sqrt(np.maximum(var, 0)) * np.sqrt(np.where(cnt >= 2, cnt / np.maximum(cnt - 1, 1), 1.0))
sd = np.maximum(sd, 1e-3)
np.savez_compressed(out, train_mu=mu.astype(np.float32), train_sd=sd.astype(np.float32),
                    test_mu=te.mean(0).astype(np.float32), test_sd=np.maximum(te.std(0, ddof=1), 1e-3).astype(np.float32),
                    n_out=cnt.astype(np.int16))
print(f"{len(files)} shadows; OUT count per training example: min {cnt.min()}, median {np.median(cnt):.0f}, "
      f"max {cnt.max()}; examples with < 2 OUT shadows: {(cnt < 2).sum()}")
