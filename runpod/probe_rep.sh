#!/usr/bin/env bash
# Timing and data probe for the replication (CIFAR-100 ResNet-18-GN, FEMNIST natural partition). Pushes probe/ to branch probe-rep.
set -uo pipefail
cd /workspace/repo/workspace/feddac-revision
git config user.name "probe"; git config user.email "runner@feddac.invalid"
pip install -q -r requirements.txt 2>&1 | tail -1
mkdir -p probe
nvidia-smi --query-gpu=name --format=csv,noheader > probe/gpu.txt
python - <<'P' > probe/femnist_stats.txt 2>&1
import numpy as np, sys
sys.path.insert(0, ".")
from feddac.data import _femnist_raw
x, y, w = _femnist_raw()
c = np.bincount(w)
el = c[c >= 100]
print("examples", len(y), "writers", len(c), "writers>=100", len(el), "median n", int(np.median(el)), "classes", y.max() + 1)
k = [len(np.unique(y[w == i])) for i in np.where(c >= 100)[0][:500]]
print("classes per writer (first 500 eligible): median", int(np.median(k)), "min", min(k), "max", max(k))
P
COMMON="--no_replacement --val_frac 0.1 --deterministic --optimizer sgd --lr 0.02 --rounds 5 --out probe/res --exp p"
( s=$(date +%s); python scripts/run.py --dataset cifar100 --clients 20 --method feddac_order $COMMON > probe/c100_N20.log 2>&1; echo "SECONDS $(( $(date +%s)-s ))" >> probe/c100_N20.log ) &
( s=$(date +%s); python scripts/run.py --dataset cifar100 --clients 50 --method rand_seq $COMMON > probe/c100_N50.log 2>&1; echo "SECONDS $(( $(date +%s)-s ))" >> probe/c100_N50.log ) &
( s=$(date +%s); python scripts/run.py --dataset femnist --clients 50 --method feddac_order $COMMON > probe/fem_N50.log 2>&1; echo "SECONDS $(( $(date +%s)-s ))" >> probe/fem_N50.log ) &
( s=$(date +%s); python scripts/run.py --dataset femnist --clients 100 --method rand_seq $COMMON > probe/fem_N100.log 2>&1; echo "SECONDS $(( $(date +%s)-s ))" >> probe/fem_N100.log ) &
wait
tail -n 3 probe/*.log > probe/summary.txt
git checkout -q -b probe-rep2 && git add -f probe && git commit -qm "replication probe" && git push -qf origin probe-rep2
if command -v runpodctl >/dev/null; then runpodctl remove pod "$RUNPOD_POD_ID"; fi
sleep infinity
