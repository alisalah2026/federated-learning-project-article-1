#!/usr/bin/env bash
# Pod entrypoint for one block of the reduced A-core plan (docs/PLAN.md section 7).
# Env: GH_TOKEN (RunPod secret), BLOCK (L1..L5), COMMIT (pinned commit on main), PROCS (default 4),
#      LR_FILE (per-method learning rates from L2, needed for L1/L4/L5),
#      TLIM_S (hard time limit of the worker, seconds), SHARD / NUM_SHARDS (default 0 / 1),
#      REVERSE=1 (helper pod: process the shard's jobs from the end; use with RUN_TAG).
# Results, logs and gpu_util.json are pushed to branch r/alite-<BLOCK>[-<SHARD>] every 10 min and at
# the end; the pod then removes itself so that billing stops.
set -uo pipefail
BR="r/alite-${BLOCK}${RUN_TAG:+-$RUN_TAG}$([ "${NUM_SHARDS:-1}" -gt 1 ] && echo "-${SHARD:-0}")"
REPO=/workspace/repo
OUT=$REPO/workspace/feddac-revision
cleanup() {
  cd "$OUT" 2>/dev/null && { git add -A results logs "runs_${BLOCK}" >/dev/null 2>&1; git commit -qm "alite ${BLOCK}: final ($1)" >/dev/null 2>&1; git push -q origin "HEAD:$BR" >/dev/null 2>&1; }
  if [ "${AUTO_REMOVE:-1}" = "1" ] && command -v runpodctl >/dev/null; then runpodctl remove pod "$RUNPOD_POD_ID"; fi
  sleep infinity
}
if [ ! -d "$REPO/.git" ]; then
  git clone -q "https://x-access-token:${GH_TOKEN}@github.com/yazanjer/paper-feddac-resub-tbd-ali" "$REPO" || { echo "clone failed"; sleep 600; exit 1; }
fi
cd "$REPO" && git fetch -q origin && git checkout -q "$COMMIT" || { echo "checkout of $COMMIT failed"; cleanup checkout_failed; }
git checkout -q -B "$BR"
git config user.name "alite-runner"; git config user.email "runner@feddac.invalid"
cd "$OUT"
mkdir -p logs results "runs_${BLOCK}"
exec > >(tee -a "logs/pod_${BLOCK}_${SHARD:-0}.txt") 2>&1
echo "start $(date -u +%FT%TZ) block=$BLOCK commit=$(git -C $REPO rev-parse HEAD) branch=$BR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; nproc; free -g | head -2
pip install -q -r requirements.txt 2>&1 | tail -2 || cleanup pip_failed
python -c "import torch,json,platform; json.dump({'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0)}, open('runs_${BLOCK}/env.json','w'))" || cleanup env_failed
export FEDDAC_DATA=/workspace/data
for d in cifar10 mnist; do python -c "from feddac.data import load_dataset as l; d=l('$d','cpu'); print('dataset ok','$d',tuple(d.x_train.shape))" || cleanup "dataset_$d"; done
python scripts/make_alite.py "$BLOCK" ${LR_FILE:+--lr_file "$LR_FILE"} || cleanup make_grid_failed
[ -s "configs/alite_${BLOCK}.jsonl" ] || cleanup no_jobs
git add -A logs "runs_${BLOCK}"; git commit -qm "alite ${BLOCK}: setup done"; git push -q origin "HEAD:$BR" || echo "WARNING: first push failed"
bash "$REPO/scripts/gpu_monitor.sh" "runs_${BLOCK}" \
  timeout "${TLIM_S:-21600}" python scripts/worker.py --jobs "configs/alite_${BLOCK}.jsonl" --sync_every 600 --shard "${SHARD:-0}" \
  --num_shards "${NUM_SHARDS:-1}" --procs "${PROCS:-4}" --branch "$BR" ${REVERSE:+--reverse}
echo "worker exit $? at $(date -u +%FT%TZ)"
cleanup done
