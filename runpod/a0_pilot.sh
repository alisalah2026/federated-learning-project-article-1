#!/usr/bin/env bash
# A0 pilot of the resubmission protocol (docs/PLAN.md section 3, A0). One GPU, PROCS concurrent runs.
# Run from the repository root of yazanjer/paper-feddac-resub-tbd-ali, normally under scripts/gpu_monitor.sh:
#   POD_NAME=a0-resnet PHASES=resnet bash scripts/gpu_monitor.sh /workspace/a0_out \
#       bash workspace/feddac-revision/runpod/a0_pilot.sh /workspace/a0_out
#   bash scripts/stage_and_upload.sh /workspace/a0_out a0-resnet     # final upload, after gpu_util.json exists
#
# Order: run_config.json, seed_map.json, env.json and the job files are written BEFORE training and are
# part of every upload. Uploads (scripts/stage_and_upload.sh, names prefixed with POD_NAME) happen after
# each phase and every UPLOAD_EVERY seconds. status.json / phase_status.json record how each step ended.
# Exit codes: 0 all phases finished and every planned run has a result file; 2 setup or protocol-test
# failure (nothing trained); 3 a phase failed, hit its time limit or left runs without results (INCOMPLETE).
# DRY_RUN=1 replaces installs, tests and the worker by stubs (used by tests/test_upload_flow.sh only).
set -uo pipefail
REPO="$(pwd)"
OUT="${1:?output dir}"
PROCS="${PROCS:-4}"
PHASES="${PHASES:-resnet cnn}"
POD="${POD_NAME:-a0-${PHASES// /-}}"
TLIM_RESNET="${TLIM_RESNET:-12240}" TLIM_CNN="${TLIM_CNN:-12240}"
UPLOAD_EVERY="${UPLOAD_EVERY:-1200}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "$OUT"
up() { bash "$REPO/scripts/stage_and_upload.sh" "$OUT" "$POD" >> "$OUT/upload_log.txt" 2>&1; local rc=$?; echo "$(date -u +%FT%TZ) upload rc=$rc ($1)" >> "$OUT/upload_log.txt"; return $rc; }
status() {  # status <state> <detail>
  printf '{"pod": "%s", "state": "%s", "detail": "%s", "utc": "%s"}\n' "$POD" "$1" "$2" "$(date -u +%FT%TZ)" > "$OUT/status.json"; }
fail_setup() {  # fail_setup <step> <rc>
  status setup_failed "$1 exit $2"; echo "SETUP FAILED: $1 (exit $2); nothing was trained" >&2
  up "setup_failed" || true; exit 2; }

cd "$REPO/workspace/feddac-revision"
export FEDDAC_DATA="${FEDDAC_DATA:-/workspace/data}" FEDDAC_RESULTS="$PWD/results"
status starting "writing run metadata"
COMMIT="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
DIRTY="$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
python3 - "$OUT" "$POD" "$COMMIT" "$DIRTY" "$PHASES" "$PROCS" "$TLIM_RESNET" "$TLIM_CNN" "$UPLOAD_EVERY" <<'PY'
import json, sys, time, platform, socket
o, pod, commit, dirty, phases, procs, tr, tc, ue = sys.argv[1:10]
json.dump({"pod": pod, "commit": commit, "uncommitted_files": int(dirty), "phases": phases.split(), "procs": int(procs),
           "time_limit_s": {"resnet": int(tr), "cnn": int(tc)}, "upload_every_s": int(ue),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
           "python": platform.python_version()}, open(f"{o}/run_config.json", "w"), indent=1)
PY
[ $? -eq 0 ] && [ -s "$OUT/run_config.json" ] || fail_setup run_config 1
if [ "$DRY_RUN" = 1 ]; then mkdir -p configs || fail_setup mkdir_configs $?; else
  python scripts/make_grid.py a0 >> "$OUT/setup_log.txt" 2>&1 || fail_setup make_grid $?
fi
for ph in $PHASES; do [ -f "configs/a0_${ph}.jsonl" ] || fail_setup "missing configs/a0_${ph}.jsonl" 1; cp "configs/a0_${ph}.jsonl" "$OUT/" || fail_setup "copy a0_${ph}.jsonl" $?; done
python3 - "$OUT" $PHASES <<'PY'
import json, sys
o, phases = sys.argv[1], sys.argv[2:]
m = []
for ph in phases:
    for i, l in enumerate(open(f"configs/a0_{ph}.jsonl")):
        j = json.loads(l); e = j.get("extra", [])
        m.append({"phase": ph, "line": i, "method": j["method"], "seed": j["seed"], "tag": e[e.index("--tag") + 1] if "--tag" in e else None,
                  "ratios": j.get("ratios"), "rounds": j.get("rounds"), "dp_eps": j.get("dp_eps")})
json.dump(m, open(f"{o}/seed_map.json", "w"), indent=0)
PY
[ $? -eq 0 ] && [ -s "$OUT/seed_map.json" ] || fail_setup seed_map 1
status setup "installing and testing"
if [ "$DRY_RUN" != 1 ]; then
  pip install -q -r requirements.txt >> "$OUT/setup_log.txt" 2>&1 || fail_setup pip_install $?
  python -m pytest -q tests/test_protocol.py > "$OUT/cpu_tests.txt" 2>&1 || fail_setup protocol_tests $?
  python -c "from feddac.data import load_dataset as l; d=l('cifar10','cpu'); print('cifar10', tuple(d.x_train.shape))" >> "$OUT/setup_log.txt" 2>&1 || fail_setup dataset_load $?
  python -c "import json,torch,subprocess; json.dump({'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(), 'pip_freeze': subprocess.run(['pip','freeze'],capture_output=True,text=True).stdout.split()}, open('$OUT/env.json','w'), indent=1)" || fail_setup env_json $?
else echo '{"dry_run": true}' > "$OUT/env.json"; fi
status running "metadata uploaded, training starts"
# Training starts only after the metadata (run_config, seed_map, job files, env) reached the upload store.
up "before_training" || fail_setup pre_training_upload $?
echo "pre-training metadata uploaded"

( while sleep "$UPLOAD_EVERY"; do
    tar czf "$OUT/live_results.tar.gz" results 2>/dev/null; tar czf "$OUT/live_logs.tar.gz" logs 2>/dev/null
    up "periodic"
  done ) &
UPLOADER=$!
echo '{}' > "$OUT/phase_status.json"
RC=0
for ph in $PHASES; do
  t0=$(date +%s)
  tl=TLIM_${ph^^}; tl="${!tl}"
  if [ "$DRY_RUN" = 1 ]; then timeout "$tl" bash -c 'touch '"$OUT"'/trained.flag; mkdir -p results/a0 logs; for i in $(seq 1 $(wc -l < configs/a0_'"$ph"'.jsonl)); do echo "{}" > results/a0/dry_'"$ph"'_$i.json; done; sleep "${DRY_SLEEP:-1}"'
  else timeout "$tl" python scripts/worker.py --jobs "configs/a0_${ph}.jsonl" --shard 0 --num_shards 1 --procs "$PROCS" --no_git; fi
  wrc=$?
  wall=$(( $(date +%s) - t0 ))
  planned=$(wc -l < "configs/a0_${ph}.jsonl")
  state=complete; [ "$wrc" = 124 ] && state=incomplete_timeout; [ "$wrc" != 0 ] && [ "$wrc" != 124 ] && state=failed
  python3 - "$OUT" "$ph" "$wrc" "$wall" "$planned" "$state" <<'PY'
import glob, json, sys
o, ph, wrc, wall, planned, state = sys.argv[1:7]
n = len(glob.glob("results/a0/*.json"))
p = json.load(open(f"{o}/phase_status.json"))
p[ph] = {"worker_exit": int(wrc), "wall_s": int(wall), "planned_runs": int(planned), "result_files_total_so_far": n, "state": state}
json.dump(p, open(f"{o}/phase_status.json", "w"), indent=1)
PY
  [ "$wrc" != 0 ] && RC=3
  echo "phase $ph: worker exit $wrc, state $state, ${wall}s"
  tar czf "$OUT/partial_results_${ph}.tar.gz" results; tar czf "$OUT/partial_logs_${ph}.tar.gz" logs
  up "after_phase_$ph" || echo "partial upload failed" >&2
done
kill "$UPLOADER" 2>/dev/null
# A phase with exit 0 but fewer result files than planned runs is also incomplete (checked on the union of phases).
planned_all=0; for ph in $PHASES; do planned_all=$(( planned_all + $(wc -l < "configs/a0_${ph}.jsonl") )); done
found=$(ls results/a0/*.json 2>/dev/null | wc -l)
[ "$found" -lt "$planned_all" ] && RC=3
[ "$DRY_RUN" != 1 ] && { python scripts/a0_summary.py results "$OUT/phase_status.json" "$OUT/a0_summary.json" 2>> "$OUT/setup_log.txt" || echo "a0_summary failed" >&2; }
tar czf "$OUT/results.tar.gz" results; tar czf "$OUT/logs.tar.gz" logs
[ -d results/models ] && tar czf "$OUT/models.tar.gz" results/models   # present only if a job ran with --save_model
if [ "$RC" = 0 ]; then status complete "all phases exit 0; $found/$planned_all result files"; else status incomplete "$found/$planned_all result files; see phase_status.json"; fi
up "final_pre_monitor" || RC=3
exit $RC
