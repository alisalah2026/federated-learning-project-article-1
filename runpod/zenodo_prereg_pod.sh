#!/usr/bin/env bash
# CPU pod: publish the replication pre-registration on Zenodo; status pushed to branch prereg-status.
set -uo pipefail
cd /workspace/repo
git config user.name "prereg-runner"; git config user.email "runner@feddac.invalid"
PY=$(command -v python || command -v python3)
$PY -m pip install -q --break-system-packages requests 2>&1 | tail -1 || $PY -m pip install -q requests 2>&1 | tail -1
$PY workspace/feddac-revision/scripts/zenodo_prereg.py "$REG_COMMIT" > /tmp/prereg.log 2>&1; rc=$?
echo "EXIT $rc" >> /tmp/prereg.log
git checkout -q --orphan pstatus && git rm -rq --cached . >/dev/null 2>&1
mkdir -p /tmp/pst && cp /tmp/prereg.log /tmp/pst/ && cp /tmp/prereg_status.json /tmp/pst/ 2>/dev/null
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} + && cp /tmp/pst/* . && git add . && git commit -qm "prereg rc=$rc" && git push -qf origin HEAD:prereg-status${PREREG:+-$PREREG}
if command -v runpodctl >/dev/null; then runpodctl remove pod "$RUNPOD_POD_ID"; fi
sleep infinity
