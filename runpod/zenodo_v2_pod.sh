#!/usr/bin/env bash
# One-shot CPU pod: refresh Zenodo deposition 22965988 with the bundle committed on branch zenodo-v2-bundle.
# Env (RunPod secrets): GH_TOKEN, ZENODO_TOKEN. The status (no credentials) is pushed to branch zenodo-v2-status.
set -uo pipefail
cd /workspace/repo
git config user.name "zenodo-runner"; git config user.email "runner@feddac.invalid"
pip install -q requests 2>&1 | tail -1
cd workspace/feddac-revision
python scripts/zenodo_refresh_v2.py ../../zenodo_bundle > /tmp/zenodo.log 2>&1; rc=$?
echo "EXIT $rc" >> /tmp/zenodo.log
sed -i "s/${ZENODO_TOKEN:-__none__}/***/g" /tmp/zenodo.log
cd /workspace/repo && git checkout -q --orphan zstatus && git rm -rq --cached . >/dev/null 2>&1
mkdir -p /tmp/zst && cp /tmp/zenodo.log /tmp/zst/ && cp /tmp/zenodo_status.json /tmp/zst/ 2>/dev/null
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} + && cp /tmp/zst/* . && git add . && git commit -qm "zenodo v2 refresh rc=$rc" && git push -qf origin HEAD:zenodo-v2-status
if command -v runpodctl >/dev/null; then runpodctl remove pod "$RUNPOD_POD_ID"; fi
sleep infinity
