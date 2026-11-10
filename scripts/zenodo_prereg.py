#!/usr/bin/env python
"""Publish the replication pre-registration (docs/REPLICATION_PLAN.md) as its own Zenodo record (env ZENODO_TOKEN).
Files: the plan, the analysis script, the job-grid script and a zip of the code at the registered commit.
The record is PUBLISHED (owner decision 2026-10-28: timestamped Zenodo record). Status (no credentials) -> /tmp/prereg_status.json
Usage (repository root): python workspace/feddac-revision/scripts/zenodo_prereg.py <commit>
"""
import json
import os
import subprocess
import sys

import requests

commit = sys.argv[1]
Z = os.environ.get("ZENODO_URL", "https://zenodo.org").rstrip("/")
H = {"Authorization": f"Bearer {os.environ['ZENODO_TOKEN']}"}
os.makedirs("/tmp/prereg", exist_ok=True)
subprocess.run(["git", "archive", "--format=zip", "-o", "/tmp/prereg/feddac-code-at-registration.zip", commit,
                "workspace/feddac-revision"], check=True)
KIND = os.environ.get("PREREG", "REP")  # REP: replication (record 23722974); L14: DP histogram budget (ORDER_PLAN section 7)
if KIND == "L14":
    files = {"ORDER_PLAN.md": "docs/ORDER_PLAN.md",
             "analyse_L14.py": "workspace/feddac-revision/scripts/analyse_L14.py",
             "make_alite.py": "workspace/feddac-revision/scripts/make_alite.py"}
else:
    files = {"REPLICATION_PLAN.md": "docs/REPLICATION_PLAN.md",
             "analyse_rep.py": "workspace/feddac-revision/scripts/analyse_rep.py",
             "make_alite.py": "workspace/feddac-revision/scripts/make_alite.py"}
for name, path in files.items():
    open(f"/tmp/prereg/{name}", "wb").write(subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, check=True).stdout)
st = {}


def chk(r):
    if r.status_code >= 400:
        st["error"] = {"code": r.status_code, "body": r.text[:800]}
        json.dump(st, open("/tmp/prereg_status.json", "w"), indent=1)
        raise SystemExit(f"HTTP {r.status_code}: {r.text[:300]}")
    return r


dep = chk(requests.post(f"{Z}/api/deposit/depositions", headers=H, json={}, timeout=60)).json()
creators = json.load(open("workspace/feddac-revision/.zenodo.json"))["creators"]
md = {"title": "Pre-registration: replication of coverage-aware chain ordering (FedDAC-Order) on CIFAR-100 with ResNet-18 and FEMNIST",
      "upload_type": "publication", "publication_type": "other",
      "description": ("Pre-registration of a replication experiment for the study 'Grouping or Ordering? Coverage-Aware Sequential "
                      "Federated Learning under Label Skew'. The plan (REPLICATION_PLAN.md), the analysis script and the job grids "
                      f"were fixed at commit {commit} of the study repository before any replication run; this record "
                      "timestamps them. Primary hypothesis: FedDAC-Order exceeds random grouping with random chain order in "
                      "final test accuracy (exact two-sided Wilcoxon, Holm over three settings). Settings: CIFAR-100 with "
                      "ResNet-18 (N = 50), FEMNIST with its natural partition by writer (N = 50 and N = 100). Code and results of "
                      "the study: Zenodo DOI 10.5281/zenodo.22965988."),
      "creators": creators, "access_right": "open", "license": "cc-by-4.0",
      "keywords": ["pre-registration", "federated learning", "label skew", "client ordering", "replication"]}
if KIND == "L14":
    md["title"] = "Pre-registration: histogram budget of coverage-aware chain ordering (FedDAC-Order) under record-level differential privacy"
    md["description"] = ("Pre-registration of an experiment for the study 'Grouping or Ordering? Coverage-Aware Sequential Federated "
                         "Learning under Label Skew'. Section 7 of ORDER_PLAN.md, the analysis script and the job grid were fixed at "
                         f"commit {commit} of the study repository before any run of block L14; this record timestamps them. "
                         "Primary hypothesis H5: at a total budget eps = 1 split as eps_h + eps_sgd, FedDAC-Order with the histogram "
                         "budget eps_h selected on the validation split (from 0.05, 0.1, 0.25, 0.5) versus random grouping with random "
                         "chain order (exact two-sided Wilcoxon over seeds 0-9, Holm over CIFAR-10 and MNIST), with a pre-stated "
                         "reading of the deficit. Related pre-registration: 10.5281/zenodo.23722974. Code and results of the study: "
                         "Zenodo DOI 10.5281/zenodo.22965988.")
    md["keywords"] = ["pre-registration", "federated learning", "differential privacy", "label skew", "client ordering"]
chk(requests.put(f"{Z}/api/deposit/depositions/{dep['id']}", headers=H, json={"metadata": md}, timeout=60))
bucket = dep["links"]["bucket"]
for name in sorted(os.listdir("/tmp/prereg")):
    with open(f"/tmp/prereg/{name}", "rb") as f:
        chk(requests.put(f"{bucket}/{name}", data=f, headers=H, timeout=600))
pub = chk(requests.post(f"{Z}/api/deposit/depositions/{dep['id']}/actions/publish", headers=H, timeout=120)).json()
st = {"deposition_id": pub["id"], "doi": pub.get("doi"), "record_url": pub["links"].get("record_html") or pub["links"].get("html"),
      "created": pub.get("created"), "published": pub.get("submitted"), "commit": commit,
      "files": sorted(f["filename"] for f in pub.get("files", []))}
json.dump(st, open("/tmp/prereg_status.json", "w"), indent=1)
print(json.dumps(st, indent=1))
