#!/usr/bin/env python
"""Refresh Zenodo deposition 22965988 with the v2.0.0 bundle (runs on a pod; env ZENODO_TOKEN from a RunPod secret).

Steps: read the deposition; if it is published, open a new version draft; otherwise edit the existing draft.
Delete the outdated code, paper-asset, README and checksum files (results and models zips of v1.0.0 are kept),
upload the files of the bundle directory, and update the metadata from .zenodo.json while keeping the reserved DOI.
The deposition is NOT published. A status file without credentials is written to /tmp/zenodo_status.json.
Usage: python scripts/zenodo_refresh_v2.py <bundle dir> [deposition id]
"""
import glob
import hashlib
import json
import os
import sys

import requests

ZURL = os.environ.get("ZENODO_URL", "https://zenodo.org").rstrip("/")
TOKEN = os.environ.get("ZENODO_TOKEN", "")
H = {"Authorization": f"Bearer {TOKEN}"}
bundle = sys.argv[1]
dep_id = sys.argv[2] if len(sys.argv) > 2 else "22965988"
status = {"deposition_requested": dep_id}


def req(method, url, **kw):
    r = requests.request(method, url, headers=H, timeout=kw.pop("timeout", 120), **kw)
    if r.status_code >= 400:
        status["error"] = {"url": url.replace(TOKEN, "***"), "code": r.status_code, "body": r.text[:800]}
        json.dump(status, open("/tmp/zenodo_status.json", "w"), indent=1)
        raise SystemExit(f"HTTP {r.status_code} on {method} {url}: {r.text[:300]}")
    return r


if not TOKEN:
    raise SystemExit("ZENODO_TOKEN is empty")
dep = req("GET", f"{ZURL}/api/deposit/depositions/{dep_id}").json()
status["initial_state"] = {"state": dep.get("state"), "submitted": dep.get("submitted"),
                           "files": [f["filename"] for f in dep.get("files", [])]}
if dep.get("submitted"):
    nv = req("POST", f"{ZURL}/api/deposit/depositions/{dep_id}/actions/newversion").json()
    dep = req("GET", nv["links"]["latest_draft"]).json()
    status["new_version_draft"] = dep["id"]
did = dep["id"]
keep = ("feddac-results-v1.0.0.zip", "feddac-models-v1.0.0.zip")
new_names = {os.path.basename(p) for p in glob.glob(os.path.join(bundle, "*"))}
deleted = []
for f in dep.get("files", []):
    if f["filename"] in keep and f["filename"] not in new_names:
        continue
    req("DELETE", f"{ZURL}/api/deposit/depositions/{did}/files/{f['id']}")
    deleted.append(f["filename"])
status["deleted"] = deleted
bucket = dep["links"]["bucket"]
up = {}
for p in sorted(glob.glob(os.path.join(bundle, "*.zip"))) + [os.path.join(bundle, "README.md"), os.path.join(bundle, "CHECKSUMS.sha256")]:
    name = os.path.basename(p)
    with open(p, "rb") as fh:
        r = req("PUT", f"{bucket}/{name}", data=fh, timeout=3600)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    up[name] = {"size": os.path.getsize(p), "sha256": h, "zenodo_checksum": r.json().get("checksum")}
    print("uploaded", name, os.path.getsize(p), flush=True)
status["uploaded"] = up
meta = json.load(open(".zenodo.json"))
md = dict(dep.get("metadata", {}))
for k in ("title", "upload_type", "description", "creators", "license", "keywords", "version"):
    md[k] = meta[k]
md.pop("doi", None) if not md.get("prereserve_doi") else None
r = req("PUT", f"{ZURL}/api/deposit/depositions/{did}", json={"metadata": md})
dep = req("GET", f"{ZURL}/api/deposit/depositions/{did}").json()
status["final"] = {"deposition_id": did, "state": dep.get("state"), "submitted": dep.get("submitted"),
                   "title": dep["metadata"].get("title"), "version": dep["metadata"].get("version"),
                   "reserved_doi": (dep["metadata"].get("prereserve_doi") or {}).get("doi"),
                   "files": sorted(f["filename"] for f in dep.get("files", [])),
                   "html": dep["links"].get("html")}
json.dump(status, open("/tmp/zenodo_status.json", "w"), indent=1)
print(json.dumps(status["final"], indent=1), flush=True)
