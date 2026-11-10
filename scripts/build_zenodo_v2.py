#!/usr/bin/env python
"""Build the v2.0.0 Zenodo bundle of the Neurocomputing revision from this repository (local, no network).

Contents:
  feddac-code-v2.0.0.zip          source tree of workspace/feddac-revision at HEAD (without results/)
  feddac-results-v2.0.0.zip       per-run result JSONs of blocks L1-L11 from the r/alite-* branches (one file per run;
                                  where an original shard and a helper both hold a run, the original is kept), blocks
                                  L12-L16 as merged on the default branch, pod logs of every r/alite-* branch,
                                  environment and GPU-utilisation records, analysis summaries, selection configs,
                                  LiRA statistics, and the analysis plans and run ledger
  feddac-paper-assets-v2.0.0.zip  generated LaTeX tables and figures of the manuscript
  README.md, CHECKSUMS.sha256

Usage (from the repository root): python workspace/feddac-revision/scripts/build_zenodo_v2.py <out dir>
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import zipfile

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
OUT = os.path.abspath(sys.argv[1])
FR = "workspace/feddac-revision"
VER = "2.0.0"
os.makedirs(OUT, exist_ok=True)


def git(*a, binary=False):
    r = subprocess.run(["git", "-C", ROOT, *a], capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


head = git("rev-parse", "HEAD").strip()

# ---------------------------------------------------------------- code
code = os.path.join(OUT, f"feddac-code-v{VER}.zip")
files = [f for f in git("ls-files", FR).split("\n") if f and not f.startswith(f"{FR}/results/")]
with zipfile.ZipFile(code, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.writestr("feddac/" + os.path.relpath(f, FR), git("show", f"HEAD:{f}", binary=True))
print("code files:", len(files))

# ---------------------------------------------------------------- results
branches = [b for b in git("for-each-ref", "--format=%(refname)", "refs/remotes/origin/r").split() if "/alite-" in b]
# originals before helpers, so that originals win
branches.sort(key=lambda b: ("helper" in b, b))
runs, other, counts = {}, {}, {}
for b in branches:
    tag = b.split("/r/", 1)[1]
    block = tag.split("-")[1]  # alite-L1-helper-3 -> L1
    for f in git("ls-tree", "-r", "--name-only", b, "--", FR).split("\n"):
        if not f:
            continue
        rel = os.path.relpath(f, FR)
        if rel.startswith(f"results/{block}/") and rel.endswith(".json"):
            if rel not in runs:
                runs[rel] = (b, f)
        elif rel.startswith("logs/") or rel.startswith("runs_"):
            other[f"pods/{tag}/{rel}"] = (b, f)
with zipfile.ZipFile(os.path.join(OUT, f"feddac-results-v{VER}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for rel, (b, f) in sorted(runs.items()):
        z.writestr(rel, git("show", f"{b}:{f}", binary=True))
        blk = rel.split("/")[1]
        counts[blk] = counts.get(blk, 0) + 1
    for rel, (b, f) in sorted(other.items()):
        z.writestr(rel, git("show", f"{b}:{f}", binary=True))
    for f in sorted(glob.glob(os.path.join(ROOT, FR, "results/analysis/*.json"))):
        z.write(f, "results/analysis/" + os.path.basename(f))
    for f in sorted(glob.glob(os.path.join(ROOT, FR, "configs/*"))):
        if os.path.isfile(f) and (f.endswith(".json") or f.endswith(".npz")):
            z.write(f, "configs/" + os.path.basename(f))
    for d in ("SEED_PROTOCOL.md", "COVERAGE_PLAN.md", "ORDER_PLAN.md", "RUNS_ALITE.md", "PLAN.md"):
        p = os.path.join(ROOT, "docs", d)
        if os.path.exists(p):
            z.write(p, "docs/" + d)
# blocks L12-L16 (replications, histogram budget, SCAFFOLD/FedDyn) as merged on the default branch
for f in git("ls-files", f"{FR}/results").split("\n"):
    rel = os.path.relpath(f, FR) if f else ""
    parts = rel.split("/")
    if len(parts) == 3 and parts[1] in ("L12", "L13", "L14", "L15", "L16") and rel.endswith(".json") and rel not in runs:
        with zipfile.ZipFile(os.path.join(OUT, f"feddac-results-v{VER}.zip"), "a", zipfile.ZIP_DEFLATED) as z:
            z.writestr(rel, git("show", f"HEAD:{f}", binary=True))
        counts[parts[1]] = counts.get(parts[1], 0) + 1
print("result files per block:", dict(sorted(counts.items())), "total", sum(counts.values()), "pod files", len(other))

# ---------------------------------------------------------------- paper assets
with zipfile.ZipFile(os.path.join(OUT, f"feddac-paper-assets-v{VER}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(os.path.join(ROOT, "workspace/paper/neuro/tables/*.tex"))):
        z.write(f, "tables/" + os.path.basename(f))
    for f in sorted(glob.glob(os.path.join(ROOT, "workspace/paper/neuro/figures/*.pdf"))):
        z.write(f, "figures/" + os.path.basename(f))

# ---------------------------------------------------------------- README
blocks = {
    "L1": "main comparison, CIFAR-10 CNN, N = 20/50/100 and 90 % extreme, 9 methods, seeds 0-4 (FedDAC and Random+Seq 0-9), IID reference",
    "L2": "learning-rate selection on the validation split and GPU repeat check",
    "L3": "record-level DP-SGD (declared capacity), CIFAR-10 and MNIST, eps 1 and 4, 7 methods, seeds 0-4",
    "L4": "label-distribution and membership inference attacks (loss attack, offline LiRA)",
    "L5": "LiRA shadow federations (8)",
    "L6": "coverage check, N = 50, all clients with two classes, default-setting learning rates",
    "L7": "learning-rate re-selection in the L6 scenario",
    "L8": "FedDAC at the re-selected rate in the L6 scenario",
    "L9": "FedDAC-Order experiment, N = 20 default and 90 % extreme, common rate 0.02",
    "L10": "FedDAC-Order at N = 50 and N = 100",
    "L11": "FedDAC-Order under record-level DP",
    "L12": "replication pilot (stage 1), CIFAR-100 ResNet-18 and FEMNIST, including the repeated FEMNIST runs",
    "L13": "pre-registered replication on CIFAR-100 ResNet-18 and FEMNIST (doi:10.5281/zenodo.23722974)",
    "L14": "histogram budget under DP, eps_h 0.05-0.5 (pre-registered, doi:10.5281/zenodo.23726867)",
    "L15": "learning-rate selection for SCAFFOLD and FedDyn",
    "L16": "SCAFFOLD and FedDyn baselines against FedDAC-Order, seeds 0-9",
}
lines = [f"# Grouping or Ordering? Coverage-Aware Sequential Federated Learning under Label Skew - code and results (v{VER})", "",
         "Code, per-run results and generated manuscript assets of the revised FedDAC study (target: Neurocomputing).",
         f"Built from repository commit `{head[:10]}`.", "",
         "## Files", "",
         f"- `feddac-code-v{VER}.zip`: the `feddac` package, run scripts, job grids, analysis and table scripts, tests (`pytest tests`), pod runner.",
         f"- `feddac-results-v{VER}.zip`: one JSON per run (`results/<block>/<run_id>.json`, with configuration, per-round history, groups, histograms, cost and device),",
         "  pod logs and environment records (`pods/`), analysis summaries (`results/analysis/`), selected learning rates and LiRA statistics (`configs/`),",
         "  and the prespecified analysis plans and run ledger (`docs/`).",
         f"- `feddac-paper-assets-v{VER}.zip`: the LaTeX tables and figures of the manuscript, generated from the result files.",
         "- `feddac-results-v1.0.0.zip`, `feddac-models-v1.0.0.zip` (if present): results and trained models of the earlier protocol (sampling with replacement),",
         "  used only for the CIFAR-100 and Tiny-ImageNet ResNet-18 table reported separately in the manuscript.", "",
         "## Experiment blocks", "", "| Block | Content | Runs |", "|---|---|---|"]
for b, d in blocks.items():
    lines.append(f"| {b} | {d} | {counts.get(b, 0)} |")
lines += ["", f"Total: {sum(counts.values())} runs. Blocks L9_ref, L10_ref and L11_ref of the repository are copies of L1/L3 runs and are not duplicated here.", "",
          "## Protocol", "",
          "Partitions without duplicated examples; 10 % stratified validation split for model selection (never the test split);",
          "deterministic CUDA kernels; declared public client capacity for DP-SGD (RDP accountant); paired seeds with exact Wilcoxon",
          "signed-rank tests, Holm correction for the prespecified families, percentile bootstrap intervals.", "",
          "## Reproduction", "",
          "```", "unzip feddac-code-v2.0.0.zip && cd feddac && pip install -r requirements.txt",
          "python scripts/make_alite.py L1 --lr_file configs/alite_lr.json   # job grid of a block",
          "python scripts/worker.py --jobs configs/alite_L1.jsonl            # run (GPU)",
          "python scripts/analyse_L9.py results paper_tables summary.json    # example analysis", "```", "",
          "License: MIT (code), CC-BY-4.0 (results)."]
open(os.path.join(OUT, "README.md"), "w").write("\n".join(lines) + "\n")
with open(os.path.join(OUT, "CHECKSUMS.sha256"), "w") as f:
    for p in sorted(glob.glob(os.path.join(OUT, "*.zip"))) + [os.path.join(OUT, "README.md")]:
        f.write(f"{sha256(p)}  {os.path.basename(p)}\n")
json.dump({"commit": head, "runs": counts}, open(os.path.join(OUT, "build_info.json"), "w"), indent=1)
print(open(os.path.join(OUT, "CHECKSUMS.sha256")).read())
