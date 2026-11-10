"""CPU tests for the multi-model baselines (IFCA, CFL) and the personalised accuracy metric."""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from feddac.multimodel import _bipartition, personal_accuracy  # noqa: E402
import run as RUN  # noqa: E402


def test_personal_accuracy_analytic():
    hist = np.array([[10, 0], [0, 30]])           # client 0 only class 0, client 1 only class 1
    acc = np.array([[80.0, 20.0], [50.0, 90.0]])  # per-client per-class accuracy of its model
    # (10/40)*80 + (30/40)*90 = 20 + 67.5
    assert abs(personal_accuracy(acc, hist) - 87.5) < 1e-9


def test_personal_accuracy_nan_class_counts_as_zero_only_if_held():
    hist = np.array([[5, 5]])
    acc = np.array([[100.0, np.nan]])
    assert abs(personal_accuracy(acc, hist) - 50.0) < 1e-9


def test_bipartition_separates_opposite_directions():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(50, generator=g)
    ups = [base + 0.05 * torch.randn(50, generator=g) for _ in range(4)]
    ups += [-base + 0.05 * torch.randn(50, generator=g) for _ in range(3)]
    halves, S = _bipartition(ups)
    assert sorted(map(sorted, halves)) == [[0, 1, 2, 3], [4, 5, 6]]
    assert S.shape == (7, 7)


def test_run_id_unchanged_without_new_keys():
    base = dict(dataset="cifar10", method="feddac", clients=20, alpha=0.2, ratios=[0.1, 0.3, 0.6], seed=0,
                dp_eps=None, rounds=100, tag="")
    with_none = dict(base, ifca_k=None, cfl_eps1=None, cfl_eps2=None, cfl_warmup=None)
    assert RUN.run_id(base) == RUN.run_id(with_none)
    assert RUN.run_id(dict(base, method="ifca")) != RUN.run_id(dict(base, method="ifca", ifca_k=4))


def _run(tmp_path, *args):
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts/run.py"), "--dataset", "synthetic",
                           "--clients", "10", "--out", str(tmp_path), *args])
    files = glob.glob(str(tmp_path / "default" / "*.json"))
    return [json.load(open(f)) for f in files]


def test_ifca_runs_and_assigns_every_client(tmp_path):
    (res,) = _run(tmp_path, "--method", "ifca", "--rounds", "3", "--ifca_k", "3")
    mm = res["multi_model"]
    assert mm["num_models"] == 3 and len(mm["assignment"]) == 10
    assert sorted(c for g in res["groups"] for c in g) == list(range(10))
    assert 0.0 <= res["final_acc"] <= 100.0 and 0.0 <= res["final_personal_acc"] <= 100.0
    assert res["cost"]["model_transfers"] == 3 * 10 * (3 + 1)


def test_cfl_split_when_thresholds_met(tmp_path):
    # eps1 large and eps2 = 0 make the split criterion true at the first allowed round
    (res,) = _run(tmp_path, "--method", "cfl", "--rounds", "3", "--cfl_eps1", "1e9", "--cfl_eps2", "0",
                  "--cfl_warmup", "1")
    mm = res["multi_model"]
    assert len(mm["splits"]) >= 1 and mm["num_models"] >= 2
    assert sorted(c for g in res["groups"] for c in g) == list(range(10))


def test_cfl_no_split_when_thresholds_not_met(tmp_path):
    (res,) = _run(tmp_path, "--method", "cfl", "--rounds", "3", "--cfl_eps1", "0", "--cfl_eps2", "1e9",
                  "--cfl_warmup", "1")
    assert res["multi_model"]["splits"] == [] and res["multi_model"]["num_models"] == 1


def test_single_model_methods_report_personal_accuracy(tmp_path):
    (res,) = _run(tmp_path, "--method", "fedavg", "--rounds", "2")
    assert 0.0 <= res["final_personal_acc"] <= 100.0


def test_offline_lira_pipeline(tmp_path):
    """Shadows on random halves -> per-example OUT statistics -> LiRA scores in the MIA output."""
    for s in (100, 101, 102):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts/run.py"), "--dataset", "synthetic",
                               "--method", "fedavg", "--clients", "10", "--rounds", "2", "--seed", str(s),
                               "--val_frac", "0.1", "--subset_frac", "0.5", "--save_conf", "--exp", "L5",
                               "--out", str(tmp_path)])
    shadows = sorted(glob.glob(str(tmp_path / "conf" / "L5" / "*.npz")))
    assert len(shadows) == 3
    z = np.load(shadows[0])
    assert len(z["subset"]) == len(z["train"]) // 2
    stats = str(tmp_path / "lira_stats.npz")
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts/lira_stats.py"), stats, *shadows])
    st = np.load(stats)
    assert st["train_mu"].shape == z["train"].shape and (st["train_sd"] > 0).all()
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts/run.py"), "--dataset", "synthetic",
                           "--method", "feddac", "--clients", "10", "--rounds", "2", "--seed", "0",
                           "--val_frac", "0.1", "--mia", "--lira_stats", stats, "--exp", "L4", "--out", str(tmp_path)])
    (res,) = [json.load(open(f)) for f in glob.glob(str(tmp_path / "L4" / "*.json"))]
    g = res["mia"]["global"][0]
    assert 0.0 <= g["lira_auc"] <= 1.0 and "lira_tpr@1%fpr" in g
    assert all("lira_auc" in u for u in res["mia"]["unit"])


def test_quantity_skew_and_coverage_measures():
    import importlib.util
    import numpy as np
    from feddac.data import partition
    labels = np.repeat(np.arange(10), 300)
    a = partition(labels, 20, (0.1, 0.3, 0.6), 0.2, 10, seed=3, replacement=False)
    b = partition(labels, 20, (0.1, 0.3, 0.6), 0.2, 10, seed=3, replacement=False, size_sigma=0.0)
    assert (a.histograms == b.histograms).all()  # sigma = 0 reproduces the equal-size partition
    q = partition(labels, 20, (0.1, 0.3, 0.6), 0.2, 10, seed=3, replacement=False, size_sigma=1.0)
    n = q.histograms.sum(1)
    assert n.max() > 2 * n.min() and n.sum() <= len(labels) and n.max() <= q.capacity
    allidx = np.concatenate(q.client_indices)
    assert len(allidx) == len(np.unique(allidx))  # no example stored twice
    spec = importlib.util.spec_from_file_location("cm", "scripts/coverage_measures.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    h = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10], [10, 10, 10]])
    m = cm.measures([[0, 1, 2], [3]], h, np.arange(3), np.random.default_rng(0), extended=True)
    assert m["m0_full"] == 1.0 and m["m1_full"] == 1.0 and abs(m["m2_mean"] - 1.0) < 1e-9
    assert m["m4_mean"] < 1e-9 and m["m3_min"] == 1.0
    m = cm.measures([[0, 1], [2, 3]], h, np.arange(3), np.random.default_rng(0), extended=True)
    assert m["m0_full"] == 0.5 and m["m3_min"] == 0.5 and 0 < m["m5_p0.3"] <= 0.5
    lab, n = cm.pool_labels(10, 100.0)
    assert n[0] == 4500 and n[-1] == 45 and len(lab) == n.sum()


def test_coverage_order():
    import numpy as np
    from feddac import clustering as CL
    h = np.array([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100], [25, 25, 25, 25], [50, 50, 0, 0]])
    rng = np.random.default_rng(0)
    for _ in range(20):
        o = CL.coverage_order(list(range(6)), h, rng)
        assert sorted(o) == list(range(6))
        assert o[-1] == 4  # the client holding every class trains last
    # two complementary blocks: the last four placed clients after 4 cover all classes
    o = CL.coverage_order([0, 1, 2, 3], h, np.random.default_rng(1))
    assert sorted(o) == [0, 1, 2, 3]
    assert CL.coverage_order([3], h, rng) == [3]


def test_order_methods_run(tmp_path):
    import json
    import subprocess
    import sys
    for m in ("feddac_order", "rand_order"):
        subprocess.run([sys.executable, "scripts/run.py", "--dataset", "synthetic", "--clients", "10", "--method", m,
                        "--rounds", "2", "--no_replacement", "--out", str(tmp_path), "--exp", "t"], check=True,
                       capture_output=True)
    files = list((tmp_path / "t").glob("*.json"))
    assert len(files) == 2
    for f in files:
        r = json.load(open(f))
        assert r["config"]["method"] in ("feddac_order", "rand_order") and len(r["history"]) == 2
