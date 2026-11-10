"""CPU tests of the resubmission protocol (docs/PLAN.md section 5): no-replacement partitioner,
declared public capacity, validation split, private-grouping specification, deterministic mode."""
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from feddac import clustering as CL  # noqa: E402
from feddac.dp import ClientDP  # noqa: E402
from feddac.data import partition, validation_split  # noqa: E402

Y = np.repeat(np.arange(10), 1000)


def test_default_partition_unchanged():
    """replacement=True reproduces the submitted protocol bit for bit (fingerprint of the old code)."""
    h = hashlib.sha1()
    for seed in range(3):
        p = partition(Y, 20, (0.1, 0.3, 0.6), 0.2, 10, seed=seed)
        for ix in p.client_indices:
            h.update(np.asarray(ix, dtype=np.int64).tobytes())
        h.update(str(p.reused_samples).encode())
    assert h.hexdigest() == "9e73887b3177ee493930f83f815f0a05d742f0e4"


def test_no_replacement_partition():
    for ratios, alpha in (((0.1, 0.3, 0.6), 0.2), ((0.1, 0.0, 0.9), 0.2), ((0.0, 1.0, 0.0), 0.1)):
        for seed in range(3):
            p = partition(Y, 20, ratios, alpha, 10, seed=seed, replacement=False)
            allix = np.concatenate(p.client_indices)
            assert len(allix) == len(np.unique(allix)), "an example is stored twice"
            assert p.reused_samples == 0
            sizes = p.histograms.sum(1)
            assert 0 < sizes.min() and sizes.max() <= p.capacity == 500
            assert sizes.sum() + p.dropped_samples == 20 * 500
            for i, ix in enumerate(p.client_indices):
                assert (np.bincount(Y[ix], minlength=10) == p.histograms[i]).all()
                if p.client_type[i] == "extreme":
                    assert (p.histograms[i] > 0).sum() <= 2


def test_capacity_makes_mechanism_size_independent():
    kw = dict(batch_size=64, rounds=5, local_epochs=1, eps_sgd=3.5, delta=1e-5)
    a = ClientDP(3000, capacity=3000, **kw)
    b = ClientDP(2999, capacity=3000, **kw)
    c = ClientDP(2999, **kw)  # submitted protocol: q from the realized size
    assert (a.q, a.total_steps, a.sigma) == (b.q, b.total_steps, b.sigma)
    assert c.q != a.q


def test_validation_split():
    tr, va = validation_split(Y, 0.1, seed=0)
    assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(Y)
    assert (np.bincount(Y[va], minlength=10) == 100).all()


def test_degenerate_noisy_histogram_goes_to_smallest_group():
    rng = np.random.default_rng(0)
    p = partition(Y, 20, (0.1, 0.3, 0.6), 0.2, 10, seed=0)
    h = CL.laplace_histograms(p.histograms, 0.5, rng)
    assert (h >= 0).all() and (h == np.round(h)).all()
    h[0] = 0  # all-zero noisy row
    ext, _, _ = CL.classify_clients(h)
    assert 0 in ext
    g = CL.dac_cluster(h, CL.num_clusters(20))
    assert sorted(c for x in g for c in x) == list(range(20))


def _run(tmp_path, out, *extra):
    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts/run.py"), "--dataset", "synthetic",
                           "--clients", "10", "--rounds", "2", "--out", str(tmp_path / out), *extra])
    (f,) = glob.glob(str(tmp_path / out / "default" / "*.json"))
    return json.load(open(f))


def test_deterministic_repeats_identical(tmp_path):
    flags = ["--method", "feddac", "--deterministic", "--no_replacement", "--val_frac", "0.1"]
    r1, r2 = _run(tmp_path, "a", *flags), _run(tmp_path, "b", *flags)
    assert r1["history"] == r2["history"] and r1["deterministic"]
    assert r1["replacement"] is False and r1["reused_samples"] == 0
    assert "final_val_acc" in r1


def test_dp_capacity_end_to_end(tmp_path):
    r = _run(tmp_path, "dp", "--method", "fedavg", "--dp_eps", "4", "--no_replacement", "--deterministic")
    assert r["dp"]["capacity"] == r["capacity"] == 6000 // 10  # synthetic: 6,000 training examples
    assert r["dp"]["q_min"] == r["dp"]["q_max"] == 64 / r["capacity"]
    assert max(r["client_sizes"]) <= r["capacity"]
