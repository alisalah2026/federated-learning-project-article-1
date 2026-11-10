"""Test of scripts/seed_analysis.py and scripts/a0_table.py on a SYNTHETIC fixture.

The fixture is generated here from a seeded random generator and written to a temporary directory.
It has no relation to any experiment; none of its values may enter the manuscript.
"""
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import seed_analysis as sa  # noqa: E402


def make_fixture(root, seeds=range(5)):
    rng = np.random.default_rng(12345)
    shift = {"fedavg": 0.0, "feddac": 2.0, "fedprox": 0.2}
    for N in (20, 50):
        for m, sh in shift.items():
            for s in seeds:
                base = 50 + 3 * s * 0.1  # seed effect shared across methods -> paired design
                acc = base + sh + rng.normal(0, 0.1)
                rec = dict(config=dict(dataset="synthetic", method=m, clients=N, ratios=[0.1, 0.3, 0.6], alpha=0.2,
                                       seed=s, dp_eps=None, tag="SYNTHETIC", commit="synthetic0"),
                           final_acc=acc, final_f1=acc - 1, history=[dict(acc=acc * (r + 1) / 10) for r in range(10)])
                os.makedirs(root, exist_ok=True)
                json.dump(rec, open(os.path.join(root, f"{m}_{N}_{s}.json"), "w"))


def test_holm_fixed_family_keeps_undefined_settings():
    out = sa.holm_fixed([0.01, np.nan, 0.03, 0.04])
    assert np.allclose(out, [0.04, 1.0, 0.09, 0.09])  # m stays 4; NaN counted as p = 1.0
    assert sa.holm([0.01, np.nan, 0.03, 0.04])[0] == 0.03  # plain holm shrinks the family
    assert np.allclose(sa.holm_fixed([1.0, 1.0, 1.0, 1.0]), 1.0)
    for bad in ([0.01, 0.02, 0.03], [0.01] * 5):
        with pytest.raises(ValueError):
            sa.holm_fixed(bad)


def test_holm_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    # sorted 0.01,0.03,0.04,0.5 -> 0.04, 0.09, 0.09 (monotone), 0.5
    assert np.allclose(sa.holm(p), [0.04, 0.09, 0.09, 0.5])
    assert np.isnan(sa.holm([np.nan, 0.2])[0]) and sa.holm([np.nan, 0.2])[1] == 0.2


def test_paired_stats_exact_p_five_seeds():
    a = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    st = sa.paired_stats(a + 1, a)  # all differences positive: exact two-sided p = 2/2^5
    assert st["n"] == 5 and abs(st["p"] - 0.0625) < 1e-12 and st["mean_diff"] == pytest.approx(1.0)


def brute_force_p(d):
    """Reference: enumerate all 2^n sign patterns (mid-ranks for tied |d|, zeros dropped)."""
    d = np.asarray([x for x in d if x != 0], float)
    n = len(d)
    if n == 0:
        return 1.0, 0
    a = np.abs(d)
    ranks = np.array([np.mean([i + 1 for i in range(n) if np.sort(a)[i] == x]) for x in a])
    obs = ranks[d > 0].sum()
    tot = ranks.sum()
    cnt = 0
    for signs in itertools.product([0, 1], repeat=n):
        w = ranks[np.array(signs, bool)].sum()
        cnt += abs(w - tot / 2) >= abs(obs - tot / 2) - 1e-9
    return cnt / 2 ** n, n


@pytest.mark.parametrize("d", [
    [1.0, 2.0, 3.0, 4.0, 5.0],            # all positive, no ties: 0.0625
    [1.0, -2.0, 3.0, 4.0, 5.0],           # mixed signs
    [0.0, 1.0, 2.0, 3.0, 4.0],            # one zero difference: n_eff = 4, p_min = 0.125
    [1.0, 1.0, -1.0, 2.0, 2.0],           # ties in |d|
    [0.0, 0.0, 0.0, 0.0, 0.0],            # all zero
    [0.0, 0.0, 0.0, 1.0, 2.0],            # n_eff = 2
    [-1.0, -1.0, -1.0, -1.0, -1.0],       # all negative with ties
])
def test_wilcoxon_exact_matches_enumeration(d):
    w = sa.wilcoxon_exact(d)
    p, n = brute_force_p(d)
    assert w["p"] == pytest.approx(p, abs=1e-12) and w["n_eff"] == n and w["n_zero"] == 5 - n


def test_wilcoxon_zero_and_tie_flags():
    assert sa.wilcoxon_exact([0.0, 1.0, 2.0, 3.0, 4.0])["p_min"] == pytest.approx(0.125)  # 2/2^4
    assert sa.wilcoxon_exact([0.0] * 5)["p"] == 1.0 and sa.wilcoxon_exact([0.0] * 5)["n_eff"] == 0
    w = sa.wilcoxon_exact([1.0, 1.0, 2.0, 3.0, 4.0])
    assert w["ties"] and not sa.wilcoxon_exact([1.0, 2.0, 3.0, 4.0, 5.0])["ties"]


def test_wilcoxon_exact_agrees_with_scipy_without_ties_or_zeros():
    from scipy import stats
    rng = np.random.default_rng(7)
    for n in (5, 8, 12):
        d = rng.normal(0.3, 1.0, n)
        assert sa.wilcoxon_exact(d)["p"] == pytest.approx(stats.wilcoxon(d, method="exact").pvalue, abs=1e-12)


def test_paired_stats_reports_effective_n():
    a = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    st = sa.paired_stats(a + np.array([0, 1, 1, 1, 1]), a)
    assert st["n"] == 5 and st["n_eff"] == 4 and st["n_zero"] == 1 and st["p"] == pytest.approx(0.125)


def _write(root, name, rec):
    os.makedirs(root, exist_ok=True)
    json.dump(rec, open(os.path.join(root, name), "w"))


def _rec(method="fedavg", seed=0, acc=50.0, **cfg):
    c = dict(dataset="synthetic", method=method, clients=20, ratios=[0.1, 0.3, 0.6], alpha=0.2, seed=seed,
             dp_eps=None, tag="SYNTHETIC", rounds=10, commit="abc1234")
    c.update(cfg)
    return dict(config=c, final_acc=acc, final_f1=acc - 1, history=[])


def test_duplicate_records_are_rejected_not_overwritten(tmp_path):
    fx, out = tmp_path / "fx", tmp_path / "out"
    _write(fx, "a.json", _rec(acc=50.0))
    _write(fx, "b.json", _rec(acc=51.0))  # same setting, method, seed
    with pytest.raises(SystemExit):
        sa.main(["--results", str(fx), "--out", str(out)])
    rep = json.load(open(out / "input_report.json"))
    assert rep["problems"][0]["kind"] == "duplicate" and len(rep["problems"][0]["files"]) == 2
    with pytest.raises(sa.InputError):
        sa.analyse(sa.load(str(fx)), "fedavg")


def test_conflicting_configuration_is_rejected(tmp_path):
    fx, out = tmp_path / "fx", tmp_path / "out"
    _write(fx, "a.json", _rec(seed=0, lr=0.01))
    _write(fx, "b.json", _rec(seed=1, lr=0.05))  # same setting and method, different learning rate
    with pytest.raises(SystemExit):
        sa.main(["--results", str(fx), "--out", str(out)])
    rep = json.load(open(out / "input_report.json"))
    assert rep["problems"][0]["kind"] == "conflicting_config" and rep["problems"][0]["fields"] == ["lr"]


def test_different_rounds_are_different_settings(tmp_path):
    fx, out = tmp_path / "fx", tmp_path / "out"
    for i, rounds in enumerate((10, 20)):
        _write(fx, f"r{i}.json", _rec(rounds=rounds))
    sa.main(["--results", str(fx), "--out", str(out)])
    assert len(json.load(open(out / "stats.json"))["rows"]) == 2


def test_unreadable_and_invalid_inputs_are_reported(tmp_path):
    fx, out = tmp_path / "fx", tmp_path / "out"
    _write(fx, "ok.json", _rec())
    os.makedirs(fx, exist_ok=True)
    (fx / "broken.json").write_text("{not json")
    _write(fx, "nometric.json", dict(config=dict(method="fedavg", seed=3)))
    with pytest.raises(SystemExit):
        sa.main(["--results", str(fx), "--out", str(out)])
    rep = json.load(open(out / "input_report.json"))
    assert len(rep["unreadable"]) == 1 and len(rep["invalid"]) == 1
    sa.main(["--results", str(fx), "--out", str(out), "--skip-invalid"])  # explicit opt-in; still listed
    assert len(json.load(open(out / "input_report.json"))["unreadable"]) == 1


def test_provenance_kept_and_mixed_commits_reported(tmp_path):
    fx, out = tmp_path / "fx", tmp_path / "out"
    _write(fx, "a.json", _rec(seed=0))
    r = _rec(seed=1)
    r["config"].pop("commit")
    r["env"] = dict(commit="def5678")
    _write(fx, "b.json", r)
    sa.main(["--results", str(fx), "--out", str(out)])
    rep = json.load(open(out / "input_report.json"))
    assert {x["commit"] for x in rep["sources"]} == {"abc1234", "def5678"}
    assert all(x["config"]["method"] == "fedavg" and x["file"].endswith(".json") for x in rep["sources"])
    assert rep["provenance"][0]["commits"] == ["abc1234", "def5678"]


def test_pipeline_outputs(tmp_path):
    fx, out = tmp_path / "fx", tmp_path / "out"
    make_fixture(str(fx))
    sa.main(["--results", str(fx), "--out", str(out), "--reference", "fedavg"])
    for f in ("stats.csv", "stats.json", "table_final_acc.tex", "fig_final_acc.png", "fig_final_acc.pdf",
              "fig_curves.png", "fig_curves.pdf"):
        assert (out / f).exists(), f
    st = json.load(open(out / "stats.json"))
    c = [x for x in st["comparisons"] if x["method"] == "feddac"]
    assert len(c) == 2 and all(1.5 < x["mean_diff"] < 2.5 for x in c)
    assert all(abs(x["p"] - 0.0625) < 1e-12 for x in c)  # 5 paired seeds, all positive
    assert all(x["p_holm"] >= x["p"] for x in st["comparisons"])
    tex = (out / "table_final_acc.tex").read_text()
    assert "\\toprule" in tex and "\\bottomrule" in tex and "smallest attainable" in tex and "n_{\\mathrm{eff}}" in tex
    from PIL import Image
    assert Image.open(out / "fig_final_acc.png").info.get("dpi", (0,))[0] > 299


def test_missing_cells_do_not_crash(tmp_path):
    fx, out = tmp_path / "fx", tmp_path / "out"
    make_fixture(str(fx), seeds=range(3))
    os.remove(fx / "fedprox_50_1.json")
    sa.main(["--results", str(fx), "--out", str(out)])
    st = json.load(open(out / "stats.json"))
    assert min(r["n"] for r in st["rows"]) == 2


def test_a0_table(tmp_path):
    d = dict(summary=dict(gpu="SYNTHETIC", pairs=1, pairs_identical=1, phases=dict(resnet=dict(packing=None), cnn=dict(packing=0.5))),
             pairs=[dict(kind="cnn", method="fedavg", rounds=10, final_acc=[1.0, 1.0], run_h=[0.1, 0.1], history_identical=True)])
    json.dump(d, open(tmp_path / "s.json", "w"))
    subprocess.check_call([sys.executable, os.path.join(HERE, "..", "scripts", "a0_table.py"), str(tmp_path / "s.json"), str(tmp_path / "t.tex")])
    assert "\\toprule" in (tmp_path / "t.tex").read_text()


def test_a0_table_tolerates_missing_values(tmp_path):
    """Missing accuracy, timing, rounds, repeat flag, GPU and phase values print as '--' (SYNTHETIC)."""
    d = dict(summary=dict(phases=dict(cnn=dict(packing=None))),
             pairs=[dict(kind="cnn", method="fedavg"),
                    dict(kind="resnet", method="feddac", rounds=None, final_acc=[None, 1.0], run_h=[None, None]),
                    dict(kind="dp", method="fedavg", rounds=5, final_acc=[], run_h=[0.2, None], history_identical=None)])
    json.dump(d, open(tmp_path / "s.json", "w"))
    subprocess.check_call([sys.executable, os.path.join(HERE, "..", "scripts", "a0_table.py"), str(tmp_path / "s.json"), str(tmp_path / "t.tex")])
    tex = (tmp_path / "t.tex").read_text()
    assert "cnn & fedavg & -- & -- & -- & --" in tex and "dp & fedavg & 5 & -- & 0.200 & --" in tex
    assert "GPU: --" in tex and "-- of --" in tex
    json.dump({}, open(tmp_path / "e.json", "w"))  # no summary, no pairs
    subprocess.check_call([sys.executable, os.path.join(HERE, "..", "scripts", "a0_table.py"), str(tmp_path / "e.json"), str(tmp_path / "e.tex")])


@pytest.mark.parametrize("bad", [None, [], [1, 2], 5, "fedavg", True])
def test_malformed_config_is_reported_not_crashing(tmp_path, bad):
    fx, out = tmp_path / "fx", tmp_path / "out"
    _write(fx, "ok.json", _rec())
    _write(fx, "badcfg.json", dict(config=bad, final_acc=50.0))
    with pytest.raises(SystemExit):  # default: rejected
        sa.main(["--results", str(fx), "--out", str(out)])
    rep = json.load(open(out / "input_report.json"))
    assert len(rep["invalid"]) == 1 and "not a dictionary" in rep["invalid"][0]["error"]
    sa.main(["--results", str(fx), "--out", str(out), "--skip-invalid"])  # explicit skip; still listed
    rep = json.load(open(out / "input_report.json"))
    assert len(rep["invalid"]) == 1 and len(rep["sources"]) == 1


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize("field", ["final_acc", "final_f1"])
def test_non_finite_metrics_are_rejected(tmp_path, bad, field):
    fx, out = tmp_path / "fx", tmp_path / "out"
    _write(fx, "ok.json", _rec(seed=0))
    r = _rec(seed=1)
    r[field] = bad  # json.dump writes Infinity/NaN, which json.load reads back
    _write(fx, "bad.json", r)
    with pytest.raises(SystemExit):
        sa.main(["--results", str(fx), "--out", str(out), "--metric", field])
    rep = json.load(open(out / "input_report.json"))
    assert len(rep["invalid"]) == 1 and "finite" in rep["invalid"][0]["error"]
    sa.main(["--results", str(fx), "--out", str(out), "--metric", field, "--skip-invalid"])
    assert len(json.load(open(out / "input_report.json"))["sources"]) == 1


def test_analyse_refuses_non_finite_metric_directly():
    recs = [dict(setting=("s",), method="fedavg", seed=0, final_acc=float("inf"), final_f1=None, file="x")]
    with pytest.raises(sa.InputError):
        sa.analyse(recs, "fedavg", "final_acc")


def _by(setting_vals):
    return {s: {"feddac": dict(zip(range(len(f)), f)), "random_seq": dict(zip(range(len(r)), r))}
            for s, (f, r) in setting_vals.items()}


def test_holm_family_rejects_nan_and_wrong_size():
    with pytest.raises(ValueError):
        sa.holm_family([0.01, float("nan"), 0.5, 0.2], 4)
    with pytest.raises(ValueError):
        sa.holm_family([0.01, 0.5, 0.2], 4)
    assert np.allclose(sa.holm_family([0.01, 0.04, 0.03, 0.5], 4), [0.04, 0.09, 0.09, 0.5])


def test_confirmatory_keeps_four_settings_with_missing_and_all_zero():
    base = [50.0, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    by = _by({
        "A": ([x + d for x, d in zip(base, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])], base),  # all positive
        "B": (base, base),                                                          # all zero
        "C": ([60.0, 61.0], [50.0, 52.0]),                                          # 2 pairs only
        "D": ([], []),                                                              # nothing
    })
    by["C"]["feddac"][5] = 70.0  # seed 5 has no reference run
    res = sa.confirmatory(by, ["A", "B", "C", "D"], "feddac", "random_seq", n_boot=200)
    assert [c["setting"] for c in res] == ["A", "B", "C", "D"]
    a, b, c, d = res
    assert a["n_eff"] == 10 and a["p"] == pytest.approx(2 / 1024)
    assert a["p_holm"] == pytest.approx(4 * 2 / 1024)
    assert b["p"] == 1.0 and b["n_eff"] == 0 and b["p_holm"] == 1.0
    assert c["seeds"] == [0, 1] and dict(seed=5, reason="no random_seq run") in c["missing"]
    assert dict(seed=2, reason="no feddac run and no random_seq run") in c["missing"] and len(c["missing"]) == 8
    assert d["p"] == 1.0 and d["n"] == 0 and d["p_holm"] == 1.0
    # the family is always 4: the all-zero setting's p = 1.0 is counted, so A's Holm factor is 4, not 3
    assert all(not np.isnan(x["p_holm"]) for x in res)


def test_confirmatory_unknown_setting_counts_as_no_pairs():
    res = sa.confirmatory({}, ["A", "B", "C", "D"], "feddac", "random_seq")
    assert [c["p_holm"] for c in res] == [1.0] * 4


def test_confirmatory_requires_exactly_four_distinct_settings():
    for fam in (["A", "B", "C"], ["A", "B", "C", "D", "E"], ["A", "A", "B", "C"]):
        with pytest.raises(ValueError):
            sa.confirmatory({}, fam, "feddac", "random_seq")


def test_confirmatory_uses_only_prespecified_seeds_and_records_nonfinite():
    ok = {k: 50.0 + k for k in range(10)}
    f = dict(ok)
    f.update({10: 99.0, 11: 99.0})              # seeds outside 0-9 are never analysed
    f[3] = float("nan")                          # non-finite is an incomplete pair, not a value
    by = {s: {"feddac": f, "random_seq": {k: v - 1 for k, v in ok.items()}} for s in "ABCD"}
    res = sa.confirmatory(by, list("ABCD"), "feddac", "random_seq", n_boot=100)
    for c in res:
        assert c["seeds"] == [0, 1, 2, 4, 5, 6, 7, 8, 9] and c["n"] == 9
        assert c["excluded_seeds"] == [10, 11]
        assert c["missing"] == [dict(seed=3, reason="no feddac run")]
