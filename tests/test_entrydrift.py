# entrydrift.py's column, metric and row-layout helpers, pinned on
# hand-checkable cases, and pages drawn from synthetic metrics.
import sys

import numpy as np
import pytest

import entrydrift


def test_spaced_steps_keeps_ends_and_skips_bursts():
    steps = [10, 20, 30, 40, 50, 100, 101, 102, 103, 104, 105]
    got = entrydrift.spaced_steps(steps, 4)
    assert got[0] == 10 and got[-1] == 105
    # targets 10, 41.7, 73.3, 105: nearest 10, 40, 50, 105, so the burst
    # of saves at 100-105 contributes only its last
    assert got == [10, 40, 50, 105]
    assert entrydrift.spaced_steps([3, 1, 2], 5) == [1, 2, 3]


def test_parse_run_and_step_label():
    assert entrydrift.parse_run("data/out/x:5000") == ("data/out/x", 5000)
    assert entrydrift.parse_run("data/out/x") == ("data/out/x", 0)
    assert entrydrift.step_label(50000) == "50k"
    assert entrydrift.step_label(372600) == "372600"
    assert entrydrift.step_label(500) == "500"


def test_mark_boundary():
    steps = [10, 30, 60, 80]
    assert entrydrift.mark_boundary(steps, 50, None) == 2
    assert entrydrift.mark_boundary(steps, 5, None) is None   # nothing before
    assert entrydrift.mark_boundary(steps, 90, None) is None  # nothing after
    assert entrydrift.mark_boundary(steps, None, 0) == 1
    assert entrydrift.mark_boundary(steps, None, None) is None


def test_entry_dirs_removes_offset_and_keeps_layout():
    # embed(code) = code_flat @ W + b: entry (a, b, e) decodes to W[f] + b,
    # f = (a*h + b)*k + e, and the offset b is removed
    l, h, k, d = 2, 3, 4, 5
    rng = np.random.default_rng(0)
    W = rng.normal(size=(l * h * k, d)).astype(np.float32)
    bias = rng.normal(size=d).astype(np.float32)
    embed = lambda c: c.reshape(len(c), -1) @ W + bias
    U = entrydrift.entry_dirs(embed, l, h, k, d, chunk=7)
    assert U.shape == (l, h, k, d)
    np.testing.assert_allclose(U[1, 2, 3], W[(1 * h + 2) * k + 3], atol=1e-5)
    np.testing.assert_allclose(U.reshape(-1, d), W, atol=1e-5)


def test_drift_cos():
    U = np.zeros((1, 1, 3, 2))
    U[0, 0] = [[1, 0], [0, 2], [1, 1]]
    R = np.zeros((1, 1, 3, 2))
    R[0, 0] = [[3, 0], [0, -1], [1, 0]]
    np.testing.assert_allclose(entrydrift.drift_cos(U, R)[0, 0],
                               [1.0, -1.0, 1 / np.sqrt(2)], atol=1e-6)


def test_head_match_undoes_a_permutation():
    rng = np.random.default_rng(1)
    R = rng.normal(size=(2, 2, 5, 8))
    p = np.array([3, 0, 4, 1, 2])
    U = R.copy()
    U[1, 0] = R[1, 0][np.argsort(p)]   # reference entry j now at index p[j]
    perm, cos = entrydrift.head_match(U, R)
    assert perm[1, 0].tolist() == p.tolist()
    assert (perm[0, 0] == np.arange(5)).all()
    np.testing.assert_allclose(cos, 1.0, atol=1e-6)
    frac = entrydrift.self_match(perm)
    assert frac[0].tolist() == [1.0, 1.0]
    assert frac[1, 0] == 0.0 and frac[1, 1] == 1.0


def test_select_heads_and_row_layout():
    pairs = entrydrift.select_heads(3, 4, [0, 2], [1, 3])
    assert pairs == [(0, 1), (0, 3), (2, 1), (2, 3)]
    usage = np.zeros((3, 4, 3))
    usage[0, 1] = [0.1, 0.7, 0.2]
    usage[0, 3] = [0.5, 0.0, 0.5]   # tie: stable, lower index first
    rows, hs, ls = entrydrift.row_layout(pairs, usage)
    assert rows[:6].tolist() == [[0, 1, 1], [0, 1, 2], [0, 1, 0],
                                 [0, 3, 0], [0, 3, 2], [0, 3, 1]]
    assert len(rows) == 12
    assert hs == [2.5, 8.5] and ls == [5.5]


def test_gather():
    V = np.arange(2 * 1 * 2 * 3).reshape(2, 1, 2, 3).astype(float)
    rows = np.array([[0, 1, 2], [0, 0, 0]])
    assert entrydrift.gather(V, rows).tolist() == [[5, 11], [0, 6]]


def synthetic_Z(c=4, l=2, h=3, k=4, seed=0):
    rng = np.random.default_rng(seed)
    usage = rng.dirichlet(np.ones(k), size=(c, l, h))
    usage[:, 0, 0] = 0
    usage[:, 0, 0, 0] = 1   # a collapsed head: three entries dead
    drift = rng.uniform(-1, 1, size=(c, l, h, k))
    drift[-1] = 1
    return {"usage": usage, "drift": drift, "match": np.abs(drift),
            "ref": c - 1, "cols": np.asarray([f"{10 * j}k" for j in range(c)]),
            "steps": np.arange(c) * 10000,
            "temps": np.geomspace(1.0, 0.03, c), "title": "t",
            "mark_step": 15000}


def test_draw_all_pages(tmp_path):
    from types import SimpleNamespace
    Z = synthetic_Z()
    cfg = SimpleNamespace(vmax=None, mark_step=None, mark_col=None,
                          layers=None, heads=None, max_heads=4,
                          metrics=["usage", "drift", "match"], no_mask=False)
    paths = entrydrift.draw_all(cfg, Z, tmp_path)
    assert [p.name for p in paths] == ["drift_p0.png", "drift_p1.png"]
    assert all(p.stat().st_size > 0 for p in paths)


def test_replot_from_npz(tmp_path, monkeypatch):
    np.savez(tmp_path / "drift.npz", **synthetic_Z())
    monkeypatch.setattr(sys, "argv", [
        "entrydrift.py", "--ckpt", "unused", "--replot", "--out",
        str(tmp_path), "--layers", "1", "--metrics", "usage", "match"])
    entrydrift.main()
    assert (tmp_path / "drift_p0.png").stat().st_size > 0
