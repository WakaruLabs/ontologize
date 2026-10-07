# headnmi.py's contingency, NMI, null and ordering helpers on
# hand-checkable partitions, and the figure drawn from synthetic labels.
import numpy as np
import pytest

import headnmi


def test_contingency_matches_bincount():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 4, size=(500, 3))
    C = headnmi.contingency(A, 4, chunk=128)
    assert C.shape == (3, 3, 4, 4)
    for a in range(3):
        for b in range(3):
            ref = np.bincount(A[:, a] * 4 + A[:, b], minlength=16)
            assert np.array_equal(C[a, b].reshape(-1), ref)
    B = rng.integers(0, 4, size=(500, 2))
    Cx = headnmi.contingency(A, 4, B)
    assert Cx.shape == (3, 2, 4, 4)
    assert np.array_equal(Cx[1, 0].reshape(-1),
                          np.bincount(A[:, 1] * 4 + B[:, 0], minlength=16))


def test_identical_and_relabelled_partitions_are_one():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, 4000)
    relabel = np.array([3, 0, 4, 1, 2])[a]
    st = headnmi.head_nmi(np.stack([a, a, relabel], 1), 5)
    assert np.allclose(st["nmi"], 1.0, atol=1e-6)
    assert np.allclose(st["adj"], 1.0, atol=1e-6)


def test_independent_partitions_near_zero_after_adjustment():
    rng = np.random.default_rng(2)
    A = rng.integers(0, 8, size=(4000, 6))
    st = headnmi.head_nmi(A, 8)
    off = ~np.eye(6, dtype=bool)
    # raw NMI carries the finite-N bias, ~ (k-1)^2 / (2 N H)
    bias = 49 / (2 * 4000 * np.log(8))
    assert st["nmi"][off].mean() == pytest.approx(bias, rel=0.3)
    assert abs(st["adj"][off].mean()) < 0.2 * bias


def test_analytic_null_agrees_with_permutation():
    rng = np.random.default_rng(3)
    A = rng.integers(0, 8, size=(4000, 5))
    I, H, r, N = headnmi.mutual_info(headnmi.contingency(A, 8))
    off = ~np.eye(5, dtype=bool)
    Ea = headnmi.analytic_null(r, N)[off].mean()
    Ep = headnmi.perm_null(A, 8, 8, 0)[off].mean()
    assert Ep == pytest.approx(Ea, rel=0.15)
    assert I[off].mean() == pytest.approx(Ea, rel=0.3)  # A is independent


def test_known_mutual_information():
    # b = a // 2: I(a; b) = H(b) = log 2 for uniform a on 4 labels
    a = np.tile(np.arange(4), 250)
    I, H, r, N = headnmi.mutual_info(headnmi.contingency(
        np.stack([a, a // 2], 1), 4))
    assert N == 1000 and r.tolist() == [4, 2]
    assert H == pytest.approx([np.log(4), np.log(2)])
    assert I[0, 1] == pytest.approx(np.log(2))
    nmi, _ = headnmi.normalized(I, H, np.zeros_like(I))
    assert nmi[0, 1] == pytest.approx(np.log(2) / (0.5 * np.log(8)))


def test_constant_partition_is_zero_not_nan():
    rng = np.random.default_rng(4)
    A = np.stack([np.zeros(1000, int), rng.integers(0, 4, 1000),
                  np.full(1000, 2)], 1)
    st = headnmi.head_nmi(A, 4)
    for key in ("nmi", "adj", "I"):
        assert np.isfinite(st[key]).all()
    assert st["H"][0] == 0 and st["r"][0] == 1
    assert st["nmi"][0, 1] == 0 and st["adj"][0, 1] == 0
    assert st["nmi"][0, 2] == 0 and st["adj"][0, 2] == 0  # both constant


def test_head_order_blocks_and_dead_last():
    # two layers of three heads; heads 0 and 2 identical, head 4 dead
    S = np.eye(6)
    S[0, 2] = S[2, 0] = 0.9
    S[3, 5] = S[5, 3] = 0.8
    layer = np.repeat([0, 1], 3)
    dead = np.zeros(6, bool)
    dead[4] = True
    idx = headnmi.head_order(S, layer, dead, "index")
    assert idx.tolist() == [0, 1, 2, 3, 5, 4]
    hc = headnmi.head_order(S, layer, dead, "hclust")
    assert sorted(hc[:3].tolist()) == [0, 1, 2] and hc[-1] == 4
    pos = {v: i for i, v in enumerate(hc)}
    assert abs(pos[0] - pos[2]) == 1  # the redundant pair sits together
    gl = headnmi.head_order(S, layer, dead, "global")
    assert gl[-1] == 4 and sorted(gl.tolist()) == list(range(6))


def test_hclust_order_recovers_blocks():
    S = np.full((6, 6), 0.05)
    for blk in ([0, 3, 5], [1, 2, 4]):
        S[np.ix_(blk, blk)] = 0.9
    o = headnmi.hclust_order(S).tolist()
    assert set(o[:3]) in ({0, 3, 5}, {1, 2, 4})


def test_pair_means_and_top_pairs():
    layer = np.repeat([0, 1, 2], 2)
    S = np.zeros((6, 6))
    S[0, 1] = S[1, 0] = 0.6          # within layer 0
    S[1, 2] = S[2, 1] = 0.4          # adjacent
    S[0, 4] = S[4, 0] = 0.2          # distant
    np.fill_diagonal(S, 1.0)
    dead = np.zeros(6, bool)
    dead[5] = True
    pm = headnmi.pair_means(S, layer, dead)
    assert pm["within"] == pytest.approx(0.6 / 2)  # pairs (0,1) and (2,3)
    assert pm["adjacent"] == pytest.approx(0.4 / 6)
    assert pm["distant"] == pytest.approx(0.2 / 2)  # head 5 is dead
    assert pm["blocks"].shape == (3, 3)
    top = headnmi.top_pairs(S, dead, 2)
    assert [(a, b) for a, b, _ in top] == [(0, 1), (1, 2)]


def test_load_ei_layout(tmp_path):
    l, h = 2, 3
    E = np.full((l, h, l, h), np.nan)
    E[0, 1, 1, 2] = 0.5
    np.save(tmp_path / "ei_live.npy", E)
    M = headnmi.load_ei(tmp_path, "ei_live", l, h)
    assert M[1, h + 2] == 0.5 and np.isfinite(M).sum() == 1
    with pytest.raises(AssertionError):
        headnmi.load_ei(tmp_path / "ei_live.npy", "", 3, 2)


@pytest.mark.parametrize("order", ["index", "hclust", "global"])
def test_draw(tmp_path, order):
    rng = np.random.default_rng(5)
    l, h, k = 3, 4, 5
    base = rng.integers(0, k, 600)
    A = rng.integers(0, k, size=(600, l * h))
    A[:, 1] = A[:, 6] = base         # a cross-layer redundant pair
    A[:, 11] = 0                     # a dead head
    st = headnmi.head_nmi(A, k)
    st["dead"] = np.exp(st["H"]) < 1.1
    assert st["dead"].tolist() == [False] * 11 + [True]
    ei = np.where(np.repeat(np.arange(l), h)[:, None]
                  < np.repeat(np.arange(l), h)[None, :],
                  rng.random((l * h, l * h)), np.nan)
    for stat, e, log in (("adj", None, True), ("nmi", ei, True),
                         ("adj", ei, False)):
        path = tmp_path / f"{order}_{stat}_{log}.png"
        lim = (1e-3, 1.0) if log else (0.0, 0.5)
        headnmi.draw(st, l, h, order, stat, lim, log, "t", path, e,
                     "ei_live")
        assert path.stat().st_size > 0


def test_ei_limits():
    ei = np.full((4, 4), np.nan)
    ei[0, 1:] = [0.0, 0.01, 0.1]
    lo, hi = headnmi.ei_limits(ei, True)
    assert 0.01 <= lo < hi <= 0.1  # zeros excluded from the log range
    assert headnmi.ei_limits(ei, False)[0] == 0.0
    assert headnmi.ei_limits(ei, True, (0.5, 2.0)) == (0.5, 2.0)
