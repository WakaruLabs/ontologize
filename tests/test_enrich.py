# enrich.py's counting, hypergeometric tails, odds ratio, BH and NMI on
# hand-checkable cases, and the figures drawn from synthetic assignments.
import importlib.util
import sys
from math import comb
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import enrich


def exact_tail(x, N, K, n, upper):
    """P(X >= x) or P(X <= x) for X ~ Hypergeom(N, K, n) by enumeration."""
    pmf = lambda i: comb(K, i) * comb(N - K, n - i) / comb(N, n)
    support = range(max(0, n - (N - K)), min(K, n) + 1)
    return sum(pmf(i) for i in support if (i >= x if upper else i <= x))


def test_count_table():
    A = np.array([[0, 1], [0, 1], [1, 0]])  # n=3 rows, m=2 heads, k=2
    y = np.array([0, 1, 1])
    C = enrich.count_table(A, y, 2, 2)
    assert C.shape == (2, 2, 2)
    assert C[0].tolist() == [[1, 1], [0, 1]]
    assert C[1].tolist() == [[0, 1], [1, 1]]


def test_soft_table_matches_hard_for_one_hot():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 4, (50, 3))
    y = rng.integers(0, 5, 50)
    P = np.eye(4)[A]  # (n, m, k)
    np.testing.assert_allclose(enrich.soft_table(P, y, 5),
                               enrich.count_table(A, y, 4, 5))


@pytest.mark.parametrize("x,N,K,n", [(0, 20, 8, 6), (3, 20, 8, 6),
                                      (6, 20, 8, 6), (1, 30, 3, 10)])
def test_hyper_tails_match_enumeration(x, N, K, n):
    a = lambda v: np.array([v])
    up = enrich.hyper_log10p(a(x), a(N), a(K), a(n), "greater")[0]
    lo = enrich.hyper_log10p(a(x), a(N), a(K), a(n), "less")[0]
    assert 10 ** up == pytest.approx(exact_tail(x, N, K, n, True), rel=1e-9)
    assert 10 ** lo == pytest.approx(exact_tail(x, N, K, n, False), rel=1e-9)
    two = 10 ** enrich.hyper_log10p(a(x), a(N), a(K), a(n), "two")[0]
    side = exact_tail(x, N, K, n, x * N >= n * K)
    assert two == pytest.approx(min(1.0, 2 * side), rel=1e-9)


def test_zero_count_is_never_enriched():
    # entry 0 holds half the rows, label 0 half, and they never meet: E =
    # 25, x = 0. The enrichment tail is exactly 1 (the old phyper(q - 1)
    # bug made it 0); the cell is significant only as a depletion
    C = np.zeros((1, 2, 2))
    C[0] = [[0, 50], [50, 0]]
    R = enrich.enrichment(C, "greater", 5.0)
    assert R.tested[0, 0, 0]
    assert R.log10p[0, 0, 0] == 0.0 and R.log10q[0, 0, 0] == 0.0
    R2 = enrich.enrichment(C, "two", 5.0)
    assert R2.log10q[0, 0, 0] < np.log10(0.05)
    assert R2.log2or[0, 0, 0] < 0  # significant, and as a depletion
    assert R2.log2or[0, 0, 1] > 0


def test_zero_count_small_expected_untested():
    # x = 0 with E = 1 is below min_count: untested, so no dot at all
    C = np.zeros((1, 2, 2))
    C[0] = [[0, 10], [10, 80]]  # E[0, 0] = 10 * 10 / 100 = 1
    R = enrich.enrichment(C, "two", 5.0)
    assert not R.tested[0, 0, 0] and np.isnan(R.log10q[0, 0, 0])
    assert R.tested[0, 1, 1]


def test_log2_odds_ratio_haldane():
    C = np.array([[[3, 1], [2, 4]]], float)  # x=3: a=3 b=1 c=2 d=4
    M = enrich.margins(C)
    want = np.log2(3.5 * 4.5 / (1.5 * 2.5))
    assert enrich.log2_odds_ratio(M)[0, 0, 0] == pytest.approx(want)
    # independence gives OR ~ 1 up to the correction
    C = np.array([[[10, 10], [10, 10]]], float)
    assert enrich.log2_odds_ratio(enrich.margins(C))[0, 0, 0] == 0.0


def test_bh_hand_case():
    # sorted p = .005 .01 .03 .04, m = 4: p*m/i = .02 .02 .04 .04, already
    # monotone; returned in input order
    p = np.array([0.01, 0.04, 0.03, 0.005])
    q = 10 ** enrich.bh_log10(np.log10(p))
    np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])


def test_bh_step_up_takes_later_minimum():
    # p*m/i = .03 .025 .5: the first is lowered to the second's .025
    p = np.array([0.01, 0.0166666667, 0.5])
    q = 10 ** enrich.bh_log10(np.log10(p))
    np.testing.assert_allclose(q, [0.025, 0.025, 0.5], rtol=1e-6)
    assert enrich.bh_log10(np.zeros(0)).shape == (0,)


def test_bh_keeps_order_below_underflow():
    lp = np.array([-400.0, -350.0, -0.5])
    lq = enrich.bh_log10(lp)
    assert np.isfinite(lq).all() and lq[0] < lq[1] < lq[2] <= 0


def test_margins_round_soft_consistently():
    C = np.array([[[0.6, 0.6], [0.4, 2.4]]])
    M = enrich.margins(C)
    assert M.x.tolist() == [[[1, 1], [0, 2]]]
    assert M.N[0, 0, 0] == 4 and M.n_e[0, :, 0].tolist() == [2, 2]


def test_enrichment_recovers_planted_cell():
    rng = np.random.default_rng(0)
    n, k, V = 4000, 6, 5
    y = rng.integers(0, V, n)
    A = rng.integers(0, k, (n, 2))
    A[y == 3, 1] = 2  # head 1 sends label 3 to entry 2
    R = enrich.enrichment(enrich.count_table(A, y, k, V))
    sig = R.tested & (R.log10q < np.log10(0.05)) & (R.log2or > 0)
    assert sig[1, 2, 3]
    assert sig[0].sum() == 0  # the random head has no enrichments


def load_headlang():
    path = Path(__file__).parents[1] / "experiments/ste-arm/headlang.py"
    spec = importlib.util.spec_from_file_location("headlang", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_nmi_table_matches_headlang():
    headlang = load_headlang()
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 5, 300), rng.integers(0, 7, 300)
    b[:100] = a[:100]
    C = enrich.count_table(a[:, None], b, 5, 7)[0]
    got = enrich.nmi_table(C)
    want = headlang.nmi(a, b, 5, 7)
    assert got[0] == pytest.approx(want[0])
    assert got[1] == pytest.approx(want[1])
    assert got[2] == pytest.approx(want[2])


def test_nmi_heads_null_and_ceiling():
    rng = np.random.default_rng(2)
    n, k, V = 3000, 4, 4
    y = rng.integers(0, V, n)
    A = np.stack([y, rng.integers(0, k, n)], 1)  # head 0 IS the label
    s = enrich.nmi_heads(A, y, k, V, nulls=4, seed=0)
    assert s["nmi"][0] == pytest.approx(1.0)
    assert s["ceiling"][0] == pytest.approx(1.0)
    assert s["z"][0] > 100
    assert abs(s["z"][1]) < 5 and s["null"][1] < 0.01


def test_entry_rows_order_and_drop():
    rng = np.random.default_rng(3)
    n, k, V = 3000, 5, 3
    y = rng.integers(0, V, n)
    A = rng.integers(0, 4, (n, 1))  # entry 4 dead
    A[y == 2, 0] = 1
    A[(y == 0) & (rng.random(n) < 0.5), 0] = 3
    R = enrich.enrichment(enrich.count_table(A, y, k, V))
    rows = enrich.entry_rows(R, 0, [0, 1, 2], 0.05, False)
    assert 4 not in rows  # dead entries are never drawn
    # entries 0 and 2 absorb label 1 (entries 1 and 3 take labels 2 and 0
    # away from them): label-0 group, then label 1's by usage, then label 2
    assert rows[0] == 3 and set(rows[1:3]) == {0, 2} and rows[3] == 1
    kept = enrich.entry_rows(R, 0, [0, 1, 2], 0.05, True)
    assert set(kept) <= set(rows)


def test_shown_labels_and_pick_heads():
    values = np.array(["a", "b", "c", "d"])
    counts = np.array([5, 9, 9, 1])
    assert enrich.shown_labels(values, counts, None, 3) == [1, 2, 0]
    assert enrich.shown_labels(values, counts, ["d", "a"], 3) == [3, 0]
    st = {"nmi": np.array([0.1, 0.5, 0.3]), "null": np.array([0.0, 0.4, 0.0])}
    assert enrich.pick_heads(st, 2) == [0, 2]


def test_enriched_labels_prefers_hits_over_frequency():
    rng = np.random.default_rng(4)
    n, k, V = 4000, 6, 5
    y = rng.integers(0, V, n)
    A = rng.integers(0, k, (n, 2))
    A[y == 3, 1] = 2
    R = enrich.enrichment(enrich.count_table(A, y, k, V))
    counts = np.bincount(y, minlength=V)
    assert enrich.enriched_labels(R, [1], counts, 2, 0.05)[0] == 3


def synthetic(n=2000, l=2, h=3, k=6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    A = rng.integers(0, k - 1, (n, l, h))
    A[y == 1, 0, 2] = 0
    labels = {"tokclass": np.array(["word", "digit", "punct", "space"])[y],
              "pos": np.array(["p0", "p1"])[rng.integers(0, 2, n)]}
    return A, labels


def cfg_for(tmp_path, **kw):
    cfg = enrich.parse_args(["--ckpt", str(tmp_path / "run"), "--by",
                             "tokclass", "--nulls", "3"])
    for a, v in kw.items():
        setattr(cfg, a, v)
    return cfg


def test_analyze_end_to_end(tmp_path):
    A, labels = synthetic()
    enrich.analyze(cfg_for(tmp_path), A, labels, None, 6, 5, tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "enrich_l0.png", "enrich_l1.png", "nmi.png"]
    rows = (tmp_path / "cells.csv").read_text().splitlines()
    head = rows[0].split(",")
    assert head[:4] == ["layer", "head", "entry", "label"]
    top = dict(zip(head, rows[1].split(",")))
    assert (top["layer"], top["head"]) == ("0", "2")
    assert (top["entry"], top["label"]) in {("0", "digit"), ("0", "word"),
                                            ("0", "punct"), ("0", "space")}
    assert float(top["FDR"]) < 1e-10
    nmi = (tmp_path / "nmi.csv").read_text().splitlines()
    assert len(nmi) == 1 + 2 * A.shape[1] * A.shape[2]


def test_analyze_soft_and_explicit_heads(tmp_path):
    A, labels = synthetic()
    n, l, h = A.shape
    k = 6
    vals, y = np.unique(labels["tokclass"], return_inverse=True)
    P = np.eye(k)[A.reshape(n, l * h)]
    S = enrich.soft_table(P, y, len(vals))
    cfg = cfg_for(tmp_path, soft=True, heads=[2], layers=[0],
                  drop_empty=True, labels=["digit", "word"])
    enrich.analyze(cfg, A, labels, S, k, 5, tmp_path)
    assert (tmp_path / "enrich_l0.png").stat().st_size > 0
    assert not (tmp_path / "enrich_l1.png").exists()


def test_available_bys(tmp_path):
    cache = tmp_path / "c.npy"
    assert enrich.available_bys(cache) == []
    (tmp_path / "c.langs.npy").touch()
    assert enrich.available_bys(cache) == ["lang"]
    (tmp_path / "c.index.npy").touch()
    assert enrich.available_bys(cache) == ["lang", "tokclass", "pos", "doc"]
