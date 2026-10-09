# quantrate.py's pure helpers: the per-slot symbol histograms and their
# entropies, the subset rate convention shared with pareto.py's index bits,
# grouped top-1 symbols, and the readouts along a rate-distortion curve.
import math

import numpy as np
import pytest

import pareto
import quantrate as qr


def brute_entropy(col):
    _, c = np.unique(col, return_counts=True)
    p = c / len(col)
    return float(-(p * np.log2(p)).sum())


def sparse_symbols(seed=0, n=200, slots=6):
    rng = np.random.default_rng(seed)
    S = rng.integers(-3, 4, size=(n, slots))
    S[rng.random((n, slots)) < 0.6] = 0         # mostly at the origin
    S[:, 5] = 0                                  # one slot never sent
    return S


def test_symbol_counts_entropy_matches_brute_force():
    S = sparse_symbols()
    sc = qr.SymbolCounts(S.shape[1])
    sc.add_dense(S[:120])                        # two batches merge
    sc.add_dense(S[120:])
    H, Hb = qr.slot_bits(sc)
    n = len(S)
    for j in range(S.shape[1]):
        kinds = len(np.unique(S[:, j]))
        want = brute_entropy(S[:, j]) + (kinds - 1) / (2 * n * math.log(2))
        assert H[j] == pytest.approx(want, abs=1e-12)
        on = S[:, j] != 0
        both = 0 < on.sum() < n
        want_b = brute_entropy(on) + both / (2 * n * math.log(2))
        assert Hb[j] == pytest.approx(want_b, abs=1e-12)
    assert H[5] == 0 and Hb[5] == 0
    assert np.array_equal(sc.nonzeros(), (S != 0).sum(1))


def test_symbol_counts_sparse_slices_equal_dense():
    # a fixed-width slice of the largest coefficients, padded with zero
    # symbols, gives the same histograms as the full width
    S = sparse_symbols(1, slots=8)
    dense = qr.SymbolCounts(8)
    dense.add_dense(S)
    sparse = qr.SymbolCounts(8)
    idx = np.argsort(-np.abs(S), axis=1)[:, :6]
    sparse.add(idx, np.take_along_axis(S, idx, 1))
    for a, b in zip(qr.slot_bits(dense), qr.slot_bits(sparse)):
        assert np.allclose(a, b)


def test_symbol_out_of_range_raises():
    sc = qr.SymbolCounts(1)
    with pytest.raises(ValueError):
        sc.add_dense(np.array([[qr.SPAN // 2]]))


def test_plugin_bits():
    assert qr.plugin_bits(np.array([10]), 10) == 0.0
    # two equiprobable symbols: one bit plus the Miller-Madow term
    n = 1000
    assert qr.plugin_bits(np.array([500, 500]), n) == pytest.approx(
        1 + 1 / (2 * n * math.log(2)))


def test_subset_bits_constant_support_is_the_binomial():
    nnz = np.full(100, 32)
    assert qr.subset_bits(nnz, 5120) == pytest.approx(
        pareto.log2_choose(5120, 32))
    # grouped top-1 with every head firing: one winner of 32 per head
    assert qr.subset_bits(np.full(50, 160), 5120, 160) == pytest.approx(800)


def test_subset_bits_adds_the_entropy_of_the_count():
    nnz = np.array([2] * 50 + [3] * 50)
    want = (0.5 * (pareto.log2_choose(10, 2) + pareto.log2_choose(10, 3))
            + 1 + 1 / (2 * 100 * math.log(2)))
    assert qr.subset_bits(nnz, 10) == pytest.approx(want)


def test_group_symbols():
    win = np.array([[0, 3], [1, 3]])
    lev = np.array([[0, 2], [5, 2]])
    sym = qr.group_symbols(win, lev)
    assert sym[0, 0] == 0                       # a silent head
    assert sym[0, 1] == sym[1, 1] == 1 + 3 * qr.LEVELS + 2
    assert sym[1, 0] == 1 + 1 * qr.LEVELS + 5
    with pytest.raises(ValueError):
        qr.group_symbols(win, -lev)


def test_fit_count():
    # the scored cache's own head stops short of the scored tail ...
    assert qr.fit_count(90_000, 65536, 32768, same=True) == 90_000 - 32768
    assert qr.fit_count(4_000_000, 65536, 32768, same=True) == 65536
    # ... a separate fit cache can be used from its first row to its last
    assert qr.fit_count(50_000, 65536, 32768, same=False) == 50_000
    assert qr.fit_count(10_000, 65536, 32768, same=True) == 0


def test_whitened_norms():
    W = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    out = qr.whitened_norms(W, np.array([4.0, 1.0]))
    assert out[:2] == pytest.approx([2.0, 2.0])
    assert 0 < out[2] < 1e-9                    # floored, not zero


def power_law(bits):
    return 3.0 * bits ** -0.5


def test_readouts_interpolate_in_log_log():
    bits = np.array([100.0, 400.0, 1600.0, 6400.0])
    pts = sorted(zip(bits, power_law(bits)))
    # exact on a power law, which is a line in log-log
    assert qr.fvu_at_bits(pts, 1000.0) == pytest.approx(power_law(1000.0))
    assert qr.fvu_at_bits(pts, 50.0) is None
    target = power_law(3000.0)
    assert qr.bits_at_fvu(pts, target) == pytest.approx(3000.0)
    assert qr.bits_at_fvu(pts, 1e-6) is None


def test_curve_drops_unquantized_rows():
    rows = [{"bits": float("nan"), "fvu_w": 0.1},
            {"bits": 50.0, "fvu_w": 0.3}, {"bits": 20.0, "fvu_w": 0.5}]
    assert qr.curve(rows) == [(20.0, 0.5), (50.0, 0.3)]
