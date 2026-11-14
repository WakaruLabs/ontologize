# experiments/next-sentence/closure.py: the one-step closure gap
# I(X_t; Z_t+1 | Z_t) of a coarse-graining Z = phi(X). The tests plant a
# four-state chain coarse-grained into two lumps. When the lumps' outgoing
# probabilities agree within each lump (a lumpable chain) the gap is 0; when
# the two states of a lump leave it differently, the gap is the information
# the state adds, 1 - H(0.9) = 0.531 bits here.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "next-sentence"))
from closure import (doc_bootstrap, fit_select, marginal_ce, probe_ce,  # noqa: E402
                     quantile_codes, table, table_ce)


def chain(n: int, lumpable: bool, rng: np.random.Generator):
    """States 0-3, lumps {0, 1} and {2, 3}. Each state is drawn uniformly;
    its successor's lump is the same lump w.p. 0.9 (lumpable: for both
    states) or w.p. 0.9 / 0.1 by state (not lumpable); the state within the
    lump is uniform. Returns one-hot states and the lump codes."""
    s = rng.integers(0, 4, n)
    if lumpable:
        stay = np.full(n, 0.9)
    else:
        stay = np.where(s % 2 == 0, 0.9, 0.1)
    same = rng.random(n) < stay
    lump = s // 2
    nxt = np.where(same, lump, 1 - lump)
    return np.eye(4)[s], lump[:, None], nxt[:, None]


def gap(lumpable: bool) -> float:
    rng = np.random.default_rng(0)
    x, zt, zn = chain(20000, lumpable, rng)
    xq, zq, nq = chain(5000, lumpable, rng)
    val = np.zeros(len(x), bool)
    val[:2000] = True
    P0, fits, best = fit_select(x, zt, zn, val, 2, [1e-4], 300, 1024, 0.05, 0)
    ce_t = table_ce(P0, zq, nq).mean()
    ce_p = (probe_ce(fits[0], xq, zq, nq).mean() if best[0] == 0 else ce_t)
    return float(ce_t - ce_p)


def test_a_lumpable_chain_is_closed():
    assert abs(gap(lumpable=True)) < 0.02


def test_a_non_lumpable_chain_has_the_planted_gap():
    g = gap(lumpable=False)
    assert abs(g - (1 - (-(0.9 * np.log2(0.9) + 0.1 * np.log2(0.1))))) < 0.03


def test_table_and_marginal():
    zt = np.array([[0], [0], [1], [1], [1]])
    zn = np.array([[0], [1], [1], [1], [0]])
    P = table(zt, zn, 2, alpha=0.0)
    assert np.allclose(P[0], [[0.5, 0.5], [1 / 3, 2 / 3]])
    assert np.allclose(table_ce(P, zt, zn)[:, 0],
                       -np.log2([0.5, 0.5, 2 / 3, 2 / 3, 1 / 3]))
    m = marginal_ce(zn, zn, 2, alpha=0.0)[:, 0]
    assert np.allclose(m, -np.log2(np.where(zn[:, 0] == 1, 0.6, 0.4)))


def test_doc_bootstrap_and_quantile_codes():
    rng = np.random.default_rng(3)
    v = rng.random(1000)
    doc = rng.integers(0, 50, 1000)
    m, se = doc_bootstrap(v, doc, 200, 0)
    assert np.isclose(m, v.mean()) and 0 < se < 0.05
    Y = rng.standard_normal((4000, 2))
    edges = np.quantile(Y, np.arange(1, 8) / 8, axis=0).T
    A = quantile_codes(Y, edges)
    assert A.shape == (2, 4000)
    for j in range(2):
        assert np.all(np.abs(np.bincount(A[j], minlength=8) - 500) <= 1)
