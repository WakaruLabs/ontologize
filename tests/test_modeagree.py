# experiments/next-sentence/modeagree.py: slow-block agreement by mode and
# by head. Planted codes pin the statistics: a second seed whose heads are
# relabelled copies of the first's predicts its modes almost exactly, an
# independent block and shuffled rows do not, a head duplicated inside its
# block reads a VIF R^2 near 1, and the margin and rank correlation are
# what they say.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "next-sentence"))
from modeagree import (best_margin, head_vif, mode_agreement, onehot,  # noqa: E402
                       ridge, r2, spearman)


def test_onehot_layout():
    A = np.array([[0, 2], [1, 1]])            # two heads, two rows, k = 3
    Z = onehot(A, 3)
    np.testing.assert_array_equal(Z, [[1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0]])


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(2000, 5))
    Y = Z @ rng.normal(size=(5, 2)) + 3.0
    f = ridge(Z, Y, 1e-6)
    assert (r2(Y, f(Z)) > 0.999).all()


def test_relabelled_copies_agree_by_mode_and_independent_blocks_do_not():
    rng = np.random.default_rng(1)
    n, h, k = 4000, 6, 4
    A = rng.integers(0, k, size=(h, n))
    B = np.stack([rng.permutation(k)[a] for a in A])    # same cells, new labels
    other = rng.integers(0, k, size=(h, n))
    res = mode_agreement(onehot(A, k), onehot(B, k), onehot(other, k),
                         n // 2, 8, 1e-4, seed=0)
    assert (res["block"] > 0.99).all()
    # a null's held-out R^2 sits near -p/n (24 features on 2,000 rows)
    assert (np.abs(res["other"]) < 0.05).all()
    assert (np.abs(res["shuffled"]) < 0.05).all()
    assert np.all(np.diff(res["share"]) <= 1e-12)


def test_vif_reads_a_duplicated_head():
    rng = np.random.default_rng(2)
    n, k = 4000, 4
    a = rng.integers(0, k, size=n)
    A = np.stack([a, rng.permutation(k)[a], rng.integers(0, k, size=n)])
    v = head_vif(onehot(A, k), 3, k, n // 2, 1e-4)
    assert v[0] > 0.99 and v[1] > 0.99 and abs(v[2]) < 0.02


def test_margin_and_spearman():
    M = np.array([[0.9, 0.1, 0.5], [0.2, 0.2, 0.1]])
    np.testing.assert_allclose(best_margin(M), [0.4, 0.0])
    x = np.arange(30.0)
    rho, p = spearman(x, -x, 500, 0)
    assert rho == -1.0 and p < 0.01
