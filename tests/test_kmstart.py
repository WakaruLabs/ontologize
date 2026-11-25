# experiments/vertices/kmstart.py: the one-head start from spherical k-means.
# The start's classifier must select spherical k-means' nearest centre for
# every shape, through the bilinear classifier's own forward, and its
# decoder must decode each entry's atom to the cell's least-squares target.
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "vertices"))
from kmstart import (cell_targets, gaps, shape_frame,  # noqa: E402
                     start_weights, unit)

D, K, E = 10, 5, 24


def bilinear(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """NLinearBlock's forward with gate and activation 'none', no bias:
    the product of the two factors over [x / |x|, 1]."""
    U = np.concatenate([X / np.linalg.norm(X, axis=1, keepdims=True),
                        np.ones((len(X), 1))], 1)
    Ys = np.einsum("bm,nhdm->nbhd", U, W)
    return Ys[0] * Ys[1]


def setup(rng):
    C = unit(rng.normal(size=(K, D)))
    w = rng.uniform(0.5, 2.0, D)
    ubar = rng.normal(size=D) * 0.1
    W = rng.normal(size=(2, 1, K, D + 1))
    A = np.abs(rng.normal(size=(K, E)))
    Wd = rng.normal(size=(D, E))
    T = rng.normal(size=(K, D))
    return C, w, ubar, W, A, Wd, T


def test_start_selects_the_nearest_centre():
    rng = np.random.default_rng(0)
    C, w, ubar, W, A, Wd, T = setup(rng)
    Wn, _ = start_weights(W, A, Wd, C, w, ubar, T, lam=0.37)
    X = rng.normal(size=(2000, D)) * rng.uniform(0.1, 5.0, (2000, 1))
    logits = bilinear(Wn, X)[:, 0]
    F = shape_frame(X, w, ubar)
    assert np.allclose(logits, 0.37 * F @ C.T, atol=1e-5)
    # the row's norm drops out: the gain-shape split hides it from the
    # classifier, and spherical assignment ignores it too
    assert (logits.argmax(1) == (unit(F) @ C.T).argmax(1)).all()


def test_start_decodes_each_entry_to_its_target():
    rng = np.random.default_rng(1)
    C, w, ubar, W, A, Wd, T = setup(rng)
    _, Wdn = start_weights(W, A, Wd, C, w, ubar, T, lam=1.0)
    assert np.allclose(Wdn @ A.T, T.T, atol=1e-8)
    # the least change: nothing outside the atoms' span moves
    P = np.eye(E) - A.T @ np.linalg.solve(A @ A.T, A)
    assert np.allclose((Wdn - Wd) @ P, 0.0, atol=1e-8)


def test_start_refuses_other_classifiers():
    rng = np.random.default_rng(2)
    C, w, ubar, W, A, Wd, T = setup(rng)
    with pytest.raises(ValueError):
        start_weights(np.zeros((2, 2, K, D + 1)), A, Wd, C, w, ubar, T, 1.0)
    with pytest.raises(ValueError):
        start_weights(np.zeros((2, 1, K, D)), A, Wd, C, w, ubar, T, 1.0)
    with pytest.raises(ValueError):
        start_weights(np.zeros((3, 1, K, D + 1)), A, Wd, C, w, ubar, T, 1.0)


def test_cell_targets_are_the_gain_weighted_least_squares_atoms():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, D)) * rng.uniform(0.2, 3.0, (600, 1))
    lab = rng.integers(0, K, 600)
    T = cell_targets(X, lab, K)
    g = np.linalg.norm(X, axis=1)
    for j in range(K):
        m = lab == j
        # the head writes g times its atom: argmin_a sum |x - g a|^2
        a, *_ = np.linalg.lstsq(np.repeat(g[m, None], 1, 1), X[m], rcond=None)
        assert np.allclose(T[j], a[0], atol=1e-10)


def test_gaps():
    K_ = np.array([[0.1, 0.5, 0.3], [2.0, -1.0, 1.5]])
    assert np.allclose(gaps(K_), [0.2, 0.5])
