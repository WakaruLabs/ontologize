# experiments/seed-stability/kmeans_seeds.py: k-means seed agreement. The
# tests pin the assignment against brute force, and the agreement on three
# planted cases: separated clusters and a dominant direction reproduce
# across seeds, and an isotropic Gaussian, whose cells can rotate at no
# cost, does not.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "seed-stability"))
from kmeans_seeds import (assign, fastica, fvu, kmeans, marginal_copy,  # noqa: E402
                          principal, seed_table, shape_stats)


def test_assign_is_the_nearest_centre():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6)).astype(np.float32)
    C = rng.normal(size=(7, 6)).astype(np.float32)
    brute = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
    assert np.array_equal(assign(X, C, chunk=128), brute)


def test_separated_clusters_reproduce_and_fit():
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(4, 8)) * 20
    X = (centres[rng.integers(4, size=6000)] + rng.normal(size=(6000, 8))
         ).astype(np.float32)
    t = seed_table(X[:4000], X[4000:], 4, [1, 2, 3])
    assert min(t["nmi"]) > 0.95
    assert max(t["null"]) < 0.01
    assert max(t["fvu"]) < 0.05


def test_a_dominant_direction_pins_a_two_cell_split():
    rng = np.random.default_rng(2)
    X = (rng.normal(size=(6000, 8)) * np.r_[10.0, np.ones(7)]).astype(np.float32)
    t = seed_table(X[:4000], X[4000:], 2, [1, 2, 3])
    assert min(t["nmi"]) > 0.9


def test_an_isotropic_gaussian_does_not_reproduce():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(12000, 8)).astype(np.float32)
    t = seed_table(X[:8000], X[8000:], 8, [1, 2, 3])
    sep = rng.normal(size=(8, 8)) * 20
    Y = (sep[rng.integers(8, size=12000)] + rng.normal(size=(12000, 8))
         ).astype(np.float32)
    u = seed_table(Y[:8000], Y[8000:], 8, [1, 2, 3])
    assert np.mean(t["nmi"]) < np.mean(u["nmi"]) - 0.2
    assert np.mean(t["nmi"]) > 10 * np.mean(t["null"])


def test_kmeans_ends_with_every_cell_used():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 3)).astype(np.float32)
    C = kmeans(X, 20, 0)
    lab = assign(X, C)
    assert len(np.unique(lab)) == 20
    assert 0 < fvu(X, C, lab) < 1


def _square_dependence(Y):
    """The largest |corr(Y_i², Y_j)| over i != j."""
    C = np.corrcoef(np.concatenate([Y ** 2, Y], 1).T)[:Y.shape[1], Y.shape[1]:]
    return np.abs(C - np.diag(np.diag(C))).max()


def test_the_pca_copy_keeps_each_principal_marginal_and_breaks_dependence():
    # z2 is a function of z1 yet uncorrelated with it, and the variances
    # differ, so the principal frame is z's own up to a rotation
    rng = np.random.default_rng(5)
    z1 = 2 * rng.standard_normal(8000)
    Z = np.stack([z1, (z1 ** 2 - 4) / 4, 0.5 * rng.standard_normal(8000)], 1)
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    X = (Z @ R + 1.0).astype(np.float32)
    T, E, st = marginal_copy(X[:6000], X[6000:], "pca")
    mu, lam, V = principal(X[:6000])
    for orig, copy in ((X[:6000], T), (X[6000:], E)):
        Y, Yc = (orig - mu) @ V, (copy - mu) @ V
        assert np.allclose(np.sort(Y, 0), np.sort(Yc, 0), atol=1e-4)
        assert _square_dependence(Y) > 0.9 > 0.1 > _square_dependence(Yc)
    assert np.allclose(np.cov(T.T), np.cov(X[:6000].T), rtol=0.05, atol=0.05)
    assert np.allclose(st["variance_share"], lam / lam.sum())


def test_fastica_recovers_planted_sources():
    rng = np.random.default_rng(6)
    n = 20000
    S = np.stack([rng.uniform(-1, 1, n), rng.laplace(size=n),
                  rng.exponential(size=n), rng.uniform(-1, 1, n) ** 3], 1)
    S = (S - S.mean(0)) / S.std(0)
    X = S @ rng.standard_normal((4, 4)) + 3.0
    mu, lam, V = principal(X)
    W, it, lim = fastica((X - mu) @ (V / np.sqrt(lam)), seed=0)
    Y = (X - mu) @ (V / np.sqrt(lam)) @ W.T
    C = np.abs(np.corrcoef(np.concatenate([Y, S], 1).T)[:4, 4:])
    assert lim < 1e-4 and it < 200
    assert (C.max(0) > 0.98).all() and (C.max(1) > 0.98).all()


def test_the_ica_copy_keeps_an_independent_model_the_pca_copy_breaks():
    # two uniform sources along non-orthogonal directions: the rows fill a
    # parallelogram. Shuffling the sources keeps it; shuffling principal
    # coordinates, which mix the sources, spills rows outside it
    rng = np.random.default_rng(7)
    n = 20000
    S = rng.uniform(-np.sqrt(3), np.sqrt(3), (n, 2))
    A = np.array([[1.0, 0.0], [0.8, 0.6]])
    X = (S @ A).astype(np.float32)
    edge = {}
    for frame in ("pca", "ica"):
        T, E, st = marginal_copy(X[:15000], X[15000:], frame)
        edge[frame] = np.abs(T.astype(np.float64) @ np.linalg.inv(A)).max()
    assert edge["ica"] < np.sqrt(3) + 0.05
    assert edge["pca"] > np.sqrt(3) + 0.3
    assert st["ica_last_change"] < 1e-4


def test_shape_stats_read_gaussian_uniform_and_two_modes():
    rng = np.random.default_rng(8)
    n = 200000
    Y = np.stack([rng.standard_normal(n), rng.uniform(size=n),
                  rng.standard_normal(n) + 4 * rng.choice([-1, 1], n)], 1)
    st = shape_stats(Y)
    assert abs(st["kurtosis"][0]) < 0.05 and abs(st["bimodality"][0] - 1 / 3) < 0.01
    assert abs(st["kurtosis"][1] + 1.2) < 0.02 and abs(st["bimodality"][1] - 5 / 9) < 0.01
    assert st["bimodality"][2] > 0.8
