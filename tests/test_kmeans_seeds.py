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
from kmeans_seeds import assign, fvu, kmeans, seed_table  # noqa: E402


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
