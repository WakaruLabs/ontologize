# experiments/planted-circle/circle.py: the planted circle's sampler and the
# cut statistics. The tests pin the tilted density's mean cosine, the cuts
# recovered from a planted arc partition and a planted interval partition,
# a boundary whose entry flickers counted once, the cut distance (0 for the
# same cuts, 1 for cuts offset by half a gap), the partition NMI at its two
# ends, and the NMI of equal arcs at independent offsets, 1 - 1/(2 ln m).
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "planted-circle"))
from circle import (angles, cut_distance, cuts, nmi, parse_args, plant,  # noqa: E402
                    rotated_nmi, sample)


def test_tilt_sets_the_mean_cosine():
    rng = np.random.default_rng(0)
    assert abs(np.cos(angles(rng, 200000, 0.0)).mean()) < 0.01
    assert abs(np.cos(angles(rng, 200000, 0.5)).mean() - 0.25) < 0.01


def test_cuts_recover_planted_arcs_and_intervals():
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 50000)
    phase = 0.3
    a = (((th - phase) % (2 * np.pi)) // (np.pi / 2)).astype(int)
    c = cuts(a, th, 0, 2 * np.pi, 360, True)
    want = np.sort((phase + np.arange(4) * np.pi / 2) % (2 * np.pi))
    assert len(c) == 4 and np.abs(c - want).max() < 2 * np.pi / 360 + 1e-9
    t = rng.uniform(-1, 1, 50000)
    b = np.digitize(t, [-0.5, 0.2])
    c = cuts(b, t, -1, 1, 200, False)
    assert len(c) == 2 and np.abs(c - [-0.5, 0.2]).max() <= 0.01 + 1e-9


def test_a_flickering_boundary_is_one_cut():
    rng = np.random.default_rng(4)
    th = rng.uniform(0, 2 * np.pi, 50000)
    a = (th // (np.pi / 2)).astype(int)
    near = np.abs(th - np.pi / 2) < 0.08
    a[near] = rng.integers(0, 2, near.sum())
    assert len(cuts(a, th, 0, 2 * np.pi, 64, True)) == 4


def test_rotated_nmi_matches_equal_arcs_at_random_offsets():
    rng = np.random.default_rng(5)
    th = rng.uniform(0, 2 * np.pi, 40000)
    m, gap = 8, 2 * np.pi / 8
    base = (th // gap).astype(int)
    vals = [nmi(base, (((th - f * gap) % (2 * np.pi)) // gap).astype(int))
            for f in rng.uniform(0, 1, 40)]
    assert abs(np.mean(vals) - rotated_nmi(m)) < 0.02


def test_cut_distance_ends():
    base = np.arange(4) * np.pi / 2
    assert cut_distance(base, base, 2 * np.pi, 2 * np.pi) == 0
    shifted = base + np.pi / 4
    assert abs(cut_distance(shifted, base, 2 * np.pi, 2 * np.pi) - 1.0) < 1e-12
    # circular: a cut just below 2π is next to one at 0
    assert cut_distance(np.array([2 * np.pi - 0.01]), np.array([0.0, np.pi]),
                        2 * np.pi, 2 * np.pi) < 0.01


def test_nmi_ends():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 8, 20000)
    assert abs(nmi(a, a) - 1) < 1e-9
    assert nmi(a, rng.integers(0, 8, 20000)) < 0.01


def test_sample_puts_the_factors_on_the_planted_directions():
    cfg = parse_args(["--sigma", "0"])
    rng = np.random.default_rng(3)
    U = plant(cfg, rng)
    X, th, t = sample(cfg, U, rng, 1000)
    assert np.allclose(X @ U[0], cfg.radius * np.cos(th), atol=1e-5)
    assert np.allclose(X @ U[1], cfg.radius * np.sin(th), atol=1e-5)
    assert np.allclose(X @ U[2], t, atol=1e-5)
