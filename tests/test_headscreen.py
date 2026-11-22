# experiments/seed-stability/headscreen.py: the multitrait-multimethod screen
# of seed pairs' heads. The tests pin the Miller-Madow correction (independent
# partitions read zero where the plug-in NMI reads well above it) and the
# screen on planted heads: a shared head passes, a shared head whose own seed
# holds a copy of it fails on its within-seed check, and independent heads
# fail on the floor.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "seed-stability"))
from headscreen import corrected_nmi, screen  # noqa: E402
from partition import nmi_matrix  # noqa: E402


def noisy(f, k, share, rng):
    """f relabelled, with `share` of its rows given a random label."""
    g = rng.permutation(k)[f]
    flip = rng.random(len(f)) < share
    g[flip] = rng.integers(0, k, flip.sum())
    return g


def test_independent_partitions_read_zero_once_corrected():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 32, (6, 4000))
    B = rng.integers(0, 32, (6, 4000))
    raw = nmi_matrix(A, 32, B, 32)
    cor = corrected_nmi(A, B, 32)
    assert raw.mean() > 0.02
    assert abs(cor.mean()) < 0.003


def test_the_screen_passes_a_shared_head_and_fails_copies_and_noise():
    rng = np.random.default_rng(1)
    k, n, h = 8, 30000, 12
    f0, f1 = rng.integers(0, k, n), rng.integers(0, k, n)
    A = rng.integers(0, k, (h, n))
    B = rng.integers(0, k, (h, n))
    A[0], A[1], A[2] = f0, rng.permutation(k)[f0], f1     # A holds f0 twice
    B[5], B[7] = noisy(f0, k, 0.2, rng), noisy(f1, k, 0.1, rng)
    r = screen(A, B, k, np.zeros(h, int))
    partner = dict(zip(r["a"], r["b"]))
    assert partner[2] == 7 and r["pass"][list(r["a"]).index(2)]
    j = [i for i, a in enumerate(r["a"]) if partner[a] == 5][0]
    assert r["a"][j] in (0, 1) and not r["pass"][j]
    assert r["z"][j, 3] <= 2.0                                  # mono_a fails
    assert r["pass"].sum() == 1
    assert abs(r["null_check"]["mean"]) < 0.002
