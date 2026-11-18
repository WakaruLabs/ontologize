# experiments/rate-penalty/ratecurve.py: a hard code's FVU against its
# realized bits per layer prefix. The tests check that a head used evenly
# reads its nominal log2(k) bits and a constant head none, that the counts
# see every entry including unused ones, and that the prefix FVU and the
# cumulative bits are what coordfvu.py's weighting gives.
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "rate-penalty"))
from ratecurve import head_bits, prefix_curve, usage_counts  # noqa: E402


def test_even_and_constant_heads():
    k, n = 8, 8000
    E = np.stack([np.arange(n) % k, np.zeros(n, int)], 1)
    C = usage_counts(E, k)
    assert C.shape == (2, k)
    assert (C[0] == n // k).all() and C[1, 0] == n and (C[1, 1:] == 0).all()
    b = head_bits(C)
    assert abs(b[0] - math.log2(k)) < 0.01
    assert b[1] == 0.0


def test_skewed_head_reads_its_entropy():
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.25, 0.125, 0.125])
    E = rng.choice(4, size=(200000, 1), p=p)
    b = head_bits(usage_counts(E, 4))[0]
    assert abs(b - 1.75) < 0.01


def test_prefix_curve_weights_and_accumulates():
    var = np.array([1.0, 2.0, 4.0])
    w = np.array([1.0, 0.5, 0.0])
    err = np.array([[0.5, 1.0, 9.0],
                    [0.1, 0.2, 9.0]])
    fvu, cum = prefix_curve(err, var, w, np.array([10.0, 6.0]))
    # the zero-weight coordinate drops out of both sums
    assert np.allclose(fvu, [(0.5 + 0.5) / 2.0, (0.1 + 0.1) / 2.0])
    assert np.allclose(cum, [10.0, 16.0])
