# experiments/rate-penalty/headdrop.py: the rates a trained hard code
# reaches by dropping heads, and the convex hull other runs are read
# against. The tests check that heads are dropped by FVU per bit saved,
# with silent heads first, that the masks accumulate in that order, and
# that the hull passes under points above a chord and reads NaN outside
# its range.
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "rate-penalty"))
from headdrop import drop_masks, drop_order, hull_at, lower_hull  # noqa: E402


def test_drop_order_by_cost_per_bit():
    dfvu = np.array([0.5, 0.125, 0.25, 0.0])
    bits = np.array([4.0, 1.0, 4.0, 0.0])
    # per bit: 0.125, 0.125, 0.0625; the silent head goes first, ties stay in index order
    assert drop_order(dfvu, bits).tolist() == [3, 2, 0, 1]


def test_drop_masks_accumulate():
    M = drop_masks(np.array([2, 0, 1]), np.array([0, 1, 3]))
    assert M.tolist() == [[0, 0, 0], [0, 0, 1], [1, 1, 1]]


def test_lower_hull_skips_points_above_the_chord():
    b = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    f = np.array([1.0, 0.8, 0.4, 0.3, 0.25])
    hb, hf = lower_hull(b, f)
    # (1, 0.8) lies above the chord from (0, 1) to (2, 0.4)
    assert hb.tolist() == [0.0, 2.0, 3.0, 4.0]
    assert np.allclose(hull_at(hb, hf, np.array([1.0, 2.5])), [0.7, 0.35])
    assert np.isnan(hull_at(hb, hf, np.array([5.0]))).all()
