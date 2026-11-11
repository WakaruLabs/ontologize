# experiments/lang-tags/tagflip.py: which heads and latents the source tag
# moves, read against random shifts of the same size. The tests pin the
# statistics: the keep and stay rates, the noise an excess is read
# against, and the two language measures.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "lang-tags"))
from tagflip import (describe, excess, jaccard, keep_rate,  # noqa: E402
                     label_nmi, spearman, stay_rate, top_label_share)


def test_keep_rate_is_per_head():
    A = np.array([[0, 1], [2, 1], [3, 0]])
    B = np.array([[0, 1], [2, 0], [1, 0]])
    assert np.allclose(keep_rate(A, B), [2 / 3, 2 / 3])


def test_stay_rate_conditions_on_active_rows():
    Z = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], bool)
    Z2 = np.array([[1, 1, 0], [0, 1, 0], [1, 1, 0]], bool)
    rate, n = stay_rate(Z, Z2)
    assert n.tolist() == [2, 2, 0]
    assert np.allclose(rate[:2], [0.5, 1.0]) and np.isnan(rate[2])
    assert np.allclose(jaccard(Z, Z2), [0.5, 0.5, 0.5])


def test_excess_noise_matches_its_sampling_spread():
    """With no tag effect, the excess's spread over units is what the
    function says it is."""
    rng = np.random.default_rng(0)
    u, n, p = 4000, 500, 0.4
    kt, k1, k2 = (rng.binomial(n, p, u) / n for _ in range(3))
    ex, noise = excess(kt, k1, k2)
    assert abs(ex.mean()) < 0.002
    assert np.isclose(ex.std(), noise, rtol=0.08)
    d = describe(ex, noise)
    assert d["above_3"] < 0.01 and d["below_3"] < 0.01
    # a unit the tag does move stands out
    kt[0] = 0.1
    ex, noise = excess(kt, k1, k2)
    assert ex[0] > 10 * noise


def test_excess_ignores_unscored_units_in_its_noise():
    k1 = np.array([0.5, 0.6, np.nan])
    k2 = np.array([0.4, 0.6, np.nan])
    ex, noise = excess(np.array([0.3, 0.6, np.nan]), k1, k2)
    assert np.isclose(noise, np.std([0.1, 0.0]) * np.sqrt(3) / 2)
    assert np.isnan(ex[2]) and describe(ex, noise)["n"] == 2


def test_label_measures():
    y = np.repeat(np.arange(4), 50)
    assert np.isclose(label_nmi(y, y, 4, 4), 1.0)
    assert np.isclose(label_nmi((y + 1) % 4, y, 4, 4), 1.0)   # a renaming
    rng = np.random.default_rng(1)
    assert label_nmi(rng.integers(0, 4, 200), y, 4, 4) < 0.05
    Z = np.zeros((200, 3), bool)
    Z[:50, 0] = True                         # one language only
    Z[::2, 1] = True                         # every language equally
    share = top_label_share(Z, y, 4)
    assert np.allclose(share[:2], [1.0, 0.25]) and np.isnan(share[2])


def test_spearman_skips_missing_units():
    a = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert np.isclose(spearman(a, b), 1.0)
    assert np.isclose(spearman(a, -b), -1.0)
