# experiments/ste-arm/errspectrum.py: a code's held-out error split over
# the whitened eigenbasis, beside reverse water-filling. The tests check the
# water level against pareto's reference, that the per-direction errors add
# up to the total, and that the diagonal rescaling recovers a planted scale.
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
from pareto import gaussian_reference  # noqa: E402
from errspectrum import (band_summary, direction_errors,  # noqa: E402
                         effective_bits, eigenbasis, shrink_fit, water_level)


def test_water_level_matches_the_reference():
    lam = np.array([8.0, 4.0, 2.0, 1.0, 0.5, 0.25])
    for bits in (0.5, 2.0, 6.0):
        theta = water_level(lam, bits)
        rate = np.maximum(0.0, np.log2(lam / theta) / 2).sum()
        assert abs(rate - bits) < 1e-6
        fvu = np.minimum(theta, lam).sum() / lam.sum()
        assert abs(fvu - gaussian_reference(lam, bits)[0]) < 1e-9
        # and the effective rate inverts it
        assert abs(effective_bits(lam, fvu) - bits) < 1e-6


def test_direction_errors_add_up_in_any_basis():
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((500, 6)) * np.array([3, 2, 1, 1, 0.5, 0.1])
    lam, V = eigenbasis(Y)
    assert np.all(np.diff(lam) <= 1e-12)
    E = rng.standard_normal((500, 6))
    e = direction_errors(E, V)
    assert np.isclose(e.sum(), (E ** 2).sum(1).mean())


def test_shrink_fit_recovers_a_planted_scale():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((2000, 3))
    Zh = Z / np.array([2.0, 1.0, 4.0]) + 0.5
    a, b = shrink_fit(Z, Zh)
    assert np.allclose(a, [2.0, 1.0, 4.0], atol=1e-8)
    assert np.allclose(Zh * a + b, Z, atol=1e-8)
    # a constant reconstruction falls back to the mean
    a0, b0 = shrink_fit(Z, np.zeros_like(Z))
    assert np.allclose(a0, 0.0) and np.allclose(b0, Z.mean(0))


def test_band_summary_shares_and_counts():
    lam = np.array([4.0, 2.0, 1.0, 1.0])
    e = np.array([0.4, 0.2, 1.5, 0.5])
    D = np.array([0.5, 0.5, 0.5, 0.5])
    rows = band_summary(lam, e, D, bands=((0, 2), (2, 4)))
    assert np.isclose(sum(r["var_share"] for r in rows), 1.0)
    assert np.isclose(sum(r["err_share"] for r in rows), 1.0)
    assert np.isclose(sum(r["excess_share"] for r in rows), 1.0)
    # only the third direction's error is above its variance
    assert [r["n_err_above_var"] for r in rows] == [0, 1]
    assert rows[0]["excess_share"] == 0.0
