# experiments/seed-stability/atomsources.py: whether a stack's atoms point
# along FastICA's sources. The tests pin the helpers: copies of some
# directions are found along them and not along the rest, the classes
# follow kurtosis and skew, the rotated reference keeps the spectrum, the
# centered atoms drop unused entries, and "both runs" reads against
# independence.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "seed-stability"))
from atomsources import (best_abs_cos, both, centered_live,  # noqa: E402
                         rotated_mixing, source_classes)


def test_copied_directions_are_found_and_others_are_not():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((10, 64))
    A = np.concatenate([-3 * B[:4] + 0.05 * rng.standard_normal((4, 64)),
                        rng.standard_normal((6, 64))])
    per_dir, per_atom = best_abs_cos(A, B)
    assert (per_dir[:4] > 0.95).all() and (per_dir[4:] < 0.6).all()
    assert (per_atom[:4] > 0.95).all() and (per_atom[4:] < 0.6).all()


def test_sources_are_classed_by_kurtosis_and_skew():
    k = np.array([8.0, 8.0, 1.5, 1.5, 4.0])
    s = np.array([-2.0, 0.5, 0.1, 0.5, 1.5])
    assert list(source_classes(k, s)) == ["skewed", "other", "symmetric", "other", "other"]


def test_the_rotated_reference_keeps_the_spectrum_and_is_a_rotation():
    rng = np.random.default_rng(1)
    V = np.linalg.qr(rng.standard_normal((6, 5)))[0]
    lam = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
    fit = {"eigenvectors": V, "eigenvalues": lam}
    M = rotated_mixing(fit, seed=3)
    Q = M @ (V / np.sqrt(lam))                 # back to the sphered frame
    assert np.allclose(Q @ Q.T, np.eye(5), atol=1e-10)
    assert np.isclose((M ** 2).sum(), lam.sum())


def test_atoms_are_centered_on_their_heads_and_unused_entries_dropped():
    rng = np.random.default_rng(2)
    D = rng.standard_normal((2, 4, 8))
    usage = np.array([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    C = centered_live(D, usage)
    assert C.shape == (6, 8)
    assert np.allclose(0.5 * C[0] + 0.5 * C[1], 0)
    assert np.allclose(C[2:].mean(0), 0)


def test_both_runs_read_against_independence():
    a = np.array([0.9, 0.9, 0.1, 0.1])
    assert both(a, a, np.ones(4, bool), 0.5) == {"both": 0.5, "independent": 0.25}
    b = np.array([0.9, 0.1, 0.9, 0.1])
    assert both(a, b, np.ones(4, bool), 0.5) == {"both": 0.25, "independent": 0.25}
