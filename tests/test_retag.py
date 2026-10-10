# experiments/lang-tags/retag.py's english-stage helpers: the English share
# counts listed words only, the linear share recovers a planted direction
# and ignores noise, centering removes group offsets, and the held-out AUC
# separates a planted label and sits at chance without one.
import importlib.util
from pathlib import Path

import numpy as np


def load_retag():
    path = Path(__file__).parents[1] / "experiments/lang-tags/retag.py"
    spec = importlib.util.spec_from_file_location("retag", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


retag = load_retag()


def test_english_share_counts_listed_words():
    assert retag.english_share("the cat and the dog") == 3 / 5
    assert retag.english_share("der Hund und die Katze") == 0.0
    assert retag.english_share("Click here: 123 !!") == 1.0
    assert retag.english_share("") == 0.0


def test_linear_share_recovers_a_planted_direction():
    rng = np.random.default_rng(0)
    s = rng.random(2000)
    X = rng.normal(size=(2000, 8)) * 0.1
    X[:, 0] += 3 * s
    w = np.ones(8)
    planted = retag.linear_share(X, s, w)
    total = (X[:, 0] - X[:, 0].mean()).var() / (X - X.mean(0)).var(0).sum()
    assert planted > 0.9 * total
    assert retag.linear_share(X, rng.random(2000), w) < 0.01
    assert np.isnan(retag.linear_share(X, np.ones(2000), w))


def test_group_centered_removes_group_offsets():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(4), 50)
    X = rng.normal(size=(200, 5)) + 10 * g[:, None]
    Z = retag.group_centered(X, g, np.ones(5))
    for k in range(4):
        np.testing.assert_allclose(Z[g == k].mean(0), 0, atol=1e-9)
    np.testing.assert_allclose(Z.std(0), 1, atol=1e-9)


def test_logistic_auc_separates_a_planted_label_and_not_noise():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(600, 6))
    y = (Z[:, 2] + 0.3 * rng.normal(size=600) > 0.5).astype(float)
    fold = rng.integers(0, 5, 600)
    assert retag.logistic_auc(Z, y, fold) > 0.9
    noise = (rng.random(600) < 0.3).astype(float)
    assert abs(retag.logistic_auc(Z, noise, fold) - 0.5) < 0.1
    assert retag.rank_auc(np.array([0., 0, 1, 1]), np.array([1., 2, 3, 4])) == 1.0
