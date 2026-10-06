# steerembed.py's supervised baselines: the difference-of-means and probe
# vectors it scores alongside the model's own steering directions. They
# are only a fair baseline if they recover a direction that is there and
# never see the rows they are about to steer.
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"
                       / "ste-arm"))
from steerembed import supervised_dirs  # noqa: E402

D, N = 32, 4096


def planted(seed=0):
    """Unit-ish rows with a rare class (~5%) defined by one direction."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=D)
    u /= np.linalg.norm(u)
    X = rng.normal(size=(N, D)) / np.sqrt(D)
    y = X @ u > np.quantile(X @ u, 0.95)
    return jnp.asarray(X, jnp.float32), y, u


def test_both_vectors_recover_the_planted_direction():
    X, y, u = planted()
    dm, probe = supervised_dirs(X, y, np.array([], int))
    dm = np.asarray(dm) / np.linalg.norm(np.asarray(dm))
    assert dm @ u > 0.95
    assert np.asarray(probe) @ u > 0.95
    assert np.isclose(np.linalg.norm(np.asarray(probe)), 1.0, atol=1e-5)


def test_held_out_rows_do_not_enter_the_fit():
    """Corrupting only the excluded rows must leave both vectors unchanged."""
    X, y, _ = planted(1)
    excl = np.arange(64)
    dm0, p0 = supervised_dirs(X, y, excl)
    X2 = X.at[excl].set(100.0 * jnp.ones((len(excl), D)))
    dm1, p1 = supervised_dirs(X2, y, excl)
    assert np.allclose(dm0, dm1, atol=1e-6)
    assert np.allclose(p0, p1, atol=1e-6)
