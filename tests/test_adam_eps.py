"""experiments/adam-eps/train_eps.py: `EpsHyperparams` hands its eps to optax."""
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "adam-eps"))
from throttle import throttle  # noqa: E402
from train_eps import EpsHyperparams  # noqa: E402


def first_step(eps: float, g: float = 1e-12, lr: float = 1e-3) -> float:
    """Size of Adam's first update on a parameter whose gradient is `g`:
    bias-corrected, m_hat = g and sqrt(v_hat) = |g|, so lr * g / (|g| + eps)."""
    tx = EpsHyperparams(4, 4, 8, lr=lr, adam_eps=eps).opt()
    params = {"w": jnp.zeros(3)}
    updates, _ = tx.update({"w": jnp.full(3, g)}, tx.init(params), params)
    return float(jnp.abs(updates["w"]).max())


def test_default_is_optax_default():
    assert EpsHyperparams(4, 4, 8).adam_eps == 1e-8


@pytest.mark.parametrize("eps", [1e-8, 1e-15])
def test_eps_reaches_the_optimizer(eps):
    assert first_step(eps) == pytest.approx(1e-3 * 1e-12 / (1e-12 + eps), rel=1e-3)


def test_throttle_is_the_step_share():
    import numpy as np
    g = np.array([1e-6, 1e-8, 1e-10])
    # after many steps the bias correction is 1, so v_hat = nu = g**2
    np.testing.assert_allclose(throttle(g ** 2, 10 ** 6, 1e-8), g / (g + 1e-8))
    # early on, v_hat = nu / (1 - b2^count) undoes the moving average's start
    np.testing.assert_allclose(throttle((1 - 0.999) * g ** 2, 1, 1e-8), g / (g + 1e-8))


def test_on_the_init_scale_class():
    # `--dict-init-scale` makes train_ste choose DictWDHyperparams, whose own
    # `opt` must still pass eps through, with and without its dictionary decay
    from dictwd import DictWDHyperparams
    from train_eps import with_eps
    HP = with_eps(DictWDHyperparams)
    assert issubclass(HP, DictWDHyperparams)
    params = {"w": jnp.zeros(3)}
    for dict_wd in (0.0, 1e-4):
        tx = HP(4, 4, 8, lr=1e-3, dict_wd=dict_wd, adam_eps=1e-15).opt()
        updates, _ = tx.update({"w": jnp.full(3, 1e-12)}, tx.init(params), params)
        assert float(jnp.abs(updates["w"]).max()) == pytest.approx(1e-3, rel=1e-2)


def test_adamw_takes_it_too():
    tx = EpsHyperparams(4, 4, 8, lr=1e-3, wd=1e-4, adam_eps=1e-15).opt()
    params = {"w": jnp.zeros(3)}
    updates, _ = tx.update({"w": jnp.full(3, 1e-12)}, tx.init(params), params)
    assert float(jnp.abs(updates["w"]).max()) == pytest.approx(1e-3, rel=1e-2)
