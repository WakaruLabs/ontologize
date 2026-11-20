"""experiments/adam-eps/train_eps.py: `EpsHyperparams` hands its eps to optax.
throttle.py: the step share, and an sae.py run's state read per parameter."""
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


def test_sae_medians_reads_sae_state(tmp_path):
    # sae.py's own save_state after real optax steps, read back per
    # parameter under the run's eps; a constant gradient g has v_hat = g**2
    import json
    import numpy as np
    import optax
    import sae
    from throttle import sae_medians
    params = {"W_dec": jnp.zeros((6, 4)), "W_enc": jnp.zeros((4, 6)),
              "b_dec": jnp.zeros(4), "b_enc": jnp.zeros(6)}
    g = {"W_dec": 1e-6, "W_enc": 1e-8, "b_dec": 1e-10, "b_enc": 1e-7}
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    for _ in range(3):
        _, opt = tx.update({k: jnp.full(v.shape, g[k]) for k, v in params.items()},
                           opt, params)
    sae.save_state(tmp_path / "state.npz", params, opt, 3, np.zeros(6, np.int64))
    eps, step, count, med, _, at_lr = sae_medians(tmp_path)
    assert (eps, step, count) == (1e-8, 3, 3)  # no meta.json: optax's eps
    assert list(med) == [f"leaf_{i}" for i in range(4)]  # no params.npz yet
    assert not at_lr  # no learning rate recorded: the step is per unit lr
    (tmp_path / "meta.json").write_text(json.dumps({"adam_eps": 1e-9}))
    np.savez(tmp_path / "params.npz", **params)
    eps, _, _, med, _, _ = sae_medians(tmp_path)
    assert eps == 1e-9
    for k in params:
        assert med[k] == pytest.approx(g[k] / (g[k] + 1e-9), rel=1e-4)


def test_relative_step_is_adams_step_over_the_norm():
    # the moments a step leaves are the ones that step used, so the
    # relative step read back from them is that update over the block's norm
    import numpy as np
    import optax
    from throttle import relative_step
    theta = jnp.asarray(np.random.default_rng(0).normal(size=40), jnp.float32)
    lr, eps = 1e-3, 1e-8
    tx = optax.adam(lr, eps=eps)
    opt = tx.init(theta)
    rng = np.random.default_rng(1)
    for _ in range(5):
        u, opt = tx.update(jnp.asarray(rng.normal(size=40) * 1e-4, jnp.float32), opt, theta)
    adam = opt[0]
    got = relative_step(theta, adam.mu, adam.nu, int(adam.count), eps, lr)
    want = float(jnp.linalg.norm(u) / jnp.linalg.norm(theta))
    assert got == pytest.approx(want, rel=1e-4)
    # eps throttles it: an eps far above the gradient's scale shrinks the step
    assert relative_step(theta, adam.mu, adam.nu, int(adam.count), 1e-2, lr) < got / 10


def test_sae_relative_step_takes_the_recorded_learning_rate(tmp_path):
    import json
    import numpy as np
    import optax
    import sae
    from throttle import sae_medians
    params = {"W_dec": jnp.ones((6, 4)), "W_enc": jnp.ones((4, 6)),
              "b_dec": jnp.ones(4), "b_enc": jnp.ones(6)}
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    for _ in range(3):
        _, opt = tx.update({k: jnp.full(v.shape, 1e-5) for k, v in params.items()},
                           opt, params)
    sae.save_state(tmp_path / "state.npz", params, opt, 3, np.zeros(6, np.int64))
    np.savez(tmp_path / "params.npz", **params)
    _, _, _, _, per_unit, at_lr = sae_medians(tmp_path)
    assert not at_lr
    (tmp_path / "meta.json").write_text(json.dumps({"lr": 1e-3}))
    _, _, _, _, rel, at_lr = sae_medians(tmp_path)
    assert at_lr
    for k in params:
        # a constant gradient well above eps: every entry steps by lr, and
        # the parameters are ones, so the relative step is lr itself
        assert rel[k] == pytest.approx(1e-3, rel=1e-3)
        assert per_unit[k] == pytest.approx(1.0, rel=1e-3)


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
