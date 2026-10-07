# experiments/hsic-bottleneck's HSICHyperparams on the current training
# path: its loss must take the setpoint controllers' arguments, zero the
# retired (NaN) stats before weighting them, and write the stock stats
# layout; its probe must take resid_const's layer-0 coordinate. Each of
# those broke silently as the package moved on.
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "hsic-bottleneck"))
from hsic_hyperparams import HSICHyperparams  # noqa: E402

from ontologize.training.ontostate import update, schedules  # noqa: E402
from conftest import KW, B, RETIRED_LOSS, finite_live  # noqa: E402


@pytest.mark.parametrize("const0", [True, False])
def test_hsic_arm_trains_on_the_stock_layout(const0, X):
    d = KW["d_in"]
    hyper = HSICHyperparams(
        d, d, B, 1, 5e-5, 0.0, 1.0,
        "batchnorm", "normal", "featvar", 0.0, 0.02, 0.1,
        p_drop=0.1, temperature_end=0.03, anneal_steps=100,
        p_drop_start=0.0, sd_K_end=0.006, ghost=False,
        s_hsic_heads=1e-4, hsic_h=KW["h"])
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"], n=2, gate="none",
        forward="resid", deepsup=True, resid_norm=True, resid_const=True,
        resid_gain=True, const0=const0,
        dtype_str="float32", dtype_p_str="float32")
    state = hyper.init(model, save_each=10)

    rng = jax.random.PRNGKey(2)
    losses = []
    for step in range(20):
        T, pd, sk, _ = schedules(step, 100, 1.0, 0.03, 0.1, 0.0, 0.02, 0.006)
        rng, r = jax.random.split(rng)
        state, L, _ = update(state, hyper.loss, r, X, X,
                             temperature=T, p_drop=pd, sd_K=sk,
                             sd_in=0.0, sd_F=0.1, grad_clip=1.0)
        assert jnp.isfinite(L), f"step {step}"
        losses.append(float(L))
    assert losses[-1] < losses[0]
    rows = state.stats[:10]
    assert rows.shape[-1] == hyper.n_stats
    assert finite_live(rows, RETIRED_LOSS)
    # the raw HSIC penalty is recorded (the default unbiased estimator can
    # dip below zero, so this checks it is there, not its sign)
    assert jnp.isfinite(rows[:, 2]).all() and (rows[:, 2] != 0).all()
