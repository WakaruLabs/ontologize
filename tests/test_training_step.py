# End-to-end training-path smoke: Hyperparams -> ontologizer flag wiring ->
# OntoState update with all schedules active, for each live run
# configuration. Guards the jit boundary (static vs traced args), the loss
# broadcast over deepsup prefixes, and NaN regressions in the noise paths.
import jax
import jax.numpy as jnp
import pytest

from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import update, schedules

from conftest import KW, B

CONFIGS = {
    "resid_joint": {},
    "resid_stagewise": {"deepsup_sg": True},
    "resid_conditioned": {"resid_norm": True, "resid_const": True},
}


@pytest.mark.parametrize("flags", CONFIGS.values(), ids=CONFIGS.keys())
def test_thirty_steps_finite_and_decreasing(flags, X):
    # regularizer mix mirrors the SONAR run config (sonar.py)
    d = KW["d_in"]
    hyper = Hyperparams(
        d, d, B, 1, 5e-5, 0.0, 1.0,
        "batchnorm", "normal", "featvar", 0.0, 0.02, 0.1,
        s_g=1e-4, s_L1F=1e-9, s_bcossim=1e-5, s_hcossim=1e-5, s_Hm=1e-6,
        p_drop=0.1, temperature_end=0.03, anneal_steps=100,
        p_drop_start=0.0, sd_K_end=0.006, ghost=False)
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"],
        n=2, gate="none", forward="resid", deepsup=True, **flags,
        dtype_str="float32", dtype_p_str="float32")
    state = hyper.init(model, save_each=10)

    rng = jax.random.PRNGKey(2)
    losses = []
    for step in range(30):
        T, pd, sk = schedules(step, 100, 1.0, 0.03, 0.1, 0.0, 0.02, 0.006)
        rng, r = jax.random.split(rng)
        state, L, _ = update(state, hyper.loss, r, X, X,
                             temperature=T, p_drop=pd, sd_K=sk,
                             sd_in=0.0, sd_F=0.1, grad_clip=1.0)
        assert jnp.isfinite(L), f"step {step}"
        losses.append(float(L))
    assert losses[-1] < losses[0]
    assert jnp.all(jnp.isfinite(state.stats[:10]))
