# `direct`: dictionary entries live in output space and there is no decoder.
import jax
import jax.numpy as jnp
import pytest

from conftest import KW, B, LAYER_WIDTH, RETIRED_LAYER, RETIRED_LOSS, finite_live
from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams
from ontologize.training.ontostate import update

DIRECT = dict(direct=True, signed=True, e_dec=KW["d_out"])


def test_no_decoder_parameters(build):
    _, params = build(**DIRECT)
    assert "decoder" not in params["params"]


def test_decode_is_the_identity(build, X):
    model, params = build(**DIRECT)
    R = jax.random.normal(jax.random.PRNGKey(5), (B, KW["d_out"]))
    assert jnp.array_equal(model.apply(params, R, method=Ontologizer.decode), R)
    assert jnp.array_equal(model.apply(params, R, method=Ontologizer.fwd_dec), R)


def test_output_is_the_accumulated_dictionary_residual(build, X):
    # with no decoder the reconstruction IS the residual the layers sum
    model, params = build(**DIRECT)
    Y = model.apply(params, X, temperature=0.5)
    R, _ = model.apply(params, X, temperature=0.5, method=Ontologizer.classify)
    assert jnp.allclose(Y, R)


@pytest.mark.parametrize("bad,match", [
    (dict(e_dec=KW["d_out"] * 2), "e_dec"),
    (dict(signed=False), "signed"),
    (dict(biased_dec=True), "decoder"),
])
def test_invalid_configurations_raise(build, bad, match):
    with pytest.raises(ValueError, match=match):
        build(**{**DIRECT, **bad})


def test_ghost_path_refuses(build, X):
    model, params = build(**DIRECT)
    with pytest.raises(NotImplementedError, match="direct"):
        model.apply(params, X, X, temperature=0.5,
                    method=Ontologizer.withGhost)


def test_withstats_finite(build, X):
    model, params = build(**DIRECT, select="ste")
    Y, stats, _ = model.apply(params, X, temperature=0.5,
                              method=Ontologizer.withStats)
    assert jnp.all(jnp.isfinite(Y))
    assert stats.shape == (KW["l"], LAYER_WIDTH)
    assert finite_live(stats, RETIRED_LAYER)


def test_training_steps_finite(X):
    d = KW["d_in"]
    hyper = Hyperparams(
        d, d, B, 1, 5e-5, 0.0, 1.0,
        "batchnorm", "normal", "featvar", 0.0, 0.02, 0.1,
        s_Hm=1e-6, p_drop=0.1, ghost=False)
    model = hyper.ontologizer(
        d, d, d, KW["k"], KW["h"], KW["l"],
        n=2, gate="none", forward="resid", deepsup=True, select="ste",
        direct=True, signed=True,
        dtype_str="float32", dtype_p_str="float32")
    state = hyper.init(model, save_each=10)

    rng = jax.random.PRNGKey(2)
    for step in range(10):
        rng, r = jax.random.split(rng)
        state, L, _ = update(state, hyper.loss, r, X, X,
                             temperature=0.5, p_drop=0.1, sd_K=0.02,
                             sd_in=0.0, sd_F=0.1, grad_clip=1.0)
        assert jnp.isfinite(L), f"step {step}"
    assert finite_live(state.stats[:10], RETIRED_LOSS)
