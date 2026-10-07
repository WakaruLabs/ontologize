# `Ontologizer.per_layer_dec`: one decoder per layer, built as a single
# Linear over an `l * e_dec` latent in which layer i writes only block i.
# The construction is right if (a) each layer's contribution lands in its
# own block and nowhere else, and (b) tiling a shared decoder's weights
# across the blocks reproduces the shared model exactly -- the per-layer
# model contains the shared one as the special case D_i = D.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.ontologizer import Ontologizer
from ontologize.training.serialize import migrate_spec

from conftest import KW, RETIRED_LAYER, finite_live

import dataclasses


def tiled(shared_params, l: int):
    """The shared model's params with its decoder repeated over l blocks."""
    p = jax.tree_util.tree_map(lambda v: v, shared_params)
    W = p["params"]["decoder"]["weights"]                   # (d_out, e_dec)
    p["params"]["decoder"]["weights"] = jnp.tile(W, (1, l))
    return p


def test_decoder_widens_by_l(build):
    model, params = build(per_layer_dec=True)
    W = params["params"]["decoder"]["weights"]
    assert W.shape == (KW["d_out"], KW["l"] * KW["e_dec"])
    # the dictionaries themselves are unchanged
    shared, sp = build()
    for i in range(KW["l"]):
        a = params["params"][f"dictencs_{i}"]["dict"]["weights"].shape
        b = sp["params"][f"dictencs_{i}"]["dict"]["weights"].shape
        assert a == b


def test_each_layer_writes_only_its_block(build, X):
    model, params = build(per_layer_dec=True)
    l, e = KW["l"], KW["e_dec"]

    def blocks(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        out = []
        for i, de in enumerate(module.dictencs):
            R_0 = R
            R, K = de.withClusts(R, Ein, temperature=1.0)
            out.append(R - R_0)
            if i < module.l - 1:
                Ein = module.nextinput(X, R, K)
        return jnp.stack(out)                                # (l, b, l*e)

    D = np.asarray(model.apply(params, X, method=blocks)).reshape(
        l, X.shape[0], l, e)
    for i in range(l):
        for j in range(l):
            nz = np.abs(D[i, :, j]).max()
            assert (nz > 0) if i == j else (nz == 0), (i, j, nz)


@pytest.mark.parametrize("method", ["__call__", "withStats"])
def test_tiled_shared_decoder_reproduces_shared_model(build, X, method):
    shared, sp = build()
    per, _ = build(per_layer_dec=True)
    pp = tiled(sp, KW["l"])
    if method == "__call__":
        a = shared.apply(sp, X, temperature=1.0)
        b = per.apply(pp, X, temperature=1.0)
    else:
        a, sa, _ = shared.apply(sp, X, 1.0, rng=jax.random.PRNGKey(2),
                                method=Ontologizer.withStats)
        b, sb, _ = per.apply(pp, X, 1.0, rng=jax.random.PRNGKey(2),
                             method=Ontologizer.withStats)
        assert finite_live(sb, RETIRED_LAYER)
        np.testing.assert_allclose(np.asarray(live_cols(sa)),
                                   np.asarray(live_cols(sb)),
                                   rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                               rtol=1e-5, atol=1e-6)


def live_cols(s):
    keep = [i for i in range(s.shape[-1]) if i not in RETIRED_LAYER]
    return jnp.asarray(s)[..., jnp.array(keep)]


def test_entries_decode_through_their_own_block(build):
    shared, sp = build()
    per, _ = build(per_layer_dec=True)
    pp = tiled(sp, KW["l"])
    a, _ = shared.apply(sp, method=Ontologizer.decodeEntries)
    b, _ = per.apply(pp, method=Ontologizer.decodeEntries)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                               rtol=1e-5, atol=1e-6)


def test_blocks_are_independent_decoders(build, X):
    """Changing layer 1's block of the decoder moves the output only
    through layer 1's contribution: layer 0's decode is untouched."""
    model, params = build(per_layer_dec=True)
    e = KW["e_dec"]

    def prefix0(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        R, _ = module.dictencs[0].withClusts(R, module.constinput(E),
                                             temperature=1.0)
        return module.decode(R)

    p2 = jax.tree_util.tree_map(lambda v: v, params)
    W = p2["params"]["decoder"]["weights"]
    p2["params"]["decoder"]["weights"] = W.at[:, e:2 * e].multiply(3.0)
    a = model.apply(params, X, method=prefix0)
    b = model.apply(p2, X, method=prefix0)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.allclose(np.asarray(model.apply(params, X, temperature=1.0)),
                           np.asarray(model.apply(p2, X, temperature=1.0)))


@pytest.mark.parametrize("bad", [dict(direct=True, signed=True,
                                      e_dec=KW["d_out"]),
                                 dict(forward="labels", resid_gain=False)])
def test_incompatible_settings_raise(X, bad):
    model = Ontologizer(**{**KW, "per_layer_dec": True, **bad})
    with pytest.raises(ValueError, match="per_layer_dec"):
        model.init(jax.random.PRNGKey(0), X)


def test_spec_without_the_field_is_shared(build):
    model, _ = build()
    spec = dataclasses.asdict(model)
    spec.pop("per_layer_dec")
    assert not Ontologizer(**migrate_spec(spec)).per_layer_dec
