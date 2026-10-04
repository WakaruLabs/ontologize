# The causal-intervention interface: DictBlock.intervene primitives,
# Ontologizer.withArgs (per-layer DictIntervention application), and
# Ontologizer.intervene (input-space steering). A no-op intervention being
# an exact identity is what makes own-entry controls meaningful in the
# tag-swap experiments.
import jax
import jax.numpy as jnp
import pytest

from ontologize.ontologizer import Ontologizer, DictIntervention
from ontologize.layers.dictblock import DictBlock

from conftest import KW, DB_KW


def intervene(db, params, P, **kwargs):
    return db.apply(params, P, method=DictBlock.intervene, **kwargs)


def test_noop_is_identity(dictblock):
    db, params, P = dictblock
    assert jnp.array_equal(intervene(db, params, P), P)


def test_set_forces_onehot(dictblock):
    db, params, P = dictblock
    out = intervene(db, params, P,
                    h_set=jnp.array([1]), k_set=jnp.array([3]))
    expect = jnp.zeros(DB_KW["k"]).at[3].set(1.0)
    assert jnp.array_equal(out[:, 1], jnp.broadcast_to(expect, out[:, 1].shape))
    # other heads untouched
    mask = jnp.arange(DB_KW["h"]) != 1
    assert jnp.array_equal(out[:, mask], P[:, mask])


def test_uniform_and_zero_ablate(dictblock):
    db, params, P = dictblock
    out_u = intervene(db, params, P, h_unif=jnp.array([0, 2]))
    assert jnp.allclose(out_u[:, jnp.array([0, 2])], 1.0 / DB_KW["k"])
    out_z = intervene(db, params, P, h_zero=jnp.array([2]))
    assert jnp.array_equal(out_z[:, 2], jnp.zeros_like(P[:, 2]))
    assert jnp.array_equal(out_z[:, :2], P[:, :2])


def test_add_sub_and_scale(dictblock):
    db, params, P = dictblock
    out = intervene(db, params, P,
                    k_add=jnp.array([5]), h_add=jnp.array([1]))
    assert jnp.allclose(out[:, 1, 5], P[:, 1, 5] + 1.0)
    out = intervene(db, params, P,
                    k_sub=jnp.array([5]), h_sub=jnp.array([1]))
    assert jnp.allclose(out[:, 1, 5], P[:, 1, 5] - 1.0)
    S = jnp.arange(1.0, DB_KW["h"] + 1.0)
    out = intervene(db, params, P, scale=S)
    assert jnp.allclose(out, P * S[:, None], atol=1e-6)


def test_dicts_nonnegative(dictblock):
    # abs()'d dictionary: ontofeatures cannot subtract. The raw weights do
    # go negative at init, so this checks the abs is really applied.
    db, params, _ = dictblock
    W = db.apply(params, method=DictBlock.dicts)
    assert jnp.all(W >= 0)
    assert jnp.any(params['params']['weights'] < 0)


@pytest.fixture(scope="module")
def conditioned(build):
    return build(resid_norm=True, resid_const=True)


def noop_arglist():
    return [DictIntervention() for _ in range(KW["l"])]


def test_withargs_noop_matches_plain_forward(conditioned, X):
    model, params = conditioned
    Y0 = model.apply(params, X, temperature=0.5)
    Y1, _, _, _ = model.apply(params, X, noop_arglist(), temperature=0.5,
                              method=Ontologizer.withArgs)
    assert jnp.allclose(Y0, Y1, atol=1e-6)


def test_withargs_set_changes_output_not_earlier_layers(conditioned, X):
    model, params = conditioned
    arglist = noop_arglist()
    arglist[1] = DictIntervention(h_set=jnp.array([2]), k_set=jnp.array([5]))
    Y0, _, stats0, _ = model.apply(params, X, noop_arglist(),
                                   temperature=0.5,
                                   method=Ontologizer.withArgs)
    Y1, _, stats1, _ = model.apply(params, X, arglist, temperature=0.5,
                                   method=Ontologizer.withArgs)
    assert not jnp.allclose(Y0, Y1, atol=1e-5)
    # the intervention is at layer 1: layer 0's stats must be untouched
    assert jnp.array_equal(stats0[0], stats1[0], equal_nan=True)


@pytest.mark.parametrize("flags", [
    dict(n=1, gate="relu"),   # transpose adjoint
    dict(n=2, gate="none"),   # BilinearBlock eigendecomposition adjoint
], ids=["linear", "bilinear"])
def test_input_space_intervene_noop_is_identity(build, X, flags):
    # rev is linear, so a no-op intervention reverses a zero delta: the
    # steered input and output must match the clean forward exactly.
    # layer=1 also exercises the resid_const constant-coordinate slice in
    # the classifier adjoint.
    model, params = build(resid_norm=True, resid_const=True, **flags)
    Y_int, X_int, _ = model.apply(params, X, 1, temperature=0.5,
                                  method=Ontologizer.intervene)
    assert jnp.allclose(X_int, X, atol=1e-6)
    Y0 = model.apply(params, X, temperature=0.5)
    assert jnp.allclose(Y_int, Y0, atol=1e-5)


def test_input_space_intervene_steers_bilinear(conditioned, X):
    # a real intervention through the bilinear adjoint: input changes,
    # norm is preserved by construction, everything stays finite
    model, params = conditioned
    Y_int, X_int, Ps = model.apply(
        params, X, 1, temperature=0.5,
        h_set=jnp.array([2]), k_set=jnp.array([5]),
        method=Ontologizer.intervene)
    assert not jnp.allclose(X_int, X, atol=1e-6)
    assert jnp.allclose(jnp.linalg.norm(X_int, axis=-1),
                        jnp.linalg.norm(X, axis=-1), atol=1e-5)
    for arr in (Y_int, X_int, Ps):
        assert jnp.all(jnp.isfinite(arr))
