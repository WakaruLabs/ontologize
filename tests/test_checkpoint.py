# Checkpoint save/restore through the training path's manager + load_params
# (the config.py/ontostate.py route, not serialize.py), including the
# n_stats width migration: checkpoints written before a stat was added
# carry a narrower (save_each, n) stats buffer and must restore into the
# live-width state via load_params' shape-mismatch fallback.
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest

from ontologize.ontologizer import Ontologizer
from ontologize.training.ontostate import OntoState, state_init, load_params
from flax.errors import ScopeParamShapeError

from ontologize.training.serialize import (migrate_spec, restore_spec,
                                           stored_params)

from conftest import KW, B


def make_manager(path):
    return ocp.CheckpointManager(
        str(path), checkpointers={'state': ocp.PyTreeCheckpointer(),
                                  'spec': ocp.PyTreeCheckpointer()})


def make_state(n_stats, seed=0):
    model = Ontologizer(**KW)
    return state_init(model, B, optax.adam(1e-4), jax.random.PRNGKey(seed),
                      save_each=10, n_stats=n_stats, ghost=False)


def params_equal(a, b):
    leaves = jax.tree_util.tree_map(jnp.array_equal, a, b)
    return all(jax.tree_util.tree_leaves(leaves))


def test_roundtrip_same_width(tmp_path):
    state = make_state(OntoState.n_stats).replace(step=5)
    manager = make_manager(tmp_path)
    assert state.save(manager)

    target = make_state(OntoState.n_stats, seed=7)  # different init: restore must overwrite
    restored = load_params(target, manager, 5)
    assert int(restored.step) == 5
    assert params_equal(restored.params, state.params)
    assert not params_equal(target.params, state.params)


@pytest.mark.parametrize("old_width", [10, 11])
def test_old_stats_width_migrates(tmp_path, old_width):
    """Each widening has to restore through load_params' shape-mismatch
    fallback: 10 predates L2_pwak, 11 predates the logged s_L1F. `new` is
    the live width, so this keeps tracking it rather than a fixed pair."""
    new_width = OntoState.n_stats
    assert old_width < new_width
    old = make_state(old_width).replace(step=5)
    manager = make_manager(tmp_path)
    assert old.save(manager)

    new = make_state(new_width, seed=7)
    restored = load_params(new, manager, 5)
    assert int(restored.step) == 5
    assert restored.stats.shape == (10, new_width)  # fresh buffer, new width
    assert params_equal(restored.params, old.params)


def test_spec_rebuilds_model(tmp_path):
    state = make_state(11)
    manager = make_manager(tmp_path)
    state.save(manager)
    spec = manager.restore(0, items={'spec': None})['spec']
    rebuilt = Ontologizer(**migrate_spec(spec))
    assert rebuilt == state.model


def test_a_prerename_spec_still_rebuilds(tmp_path):
    """Serialization is `asdict(model)` splatted back into the class, so a
    field rename makes every earlier checkpoint unconstructable -- 105223c
    renamed the three router fields and did exactly that. `migrate_spec`
    maps them forward, and every loader goes through it."""
    state = make_state(11)
    manager = make_manager(tmp_path)
    state.save(manager)
    spec = dict(manager.restore(0, items={'spec': None})['spec'])
    for new, old in (("activation_router", "activation_scale"),
                     ("gate_router", "gate_scale"),
                     ("biased_router", "biased_scale")):
        spec[old] = spec.pop(new)
    with pytest.raises(TypeError):
        Ontologizer(**spec)                    # the failure being migrated
    assert Ontologizer(**migrate_spec(spec)) == state.model


def test_migrate_spec_leaves_unknown_keys_to_raise(tmp_path):
    """Dropping a key it does not recognize would build a model configured
    differently from the one that trained, which is worse than refusing to
    build one."""
    bad = {"d_in": 4, "d_out": 4, "not_a_field": 1}
    assert migrate_spec(bad)["not_a_field"] == 1
    with pytest.raises(TypeError):
        Ontologizer(**migrate_spec(bad))


def save_with_spec(path, model, spec):
    """A checkpoint of `model` whose stored spec is `spec`, as an older
    branch of the code would have written it."""
    state = state_init(model, B, optax.adam(1e-4), jax.random.PRNGKey(0),
                       save_each=10, n_stats=OntoState.n_stats, ghost=False)
    manager = make_manager(path)
    manager.save(0, items={'state': state, 'spec': spec})
    manager.wait_until_finished()
    return manager, state


def old_spec(model):
    """`model`'s spec as written before `const0` and `resid_gain` existed."""
    import dataclasses
    spec = {f.name: getattr(model, f.name) for f in dataclasses.fields(model)
            if f.name not in ("parent", "name")}
    for k in ("const0", "resid_gain"):
        spec.pop(k)
    return spec


def test_a_pre_resid_gain_spec_keeps_it_off():
    """`resid_gain` was added at False and defaulted to True minutes later,
    so a spec without the key trained without it; the dataclass default
    would silently build a different forward pass."""
    model = Ontologizer(**{**KW, "resid_gain": False})
    spec = old_spec(model)
    assert Ontologizer(**spec).resid_gain                  # the silent drift
    assert Ontologizer(**migrate_spec(spec)) == \
        Ontologizer(**{**KW, "resid_gain": False, "const0": True})


def test_a_layer0_without_the_constant_restores(tmp_path, X):
    """Checkpoints from before layer 0 took resid_const's coordinate
    (`resid_nc`, `resid_nc_hm`) have a `d_in`-wide first classifier. Nothing
    in their spec says so; `restore_spec` reads it off the stored shapes, and
    the model rebuilt from it runs on the stored weights."""
    model = Ontologizer(**{**KW, "resid_const": True, "resid_gain": False,
                           "const0": False})
    manager, state = save_with_spec(tmp_path, model, old_spec(model))

    raw = manager.restore(0, items={'spec': None})['spec']
    params = stored_params(manager.restore(0, items={'state': None})['state'])
    with pytest.raises(ScopeParamShapeError):              # the loud failure
        Ontologizer(**migrate_spec(raw)).apply({'params': params}, X)

    rebuilt = Ontologizer(**restore_spec(manager, 0))
    assert rebuilt == model
    Y = rebuilt.apply({'params': params}, X)
    assert jnp.allclose(Y, model.apply(
        {'params': stored_params(state.params)}, X))


def test_a_layer0_with_the_constant_still_restores(tmp_path, X):
    """The same spec without `const0`, from a checkpoint whose layer 0 does
    take the coordinate (everything trained since), must keep it."""
    model = Ontologizer(**{**KW, "resid_const": True, "resid_gain": False})
    manager, _ = save_with_spec(tmp_path, model, old_spec(model))
    assert Ontologizer(**restore_spec(manager, 0)) == model


def headline_spec(model):
    """`model`'s spec as the headline branch writes it: its names for signed
    and concatenated heads, plus every headline-only field at its default
    (the hs_* sizes are non-trivial defaults, inert while head_sparse is
    off), with fast_stats on."""
    import dataclasses
    spec = {f.name: getattr(model, f.name) for f in dataclasses.fields(model)
            if f.name not in ("parent", "name")}
    spec["signed_dict"] = spec.pop("signed")
    spec["private_heads"] = spec.pop("concat")
    spec.update(resid_first=False, logit_norm=False,
                gain_clip=False, head_sparse="none", hs_shared=False,
                hs_decoder=False, hs_auxk=0, m_h=32, k_z=4,
                hs_bandwidth=1e-3, hs_init_threshold=1e-3, fast_stats=True)
    return spec


@pytest.mark.parametrize("signed,concat", [(False, False), (True, True)])
def test_a_headline_spec_rebuilds(signed, concat):
    """The headline branch names signed and block-diagonal heads
    `signed_dict` and `private_heads`, and carries fields this branch lacks.
    Its private heads are ConcatDictBlock under another name (same
    `(h, k, d // h)` weights), so the spec maps onto `concat`."""
    model = Ontologizer(**{**KW, "signed": signed, "concat": concat})
    spec = headline_spec(model)
    with pytest.raises(TypeError):
        Ontologizer(**spec)                    # the failure being migrated
    assert Ontologizer(**migrate_spec(spec)) == model


def test_a_headline_direct_spec_rebuilds():
    """`direct` is ported, so a headline model that used it is no longer
    refused: it maps onto this branch's field of the same name."""
    kw = {**KW, "signed": True, "direct": True, "e_dec": KW["d_out"]}
    model = Ontologizer(**kw)
    assert Ontologizer(**migrate_spec(headline_spec(model))) == model


@pytest.mark.parametrize("field,value", [("resid_first", True),
                                         ("logit_norm", True), ("gain_clip", True),
                                         ("head_sparse", "jumprelu")])
def test_an_active_headline_feature_refuses(field, value):
    """Dropping a headline-only field is safe only at its no-op value; a
    model that used the feature cannot be built here."""
    spec = headline_spec(Ontologizer(**KW))
    spec[field] = value
    with pytest.raises(ValueError, match=field):
        migrate_spec(spec)
