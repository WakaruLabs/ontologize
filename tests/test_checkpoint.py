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
from ontologize.training.serialize import migrate_spec

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
    assert migrate_spec(bad) == bad
    with pytest.raises(TypeError):
        Ontologizer(**migrate_spec(bad))


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
    spec.update(resid_first=False, logit_norm=False, direct=False,
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


@pytest.mark.parametrize("field,value", [("direct", True), ("resid_first", True),
                                         ("logit_norm", True), ("gain_clip", True),
                                         ("head_sparse", "jumprelu")])
def test_an_active_headline_feature_refuses(field, value):
    """Dropping a headline-only field is safe only at its no-op value; a
    model that used the feature cannot be built here."""
    spec = headline_spec(Ontologizer(**KW))
    spec[field] = value
    with pytest.raises(ValueError, match=field):
        migrate_spec(spec)
