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
    """Renames only. Dropping a key it does not recognize would build a
    model configured differently from the one that trained, which is worse
    than refusing to build one."""
    bad = {"d_in": 4, "d_out": 4, "not_a_field": 1}
    assert migrate_spec(bad) == bad
    with pytest.raises(TypeError):
        Ontologizer(**migrate_spec(bad))
