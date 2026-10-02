# Checkpoint save/restore through the training path's manager + load_params
# (the config.py/ontostate.py route, not serialize.py), including the
# n_stats width migration: checkpoints written before the KL_m stat carry an
# (save_each, 8) stats buffer and must restore into an n_stats=9 state via
# load_params' shape-mismatch fallback.
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest

from ontologize.ontologizer import Ontologizer
from ontologize.training.ontostate import state_init, load_params

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
    state = make_state(9).replace(step=5)
    manager = make_manager(tmp_path)
    assert state.save(manager)

    target = make_state(9, seed=7)  # different init: restore must overwrite
    restored = load_params(target, manager, 5)
    assert int(restored.step) == 5
    assert params_equal(restored.params, state.params)
    assert not params_equal(target.params, state.params)


def test_old_stats_width_migrates(tmp_path):
    # pre-KL_m checkpoint: 8-wide stats buffer
    old = make_state(8).replace(step=5)
    manager = make_manager(tmp_path)
    assert old.save(manager)

    new = make_state(9, seed=7)
    restored = load_params(new, manager, 5)
    assert int(restored.step) == 5
    assert restored.stats.shape == (10, 9)  # fresh buffer, new width
    assert params_equal(restored.params, old.params)


def test_spec_rebuilds_model(tmp_path):
    state = make_state(9)
    manager = make_manager(tmp_path)
    state.save(manager)
    spec = manager.restore(0, items={'spec': None})['spec']
    rebuilt = Ontologizer(**spec)
    assert rebuilt == state.model
