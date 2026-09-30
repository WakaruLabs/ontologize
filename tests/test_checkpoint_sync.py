"""`OntoState.save(manager, keep_period)`: sync only retained checkpoints.

`wait_until_finished` waits for orbax's writes to complete, not for them
to reach the device. Syncing every save would force writes that currently
never happen, since most checkpoints are unlinked within `max_to_keep`
while their pages are still dirty. These tests pin which steps get synced
and that the selection does not change what is written.
"""
import os
import jax
import optax
import orbax.checkpoint as ocp
import pytest

from ontologize.ontologizer import Ontologizer
from ontologize.training.ontostate import (OntoState, fsync_tree, load_params,
                                           state_init)

from conftest import KW, B


def make_manager(path, keep_period=None):
    opts = (ocp.CheckpointManagerOptions(keep_period=keep_period)
            if keep_period else None)
    return ocp.CheckpointManager(
        str(path), checkpointers={'state': ocp.PyTreeCheckpointer(),
                                  'spec': ocp.PyTreeCheckpointer()},
        options=opts)


def make_state(step):
    model = Ontologizer(**KW)
    s = state_init(model, B, optax.adam(1e-4), jax.random.PRNGKey(0),
                   save_each=10, ghost=False)
    return s.replace(step=step)


def test_syncs_on_a_retained_step(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("ontologize.training.ontostate.fsync_tree",
                        lambda root: seen.append(root))
    m = make_manager(tmp_path)
    assert make_state(200).save(m, keep_period=100)
    assert [p.name for p in seen] == ["200"]


def test_does_not_sync_a_transient_step(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("ontologize.training.ontostate.fsync_tree",
                        lambda root: seen.append(root))
    m = make_manager(tmp_path)
    assert make_state(250).save(m, keep_period=100)
    assert seen == []


def test_keep_period_zero_never_syncs(tmp_path, monkeypatch):
    """Default stays the old behaviour, so callers that do not pass it are
    unaffected."""
    seen = []
    monkeypatch.setattr("ontologize.training.ontostate.fsync_tree",
                        lambda root: seen.append(root))
    m = make_manager(tmp_path)
    assert make_state(200).save(m)
    assert seen == []


@pytest.mark.parametrize("keep_period", [0, 100])
def test_checkpoint_is_restorable_either_way(tmp_path, keep_period):
    """Syncing must not change what was written."""
    state = make_state(200)
    m = make_manager(tmp_path)
    assert state.save(m, keep_period=keep_period)
    restored = load_params(make_state(0), m, 200)
    assert int(restored.step) == 200
    leaves = jax.tree_util.tree_map(
        lambda a, b: bool((a == b).all()), restored.params, state.params)
    assert all(jax.tree_util.tree_leaves(leaves))


def test_fsync_tree_walks_files_and_dirs(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.bin").write_bytes(b"x" * 1024)
    (tmp_path / "b.bin").write_bytes(b"y" * 1024)
    fsync_tree(tmp_path)                      # must not raise
    assert (tmp_path / "sub" / "a.bin").read_bytes() == b"x" * 1024
    assert (tmp_path / "b.bin").read_bytes() == b"y" * 1024


def test_fsync_tree_tolerates_a_missing_root(tmp_path):
    fsync_tree(tmp_path / "nope")             # must not raise
