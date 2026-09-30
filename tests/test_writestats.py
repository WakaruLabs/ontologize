"""`OntoState.writestats`: the loss.csv append, and that it is durable.

A checkpoint can be reloaded after a hard stop; a lost span of loss.csv
cannot be reconstructed from anything, so this writer is the one that has
to reach disk before it returns. The failure it guards against is not a
short file but a file whose size is right and whose tail reads as NUL,
because that looks like valid data to anything counting rows.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

from ontologize.ontologizer import Ontologizer
from ontologize.training.ontostate import OntoState, state_init

from conftest import KW, B


class FakeManager:
    """`writestats` only uses `manager.directory`."""
    def __init__(self, directory):
        self.directory = directory


def make_state(tmp_path, save_each=4):
    model = Ontologizer(**KW)
    return state_init(model, B, optax.adam(1e-4), jax.random.PRNGKey(0),
                      save_each=save_each, ghost=False)


def test_appends_one_row_per_buffered_step(tmp_path):
    state = make_state(tmp_path)
    rows = jnp.arange(4 * state.n_stats, dtype=jnp.float32).reshape(
        4, state.n_stats)
    state = state.replace(stats=rows)
    assert state.writestats(FakeManager(tmp_path))
    assert state.writestats(FakeManager(tmp_path))     # appends, not truncates
    out = np.loadtxt(tmp_path / "loss.csv", delimiter=",")
    assert out.shape == (8, state.n_stats)
    np.testing.assert_allclose(out[:4], np.asarray(rows), rtol=1e-6)
    np.testing.assert_allclose(out[4:], np.asarray(rows), rtol=1e-6)


def test_row_is_readable_before_the_process_could_exit(tmp_path):
    """Durability: the bytes are on disk when the call returns, so a hard
    stop immediately afterwards cannot lose them. Reading through a fresh
    descriptor that bypasses this process's buffers is the observable part;
    the fsync itself is verified by the call not raising."""
    state = make_state(tmp_path)
    rows = jnp.ones((4, state.n_stats), dtype=jnp.float32)
    state = state.replace(stats=rows)
    assert state.writestats(FakeManager(tmp_path))

    path = tmp_path / "loss.csv"
    fd = os.open(path, os.O_RDONLY)
    try:
        raw = os.read(fd, 1 << 20)
    finally:
        os.close(fd)
    assert raw, "nothing readable after writestats returned"
    assert b"\x00" not in raw, "NUL padding means an unflushed extension"
    assert raw.count(b"\n") == 4


def test_no_nul_padding_and_size_matches_content(tmp_path):
    """The specific corruption seen after a mid-run reboot: file size
    extended past the flushed data, the remainder reading as NUL."""
    state = make_state(tmp_path)
    state = state.replace(
        stats=jnp.full((4, state.n_stats), 0.5, dtype=jnp.float32))
    assert state.writestats(FakeManager(tmp_path))
    path = tmp_path / "loss.csv"
    raw = path.read_bytes()
    assert len(raw) == path.stat().st_size
    assert b"\x00" not in raw
    assert np.loadtxt(path, delimiter=",").shape == (4, state.n_stats)
