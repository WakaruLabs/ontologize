# Resuming continues the run's batch stream instead of replaying it.
# `epochs` fixes the length of the whole run; a loader built at `step_0`
# must yield exactly the batches an uninterrupted loader yields after its
# first `step_0`, and a resumed TrainingEnv must stop at the same final step.
import numpy as np
import pytest

from ontologize.data.loaders import EmbeddingLoader, NpyDataSource
from ontologize.training.config import Hyperparams, Metadata, TrainingEnv

from conftest import KW

N, D, B, EPOCHS = 50, 4, 8, 3     # B does not divide N: batches straddle epochs


@pytest.fixture
def src(tmp_path):
    arr = np.arange(N * D, dtype=np.float32).reshape(N, D)
    np.save(tmp_path / "cache.npy", arr)
    return NpyDataSource(tmp_path / "cache.npy")


def batches(src, step_0=0, **kw):
    loader = EmbeddingLoader(B, EPOCHS, src, d=D, shuffle=True,
                             step_0=step_0, **kw)
    return loader, [np.asarray(x) for x in loader]


@pytest.mark.parametrize("step_0", [1, 6, 7, 13])
def test_resumed_stream_is_the_uninterrupted_tail(src, step_0):
    # 6 and 7 bracket the first epoch boundary (50 / 8 = 6.25 batches)
    _, full = batches(src)
    loader, tail = batches(src, step_0)
    assert len(full) == N * EPOCHS // B
    assert loader.steps == len(tail) == len(full) - step_0
    for a, b in zip(full[step_0:], tail):
        assert np.array_equal(a, b)


def test_epochs_reshuffle(src):
    # the skip is only meaningful if later epochs differ from the first
    _, full = batches(src)
    assert not np.array_equal(full[0], full[7])


@pytest.mark.parametrize("step_0", [N * EPOCHS // B, N * EPOCHS // B + 5])
def test_past_the_end_yields_nothing(src, step_0):
    loader, tail = batches(src, step_0)
    assert loader.steps == 0 and tail == []


def test_keep_remainder_counts_the_partial_batch(src):
    loader, tail = batches(src, 2, drop_remainder=False)
    assert loader.steps == len(tail) == -(-N * EPOCHS // B) - 2
    assert tail[-1].shape[0] == N * EPOCHS % B


def test_iterable_source_skips_batches():
    rows = [np.full(D, i, np.float32) for i in range(40)]

    class Stream:
        def __iter__(self):
            return iter(rows)

    full = list(EmbeddingLoader(B, 1, Stream(), d=D))
    loader = EmbeddingLoader(B, 1, Stream(), d=D, step_0=2)
    assert loader.steps is None
    tail = list(loader)
    assert len(tail) == len(full) - 2
    for a, b in zip(full[2:], tail):
        assert np.array_equal(a, b)


def env(tmp_path, cache, epochs):
    d = KW["d_in"]
    hyper = Hyperparams(d, d, B, epochs, 5e-5, 0.0, 1.0,
                        "none", "normal", "featvar", 0.0, 0.0, 0.0,
                        ghost=False)
    model = hyper.ontologizer(d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"],
                              n=2, gate="none", forward="resid",
                              dtype_str="float32", dtype_p_str="float32")
    meta = Metadata("embedding", out_path=tmp_path / "run",
                    save_each=2, checkpoint_each=2)
    return TrainingEnv(model, hyper, meta,
                       kwargs_loader={"d": d, "shuffle": True})


def test_training_env_resume_stops_at_the_run_total(tmp_path):
    n = 40
    X = np.random.default_rng(0).normal(size=(n, KW["d_in"]))
    np.save(tmp_path / "cache.npy", X.astype(np.float32))
    cache = NpyDataSource(tmp_path / "cache.npy")
    per_epoch = n // B

    # a finished 1-epoch run, then resumed asking for 2 epochs in all:
    # it trains the second epoch only, not two more
    assert int(env(tmp_path, cache, 1).train(cache).step) == per_epoch
    assert int(env(tmp_path, cache, 2).train(cache).step) == 2 * per_epoch

    # resuming a finished run trains nothing
    assert int(env(tmp_path, cache, 2).train(cache).step) == 2 * per_epoch
