# experiments/next-sentence: adjacent sentence pairs from a document-ordered
# cache, the split that keeps every held-out document out of training, the
# pair source, and train()'s (input, target) batches.
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from ontologize.training import ontostate
from ontologize.training.config import Hyperparams, Metadata, TrainingEnv

from conftest import KW

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "next-sentence"))
from pairs import PairSource, adjacent, pair_rows  # noqa: E402


def test_adjacent_needs_the_same_document_and_the_next_segment():
    # doc 0: segments 0 1 3 (2 was unusable and skipped); doc 1: 0 1;
    # doc 2: a single sentence
    index = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 3],
                      [1, 0, 0], [1, 1, 1],
                      [2, 0, 0]], np.int32)
    assert adjacent(index, 0, len(index)).tolist() == [0, 3]
    # a window that ends inside a document keeps only pairs inside it
    assert adjacent(index, 0, 4).tolist() == [0]
    assert adjacent(index, 3, 6).tolist() == [3]


@pytest.fixture
def cache(tmp_path):
    """60 documents of 5 sentences each, row i holding the value i, with
    every document's third segment skipped."""
    n_docs, per, d = 60, 5, KW["d_in"]
    seg = np.array([0, 1, 3, 4, 5])
    rows = np.arange(n_docs * per)
    index = np.stack([rows // per, rows % per, seg[rows % per]], 1)
    np.save(tmp_path / "c.npy",
            np.repeat(rows[:, None], d, 1).astype(np.float32))
    np.save(tmp_path / "c.index.npy", index.astype(np.int32))
    np.save(tmp_path / "c.docstart.npy", np.arange(0, n_docs * per, per))
    return tmp_path / "c.npy"


def test_pair_rows_keeps_the_held_out_documents_whole(cache):
    train, held = pair_rows(cache, train_rows=200, holdout=12)
    index = np.load(cache.with_name("c.index.npy"))
    n = len(index)
    # a 12-row request snaps up to 3 whole documents (15 rows)
    assert held.min() >= n - 15 and held.max() < n - 1
    assert train.max() + 1 < 200
    assert not set(index[train, 0]) & set(index[held, 0])
    # 3 adjacent pairs per document: 0-1, 3-4, 4-5 (the skip breaks 1-3)
    assert len(held) == 3 * 3


def test_pair_source_serves_the_next_row_or_the_same_one(cache):
    rows = pair_rows(cache, 300, 12)[0]
    nxt = PairSource(cache, rows, "next")
    slf = PairSource(cache, rows, "self")
    for j in (0, 5, len(rows) - 1):
        i = rows[j]
        (x, y), (x_s, y_s) = nxt[j], slf[j]
        assert x[0] == i and y[0] == i + 1
        assert x_s[0] == i and y_s[0] == i
    # pickles without its memmap, as grain workers need
    assert pickle.loads(pickle.dumps(nxt))[3][1][0] == rows[3] + 1
    with pytest.raises(ValueError):
        PairSource(cache, rows, "previous")


@pytest.mark.parametrize("objective", ["next", "self"])
def test_train_takes_the_target_from_the_pair(cache, tmp_path, monkeypatch,
                                              objective):
    seen = []
    update = ontostate.update

    def spy(state, lossfn, rng, X, Y, *args, **kwargs):
        seen.append((np.asarray(X)[:, 0], np.asarray(Y)[:, 0]))
        return update(state, lossfn, rng, X, Y, *args, **kwargs)

    monkeypatch.setattr(ontostate, "update", spy)
    d = KW["d_in"]
    hyper = Hyperparams(d, d, 8, 1, 5e-5, 0.0, 1.0,
                        "none", "normal", "featvar", 0.0, 0.0, 0.0,
                        ghost=False)
    model = hyper.ontologizer(d, d, KW["e_dec"], KW["k"], KW["h"], 1, n=2,
                              gate="none", forward="resid",
                              dtype_str="float32", dtype_p_str="float32")
    meta = Metadata("embedding", out_path=tmp_path / "run", save_each=2,
                    checkpoint_each=2)
    src = PairSource(cache, pair_rows(cache, 80, 12)[0], objective)
    TrainingEnv(model, hyper, meta,
                kwargs_loader={"d": d, "shuffle": True}).train(src)
    assert seen
    step = 1 if objective == "next" else 0
    for x, y in seen:
        assert np.array_equal(y, x + step)


def test_next_with_residual_layers_is_refused():
    import train_next
    cfg = train_next.parser().parse_args([])
    assert (cfg.l, cfg.h, cfg.dict_init_scale, cfg.s_hm) == (1, 380, 0.1, 1e-4)
    train_next.check(cfg)
    for argv in (["--l", "2"], ["--l", "2", "--forward", "resid"]):
        with pytest.raises(SystemExit):
            train_next.check(train_next.parser().parse_args(argv))
    train_next.check(train_next.parser().parse_args(
        ["--l", "2", "--forward", "labels"]))
    train_next.check(train_next.parser().parse_args(
        ["--l", "2", "--objective", "self"]))
