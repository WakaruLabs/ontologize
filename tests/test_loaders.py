# NpyDataSource: the O(1) random-access view of the embedding cache that
# the whole cached-training path (sonar.py `cache`) reads through.
import pickle

import numpy as np
import pytest

from ontologize.data.loaders import NpyDataSource


def make_source(tmp_path, **kwargs):
    arr = np.arange(40, dtype=np.float32).reshape(10, 4)
    path = tmp_path / "cache.npy"
    np.save(path, arr)
    return NpyDataSource(path, **kwargs), arr


def test_len_and_random_access(tmp_path):
    src, arr = make_source(tmp_path)
    assert len(src) == 10
    assert np.array_equal(src[3], arr[3])
    assert np.array_equal(src[9], arr[9])


def test_rows_are_copies_not_memmap_views(tmp_path):
    # batches must not pin the memmap
    src, _ = make_source(tmp_path)
    row = src[0]
    assert type(row) is np.ndarray
    assert not isinstance(row, np.memmap)


def test_pickles_into_workers(tmp_path):
    # grain worker processes receive the source by pickle; the memmap is
    # dropped in __getstate__ and lazily reopened on first access
    src, arr = make_source(tmp_path)
    src2 = pickle.loads(pickle.dumps(src))
    assert src2._arr is None
    assert len(src2) == 10
    assert np.array_equal(src2[7], arr[7])


def test_holdout_hides_tail(tmp_path):
    # the tail sae.py scores on (--eval-rows) must be unreachable when
    # sonar.py trains with the matching `holdout`
    src, arr = make_source(tmp_path, holdout=3)
    assert len(src) == 7
    assert np.array_equal(src[6], arr[6])
    with pytest.raises(IndexError):
        src[7]
    with pytest.raises(IndexError):
        src[-1]  # numpy wraparound must not reach the tail either


def test_holdout_survives_pickle(tmp_path):
    src, arr = make_source(tmp_path, holdout=3)
    src2 = pickle.loads(pickle.dumps(src))
    assert len(src2) == 7
    with pytest.raises(IndexError):
        src2[9]


def test_holdout_out_of_range(tmp_path):
    with pytest.raises(ValueError):
        make_source(tmp_path, holdout=10)
    with pytest.raises(ValueError):
        make_source(tmp_path, holdout=-1)
