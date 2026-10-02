# NpyDataSource: the O(1) random-access view of the embedding cache that
# the whole cached-training path (sonar.py `cache`) reads through.
import pickle

import numpy as np

from ontologize.data.loaders import NpyDataSource


def make_source(tmp_path):
    arr = np.arange(40, dtype=np.float32).reshape(10, 4)
    path = tmp_path / "cache.npy"
    np.save(path, arr)
    return NpyDataSource(path), arr


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
