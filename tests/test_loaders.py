# NpyDataSource: the O(1) random-access view of the embedding cache that
# the whole cached-training path (sonar.py `cache`) reads through. Also
# SampleLoader's batching of (input, target) records, which train() reads
# as a tuple on both of its paths.
import pickle

import numpy as np
import pytest

from ontologize.data.loaders import EmbeddingLoader, NpyDataSource


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


class Pairs:
    """Records (i, i + 1) as rows of 4. No length, so SampleLoader iterates
    and collates them itself."""

    def __init__(self, n):
        self.n = n

    def __getitem__(self, i):
        return (np.full(4, i, np.float32), np.full(4, i + 1, np.float32))

    def __iter__(self):
        return (self[i] for i in range(self.n))


class RandomAccessPairs(Pairs):
    """The same with a length, so grain's DataLoader batches them."""

    def __len__(self):
        return self.n


@pytest.mark.parametrize("src", [RandomAccessPairs(16), Pairs(16)],
                         ids=["grain", "iterable"])
def test_pair_records_batch_to_a_tuple(src):
    # grain's Batch serves a source with a length; SampleLoader's own
    # collation serves an iterable, and np.stack there would fuse the pair
    # into one (b, 2, d) array that train() would read as the input
    batches = list(EmbeddingLoader(4, 1, src, d=4))
    assert len(batches) == 4
    for X, Y in batches:
        assert X.shape == Y.shape == (4, 4)
        assert np.array_equal(Y, X + 1)
