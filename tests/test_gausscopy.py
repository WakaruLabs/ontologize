# experiments/seed-stability/gausscopy.py: the Gaussian copy of a cache.
# The tests pin the chunked moments against numpy's, the square root of a
# rank-deficient covariance, and the source: its rows have the planted
# moments, are the same after a pickle round trip and across a block
# boundary, differ between streams, and the held-out file is the same
# generator under its own stream.
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "seed-stability"))
from gausscopy import (BLOCK, GaussianSource, factor, label_moments,  # noqa: E402
                       mixture_rows, moments, write_rows)


def _planted(tmp_path, d=6, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    cov = A @ A.T + 0.1 * np.eye(d)
    mean = rng.normal(size=d) * 5
    path = tmp_path / "moments.npz"
    np.savez(path, mean=mean, cov=cov, factor=factor(cov))
    return path, mean, cov


def test_moments_match_numpy_with_a_large_mean_and_ragged_chunks():
    rng = np.random.default_rng(1)
    X = (rng.normal(size=(1000, 5)) + 1e3).astype(np.float32)
    m, c = moments(X, chunk=128)
    assert np.allclose(m, X.astype(np.float64).mean(0), atol=1e-9)
    assert np.allclose(c, np.cov(X.astype(np.float64).T, bias=True), atol=1e-8)


def test_factor_is_a_square_root_of_a_rank_deficient_covariance():
    rng = np.random.default_rng(2)
    B = rng.normal(size=(8, 3))
    cov = B @ B.T
    F = factor(cov)
    assert np.allclose(F @ F.T, cov, atol=1e-10)


def test_source_rows_have_the_planted_moments(tmp_path):
    path, mean, cov = _planted(tmp_path)
    src = GaussianSource(path, 6 * BLOCK, seed=0)
    X = np.stack([src[i] for i in range(len(src))]).astype(np.float64)
    m, c = moments(X)
    assert np.abs(m - mean).max() < 0.1
    assert np.linalg.norm(c - cov) / np.linalg.norm(cov) < 0.03


def test_source_is_deterministic_pickles_and_crosses_blocks(tmp_path):
    path, _, _ = _planted(tmp_path)
    src = GaussianSource(path, 3 * BLOCK, seed=0)
    idx = [0, BLOCK - 1, BLOCK, 2 * BLOCK + 5, 7]
    rows = [src[i] for i in idx]
    again = pickle.loads(pickle.dumps(src))
    assert all(np.array_equal(r, again[i]) for r, i in zip(rows, idx))
    other = GaussianSource(path, 3 * BLOCK, seed=1)
    assert not np.allclose(rows[0], other[0])


def test_concurrent_reads_generate_each_block_once(tmp_path, monkeypatch):
    import threading
    import gausscopy
    path, _, _ = _planted(tmp_path)
    src = GaussianSource(path, 3 * BLOCK, seed=0)
    calls = []
    real = gausscopy.block
    monkeypatch.setattr(gausscopy, "block",
                        lambda *a, **k: calls.append(a[3]) or real(*a, **k))
    idx = np.arange(3 * BLOCK)
    out = np.zeros((len(idx), 6), np.float32)

    def read(part):
        for i in part:
            out[i] = src[i]

    threads = [threading.Thread(target=read, args=(idx[t::16],)) for t in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(calls) == [0, 1, 2]
    fresh = GaussianSource(path, 3 * BLOCK, seed=0)
    assert np.array_equal(out[[5, BLOCK + 7, 2 * BLOCK + 9]],
                          np.stack([fresh[5], fresh[BLOCK + 7], fresh[2 * BLOCK + 9]]))


def test_heldout_file_is_the_generator_under_its_own_stream(tmp_path):
    path, _, _ = _planted(tmp_path)
    z = np.load(path)
    n = BLOCK + 100
    write_rows(tmp_path / "eval.npy", z["mean"].astype(np.float32),
               z["factor"].astype(np.float32), n, seed=1)
    E = np.load(tmp_path / "eval.npy")
    assert E.shape == (n, 6)
    held = GaussianSource(path, n, seed=1)
    assert np.allclose(E[[0, BLOCK, n - 1]],
                       np.stack([held[0], held[BLOCK], held[n - 1]]))
    train = GaussianSource(path, n, seed=0)
    assert not np.allclose(E[0], train[0])


def test_label_moments_match_numpy_per_label_across_ragged_chunks():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1003, 4)) + 7.0
    labels = rng.choice(np.array(["b", "a", "c"]), size=1003, p=[0.6, 0.3, 0.1])
    names, cnt, means, covs = label_moments(X, labels, chunk=97)
    assert names.tolist() == ["a", "b", "c"]
    for j, name in enumerate(names):
        R = X[labels == name]
        assert cnt[j] == len(R)
        assert np.allclose(means[j], R.mean(0))
        assert np.allclose(covs[j], np.cov(R.T, bias=True), atol=1e-10)
    # the law of total covariance recovers the pooled moments
    p = cnt / cnt.sum()
    mu = p @ means
    total = (np.einsum("l,lij->ij", p, covs)
             + np.einsum("l,li,lj->ij", p, means - mu, means - mu))
    assert np.allclose(total, moments(X, chunk=97)[1], atol=1e-10)


def test_mixture_rows_follow_their_labels_and_are_deterministic():
    names = np.array(["x", "y"])
    means = np.array([[0.0, 0.0], [10.0, -10.0]], np.float32)
    factors = np.stack([np.eye(2), np.diag([0.1, 2.0])]).astype(np.float32)
    labels = np.array(["y", "x"] * 20000)
    R = mixture_rows(labels, names, means, factors, seed=0)
    Y = R[labels == "y"]
    assert np.allclose(Y.mean(0), [10, -10], atol=0.05)
    assert np.allclose(Y.std(0), [0.1, 2.0], rtol=0.03)
    assert np.allclose(R[labels == "x"].std(0), [1, 1], rtol=0.03)
    assert np.array_equal(R, mixture_rows(labels, names, means, factors, 0))
    assert not np.allclose(R, mixture_rows(labels, names, means, factors, 1))
    # a row depends on its label's stream, not on the other label's rows
    alone = mixture_rows(labels[labels == "y"], names, means, factors, 0)
    assert np.array_equal(alone, Y)
    try:
        mixture_rows(np.array(["z"]), names, means, factors, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown label was drawn")
