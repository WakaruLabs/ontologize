"""Generate data/out/sonar/mse_weights.npy, the per-dimension
inverse-variance whitening weights that sonar.py and the eval suite
(sae.py, compose.py, refit.py, headcoh.py) load but nothing in the tree
previously produced.

Recipe reconstructed from usage: per-dimension variance over the
embedding cache, inverted, normalized to mean 1 (so the whitened MSE is
on the same scale as plain MSE). VERIFY against the original weights
before regenerating on top of an existing run: a cache trained with one
weight vector must be evaluated with the same one.

    uv run python make_mse_weights.py [cache.npy] [out.npy]
"""

import sys

import numpy as np

CACHE = sys.argv[1] if len(sys.argv) > 1 else "data/sonar_embeddings/mc4_4M.npy"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/out/sonar/mse_weights.npy"


def main() -> None:
    cache = np.load(CACHE, mmap_mode="r")
    # accumulate variance in chunks; the cache may be far larger than RAM
    n, d = cache.shape
    chunk = 100_000
    count = 0
    mean = np.zeros(d, dtype=np.float64)
    m2 = np.zeros(d, dtype=np.float64)
    for start in range(0, n, chunk):
        x = np.asarray(cache[start : start + chunk], dtype=np.float64)
        c = x.shape[0]
        delta = x.mean(axis=0) - mean
        mean += delta * c / (count + c)
        m2 += x.var(axis=0) * c + delta**2 * count * c / (count + c)
        count += c
    var = m2 / count

    weights = 1.0 / np.maximum(var, np.finfo(np.float64).tiny)
    weights /= weights.mean()

    np.save(OUT, weights.astype(np.float32))
    print(f"{OUT}: d={d}, min={weights.min():.4g}, max={weights.max():.4g}, mean={weights.mean():.4g}")


if __name__ == "__main__":
    main()
