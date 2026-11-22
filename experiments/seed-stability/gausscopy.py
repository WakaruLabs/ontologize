"""A Gaussian copy of a cache: rows drawn from the Gaussian with the mean and
covariance of the cache's training prefix.

Seed agreement on SONAR can only exceed what a copy of the data's second
moments gives if the seeds share structure beyond the covariance. A model
trained on rows from that Gaussian, at two seeds and with the real arm's
flags, is the baseline every seed statistic is read against at once:
partitions, head contributions, subspaces and fit.

`prepare` computes the moments of the cache's first `--rows` rows (the
rows the recipe ladder's 50,000-step arms train on) and writes a held-out
file of Gaussian rows from another stream, in the cache layout, so the
scorers read it with `--cache` as they read a cache's tail. Its first rows
also serve the trainer's initialization probe, which reads a cache's head;
the scorers read only its last rows, so the two never overlap.

`GaussianSource` is the training data: row i is block i // B of a stream
keyed by (seed, block), so a run that resumes sees the rows it would have
seen, and two seeds of an arm see the same rows in the same order, as two
seeds of a cache arm do.

`mixture` keeps one more thing: each language's own mean and covariance
(mixture.npz, from the `.langs.npy` sidecar), so a copy drawn with the real
rows' languages (`mixture_rows`) has the data's language structure and
nothing beyond it. Its pooled covariance is the cache's, by the law of
total covariance, which `mixture.meta.json` checks against `prepare`'s.

    uv run python experiments/seed-stability/gausscopy.py prepare
    uv run python experiments/seed-stability/gausscopy.py mixture
"""
import argparse
import json
import threading
from collections import OrderedDict
from pathlib import Path

import grain.python as gp
import numpy as np
from jaxtyping import Float, Int

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/sonar_embeddings/mc4_sent16.npy"
OUT = ROOT / "data/out/sonar/gausscopy"
BLOCK = 4096


def moments(X: Float[np.ndarray, "n d"], chunk: int = 131072
            ) -> tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d d"]]:
    """Mean and (biased) covariance of the rows of X in float64, in chunks
    shifted by the first chunk's mean so the second moment does not lose
    precision to a large mean."""
    n, d = X.shape
    shift = np.asarray(X[:min(chunk, n)], np.float64).mean(0)
    s1, s2 = np.zeros(d), np.zeros((d, d))
    for i in range(0, n, chunk):
        Xc = np.asarray(X[i:i + chunk], np.float64) - shift
        s1 += Xc.sum(0)
        s2 += Xc.T @ Xc
    m = s1 / n
    return shift + m, s2 / n - np.outer(m, m)


def factor(cov: Float[np.ndarray, "d d"]) -> Float[np.ndarray, "d d"]:
    """F with F Fᵀ = cov, from the eigendecomposition (negative rounding
    error in the smallest eigenvalues is clipped to zero), so a covariance
    too ill-conditioned for Cholesky still has a square root."""
    lam, V = np.linalg.eigh(cov)
    return V * np.sqrt(np.clip(lam, 0.0, None))


def label_moments(X: Float[np.ndarray, "n d"], labels: np.ndarray,
                  chunk: int = 131072
                  ) -> tuple[np.ndarray, Int[np.ndarray, "L"],
                             Float[np.ndarray, "L d"],
                             Float[np.ndarray, "L d d"]]:
    """(the labels in sorted order, each label's row count, mean and
    biased covariance), in float64, over the rows of X with their labels,
    in chunks shifted by the first chunk's mean as `moments` is."""
    n, d = X.shape
    names = np.unique(np.asarray(labels[:n]))
    shift = np.asarray(X[:min(chunk, n)], np.float64).mean(0)
    cnt = np.zeros(len(names), np.int64)
    s1, s2 = np.zeros((len(names), d)), np.zeros((len(names), d, d))
    for i in range(0, n, chunk):
        Xc = np.asarray(X[i:i + chunk], np.float64) - shift
        lab = np.searchsorted(names, np.asarray(labels[i:i + chunk]))
        for j in np.unique(lab):
            R = Xc[lab == j]
            cnt[j] += len(R)
            s1[j] += R.sum(0)
            s2[j] += R.T @ R
    m = s1 / cnt[:, None]
    cov = s2 / cnt[:, None, None] - m[:, :, None] * m[:, None, :]
    return names, cnt, shift + m, cov


def mixture_rows(labels: np.ndarray, names: np.ndarray,
                 means: Float[np.ndarray, "L d"],
                 factors: Float[np.ndarray, "L d d"], seed: int
                 ) -> Float[np.ndarray, "n d"]:
    """One row per entry of `labels`, drawn from that label's Gaussian
    (mean + F z). Label j's rows come from the stream keyed (seed, j), in
    the order they appear, so a row depends only on its label, its rank
    among that label's rows and the seed."""
    labels = np.asarray(labels)
    j_of = np.searchsorted(names, labels)
    if (j_of >= len(names)).any() or (names[np.minimum(j_of, len(names) - 1)]
                                      != labels).any():
        raise ValueError("a label has no moments")
    out = np.empty((len(labels), means.shape[1]), np.float32)
    for j in np.unique(j_of):
        idx = np.flatnonzero(j_of == j)
        z = np.random.default_rng([seed, int(j)]).standard_normal(
            (len(idx), factors.shape[2]), dtype=np.float32)
        out[idx] = means[j] + z @ factors[j].T
    return out


def block(mean: Float[np.ndarray, "d"], F: Float[np.ndarray, "d d"],
          seed: int, j: int, size: int = BLOCK) -> Float[np.ndarray, "size d"]:
    """Block j of the stream keyed by seed: `size` rows mean + F z."""
    z = np.random.default_rng([seed, j]).standard_normal(
        (size, F.shape[1]), dtype=np.float32)
    return (mean + z @ F.T).astype(np.float32)


class GaussianSource(gp.RandomAccessDataSource):
    """`n` rows of the Gaussian with the moments in `path` (`prepare`'s
    moments.npz), generated a block at a time. Grain reads a source from
    several threads at once, so the recent blocks are cached under a lock
    and each is generated once while reads stay within them. Reads must
    therefore come in order: under a shuffled sampler a batch's rows lie in
    nearly as many blocks as it has rows, each generated afresh, which held
    training to a step every 18 seconds. The rows are independent draws, so
    reading them in order costs nothing (`train_gauss.py` turns the shuffle
    off)."""

    CACHED = 8

    def __init__(self, path, n: int, seed: int = 0):
        self.path, self.n, self.seed = str(path), int(n), int(seed)
        self._load()

    def _load(self):
        z = np.load(self.path)
        self.mean = z["mean"].astype(np.float32)
        self.F = z["factor"].astype(np.float32)
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        idx = int(idx)
        if not 0 <= idx < self.n:
            raise IndexError(f"index {idx} out of range for {self.n} rows")
        j = idx // BLOCK
        with self._lock:
            rows = self._cache.get(j)
            if rows is None:
                rows = block(self.mean, self.F, self.seed, j)
                self._cache[j] = rows
                while len(self._cache) > self.CACHED:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(j)
        return np.array(rows[idx % BLOCK])

    def __getstate__(self):
        return {"path": self.path, "n": self.n, "seed": self.seed}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load()


def write_rows(path: Path, mean: Float[np.ndarray, "d"],
               F: Float[np.ndarray, "d d"], n: int, seed: int) -> None:
    """n rows of the stream keyed by seed, as a .npy cache."""
    X = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                  shape=(n, F.shape[0]))
    for j in range(0, -(-n // BLOCK)):
        rows = block(mean, F, seed, j)
        X[j * BLOCK:(j + 1) * BLOCK] = rows[:n - j * BLOCK]
    X.flush()


def prepare(cfg: argparse.Namespace) -> None:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(cfg.cache, mmap_mode="r")
    assert cfg.rows <= X.shape[0], f"{cfg.cache} has {X.shape[0]} rows"
    print(f"moments of the first {cfg.rows:,} rows of {Path(cfg.cache).name}")
    mean, cov = moments(X[:cfg.rows])
    F = factor(cov)
    lam = np.linalg.eigvalsh(cov)
    np.savez(out / "moments.npz", mean=mean, cov=cov, factor=F)
    write_rows(out / "eval.npy", mean.astype(np.float32),
               F.astype(np.float32), cfg.eval_rows, cfg.eval_seed)
    E = np.load(out / "eval.npy", mmap_mode="r")
    em, ec = moments(E)
    meta = {"kind": "gaussian copy", "cache": str(cfg.cache),
            "rows": cfg.rows, "d": int(X.shape[1]),
            "eval_rows": cfg.eval_rows, "eval_seed": cfg.eval_seed,
            "block": BLOCK, "eig_min": float(lam[0]), "eig_max": float(lam[-1]),
            "eval_mean_err": float(np.abs(em - mean).max()),
            "eval_cov_rel_err": float(np.linalg.norm(ec - cov)
                                      / np.linalg.norm(cov))}
    (out / "eval.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def mixture(cfg: argparse.Namespace) -> None:
    """Per-language moments of the training prefix, for a copy that keeps
    each language's mean and covariance and nothing else: mixture.npz."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.load(cfg.cache, mmap_mode="r")
    L = np.load(Path(cfg.cache).with_suffix(".langs.npy"), mmap_mode="r")
    print(f"per-language moments of the first {cfg.rows:,} rows of "
          f"{Path(cfg.cache).name}")
    names, cnt, means, covs = label_moments(X[:cfg.rows], L[:cfg.rows])
    factors = np.stack([factor(c) for c in covs]).astype(np.float32)
    np.savez(out / "mixture.npz", names=names, counts=cnt,
             means=means.astype(np.float32), factors=factors)
    # the law of total covariance against the pooled moments, as a check
    p = cnt / cnt.sum()
    mu = p @ means
    within = np.einsum("l,lij->ij", p, covs)
    between = np.einsum("l,li,lj->ij", p, means - mu, means - mu)
    meta = {"kind": "language mixture copy", "cache": str(cfg.cache),
            "rows": cfg.rows, "languages": len(names),
            "rows_min": int(cnt.min()), "rows_max": int(cnt.max()),
            "between_share": float(np.trace(between)
                                   / np.trace(within + between))}
    pooled = out / "moments.npz"
    if pooled.exists():   # `prepare`'s moments, of the same prefix
        z = np.load(pooled)
        meta["total_cov_rel_err"] = float(
            np.linalg.norm(within + between - z["cov"])
            / np.linalg.norm(z["cov"]))
    (out / "mixture.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare", help="moments and the held-out file")
    q.add_argument("--cache", default=str(CACHE))
    q.add_argument("--rows", type=int, default=12_800_000,
                   help="training prefix: the ladder's 50,000 steps of 256")
    q.add_argument("--eval-rows", type=int, default=65536,
                   help="held-out file; the scorers read its last rows")
    q.add_argument("--eval-seed", type=int, default=1,
                   help="its stream (training uses seed 0)")
    q.add_argument("--out", default=str(OUT))
    m = sub.add_parser("mixture", help="per-language moments (mixture.npz)")
    m.add_argument("--cache", default=str(CACHE))
    m.add_argument("--rows", type=int, default=12_800_000)
    m.add_argument("--out", default=str(OUT))
    cfg = p.parse_args()
    {"prepare": prepare, "mixture": mixture}[cfg.cmd](cfg)


if __name__ == "__main__":
    main()
