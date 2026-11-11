"""Differences of two sentences from one document, as training rows.

A sentence's embedding carries what its document holds fixed (language,
topic, register) and what the sentence itself says. The pair objective's
slow heads are trained to keep the first. The difference of two sentences
from one document cancels it, to the extent it is additive, and leaves
what varies between them. Joshi et al.'s identifiability result for
sparse shift autoencoders covers that side: from paired samples, the
concepts that vary within a pair are identified up to permutation and
scale; those every pair holds fixed drop out.

`DiffSource` serves x[row] - x[partner] for the pairs
`experiments/next-sentence/train_pair.py` trains on (the same draw:
`pairs.doc_partners`, a partner at least `min_d` segments away). Run as a
script, this file writes what a difference model needs beside the cache:

  mse_weights_diff.npy  per-coordinate inverse variance of the differences
                        over a sample of training pairs, zero where the
                        sentence weights are (coordinate 837) and scaled
                        to their sum
  tail.npy              the differences of the held-out tail's pairs
  tail_rows.npy         their (row, partner) cache rows
  tail_langs.npy        their documents' language labels

    uv run python experiments/pair-diff/diffs.py
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple

import grain.python as gp
import numpy as np
from jaxtyping import Float, Int

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "next-sentence"))

from pairs import doc_partners  # noqa: E402

from ontologize.data.loaders import doc_holdout  # noqa: E402

CACHE = ROOT / "data" / "sonar_embeddings" / "mc4_sent16.npy"
WEIGHTS = ROOT / "data" / "out" / "sonar" / "sentcache" / "mse_weights_no837.npy"
OUT = ROOT / "data" / "out" / "sonar" / "pair-diff"


class DiffSource(gp.RandomAccessDataSource):
    """Grain source of x[rows[m]] - x[partners[m]], (d,). Memory-maps
    lazily, so it pickles cleanly into grain workers."""

    def __init__(self, cache, rows: Int[np.ndarray, "p"],
                 partners: Int[np.ndarray, "p"]):
        if len(rows) != len(partners):
            raise ValueError(f"{len(rows)} rows but {len(partners)} partners")
        self.cache = str(cache)
        self.rows = np.asarray(rows, np.int64)
        self.partners = np.asarray(partners, np.int64)
        self._arr = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, m) -> Float[np.ndarray, "d"]:
        if self._arr is None:
            self._arr = np.load(self.cache, mmap_mode="r")
        m = int(m)
        return (self._arr[self.rows[m]].astype(np.float32)
                - self._arr[self.partners[m]].astype(np.float32))

    def __getstate__(self) -> dict:
        return {"cache": self.cache, "rows": self.rows,
                "partners": self.partners, "_arr": None}


def differences(arr: Float[np.ndarray, "n d"], rows: Int[np.ndarray, "p"],
                partners: Int[np.ndarray, "p"], chunk: int = 65536
                ) -> Float[np.ndarray, "p d"]:
    """x[rows] - x[partners], read a chunk at a time."""
    out = np.empty((len(rows), arr.shape[1]), np.float32)
    for i in range(0, len(rows), chunk):
        r, q = rows[i:i + chunk], partners[i:i + chunk]
        # a memory map wants sorted indices to read without thrashing
        ur, ir = np.unique(r, return_inverse=True)
        uq, iq = np.unique(q, return_inverse=True)
        out[i:i + chunk] = (np.asarray(arr[ur], np.float32)[ir]
                            - np.asarray(arr[uq], np.float32)[iq])
    return out


def diff_weights(D: Float[np.ndarray, "p d"], base: Float[np.ndarray, "d"]
                 ) -> Float[np.ndarray, "d"]:
    """Inverse variance of each coordinate of `D`, zero where `base` is,
    scaled to `base`'s sum: the whitened error of a difference model then
    weighs against the auxiliary terms as a sentence model's does under
    `base`."""
    var = D.astype(np.float64).var(0)
    w = np.where(base > 0, 1.0 / np.maximum(var, 1e-30), 0.0)
    return (w * base.sum() / w.sum()).astype(np.float32)


def tail_pairs(cache, holdout: int, min_d: int, seed: int
               ) -> Tuple[Int[np.ndarray, "p"], Int[np.ndarray, "p"]]:
    """(rows, partners) within the cache's document-snapped held-out tail,
    drawn as the training pairs are."""
    cache = Path(cache)
    n = len(np.load(cache, mmap_mode="r"))
    index = np.load(cache.with_name(cache.stem + ".index.npy"), mmap_mode="r")
    return doc_partners(index, n - doc_holdout(cache, holdout), n, min_d,
                        np.random.default_rng([seed, 11]))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--base-weights", default=str(WEIGHTS))
    p.add_argument("--rows", type=int, default=4_000_000,
                   help="the training prefix the weights are sampled from")
    p.add_argument("--sample", type=int, default=262144)
    p.add_argument("--holdout", type=int, default=32768)
    p.add_argument("--min-d", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUT))
    cfg = p.parse_args()

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg.cache)
    arr = np.load(cache, mmap_mode="r")
    index = np.load(cache.with_name(cache.stem + ".index.npy"), mmap_mode="r")
    rng = np.random.default_rng(cfg.seed)
    hi = min(cfg.rows, len(arr) - doc_holdout(cache, cfg.holdout))
    rows, partners = doc_partners(index, 0, hi, cfg.min_d, rng)
    pick = np.sort(rng.choice(len(rows), min(cfg.sample, len(rows)),
                              replace=False))
    D = differences(arr, rows[pick], partners[pick])
    base = np.load(cfg.base_weights)
    w = diff_weights(D, base)
    np.save(out / "mse_weights_diff.npy", w)
    norms = np.linalg.norm(D, axis=1)
    print(f"{len(pick)} of {len(rows)} training pairs: |x_a - x_b| median "
          f"{np.median(norms):.3f}; {int((w == 0).sum())} coordinates masked")

    tr, tp = tail_pairs(cache, cfg.holdout, cfg.min_d, cfg.seed)
    T = differences(arr, tr, tp)
    np.save(out / "tail.npy", T)
    np.save(out / "tail_rows.npy", np.stack([tr, tp], 1))
    langs = np.load(cache.with_name(cache.stem + ".langs.npy"),
                    mmap_mode="r")
    np.save(out / "tail_langs.npy", np.asarray(langs[np.sort(tr)])[
        np.argsort(np.argsort(tr))])
    print(f"{len(tr)} held-out pairs -> {out}")


if __name__ == "__main__":
    main()
