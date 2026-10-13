"""Pairs of adjacent sentences from a document-ordered sentence cache.

`mc4_sent16` stores each document's run of up to 16 sentences on
consecutive rows, with a `(doc, pos, segment)` sidecar (`encode_sentences.py`).
A pair is a row and the row after it when both belong to one document and
the second is the next segment of that document's text. The sentence
builder skips unusable segments inside a run, so two neighbouring rows of a
document are not always neighbouring sentences, and such pairs are dropped.

`PairSource` serves `(sentence, target)` pairs for `train()`: the target is
the next sentence (objective "next") or the sentence itself ("self"), over
the same input rows, so the two objectives see identical inputs.

For the pair objective (`pair_objective.md`), `doc_partners` draws any other
sentence of the same document at a minimum segment distance, and
`StackedPairSource` serves both sentences as one record.
"""
from pathlib import Path
from typing import Tuple

import grain.python as gp
import numpy as np
from jaxtyping import Float, Int

from ontologize.data.loaders import doc_holdout


def adjacent(index: Int[np.ndarray, "n 3"], lo: int, hi: int
             ) -> Int[np.ndarray, "p"]:
    """Rows i in [lo, hi - 1) whose row i + 1 is the next segment of the same
    document: the same `doc` and a `segment` one higher."""
    doc, seg = index[lo:hi, 0], index[lo:hi, 2]
    ok = (doc[1:] == doc[:-1]) & (seg[1:] == seg[:-1] + 1)
    return lo + np.flatnonzero(ok).astype(np.int64)


def pair_rows(cache, train_rows: int, holdout: int
              ) -> Tuple[Int[np.ndarray, "p_train"], Int[np.ndarray, "p_eval"]]:
    """First rows of the training pairs, all within the cache's first
    `train_rows` rows, and of the evaluation pairs, all within its tail.
    The tail is `holdout` rows snapped up to whole documents
    (`doc_holdout`), as for every other model trained on the cache, so no
    document contributes to both."""
    cache = Path(cache)
    n = len(np.load(cache, mmap_mode="r"))
    tail = doc_holdout(cache, holdout)
    index = np.load(cache.with_name(cache.stem + ".index.npy"), mmap_mode="r")
    hi = min(train_rows, n - tail)
    return adjacent(index, 0, hi), adjacent(index, n - tail, n)


class PairSource(gp.RandomAccessDataSource):
    """Grain source of `(x_t, target)` cache rows for the pairs starting at
    `rows`. Memory-maps lazily, like `NpyDataSource`, so it pickles cleanly
    into grain workers."""

    def __init__(self, cache, rows: Int[np.ndarray, "p"],
                 objective: str = "next"):
        if objective not in ("next", "self"):
            raise ValueError(f"objective {objective!r}: 'next' or 'self'")
        self.cache = str(cache)
        self.rows = np.asarray(rows, np.int64)
        self.objective = objective
        self._arr = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, j
                    ) -> Tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d"]]:
        if self._arr is None:
            self._arr = np.load(self.cache, mmap_mode="r")
        i = int(self.rows[int(j)])
        x = np.array(self._arr[i], np.float32)
        y = (np.array(self._arr[i + 1], np.float32)
             if self.objective == "next" else x)
        return x, y

    def __getstate__(self) -> dict:
        return {"cache": self.cache, "rows": self.rows,
                "objective": self.objective, "_arr": None}


def doc_partners(index: Int[np.ndarray, "n 3"], lo: int, hi: int,
                 min_d: int, rng: np.random.Generator, chunk: int = 1 << 20
                 ) -> Tuple[Int[np.ndarray, "p"], Int[np.ndarray, "p"]]:
    """For each row in [lo, hi) that has one, a partner drawn uniformly from
    the rows of its own document in [lo, hi) whose segment is at least
    `min_d` away. Returns (rows, partners). A document's rows are contiguous
    in the cache, so each document is a run of rows."""
    doc = np.asarray(index[lo:hi, 0])
    seg = np.asarray(index[lo:hi, 2]).astype(np.int64)
    n = hi - lo
    start = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
    length = np.diff(np.r_[start, n])
    run = np.repeat(np.arange(len(start)), length)
    pos = np.arange(n) - start[run]
    width = int(length.max())
    S = np.zeros((len(start), width), np.int64)
    S[run, pos] = seg
    real = np.arange(width)[None, :] < length[:, None]
    rows, partners = [], []
    for a in range(0, n, chunk):
        r = run[a:a + chunk]
        ok = real[r] & (np.abs(S[r] - seg[a:a + chunk, None]) >= min_d)
        score = np.where(ok, rng.random(ok.shape, dtype=np.float32), -1.0)
        has = ok.any(1)
        i = a + np.flatnonzero(has)
        rows.append(i)
        partners.append(start[run[i]] + score.argmax(1)[has])
    return lo + np.concatenate(rows), lo + np.concatenate(partners)


class StackedPairSource(gp.RandomAccessDataSource):
    """Grain source of two cache rows concatenated, (x_i, x_j) -> (2d,),
    for the pairs (rows[m], partners[m]). Every sentence is its own target,
    so a batch (b, 2d) reaches `train()` as an ordinary self-target batch,
    and `pairloss.pair_probe` splits it back into (2b, d). Memory-maps
    lazily, like `PairSource`, so it pickles cleanly into grain workers."""

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

    def __getitem__(self, m) -> Float[np.ndarray, "two_d"]:
        if self._arr is None:
            self._arr = np.load(self.cache, mmap_mode="r")
        m = int(m)
        return np.concatenate([self._arr[self.rows[m]],
                               self._arr[self.partners[m]]]).astype(np.float32)

    def __getstate__(self) -> dict:
        return {"cache": self.cache, "rows": self.rows,
                "partners": self.partners, "_arr": None}
