"""Pairs of adjacent sentences from a document-ordered sentence cache.

`mc4_sent16` stores each document's run of up to 16 sentences on
consecutive rows, with a `(doc, pos, segment)` sidecar (`encode_sentences.py`).
A pair is a row and the row after it when both belong to one document and
the second is the next segment of that document's text. The sentence
builder skips unusable segments inside a run, so two neighbouring rows of a
document are not always neighbouring sentences, and such pairs are dropped.

`PairSource` serves `{"x": sentence, "y": target}` for `train()`: the target
is the next sentence (objective "next") or the sentence itself ("self"), over
the same input rows, so the two objectives see identical inputs.
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
    """Grain source of `{"x": x_t, "y": target}` cache rows for the pairs
    starting at `rows`. Memory-maps lazily, like `NpyDataSource`, so it
    pickles cleanly into grain workers."""

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

    def __getitem__(self, j) -> dict:
        if self._arr is None:
            self._arr = np.load(self.cache, mmap_mode="r")
        i = int(self.rows[int(j)])
        x: Float[np.ndarray, "d"] = np.array(self._arr[i], np.float32)
        y = (np.array(self._arr[i + 1], np.float32)
             if self.objective == "next" else x)
        return {"x": x, "y": y}

    def __getstate__(self) -> dict:
        return {"cache": self.cache, "rows": self.rows,
                "objective": self.objective, "_arr": None}
