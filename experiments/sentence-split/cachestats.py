"""What the sentence cache (`mc4_sent16`) looks like, against the document
cache it was split from (`mc4_4M`).

Stages (outputs under --out):

  scan    one pass over the embeddings (row norms, rows with a non-finite
          entry, per-coordinate sums of x and x^2), read in chunks that
          are then dropped from the page cache, and one in parallel over
          the sentences file (each document's row count and config, each
          row's characters and a hash of its config and text); then the
          texts of the --top most repeated hashes
          -> scan.npz, repeats.csv
  report  from scan.npz: norms, repeated texts overall and per config. On
          --rows uniform rows of each cache: the mean cosine between rows,
          the covariance spectrum of unit rows and of each cache's own
          whitened rows, the language's share of variance (raw, unit,
          whitened), a shrunk LDA's held-out language accuracy, and the
          sentence kinds' share. On --docs uniform documents: the mean of a
          document's sentence rows against its document row (cosine, and
          whether it ranks first among the sample's documents, all or of
          its language), split by whether the document cache tagged the
          config with its own code; and the cosine of sentence pairs
          (adjacent, one document, one language, two languages). The MSE
          weights: their spread, the whitened loss scale, the same without
          the largest weight, and the near-constant coordinates behind it;
          and the hyperplane the encoder's final LayerNorm puts every row
          on (`plane`), against the norm's weights
          -> summary.json

  uv run python experiments/sentence-split/cachestats.py scan
  uv run python experiments/sentence-split/cachestats.py report

Each cache is described in its own representation: the sentence cache's
rows are SONAR's masked mean, the document cache's are unit vectors, and
"unit" quantities put the sentence rows on the unit sphere too.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from jaxtyping import Float, Int, UInt
from scipy.stats import spearmanr
from tqdm import tqdm

from ontologize.data.langs import MC4_4M_TAGS, MC4_TO_SONAR

LANGS = sorted(MC4_TO_SONAR)
CONFIG = {lang: i for i, lang in enumerate(LANGS)}
KINDS = ["sentence", "line", "piece"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["scan", "report"])
    p.add_argument("--name", default="mc4_sent16", help="the sentence cache")
    p.add_argument("--docs-cache", default="mc4_4M",
                   help="the document cache it was split from")
    p.add_argument("--dir", default="data/sonar_embeddings")
    p.add_argument("--docs-weights", default="data/out/sonar/mse_weights.npy",
                   help="the document cache's MSE weights")
    p.add_argument("--out", default="data/out/sonar/sentcache")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--chunk", type=int, default=65536,
                   help="rows per read in the embedding pass")
    p.add_argument("--top", type=int, default=40,
                   help="most repeated texts written to repeats.csv")
    p.add_argument("--rows", type=int, default=262144,
                   help="uniform rows of each cache for the geometry")
    p.add_argument("--docs", type=int, default=20000,
                   help="uniform documents for the same-document and pair checks")
    p.add_argument("--pairs", type=int, default=400000,
                   help="draws for each sampled kind of sentence pair")
    p.add_argument("--coords", type=int, default=2,
                   help="near-constant coordinates (largest |mean| / sd) to describe")
    p.add_argument("--shrink", type=float, default=1e-3,
                   help="LDA's ridge, as a multiple of the mean within-language variance")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------- stage: scan ----------

def chunks(path: Path, parts: int) -> List[Tuple[int, int]]:
    """Up to `parts` nonempty byte ranges covering `path`, each starting at
    a line start."""
    size = path.stat().st_size
    cuts = [0]
    with open(path, "rb") as f:
        for i in range(1, parts):
            f.seek(max(i * size // parts, cuts[-1]))
            f.readline()
            cuts.append(min(f.tell(), size))
    cuts.append(size)
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def text_hash(lang: str, text: str) -> int:
    """A 64-bit hash of a row's config and text that is the same in every
    process (Python's `hash` of a str is salted per process)."""
    return int.from_bytes(hashlib.blake2b(f"{lang}\t{text}".encode(), digest_size=8).digest(),
                          "little")


def scan_part(job: Tuple[str, int, int, frozenset]
              ) -> Tuple[Int[np.ndarray, "docs 3"], UInt[np.ndarray, "rows"],
                         UInt[np.ndarray, "rows"], Dict[int, Tuple[str, str]]]:
    """One byte range of a sentences file: each document's index, row count
    and config index, each row's characters and `text_hash`, and the first
    config and text found for each hash in `want`."""
    path, a, b, want = job
    docs, chars, hashes, found = [], [], [], {}
    with open(path, "rb") as f:
        f.seek(a)
        while f.tell() < b:
            r = json.loads(f.readline())
            lang = r["lang"]
            docs.append((r["doc"], len(r["sents"]), CONFIG[lang]))
            for s in r["sents"]:
                h = text_hash(lang, s[3])
                chars.append(len(s[3]))
                hashes.append(h)
                if h in want and h not in found:
                    found[h] = (lang, s[3])
    return (np.array(docs, np.int64).reshape(-1, 3), np.array(chars, np.uint16),
            np.array(hashes, np.uint64), found)


def scan_texts(path: Path, workers: int, want: frozenset = frozenset()
               ) -> Tuple[Int[np.ndarray, "docs 3"], UInt[np.ndarray, "rows"],
                          UInt[np.ndarray, "rows"], Dict[int, Tuple[str, str]]]:
    """`scan_part` over the whole file, results in file order."""
    jobs = [(str(path), a, b, want) for a, b in chunks(path, 4 * workers)]
    with Pool(workers) as pool:
        res = pool.map(scan_part, jobs)
    return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]),
            np.concatenate([r[2] for r in res]), {k: v for r in res for k, v in r[3].items()})


def scan_rows(path: Path, chunk: int) -> Dict[str, np.ndarray]:
    """Row norms, the rows with a non-finite entry, and per-coordinate sums
    of x and x^2 (float64) over a float32 .npy. Reads `chunk` rows at a time
    with pread and drops each from the page cache after, so a pass over a
    cache larger than memory does not evict other files. A non-finite row
    counts as zeros in the sums."""
    X = np.load(path, mmap_mode="r")
    if X.dtype != np.float32 or X.ndim != 2 or not X.flags.c_contiguous:
        raise ValueError(f"{path}: expected a C-ordered 2-d float32 array")
    n, d = X.shape
    off = X.offset
    del X
    buf = np.empty((chunk, d), np.float32)
    norms = np.empty(n, np.float32)
    bad, s1, s2 = [], np.zeros(d), np.zeros(d)
    fd = os.open(path, os.O_RDONLY)
    try:
        for s in tqdm(range(0, n, chunk), desc="rows", unit="chunk"):
            m = min(chunk, n - s)
            v = buf[:m]
            at, nb = off + s * d * 4, m * d * 4
            if os.preadv(fd, [memoryview(v).cast("B")], at) != nb:
                raise IOError(f"{path}: short read at row {s}")
            os.posix_fadvise(fd, at, nb, os.POSIX_FADV_DONTNEED)
            ok = np.isfinite(v).all(1)
            if not ok.all():
                bad.extend((s + np.flatnonzero(~ok)).tolist())
                v = np.where(ok[:, None], v, np.float32(0))
            norms[s:s + m] = np.sqrt(np.einsum("ij,ij->i", v, v))
            s1 += v.sum(0, dtype=np.float64)
            s2 += np.einsum("ij,ij->j", v, v, dtype=np.float64)
    finally:
        os.close(fd)
    return {"norms": norms, "nonfinite": np.array(bad, np.int64), "s1": s1, "s2": s2,
            "n": np.int64(n)}


def repeats(h: UInt[np.ndarray, "n"], g: Int[np.ndarray, "n"], groups: int,
            at: Tuple[int, ...] = (2, 10, 100, 1000)
            ) -> Tuple[Dict[str, float], Float[np.ndarray, "groups"],
                       UInt[np.ndarray, "u"], Int[np.ndarray, "u"]]:
    """How repetitive the rows are, from a hash of each row's text and
    group: the distinct hashes, the share of rows that copy an earlier row,
    and for each c in `at` the share of rows whose hash occurs at least c
    times; per group, the share of its rows that copy an earlier row (a
    hash counts toward its first row's group, exact when the hash includes
    the group; 0 for a group without rows); and the distinct hashes with
    their counts."""
    u, first, inv, cnt = np.unique(h, return_index=True, return_inverse=True,
                                   return_counts=True)
    rc = cnt[inv.reshape(-1)]
    stats = {"rows": int(len(h)), "distinct": int(len(u)),
             "copies": float(1 - len(u) / len(h))}
    for c in at:
        stats[f"rows_in_texts_seen_{c}x"] = float((rc >= c).mean())
    rows = np.bincount(g, minlength=groups)
    distinct = np.bincount(g[first], minlength=groups)
    return stats, np.where(rows > 0, 1 - distinct / np.maximum(rows, 1), 0.0), u, cnt


def scan(cfg: argparse.Namespace) -> None:
    d, out = Path(cfg.dir), Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((d / f"{cfg.name}.meta.json").read_text())
    sentences = d / f"{cfg.name}.sentences.jsonl"
    docs, chars, hashes, _ = scan_texts(sentences, cfg.workers)
    starts = np.load(d / f"{cfg.name}.docstart.npy")
    if (len(docs) != meta["n_docs"] or not (docs[:, 0] == np.arange(len(docs))).all()
            or len(chars) != meta["n"]
            or not (np.diff(np.append(starts, meta["n"])) == docs[:, 1]).all()):
        sys.exit(f"{sentences} does not match {cfg.name}'s documents and rows")
    print(f"{len(docs)} documents, {len(chars)} rows")
    rows = scan_rows(d / f"{cfg.name}.npy", cfg.chunk)
    if int(rows["n"]) != meta["n"]:
        sys.exit(f"{cfg.name}.npy has {int(rows['n'])} rows, the metadata {meta['n']}")
    config = np.repeat(docs[:, 2], docs[:, 1])
    _, _, u, cnt = repeats(hashes, config, len(LANGS))
    top = np.argsort(cnt, kind="stable")[::-1][:cfg.top]
    _, _, _, found = scan_texts(sentences, cfg.workers, frozenset(int(x) for x in u[top]))
    with open(out / "repeats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["count", "config", "characters", "text"])
        for i in top:
            lang, text = found[int(u[i])]
            w.writerow([int(cnt[i]), lang, len(text), text])
    np.savez(out / "scan.npz", chars=chars, hashes=hashes, counts=docs[:, 1].astype(np.int32),
             config=docs[:, 2].astype(np.int8), **rows)
    print(f"nonfinite rows {len(rows['nonfinite'])}, zero rows {int((rows['norms'] == 0).sum())}"
          f" -> {out / 'scan.npz'}, {out / 'repeats.csv'}")


# ---------- stage: report ----------

def unit(X: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n d"]:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def anisotropy(U: Float[np.ndarray, "n d"]) -> float:
    """Mean cosine between distinct rows of the unit rows U."""
    n = len(U)
    t = U.sum(0)
    return float((t @ t - n) / (n * (n - 1)))


def spectrum(X: Float[np.ndarray, "n d"], w: Optional[Float[np.ndarray, "d"]] = None
             ) -> Dict[str, float]:
    """The covariance spectrum of X, each coordinate scaled by sqrt(w) when
    given: its participation ratio (sum lambda)^2 / sum lambda^2, the
    variance shares of the top 1, 10 and 100 components, and how many
    components reach 50% and 90% of the variance."""
    Xc = X - X.mean(0)
    if w is not None:
        Xc = Xc * np.sqrt(w)
    lam = np.clip(np.linalg.eigvalsh(Xc.T @ Xc / len(X))[::-1], 0, None)
    c = np.cumsum(lam) / lam.sum()
    return {"participation": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "top1": float(c[0]), "top10": float(c[min(9, len(c) - 1)]),
            "top100": float(c[min(99, len(c) - 1)]),
            "k50": int(np.searchsorted(c, 0.5) + 1), "k90": int(np.searchsorted(c, 0.9) + 1)}


def eta2(X: Float[np.ndarray, "n d"], g: Int[np.ndarray, "n"],
         w: Float[np.ndarray, "d"]) -> float:
    """Between-group share of (w-weighted) variance, as retag.py's."""
    Xc = (X - X.mean(0)) * np.sqrt(w)
    n = int(g.max()) + 1
    cnt = np.bincount(g, minlength=n)
    M = np.zeros((n, X.shape[1]))
    np.add.at(M, g, Xc)
    M /= np.maximum(cnt[:, None], 1)
    return float(((cnt / len(g))[:, None] * M ** 2).sum() / (Xc ** 2).sum(1).mean())


def acc_lda(X: Float[np.ndarray, "n d"], g: Int[np.ndarray, "n"], shrink: float,
            rng: np.random.Generator) -> float:
    """Held-out accuracy of linear discriminant analysis: class means and the
    pooled within-class covariance (plus `shrink` times its mean diagonal)
    from a random half of the rows, scored on the other half."""
    train = rng.random(len(X)) < 0.5
    n = int(g.max()) + 1
    Xt, gt = X[train], g[train]
    cnt = np.bincount(gt, minlength=n).astype(np.float64)
    M = np.zeros((n, X.shape[1]))
    np.add.at(M, gt, Xt)
    M /= np.maximum(cnt, 1)[:, None]
    R = Xt - M[gt]
    Sw = R.T @ R / len(Xt)
    Sw += shrink * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    A = np.linalg.solve(Sw, M.T)
    prior = np.where(cnt > 0, np.log(np.maximum(cnt, 1) / cnt.sum()), -np.inf)
    score = X[~train] @ A - 0.5 * np.einsum("gd,dg->g", M, A) + prior
    return float((score.argmax(1) == g[~train]).mean())


def rank_self(A: Float[np.ndarray, "n d"], B: Float[np.ndarray, "n d"],
              g: Optional[Int[np.ndarray, "n"]] = None, block: int = 2048) -> Int[np.ndarray, "n"]:
    """For each row i, how many rows of B are closer to A[i] by inner product
    than B[i] is (0: B[i] is the nearest), counting only rows of B in i's
    group when g is given."""
    A32, B32 = A.astype(np.float32), B.astype(np.float32)
    out = np.empty(len(A), np.int64)
    for s in range(0, len(A), block):
        sim = A32[s:s + block] @ B32.T
        own = sim[np.arange(len(sim)), np.arange(s, s + len(sim))]
        above = sim > own[:, None]
        if g is not None:
            above &= g[None, :] == g[s:s + block, None]
        out[s:s + block] = above.sum(1)
    return out


def pairs(doc: Int[np.ndarray, "n"], g: Int[np.ndarray, "n"], m: int,
          rng: np.random.Generator) -> Dict[str, Tuple[Int[np.ndarray, "p"], Int[np.ndarray, "p"]]]:
    """Row pairs from whole documents whose rows are contiguous (doc
    nondecreasing): every adjacent pair of one document; and from m draws
    each, rows of one document at least two apart, rows of two documents
    in one group, and rows in two groups."""
    if (np.diff(doc) < 0).any():
        raise ValueError("doc must be nondecreasing")
    n = len(doc)
    adj = np.flatnonzero(doc[1:] == doc[:-1])
    lo, hi = np.searchsorted(doc, doc, "left"), np.searchsorted(doc, doc, "right")
    i = rng.integers(0, n, m)
    j = lo[i] + (rng.random(m) * (hi[i] - lo[i])).astype(np.int64)
    apart = np.abs(j - i) >= 2
    order = np.argsort(g, kind="stable")
    gs = g[order]
    a = rng.integers(0, n, m)
    glo, ghi = np.searchsorted(gs, g[a], "left"), np.searchsorted(gs, g[a], "right")
    b = order[glo + (rng.random(m) * (ghi - glo)).astype(np.int64)]
    other_doc = doc[a] != doc[b]
    c, e = rng.integers(0, n, m), rng.integers(0, n, m)
    other_g = g[c] != g[e]
    return {"adjacent": (adj, adj + 1), "one_doc_apart": (i[apart], j[apart]),
            "one_lang_two_docs": (a[other_doc], b[other_doc]), "two_langs": (c[other_g], e[other_g])}


def group_means(X: Float[np.ndarray, "n d"], g: Int[np.ndarray, "n"]) -> Float[np.ndarray, "n d"]:
    """Each row's group mean."""
    n = int(g.max()) + 1
    M = np.zeros((n, X.shape[1]))
    np.add.at(M, g, X)
    M /= np.maximum(np.bincount(g, minlength=n), 1)[:, None]
    return M[g]


def weight_spread(w: Float[np.ndarray, "d"], var: Float[np.ndarray, "d"]) -> Dict[str, float]:
    """MSE weights (mean 1) as the losses use them: largest, median, max/min,
    the share of their sum on the largest (the share of an isotropic
    direction's expected whitened squared norm on that coordinate), and the
    raw and whitened variance per row, sum_d var_d and sum_d w_d var_d; the
    second sets the whitened MSE's scale."""
    return {"argmax": int(w.argmax()), "max": float(w.max()), "median": float(np.median(w)),
            "max_over_min": float(w.max() / w.min()), "share_of_sum_on_max": float(w.max() / w.sum()),
            "raw_var_per_row": float(var.sum()), "whitened_var_per_row": float((w * var).sum())}


def final_norm(model_id: str, revision: str
               ) -> Tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d"]]:
    """The gain and bias of the encoder's final LayerNorm, the last
    operation before the masked mean."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    t = load_file(hf_hub_download(model_id, "model.safetensors", revision=revision))
    return t["layer_norm.weight"].astype(np.float64), t["layer_norm.bias"].astype(np.float64)


def plane(X: Float[np.ndarray, "n d"], gamma: Float[np.ndarray, "d"],
          beta: Float[np.ndarray, "d"]) -> Dict:
    """Where a final LayerNorm puts the masked means of its outputs (rows
    without an L2 step). The normalized vector sums to zero over
    coordinates, so every token, and so every mean of tokens, satisfies
    sum_d (x_d - beta_d) / gamma_d = 0: a hyperplane whose normal is
    proportional to 1 / gamma. Reports, for the coordinate of smallest
    |gamma|: how well the rows satisfy the constraint (median |sum| over
    the sum of |terms|), the normal's share of squared length on that
    coordinate, the covariance's two smallest eigenvalues against its
    median, the smallest eigenvector's |cosine| with the normal and its
    share on the coordinate, and the largest error of the coordinate
    solved from the constraint and the other coordinates, against its
    standard deviation."""
    T = (X - beta) / gamma
    c = int(np.argmin(np.abs(gamma)))
    normal = (1 / gamma) / np.linalg.norm(1 / gamma)
    Xc = X - X.mean(0)
    lam, V = np.linalg.eigh(Xc.T @ Xc / len(X))
    solved = beta[c] - gamma[c] * (T.sum(1) - T[:, c])
    return {"coord": c, "gamma": float(gamma[c]), "median_abs_gamma": float(np.median(np.abs(gamma))),
            "residual_relative_median": float(np.median(np.abs(T.sum(1)) / np.abs(T).sum(1))),
            "normal_share_on_coord": float(normal[c] ** 2),
            "eig_smallest": lam[:2].tolist(), "eig_median": float(np.median(lam)),
            "eigvec_cos_normal": float(abs(V[:, 0] @ normal)),
            "eigvec_share_on_coord": float(V[c, 0] ** 2),
            "solved_max_abs_error": float(np.abs(solved - X[:, c]).max()),
            "coord_sd": float(X[:, c].std())}


def geometry(X: Float[np.ndarray, "n d"], g: Int[np.ndarray, "n"], w: Float[np.ndarray, "d"],
             shrink: float, rng: np.random.Generator) -> Dict:
    U = unit(X)
    one = np.ones(X.shape[1])
    return {"mean_cosine": anisotropy(U), "spectrum_unit": spectrum(U),
            "spectrum_whitened": spectrum(X, w),
            "lang_eta2_raw": eta2(X, g, one), "lang_eta2_unit": eta2(U, g, one),
            "lang_eta2_whitened": eta2(X, g, w), "lang_acc_lda_unit": acc_lda(U, g, shrink, rng)}


def report(cfg: argparse.Namespace) -> None:
    d, out = Path(cfg.dir), Path(cfg.out)
    z = np.load(out / "scan.npz")
    n, norms, chars = int(z["n"]), z["norms"], z["chars"]
    counts, config_doc = z["counts"].astype(np.int64), z["config"].astype(np.int64)
    config = np.repeat(config_doc, counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    S = np.load(d / f"{cfg.name}.npy", mmap_mode="r")
    D = np.load(d / f"{cfg.docs_cache}.npy", mmap_mode="r")
    ws = np.load(d / f"{cfg.name}.mse_weights.npy").astype(np.float64)
    wd = np.load(cfg.docs_weights).astype(np.float64)
    K = np.asarray(np.load(d / f"{cfg.name}.kind.npy", mmap_mode="r"))
    kind = np.full(n, -1, np.int64)
    for i, k in enumerate(KINDS):
        kind[K == k] = i
    del K
    if (kind < 0).any():
        sys.exit(f"{cfg.name}.kind.npy has kinds outside {KINDS}")
    rng = np.random.default_rng(cfg.seed)
    s: Dict = {}

    s["rows"] = {"n": n, "docs": int(len(counts)), "per_doc": n / len(counts),
                 "docs_without_rows": float((counts == 0).mean()),
                 "nonfinite": int(len(z["nonfinite"])), "zero": int((norms == 0).sum())}
    stats, copies, _, _ = repeats(z["hashes"], config, len(LANGS))
    worst = np.argsort(copies)[::-1][:8]
    s["repeats"] = {**stats, "copies_by_config_highest": {LANGS[i]: float(copies[i]) for i in worst}}
    ri = np.sort(rng.choice(n, cfg.rows, replace=False))
    med = np.array([np.median(norms[config == i]) for i in range(len(LANGS))])
    o = np.argsort(med)
    s["norms"] = {"quantiles": {str(q): float(np.quantile(norms, q)) for q in [0, 0.01, 0.5, 0.99, 1]},
                  "by_kind_q10_q50_q90": {k: np.quantile(norms[kind == i], [0.1, 0.5, 0.9]).tolist()
                                          for i, k in enumerate(KINDS)},
                  "spearman_chars": float(spearmanr(norms[ri], chars[ri])[0]),
                  "median_lowest_configs": {LANGS[i]: float(med[i]) for i in o[:5]},
                  "median_highest_configs": {LANGS[i]: float(med[i]) for i in o[-5:]}}
    s["chars"] = {"quantiles": {str(q): float(np.quantile(chars, q)) for q in [0.1, 0.5, 0.9]},
                  "median_by_kind": {k: float(np.median(chars[kind == i])) for i, k in enumerate(KINDS)},
                  "share_by_kind": {k: float((kind == i).mean()) for i, k in enumerate(KINDS)}}

    # uniform rows of both caches
    Xs = np.asarray(S[ri], np.float64)
    gs, ks = config[ri], kind[ri]
    di = np.sort(rng.choice(len(D), cfg.rows, replace=False))
    Xd = np.asarray(D[di], np.float64)
    DL = np.load(d / f"{cfg.docs_cache}.langs.npy", mmap_mode="r")
    gd = np.array([CONFIG[str(x)] for x in DL[di]])
    s[cfg.name] = geometry(Xs, gs, ws, cfg.shrink, rng)
    s[cfg.docs_cache] = geometry(Xd, gd, wd, cfg.shrink, rng)
    s[cfg.name]["kind_eta2_whitened"] = eta2(Xs, ks, ws)
    for i, k in enumerate(KINDS):
        s[cfg.name][f"lang_eta2_whitened_{k}"] = eta2(Xs[ks == i], gs[ks == i], ws)

    # the MSE weights and the near-constant coordinates behind them
    mean = z["s1"] / n
    var = z["s2"] / n - mean ** 2
    vd = Xd.var(0)
    keep = np.arange(len(ws)) != ws.argmax()
    w2 = (1 / var[keep]) / (1 / var[keep]).mean()
    s["weights"] = {cfg.name: weight_spread(ws, var), cfg.docs_cache: weight_spread(wd, vd),
                    f"{cfg.name}_without_max": weight_spread(w2, var[keep]),
                    "log_weights_pearson": float(np.corrcoef(np.log(ws), np.log(wd))[0, 1]),
                    "weights_match_inverse_variance": float(np.corrcoef(ws, 1 / var)[0, 1])}
    meta = json.loads((d / f"{cfg.name}.meta.json").read_text())
    gamma, beta = final_norm(meta["encoder"], meta["encode_versions"]["encoder_revision"])
    s["plane"] = plane(Xs, gamma, beta)
    Us = unit(Xs)
    s["coords"] = {}
    for c in np.argsort(-np.abs(mean) / np.sqrt(var))[:cfg.coords]:
        x, ul = Xs[:, c], Us[:, c]
        s["coords"][str(int(c))] = {
            "mean": float(mean[c]), "sd": float(np.sqrt(var[c])),
            "abs_mean_over_sd": float(abs(mean[c]) / np.sqrt(var[c])),
            "weight": float(ws[c]), "docs_cache_weight": float(wd[c]),
            "norm_gamma": float(gamma[c]), "norm_beta": float(beta[c]),
            "lang_eta2": eta2(x[:, None], gs, np.ones(1)),
            "spearman_chars": float(spearmanr(x, chars[ri])[0]),
            "spearman_norm": float(spearmanr(x, norms[ri])[0]),
            "unit_sd_over_abs_mean": float(ul.std() / abs(ul.mean())),
            "unit_spearman_inv_norm": float(spearmanr(ul, 1 / norms[ri])[0]),
            "docs_cache_mean": float(Xd[:, c].mean()), "docs_cache_sd": float(Xd[:, c].std())}
    del Xs, Us, Xd

    # the same documents in both caches
    docs = np.sort(rng.choice(len(counts), cfg.docs, replace=False))
    kept = docs[counts[docs] > 0]
    rows = [np.asarray(S[starts[i]:starts[i] + counts[i]], np.float64) for i in kept]
    Ms = unit(np.stack([r.mean(0) for r in rows]))
    Md = np.asarray(D[kept], np.float64)
    gdoc = config_doc[kept]
    tagged = np.array([MC4_4M_TAGS[LANGS[i]] == MC4_TO_SONAR[LANGS[i]] for i in gdoc])
    A, B = unit(Ms - Ms.mean(0)), unit(Md - Md.mean(0))
    cos = (A * B).sum(1)
    r_all, r_lang = rank_self(A, B), rank_self(A, B, gdoc)
    s["same_docs"] = {"docs": int(len(kept)), "drawn": int(len(docs)), "own_tag_docs": int(tagged.sum())}
    for name, sel in [("own_tag", tagged), ("english_tag", ~tagged)]:
        s["same_docs"][name] = {"cos_centered_median": float(np.median(cos[sel])),
                                "first_among_all": float((r_all[sel] == 0).mean()),
                                "first_among_lang": float((r_lang[sel] == 0).mean())}

    # sentence pairs within and across those documents
    U = unit(np.concatenate(rows))
    doc_of = np.repeat(np.arange(len(kept)), counts[kept])
    lang_of = np.repeat(gdoc, counts[kept])
    Uc = unit(U - U.mean(0))
    Ul = unit(U - group_means(U, lang_of))
    s["pairs"] = {k: {"n": int(len(i)), "cos_centered": float((Uc[i] * Uc[j]).sum(1).mean()),
                      "cos_lang_centered": float((Ul[i] * Ul[j]).sum(1).mean())}
                  for k, (i, j) in pairs(doc_of, lang_of, cfg.pairs, rng).items()}

    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    print(f"-> {out / 'summary.json'}")


def main() -> None:
    cfg = parse_args()
    {"scan": scan, "report": report}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
