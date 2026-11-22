"""Do k-cell partitions of the data reproduce across seeds at all?

A head of the stack's layer 0 partitions the data itself, and two seeds'
best-matched layer-0 heads agree at a partition NMI of about 0.025 against
a null of 0.005 (experiments/recipe-ladder/notes.md). This asks what a
reproducible partition of the same rows looks like: k-means (k-means++
seeding, Lloyd's iterations to a fixed point) in the loss's whitened frame,
fit on training rows at several seeds and assigned on the held-out tail.
Between every pair of seeds it reports the tail partitions' NMI
(`partition.py`'s `nmi_matrix`, so the numbers sit beside the heads'),
beside the null with one partition's rows shuffled, and each seed's FVU on
the tail. For every data set whose tail rows stand for the cache's (all
but `--gauss`), it also gives each seed's NMI between its cells and the
rows' languages (the `.langs.npy` sidecar).

`--gauss` runs the same on the Gaussian copy (`gausscopy.py`), whose
training rows are the training stream and whose tail is the held-out file:
the agreement a partition reaches when nothing beyond the covariance is
there to agree on. `--mixture` runs it on a copy that keeps each
language's mean and covariance too (`gausscopy.py mixture`), each row drawn
with the real row's language: if it reproduces as SONAR does, language is
the structure that makes SONAR's partitions reproducible.
`--within-language` runs it on the cache's own rows less each row's
language mean (from the same mixture.npz), which removes the part of that
structure the means carry. `--marginals pca ica` runs it on copies of the
cache's own rows that keep each coordinate's distribution in a frame and
break every tie between coordinates (`marginal_copy`): "pca" keeps each
whitened principal direction's marginal, its skew, tails and modes; "ica"
keeps each FastICA source's, so the copy is an independent-components model
of the rows. Between the Gaussian copy and the cache they say whether the
reproducible cells come from a few oddly shaped directions or from how
directions combine; `coordinates_<frame>.csv` gives each direction's
variance share, skew, excess kurtosis and bimodality coefficient. `--ckpt`
adds, per layer of a stack, each head's best
NMI against the k-means partitions at the model's k: whether the heads are
near the data's own k-means cells.

    JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py \\
        --gauss data/out/sonar/gausscopy --ckpt <run>
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
sys.path.insert(0, str(HERE))

# the data sets whose tail row i is drawn for the cache's tail row i, so the
# row's language applies to it
ROW_FOR_ROW = {"cache", "mixture", "within"}


def kmeans_pp(X: Float[np.ndarray, "n d"], k: int,
              rng: np.random.Generator) -> Float[np.ndarray, "k d"]:
    """k-means++ seeding: each centre drawn with probability proportional
    to the squared distance to the nearest centre so far."""
    n = len(X)
    C = np.empty((k, X.shape[1]), X.dtype)
    C[0] = X[rng.integers(n)]
    d2 = ((X - C[0]) ** 2).sum(1)
    for j in range(1, k):
        p = d2 / d2.sum()
        C[j] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, ((X - C[j]) ** 2).sum(1))
    return C


def assign(X: Float[np.ndarray, "n d"], C: Float[np.ndarray, "k d"],
           chunk: int = 65536) -> Int[np.ndarray, "n"]:
    """Nearest centre of each row, by ||c||² − 2 x·c."""
    cc = (C.astype(np.float64) ** 2).sum(1)
    out = np.empty(len(X), np.int64)
    for i in range(0, len(X), chunk):
        D = cc[None, :] - 2.0 * (X[i:i + chunk] @ C.T).astype(np.float64)
        out[i:i + chunk] = D.argmin(1)
    return out


def kmeans(X: Float[np.ndarray, "n d"], k: int, seed: int,
           iters: int = 300) -> Float[np.ndarray, "k d"]:
    """Lloyd's iterations from k-means++ until no row changes cell (or
    `iters`). An emptied cell is reseeded at the row farthest from its
    centre, so every run ends with k cells."""
    rng = np.random.default_rng(seed)
    C = kmeans_pp(X, k, rng)
    lab = None
    for _ in range(iters):
        new = assign(X, C)
        if lab is not None and np.array_equal(new, lab):
            break
        lab = new
        cnt = np.bincount(lab, minlength=k)
        H = np.zeros((len(X), k), X.dtype)
        H[np.arange(len(X)), lab] = 1
        S = (H.T @ X).astype(np.float64)
        live = cnt > 0
        C[live] = (S[live] / cnt[live, None]).astype(X.dtype)
        for j in np.flatnonzero(~live):
            far = ((X - C[lab]) ** 2).sum(1).argmax()
            C[j] = X[far]
            lab[far] = j
    return C


def fvu(X: Float[np.ndarray, "n d"], C: Float[np.ndarray, "k d"],
        lab: Int[np.ndarray, "n"]) -> float:
    """Squared error to the assigned centre over the rows' total variance."""
    X64 = X.astype(np.float64)
    err = ((X64 - C[lab].astype(np.float64)) ** 2).sum()
    return float(err / ((X64 - X64.mean(0)) ** 2).sum())


def principal(X: Float[np.ndarray, "n d"], rel_tol: float = 1e-9
              ) -> tuple[Float[np.ndarray, "d"], Float[np.ndarray, "p"],
                         Float[np.ndarray, "d p"]]:
    """(mean, eigenvalues in descending order, eigenvectors) of the rows'
    covariance, keeping the directions whose variance exceeds rel_tol of
    the largest."""
    mu = X.astype(np.float64).mean(0)
    Xc = X.astype(np.float64) - mu
    lam, V = np.linalg.eigh(Xc.T @ Xc / len(X))
    lam, V = lam[::-1], V[:, ::-1]
    keep = lam > rel_tol * lam[0]
    return mu, lam[keep], V[:, keep]


def symmetric_orth(W: Float[np.ndarray, "p p"]) -> Float[np.ndarray, "p p"]:
    """(W Wᵀ)^(-1/2) W: the orthogonal matrix nearest W."""
    s, U = np.linalg.eigh(W.astype(np.float64) @ W.T.astype(np.float64))
    return (U / np.sqrt(np.clip(s, 1e-12, None))) @ U.T @ W.astype(np.float64)


def fastica(Z: Float[np.ndarray, "n p"], seed: int, iters: int = 200,
            tol: float = 1e-4) -> tuple[Float[np.ndarray, "p p"], int, float]:
    """Symmetric FastICA (Hyvärinen 1999) with the log-cosh contrast on
    sphered rows (zero mean, identity covariance): the orthogonal W whose
    rows' projections Z Wᵀ are as far from Gaussian as the contrast finds.
    Returns W, the iterations run, and the last step's largest change,
    max |1 − |w_new · w||| over rows (below tol: converged)."""
    rng = np.random.default_rng(seed)
    p = Z.shape[1]
    W = symmetric_orth(rng.standard_normal((p, p)))
    Z32 = Z.astype(np.float32)
    lim = np.inf
    for it in range(1, iters + 1):
        G = np.tanh(Z32 @ W.T.astype(np.float32))
        gp = (1.0 - G.astype(np.float64) ** 2).mean(0)
        W_new = symmetric_orth((G.T @ Z32).astype(np.float64) / len(Z)
                               - gp[:, None] * W)
        lim = float(np.abs(np.abs((W_new * W).sum(1)) - 1.0).max())
        W = W_new
        if lim < tol:
            break
    return W, it, lim


def shuffle_columns(Y: Float[np.ndarray, "n p"], seed: int
                    ) -> Float[np.ndarray, "n p"]:
    """Each column permuted on its own: every column keeps its values,
    and nothing ties one column's value in a row to another's."""
    rng = np.random.default_rng(seed)
    return np.stack([Y[rng.permutation(len(Y)), j] for j in range(Y.shape[1])], 1)


def shape_stats(Y: Float[np.ndarray, "n p"]) -> dict:
    """Per column: skewness, excess kurtosis, and Sarle's bimodality
    coefficient (skew² + 1) / (kurtosis + 3(n−1)²/((n−2)(n−3))), which is
    1/3 for a Gaussian, 5/9 for a uniform, and above 5/9 suggests two
    modes unless the column is strongly skewed (an exponential reads 5/9)."""
    n = len(Y)
    Yc = Y.astype(np.float64) - Y.mean(0)
    sd = Yc.std(0)
    skew = (Yc ** 3).mean(0) / sd ** 3
    kurt = (Yc ** 4).mean(0) / sd ** 4 - 3.0
    bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return {"skew": skew, "kurtosis": kurt, "bimodality": bc}


def marginal_copy(train: Float[np.ndarray, "n d"], tail: Float[np.ndarray, "m d"],
                  frame: str, seed: int = 0, ica_iters: int = 200
                  ) -> tuple[Float[np.ndarray, "n d"], Float[np.ndarray, "m d"], dict]:
    """Copies of the training and tail rows that keep each coordinate's
    distribution in a frame fitted on the training rows and nothing that
    ties coordinates together: every coordinate's column is permuted on its
    own, in the training rows and the tail separately. `frame` "pca" is the
    rows' principal directions, so the copy keeps the covariance and every
    principal direction's marginal (its skew, tails and modes); "ica" is
    FastICA's sources in the sphered principal frame, so the copy is
    exactly an independent-components model with the rows' own source
    marginals. Directions with no variance are left out of the copy. The
    dict has, per coordinate of the frame, its share of the variance and
    `shape_stats` on the training rows, and for "ica" how FastICA ended."""
    mu, lam, V = principal(train)
    if frame == "pca":
        M = V.T                                          # (p, d): coordinate -> row
        to = lambda X: (X.astype(np.float64) - mu) @ V
        info = {}
    elif frame == "ica":
        sphere = V / np.sqrt(lam)
        W, it, lim = fastica((train.astype(np.float64) - mu) @ sphere, seed,
                             ica_iters)
        M = W @ (V * np.sqrt(lam)).T                     # Y M + mu = X
        to = lambda X: (X.astype(np.float64) - mu) @ sphere @ W.T
        info = {"ica_iterations": it, "ica_last_change": lim}
    else:
        raise ValueError(f"unknown frame {frame!r}")
    Yt, Ye = to(train), to(tail)
    var = (M ** 2).sum(1) * Yt.var(0)
    stats = {"variance_share": var / var.sum(), **shape_stats(Yt), **info}
    back = lambda Y: (Y @ M + mu).astype(np.float32)
    return (back(shuffle_columns(Yt, 2 * seed)),
            back(shuffle_columns(Ye, 2 * seed + 1)), stats)


def seed_table(train: Float[np.ndarray, "n d"], tail: Float[np.ndarray, "m d"],
               k: int, seeds: list[int], perm_seed: int = 0) -> dict:
    """Every seed's tail partition and FVU, and every pair's NMI beside the
    rows-shuffled null."""
    from partition import nmi_matrix
    labs, fv = [], []
    for s in seeds:
        C = kmeans(train, k, s)
        lab = assign(tail, C)
        labs.append(lab)
        fv.append(fvu(tail, C, lab))
    rng = np.random.default_rng(perm_seed)
    pair, null = [], []
    for i, j in itertools.combinations(range(len(seeds)), 2):
        a, b = labs[i][None], labs[j][None]
        pair.append(float(nmi_matrix(a, k, b, k)[0, 0]))
        null.append(float(nmi_matrix(a, k, rng.permutation(labs[j])[None], k)[0, 0]))
    return {"k": k, "seeds": seeds, "fvu": fv, "nmi": pair, "null": null,
            "labels": labs}


def head_table(codes_: Int[np.ndarray, "heads m"], layer: Int[np.ndarray, "heads"],
               labs: list, k: int) -> dict:
    """Per layer: each head's best NMI against the k-means partitions."""
    from partition import nmi_matrix
    K = np.stack(labs)
    M = nmi_matrix(codes_, k, K, k)                        # (heads, seeds)
    best = M.max(1)
    return {int(l): {"median": float(np.median(best[layer == l])),
                     "max": float(best[layer == l].max())}
            for l in np.unique(layer)}


def rows(cfg) -> tuple[dict, dict]:
    """Training rows and tail rows, whitened, for each data set, with the
    tail in the cache's own frame for the heads (None for a copy made in
    the whitened frame); and each marginal copy's per-coordinate table."""
    sw = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    X = np.load(cfg.cache, mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(cfg.prefix, cfg.train_rows,
                                                  replace=False))
    out = {"cache": (np.asarray(X[idx], np.float32) * sw,
                     np.asarray(X[-cfg.tail_rows:], np.float32) * sw,
                     np.asarray(X[-cfg.tail_rows:], np.float32))}
    shapes = {}
    for frame in cfg.marginals:
        print(f"{frame} copy: fitting the frame", flush=True)
        T, E, shapes[frame] = marginal_copy(*out["cache"][:2], frame,
                                            ica_iters=cfg.ica_iters)
        out[frame] = (T, E, None)
    if cfg.gauss:
        from gausscopy import BLOCK, block
        g = Path(cfg.gauss)
        z = np.load(g / "moments.npz")
        mean, F = z["mean"].astype(np.float32), z["factor"].astype(np.float32)
        T = np.concatenate([block(mean, F, 0, j)
                            for j in range(-(-cfg.train_rows // BLOCK))])[:cfg.train_rows]
        E = np.asarray(np.load(g / "eval.npy", mmap_mode="r")[-cfg.tail_rows:],
                       np.float32)
        out["gauss"] = (T * sw, E * sw, E)
    if cfg.mixture:
        from gausscopy import mixture_rows
        z = np.load(cfg.mixture)
        L = np.load(Path(cfg.cache).with_suffix(".langs.npy"), mmap_mode="r")
        moments_ = (z["names"], z["means"], z["factors"])
        T = mixture_rows(np.asarray(L[idx]), *moments_, seed=0)
        E = mixture_rows(np.asarray(L[-cfg.tail_rows:]), *moments_, seed=1)
        out["mixture"] = (T * sw, E * sw, E)
    if cfg.within_language:
        z = np.load(cfg.within_language)
        names, means = z["names"], z["means"]
        L = np.load(Path(cfg.cache).with_suffix(".langs.npy"), mmap_mode="r")
        Mt = means[np.searchsorted(names, np.asarray(L[idx]))]
        Me = means[np.searchsorted(names, np.asarray(L[-cfg.tail_rows:]))]
        T = np.asarray(X[idx], np.float32) - Mt
        E = np.asarray(X[-cfg.tail_rows:], np.float32) - Me
        out["within"] = (T * sw, E * sw, E)
    return out, shapes


def write_shapes(path: Path, st: dict) -> dict:
    """A copy's per-coordinate table as CSV, and a summary of it."""
    cols = ("variance_share", "skew", "kurtosis", "bimodality")
    with open(path, "w") as f:
        f.write("coordinate," + ",".join(cols) + "\n")
        for i in range(len(st["skew"])):
            f.write(f"{i}," + ",".join(f"{st[c][i]:.6g}" for c in cols) + "\n")
    kurt, bc = st["kurtosis"], st["bimodality"]
    top = np.argsort(-np.abs(kurt))[:5]
    return {"coordinates": int(len(kurt)),
            "abs_kurtosis_over_1": int((np.abs(kurt) > 1).sum()),
            "bimodality_over_5/9": int((bc > 5 / 9).sum()),
            "top_abs_kurtosis": [{"coordinate": int(i),
                                  "variance_share": float(st["variance_share"][i]),
                                  "skew": float(st["skew"][i]),
                                  "kurtosis": float(kurt[i]),
                                  "bimodality": float(bc[i])} for i in top],
            **{k: st[k] for k in ("ica_iterations", "ica_last_change") if k in st}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=str(ROOT / "data/sonar_embeddings/mc4_sent16.npy"))
    p.add_argument("--mse-weights",
                   default=str(ROOT / "data/out/sonar/sentcache/mse_weights_med837.npy"))
    p.add_argument("--prefix", type=int, default=12_800_000,
                   help="training rows are drawn from the cache's first rows")
    p.add_argument("--train-rows", type=int, default=131072)
    p.add_argument("--tail-rows", type=int, default=32768)
    p.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--gauss", default=None, help="gausscopy.py's output directory")
    p.add_argument("--mixture", default=None,
                   help="gausscopy.py mixture's mixture.npz: a copy that keeps "
                        "each language's mean and covariance")
    p.add_argument("--within-language", default=None,
                   help="mixture.npz again: the cache's own rows less their "
                        "language's mean")
    p.add_argument("--marginals", nargs="*", default=[], choices=["pca", "ica"],
                   help="copies of the cache's rows that keep each coordinate's "
                        "distribution in this frame and nothing that ties them")
    p.add_argument("--ica-iters", type=int, default=200)
    p.add_argument("--ckpt", default=None, help="a stack whose heads to compare")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--out", default=str(ROOT / "data/out/sonar/kmeans_seeds"))
    cfg = p.parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    data, shapes = rows(cfg)
    summary = {}
    for frame, st in shapes.items():
        summary[f"{frame}_coordinates"] = s = write_shapes(
            out / f"coordinates_{frame}.csv", st)
        print(f"{frame} frame: {s['abs_kurtosis_over_1']} of {s['coordinates']} "
              f"coordinates with |excess kurtosis| > 1, "
              f"{s['bimodality_over_5/9']} with bimodality > 5/9"
              + (f"; FastICA {s['ica_iterations']} iterations, last change "
                 f"{s['ica_last_change']:.2e}" if frame == "ica" else ""), flush=True)
    L = np.load(Path(cfg.cache).with_suffix(".langs.npy"), mmap_mode="r")
    _, lang = np.unique(np.asarray(L[-cfg.tail_rows:]), return_inverse=True)
    n_lang = int(lang.max()) + 1
    heads = None
    if cfg.ckpt:
        from partition import codes
        heads = codes(cfg.ckpt, cfg.step, cfg.temperature, data["cache"][2], 256)
    for name, (train, tail, _) in data.items():
        summary[name] = {}
        for k in cfg.ks:
            t = seed_table(train, tail, k, cfg.seeds)
            nm, nl = np.array(t["nmi"]), np.array(t["null"])
            print(f"{name:<6} k={k:<4} seed NMI mean {nm.mean():.4f} "
                  f"[{nm.min():.4f}, {nm.max():.4f}]  null {nl.mean():.4f}  "
                  f"FVU {np.mean(t['fvu']):.4f} [{min(t['fvu']):.4f}, "
                  f"{max(t['fvu']):.4f}]", flush=True)
            row = {key: v for key, v in t.items() if key != "labels"}
            if name in ROW_FOR_ROW:   # rows that stand for the tail's own rows
                from partition import nmi_matrix
                row["language_nmi"] = [
                    float(nmi_matrix(lab[None], k, lang[None], n_lang)[0, 0])
                    for lab in t["labels"]]
                print(f"         cells' NMI with the rows' language: mean "
                      f"{np.mean(row['language_nmi']):.4f}")
            if heads is not None and name == "cache" and heads[2] == k:
                A, layer, _, used = heads
                row["heads"] = head_table(A, layer, t["labels"], k)
                row["heads_ckpt"], row["heads_step"] = cfg.ckpt, used
                for l, v in row["heads"].items():
                    print(f"         layer {l} heads' best NMI with k-means: "
                          f"median {v['median']:.4f}  max {v['max']:.4f}")
            summary[name][str(k)] = row
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
