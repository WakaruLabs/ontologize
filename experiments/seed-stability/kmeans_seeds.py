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
variance share, skew, excess kurtosis and bimodality coefficient.
`--groups` keeps groups of FastICA sources whose magnitudes correlate
(`group_copy`), and `--scale-copy` keeps each row's scale shared by all its
sources and shuffles the sources apart under it (`scale_copy`). `--ckpt`
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
            tol: float = 1e-4, W0: Float[np.ndarray, "p p"] | None = None,
            log_every: int = 0, patience: int = 0, min_step: float = 1 / 64,
            checkpoint=None, checkpoint_every: int = 100,
            step0: float | Float[np.ndarray, "p"] = 1.0, shared: bool = False
            ) -> tuple[Float[np.ndarray, "p p"], int, float]:
    """Symmetric FastICA (Hyvärinen 1999) with the log-cosh contrast on
    sphered rows (zero mean, identity covariance): the orthogonal W whose
    rows' projections Z Wᵀ are as far from Gaussian as the contrast finds.
    It starts from W0 when given (an earlier fit to continue), else from a
    random rotation.

    Each iteration computes the fixed-point update F(W) and each row's
    change, |1 − |f · w||; once the largest is below tol, F(W) is returned
    as converged. With `patience` 0 every step is the update itself.
    Symmetric FastICA can cycle instead of converging, so with `patience`
    > 0 the steps are damped: row i of W moves a share step_i of the way
    to F(W)'s (signs matched to W's), and W is orthogonalized again. The
    fixed points are the plain algorithm's, and convergence is read on the
    full update either way. The steps start at `step0` (a number, or a
    continued fit's steps) and halve, down to `min_step`:
      - by default, row by row, whenever row i, still changing by tol or
        more, has gone `patience` iterations without a new low of its own;
      - with `shared`, one step for all rows, whenever the largest change
        has gone `patience` iterations without a new low. That step answers
        to the worst row, and from a random start it can reach `min_step`
        while most rows still change by more than tol.
    Every log_every iterations it prints the largest and median change,
    how many rows change by tol or more, the step (its smallest and median
    by row), and the largest change over two iterations, which stays small
    while rows flip between two states. checkpoint(W, it, lim, step) is
    called every checkpoint_every iterations, with the shared step or the
    per-row steps. Returns W, the iterations run, and the last largest
    change."""
    rng = np.random.default_rng(seed)
    p = Z.shape[1]
    W = symmetric_orth(rng.standard_normal((p, p)) if W0 is None else W0)
    Z32 = Z.astype(np.float32)
    lim, W_prev = np.inf, None
    if shared:
        if np.ndim(step0):
            raise ValueError("a shared step starts from one value, not one per row")
        best, since, step = np.inf, 0, float(step0)
    else:
        step = np.broadcast_to(np.asarray(step0, np.float64), (p,)).copy()
        best, since = np.full(p, np.inf), np.zeros(p, int)
    for it in range(1, iters + 1):
        G = np.tanh(Z32 @ W.T.astype(np.float32))
        gp = (1.0 - G.astype(np.float64) ** 2).mean(0)
        F = symmetric_orth((G.T @ Z32).astype(np.float64) / len(Z)
                           - gp[:, None] * W)
        cos = (F * W).sum(1)
        change = np.abs(np.abs(cos) - 1.0)
        lim = float(change.max())
        if lim < tol:
            W = F
            break
        if shared:
            if lim < best:
                best, since = lim, 0
            else:
                since += 1
                if patience and since >= patience and step > min_step:
                    step, since = max(step / 2, min_step), 0
            full, share = step == 1.0, step
        else:
            low = change < best
            best[low], since[low] = change[low], 0
            since[~low] += 1
            if patience:
                cut = (since >= patience) & (change >= tol) & (step > min_step)
                step[cut], since[cut] = np.maximum(step[cut] / 2, min_step), 0
            full, share = bool((step == 1.0).all()), step[:, None]
        W_new = F if full else symmetric_orth(
            W + share * (np.sign(cos)[:, None] * F - W))
        if log_every and it % log_every == 0:
            two = (float(np.abs(np.abs((W_new * W_prev).sum(1)) - 1.0).max())
                   if W_prev is not None else np.nan)
            steps = (f"step {step:g}" if shared else
                     f"step min {step.min():g} median {np.median(step):g}")
            print(f"FastICA iteration {it}: largest change {lim:.2e}, median "
                  f"{np.median(change):.2e}, {int((change >= tol).sum())} of "
                  f"{p} rows at or above {tol:g}, {steps}, largest change "
                  f"over two iterations {two:.2e}", flush=True)
        W_prev, W = W, W_new
        if checkpoint is not None and it % checkpoint_every == 0:
            checkpoint(W, it, lim, step if shared else step.copy())
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


def ica_frame(train: Float[np.ndarray, "n d"], seed: int = 0, iters: int = 200,
              init: dict | None = None, log_every: int = 0, patience: int = 0,
              checkpoint_path: Path | None = None, step0: float | None = None,
              shared: bool = False) -> dict:
    """FastICA's frame of the training rows: their principal frame (mean,
    eigenvalues, eigenvectors, directions with no variance left out) and
    the unmixing W of symmetric FastICA in its sphered coordinates, with
    the iterations run and the last step's largest change. `init` is an
    earlier fit's frame on the same rows, which FastICA continues from (up
    to `iters` more iterations; at 0 the frame is returned as it is); it
    raises if its principal frame is not these rows'; FastICA continues at
    `step0` when given, else at the step the frame was saved with, else at
    1, and `iterations` counts the earlier fit's too.
    `patience` damps the steps (`fastica`, row by row, or with `shared` one
    step for all rows), and with `checkpoint_path` the frame so far is
    saved there every 100 iterations, with its step, in the form `init`
    takes."""
    mu, lam, V = principal(train)
    W0, done = None, 0
    if init is not None:
        if not (init["eigenvectors"].shape == V.shape
                and np.allclose(init["eigenvectors"], V, atol=1e-6)):
            raise ValueError("the initial frame was fit on other rows")
        W0, done = init["unmixing"], int(init.get("iterations", 0))
        if step0 is None and "step" in init:
            step0 = np.asarray(init["step"], np.float64)
    step0 = 1.0 if step0 is None else step0
    frame = {"mean": mu, "eigenvalues": lam, "eigenvectors": V}
    if iters == 0:
        if init is None:
            raise ValueError("no iterations and no initial frame")
        return {**frame, "unmixing": W0, "iterations": done,
                "last_change": float(init.get("last_change", np.inf))}

    def save(W, it, lim, step):
        tmp = Path(checkpoint_path).with_suffix(".tmp.npz")
        np.savez(tmp, **frame, unmixing=W, iterations=done + it,
                 last_change=lim, step=step)
        os.replace(tmp, checkpoint_path)

    W, it, lim = fastica((train.astype(np.float64) - mu) @ (V / np.sqrt(lam)),
                         seed, iters, W0=W0, log_every=log_every,
                         patience=patience, step0=step0, shared=shared,
                         checkpoint=save if checkpoint_path else None)
    return {**frame, "unmixing": W, "iterations": done + it, "last_change": lim}


def to_sources(X: Float[np.ndarray, "n d"], fit: dict) -> Float[np.ndarray, "n p"]:
    """Rows as `ica_frame`'s sources (unit variance on the training rows)."""
    V, lam = fit["eigenvectors"], fit["eigenvalues"]
    return (X.astype(np.float64) - fit["mean"]) @ (V / np.sqrt(lam)) @ fit["unmixing"].T


def mixing(fit: dict) -> Float[np.ndarray, "p d"]:
    """Each source's direction in the rows' frame: sources @ mixing + mean
    gives back the rows (their part with variance)."""
    return fit["unmixing"] @ (fit["eigenvectors"] * np.sqrt(fit["eigenvalues"])).T


def marginal_copy(train: Float[np.ndarray, "n d"], tail: Float[np.ndarray, "m d"],
                  frame: str, seed: int = 0, ica_iters: int = 200,
                  ica_init: dict | None = None, ica_log: int = 0,
                  ica_fit: dict | None = None
                  ) -> tuple[Float[np.ndarray, "n d"], Float[np.ndarray, "m d"], dict]:
    """Copies of the training and tail rows that keep each coordinate's
    distribution in a frame fitted on the training rows and nothing that
    ties coordinates together: every coordinate's column is permuted on its
    own, in the training rows and the tail separately. `frame` "pca" is the
    rows' principal directions, so the copy keeps the covariance and every
    principal direction's marginal (its skew, tails and modes); "ica" is
    FastICA's sources in the sphered principal frame (`ica_frame`, or
    `ica_fit` when it has been fitted already), so the copy is exactly an
    independent-components model with the rows' own source marginals.
    Directions with no variance are left out of the copy. The dict has, per
    coordinate of the frame, its share of the variance and `shape_stats` on
    the training rows; `frame`, the fitted frame (mean, eigenvalues,
    eigenvectors, and for "ica" the unmixing W, sources = sphered rows Wᵀ);
    and for "ica" how FastICA ended."""
    if frame == "pca":
        mu, lam, V = principal(train)
        fit = {"mean": mu, "eigenvalues": lam, "eigenvectors": V}
        M = V.T                                          # (p, d): coordinate -> row
        to = lambda X: (X.astype(np.float64) - mu) @ V
        info = {}
    elif frame == "ica":
        fit = ica_fit if ica_fit is not None else ica_frame(
            train, seed, ica_iters, ica_init, ica_log)
        mu, M = fit["mean"], mixing(fit)                 # Y M + mu = X
        to = lambda X: to_sources(X, fit)
        info = {"ica_iterations": fit["iterations"],
                "ica_last_change": fit["last_change"]}
    else:
        raise ValueError(f"unknown frame {frame!r}")
    Yt, Ye = to(train), to(tail)
    var = (M ** 2).sum(1) * Yt.var(0)
    stats = {"variance_share": var / var.sum(), **shape_stats(Yt), **info,
             "frame": fit}
    back = lambda Y: (Y @ M + mu).astype(np.float32)
    return (back(shuffle_columns(Yt, 2 * seed)),
            back(shuffle_columns(Ye, 2 * seed + 1)), stats)


def energy_dependence(S: Float[np.ndarray, "n p"]) -> Float[np.ndarray, "p p"]:
    """Spearman correlation between the sources' magnitudes |s_i|: the
    dependence ICA leaves when sources are uncorrelated but tend to be
    large together. Ranks keep heavy tails from dominating it. Independent
    sources read zero, within about 1/sqrt(n)."""
    from scipy.stats import rankdata
    R = rankdata(np.abs(S), axis=0)
    R = (R - R.mean(0)) / R.std(0)
    return R.T @ R / len(S)


def energy_groups(R: Float[np.ndarray, "p p"], tau: float) -> Int[np.ndarray, "p"]:
    """Sources grouped by average-linkage clustering on 1 − R, cut so that
    the mean correlation between a group's members' magnitudes is at least
    tau: group labels 0, 1, ..."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    D = 1.0 - R
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(np.maximum(D, 0.0), checks=False), "average")
    _, lab = np.unique(fcluster(Z, 1.0 - tau, "distance"), return_inverse=True)
    return lab


def group_copy(train: Float[np.ndarray, "n d"], tail: Float[np.ndarray, "m d"],
               fit: dict, groups: Int[np.ndarray, "p"], seed: int = 0
               ) -> tuple[Float[np.ndarray, "n d"], Float[np.ndarray, "m d"]]:
    """Copies in the ICA frame that keep each group of sources' joint
    distribution and nothing that ties groups together: one permutation of
    the rows per group, in the training rows and the tail separately. All
    singletons is the ICA copy; one group is the rows themselves."""
    M = mixing(fit)

    def copy(X, s):
        Y = to_sources(X, fit)
        rng = np.random.default_rng(s)
        out = np.empty_like(Y)
        for g in range(int(groups.max()) + 1):
            cols = np.flatnonzero(groups == g)
            out[:, cols] = Y[np.ix_(rng.permutation(len(Y)), cols)]
        return (out @ M + fit["mean"]).astype(np.float32)

    return copy(train, 2 * seed), copy(tail, 2 * seed + 1)


def row_scale(S: Float[np.ndarray, "n p"]) -> Float[np.ndarray, "n"]:
    """Each row's scale, shared by its sources: the median of |s_i| over
    them, which a few large sources hardly move."""
    return np.maximum(np.median(np.abs(S), axis=1), 1e-30)


def scale_copy(train: Float[np.ndarray, "n d"], tail: Float[np.ndarray, "m d"],
               fit: dict, seed: int = 0, bins: int = 64
               ) -> tuple[Float[np.ndarray, "n d"], Float[np.ndarray, "m d"]]:
    """Copies in the ICA frame that keep each row's scale (`row_scale`)
    and nothing else that ties sources together. The rows are sorted by
    their scale and cut into `bins` bins of equal count, and within each
    bin every source's column is permuted on its own, in the training rows
    and the tail separately. Every source keeps its values exactly and,
    up to the bin's width, how they depend on the scale; the ICA copy
    breaks that dependence as well. (Dividing by the scale, permuting and
    multiplying back would assume s = r u with u independent of r, which
    a source whose spikes grow less than in proportion to r breaks.)"""
    M = mixing(fit)

    def copy(X, s):
        Y = to_sources(X, fit)
        rng = np.random.default_rng(s)
        out = np.empty_like(Y)
        cols = np.arange(Y.shape[1])
        for b in np.array_split(np.argsort(row_scale(Y)), bins):
            idx = rng.permuted(np.tile(np.arange(len(b)), (len(cols), 1)), axis=1)
            out[b] = Y[b][idx.T, cols]
        return (out @ M + fit["mean"]).astype(np.float32)

    return copy(train, 2 * seed), copy(tail, 2 * seed + 1)


def scale_stats(train: Float[np.ndarray, "n d"], copy: Float[np.ndarray, "n d"],
                fit: dict, sources: int = 300) -> dict:
    """What the scale copy keeps, on the training rows and their copy:
    the row scale's coefficient of variation, the median magnitude
    correlation between the first `sources` sources, and the sources'
    median excess kurtosis."""
    out = {}
    for name, X in (("rows", train), ("copy", copy)):
        S = to_sources(X, fit)
        r = row_scale(S)
        R = energy_dependence(S[:, :sources])
        out[name] = {"scale_cv": float(r.std() / r.mean()),
                     "magnitude_correlation_median":
                         float(np.median(R[~np.eye(len(R), dtype=bool)])),
                     "kurtosis_median": float(np.median(shape_stats(S)["kurtosis"]))}
    return out


def group_stats(groups: Int[np.ndarray, "p"], R: Float[np.ndarray, "p p"],
                variance_share: Float[np.ndarray, "p"]) -> dict:
    """A grouping's sizes, the share of sources and of variance outside
    singletons, and its groups' mean within-group magnitude correlation."""
    size = np.bincount(groups)
    multi = size[groups] > 1
    within = [float(R[np.ix_(c, c)][~np.eye(len(c), dtype=bool)].mean())
              for c in (np.flatnonzero(groups == g) for g in np.flatnonzero(size > 1))]
    return {"groups": int(len(size)), "largest": int(size.max()),
            "groups_over_1": int((size > 1).sum()),
            "sources_in_groups": int(multi.sum()),
            "variance_in_groups": float(variance_share[multi].sum()),
            "within_correlation_median": float(np.median(within)) if within else None}


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


def train_index(prefix: int, n: int, seed: int = 0,
                exclude: Int[np.ndarray, "m"] | None = None) -> Int[np.ndarray, "n"]:
    """The training rows' indices: n of the first `prefix` rows drawn
    without replacement (seed 0 is every data set's here), sorted, none of
    them in `exclude`."""
    pool = prefix if exclude is None else np.setdiff1d(np.arange(prefix), exclude)
    return np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))


def rows(cfg) -> tuple[dict, dict, dict]:
    """Training rows and tail rows, whitened, for each data set, with the
    tail in the cache's own frame for the heads (None for a copy made in
    the whitened frame); each marginal copy's per-coordinate table; and
    the ICA fit (None unless a copy needs it), with --groups the sources'
    magnitude correlations and each grouping, and with --scale-copy what
    that copy keeps (`scale_stats`)."""
    sw = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    X = np.load(cfg.cache, mmap_mode="r")
    idx = train_index(cfg.prefix, cfg.train_rows)
    out = {"cache": (np.asarray(X[idx], np.float32) * sw,
                     np.asarray(X[-cfg.tail_rows:], np.float32) * sw,
                     np.asarray(X[-cfg.tail_rows:], np.float32))}
    shapes, extra = {}, {"ica_fit": None, "energy": None, "scale": None}
    if "ica" in cfg.marginals or cfg.groups or cfg.scale_copy:
        print("ica frame: fitting", flush=True)
        init = dict(np.load(cfg.ica_init)) if cfg.ica_init else None
        extra["ica_fit"] = ica_frame(out["cache"][0], 0, cfg.ica_iters, init,
                                     log_every=25, patience=cfg.ica_patience,
                                     checkpoint_path=Path(cfg.out) / "frame_ica.partial.npz",
                                     step0=cfg.ica_step, shared=cfg.ica_shared_step)
    for frame in cfg.marginals:
        print(f"{frame} copy", flush=True)
        T, E, shapes[frame] = marginal_copy(*out["cache"][:2], frame,
                                            ica_fit=extra["ica_fit"])
        out[frame] = (T, E, None)
    if cfg.groups:
        fit = extra["ica_fit"]
        S = to_sources(out["cache"][0], fit)
        R = energy_dependence(S)
        var = (mixing(fit) ** 2).sum(1) * S.var(0)
        extra["energy"] = {"R": R, "groups": {}}
        for tau in cfg.groups:
            g = energy_groups(R, tau)
            name = f"ica_g{tau:g}"
            print(f"{name} copy", flush=True)
            T, E = group_copy(*out["cache"][:2], fit, g)
            out[name] = (T, E, None)
            extra["energy"]["groups"][name] = {
                "tau": tau, "labels": g, **group_stats(g, R, var / var.sum())}
    if cfg.scale_copy:
        print("ica_scale copy", flush=True)
        T, E = scale_copy(*out["cache"][:2], extra["ica_fit"])
        out["ica_scale"] = (T, E, None)
        extra["scale"] = scale_stats(out["cache"][0], T, extra["ica_fit"])
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
    return out, shapes, extra


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
    p.add_argument("--ica-iters", type=int, default=200,
                   help="FastICA's iterations (with --ica-init, more of them; "
                        "0 uses --ica-init's frame as it is)")
    p.add_argument("--ica-init", default=None,
                   help="frame_ica.npz (or the frame_ica.partial.npz a run saves "
                        "every 100 iterations) from an earlier run on the same "
                        "rows: FastICA continues from its unmixing")
    p.add_argument("--ica-patience", type=int, default=0,
                   help="damp FastICA's steps, halving a row's whenever its "
                        "change goes this many iterations without a new low "
                        "(0: plain FastICA)")
    p.add_argument("--ica-shared-step", action="store_true",
                   help="damp with one step for all rows, halved whenever the "
                        "largest change stalls (the schedule "
                        "frame_ica.iter800.npz was fitted with)")
    p.add_argument("--ica-step", type=float, default=None,
                   help="the damped step to start at (default: --ica-init's "
                        "saved step, else 1)")
    p.add_argument("--groups", type=float, nargs="*", default=[],
                   help="copies that keep groups of FastICA sources whose "
                        "magnitudes correlate at a mean of at least each value, "
                        "and nothing that ties groups together")
    p.add_argument("--scale-copy", action="store_true",
                   help="a copy that keeps each row's scale shared by its "
                        "FastICA sources and shuffles the sources apart under it")
    p.add_argument("--no-cache", action="store_true",
                   help="leave the cache's own rows unscored (their numbers do "
                        "not change from run to run)")
    p.add_argument("--ckpt", default=None, help="a stack whose heads to compare")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--out", default=str(ROOT / "data/out/sonar/kmeans_seeds"))
    cfg = p.parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    data, shapes, extra = rows(cfg)
    summary = {}
    frames = {f: st["frame"] for f, st in shapes.items()}
    if extra["ica_fit"] is not None:
        frames["ica"] = extra["ica_fit"]
        print(f"FastICA: {extra['ica_fit']['iterations']} iterations, last change "
              f"{extra['ica_fit']['last_change']:.2e}", flush=True)
    for frame, fr in frames.items():
        np.savez(out / f"frame_{frame}.npz", **fr)
    if extra["scale"] is not None:
        summary["ica_scale_stats"] = s = extra["scale"]
        print("scale copy, rows / copy: scale CV {:.3f} / {:.3f}, magnitude "
              "correlation median {:+.4f} / {:+.4f}, kurtosis median {:.2f} / {:.2f}".format(
                  s["rows"]["scale_cv"], s["copy"]["scale_cv"],
                  s["rows"]["magnitude_correlation_median"],
                  s["copy"]["magnitude_correlation_median"],
                  s["rows"]["kurtosis_median"], s["copy"]["kurtosis_median"]), flush=True)
    if extra["energy"] is not None:
        R = extra["energy"]["R"]
        np.save(out / "energy_ica.npy", R.astype(np.float32))
        off = R[~np.eye(len(R), dtype=bool)]
        summary["energy"] = {
            "null_sd": float(1 / np.sqrt(cfg.train_rows)),
            "quantiles": {q: float(np.quantile(off, float(q)))
                          for q in ("0.01", "0.5", "0.99", "0.999")},
            "pairs_over": {t: int((off > float(t)).sum() // 2)
                           for t in ("0.05", "0.1", "0.2", "0.3")}}
        e = summary["energy"]
        print(f"magnitude correlations: median {e['quantiles']['0.5']:+.4f}, "
              f"99.9th percentile {e['quantiles']['0.999']:+.4f} (null SD "
              f"{e['null_sd']:.4f}); pairs over 0.1 {e['pairs_over']['0.1']}, "
              f"over 0.3 {e['pairs_over']['0.3']}", flush=True)
        np.savez(out / "groups_ica.npz",
                 **{n: g["labels"] for n, g in extra["energy"]["groups"].items()})
        for n, g in extra["energy"]["groups"].items():
            summary[f"{n}_groups"] = s = {k: v for k, v in g.items() if k != "labels"}
            print(f"{n}: {s['groups']} groups, {s['groups_over_1']} of them "
                  f"larger than one (largest {s['largest']}), holding "
                  f"{s['sources_in_groups']} sources and "
                  f"{s['variance_in_groups']:.1%} of the variance", flush=True)
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
        if name == "cache" and cfg.no_cache:
            continue
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
