"""Within-model head x head partition NMI heatmap.

Each head's argmax entry partitions the held-out rows into k cells. This
script compares every (layer, head) partition with every other one by
normalized mutual information, on the tail --eval-rows of the cache (the
sae.py holdout; assignmap.py's convention), and draws the (l*h) x (l*h)
matrix. High NMI between heads of one layer says they are not exclusive
(they split the rows the same way); high NMI between layers says a later
residual layer re-partitions what an earlier one already split.

  NMI  I(a; b) / mean(H(a), H(b)), arithmetic normalization, in nats, from
       the k x k contingency table of each pair (as in
       experiments/gpt2/seedstab.py and experiments/ste-arm/partition.py).
  null Two independent partitions on finite N have I > 0. Its expectation
       under independence with the observed marginals is approximated by
       the Miller-Madow term E[I] ~ (r_a - 1)(r_b - 1) / 2N, with r the
       number of entries the head actually uses (effinfo.py's mi_obs uses
       the same correction). The default statistic is the adjusted NMI
         adj = (I - E[I]) / (mean(H_a, H_b) - E[I]),
       which is ~0 for independent heads and 1 for identical ones (the
       AMI form with the approximate expectation). --stat nmi shows the raw
       NMI instead and marks the median null NMI on the colourbar. A one-
       draw row-permutation null is always computed and printed against
       the analytic one as a check; --null perm uses the mean of --n-perm
       permutation draws in place of the analytic term.
  dead A head whose argmax usage has exp H < --min-eff effective entries
       (collapsed onto one entry) carries no partition: its row and column
       are drawn grey and marked in the strip. Its NMI is 0, never NaN,
       and it is left out of every summary mean.

Layout (--order):
  index   blocked by layer, heads in index order within each layer
  hclust  blocked by layer, heads within each layer ordered by average-
          linkage hierarchical clustering on 1 - adj (optimal leaf order)
  global  one hclust over all live heads, ignoring layers; the layer-
          membership strip then shows whether clustering recovers layers
Dead heads go last within their block (last overall under global).
Colour is a fixed sequential scale with a colourbar: logarithmic on
[--vmin, --vmax] by default, since most pairs sit within a decade of
chance and a few near 1 (values at or below --vmin, including negative
adjusted NMI, take the lowest colour), or linear on [0, --vmax]. The
diagonal is masked.

Pairing with effective information (--ei PATH): an `effinfo.py` output,
either a (l, h, l, h) .npy indexed [source layer, source head, effect
layer, effect head] or its output directory (then --ei-key picks the file,
default ei_live, the on-distribution EI for a hard code). It is drawn as
a second panel in the same head order (rows = source, columns = effect;
EI is directed and defined only for later effect layers, the rest grey),
with a scatter of adjusted NMI against it over the pairs where it is
finite. --ei-key mi_obs, the observational MI, is a layout check: it
should track NMI closely.

  uv run python headnmi.py --ckpt data/out/sonar/multilingual/resid_nc
  uv run python headnmi.py --ckpt data/out/sonar/multilingual/ste_h76 \\
      --ei data/out/sonar/multilingual/ste_h76/effinfo
  uv run python headnmi.py --ckpt data/out/gpt2_l8/ste_h76 \\
      --cache data/activations/gpt2_l8.npy --order global

The temperature defaults to the run's schedule at the restored step (its
log.jsonl env_config); with a residual forward it changes what later
layers classify. Writes <out>/headnmi_<order>.png and headnmi.npz (the
argmax labels, the NMI, null and adjusted matrices), so the figure can be
redrawn without the model (--replot, which takes --out or --ckpt to find
it, and honours --order/--stat/--scale/--vmin/--vmax/--ei). NOTE: Ontologizer
checkpoints restore on GPU JAX only.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from jaxtyping import Bool, Float, Int


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Ontologizer checkpoint dir")
    p.add_argument("--step", type=int, default=0, help="default: latest")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: the run's schedule at --step")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="held-out tail rows of the cache (sae.py's default)")
    p.add_argument("--order", choices=["index", "hclust", "global"],
                   default="hclust")
    p.add_argument("--stat", choices=["adj", "nmi"], default="adj",
                   help="adjusted (null-corrected) or raw NMI")
    p.add_argument("--null", choices=["analytic", "perm"], default="analytic",
                   help="expected MI under independence: Miller-Madow "
                   "approximation, or the mean of --n-perm row permutations")
    p.add_argument("--n-perm", type=int, default=4)
    p.add_argument("--min-eff", type=float, default=1.1,
                   help="heads with fewer effective entries exp H are dead")
    p.add_argument("--scale", choices=["log", "linear"], default="log",
                   help="colour scale; log spreads the 1e-3..1 range most "
                   "head pairs span")
    p.add_argument("--vmin", type=float, default=1e-3,
                   help="bottom of the log scale (values below, chance "
                   "included, take the lowest colour)")
    p.add_argument("--vmax", type=float, default=1.0,
                   help="top of the fixed colour scale (values above "
                   "saturate); compare runs only at the same limits")
    p.add_argument("--ei", default=None,
                   help="effinfo.py output dir or (l, h, l, h) .npy")
    p.add_argument("--ei-key", default="ei_live",
                   help="file in an --ei directory: ei_live, ei, mi_obs, ...")
    p.add_argument("--ei-lim", type=float, nargs=2, default=None,
                   help="the EI panel's scale (default: its 1st (log) or 0 "
                   "(linear) to 99th percentile, printed in the title)")
    p.add_argument("--top", type=int, default=10,
                   help="most redundant pairs to print")
    p.add_argument("--replot", action="store_true",
                   help="redraw from <out>/headnmi.npz without the model")
    p.add_argument("--b", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="default <ckpt>/headnmi")
    return p.parse_args()


# ---------- statistics ----------

def _onehot(A: Int[np.ndarray, "b m"], k: int) -> Float[np.ndarray, "b mk"]:
    """Row-wise one-hot of every column, column a's block at [a*k, a*k+k)."""
    O = np.zeros((A.shape[0], A.shape[1] * k), np.float32)
    np.put_along_axis(O, A + np.arange(A.shape[1]) * k, 1.0, axis=1)
    return O


def contingency(A: Int[np.ndarray, "n m"], k: int,
                B: Optional[Int[np.ndarray, "n m2"]] = None,
                chunk: int = 8192) -> Float[np.ndarray, "m m2 k k"]:
    """Joint count table of every column of A against every column of B
    (default A itself; labels in [0, k)), C[a, b, i, j] = #{rows with
    A[:, a] = i and B[:, b] = j}. One one-hot matmul per chunk of rows;
    float32 is exact while n < 2**24."""
    B = A if B is None else B
    n, m = A.shape
    m2 = B.shape[1]
    assert n < 2 ** 24 and B.shape[0] == n
    assert min(A.min(), B.min()) >= 0 and max(A.max(), B.max()) < k
    C = np.zeros((m * k, m2 * k), np.float32)
    for s in range(0, n, chunk):
        Oa = _onehot(A[s:s + chunk], k)
        Ob = Oa if B is A else _onehot(B[s:s + chunk], k)
        C += Oa.T @ Ob
    return C.reshape(m, k, m2, k).transpose(0, 2, 1, 3)


def _plogp(p: Float[np.ndarray, "..."]) -> Float[np.ndarray, "..."]:
    """p log p with 0 log 0 = 0 (nats)."""
    return np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)


def mutual_info(C: Float[np.ndarray, "m m k k"], block: int = 32
                ) -> tuple[Float[np.ndarray, "m m"], Float[np.ndarray, "m"],
                           Int[np.ndarray, "m"], int]:
    """(I, H, r, N) from a contingency tensor: plug-in mutual information
    of every pair and entropy of every head in nats, the number of entries
    r each head uses, and the row count N. Blocked over rows of heads so
    only `block` x m x k x k float64 values exist at once."""
    m = C.shape[0]
    N = int(round(float(C[0, 0].sum())))
    marg = np.stack([np.diag(C[a, a]) for a in range(m)]).astype(np.float64)
    p = marg / N                                         # (m, k)
    H = -_plogp(p).sum(-1)
    r = (marg > 0).sum(-1)
    I = np.zeros((m, m))
    for s in range(0, m, block):
        J = C[s:s + block].astype(np.float64) / N        # (b, m, k, k)
        Hj = -_plogp(J).sum((-2, -1))
        I[s:s + block] = H[s:s + block, None] + H[None, :] - Hj
    I = np.maximum(0.5 * (I + I.T), 0.0)  # symmetric; clip rounding below 0
    return I, H, r, N


def analytic_null(r: Int[np.ndarray, "m"], N: int) -> Float[np.ndarray, "m m"]:
    """Expected plug-in MI (nats) of independent partitions using r_a and
    r_b cells, to first order: (r_a - 1)(r_b - 1) / 2N."""
    return np.outer(r - 1, r - 1) / (2.0 * N)


def perm_null(A: Int[np.ndarray, "n m"], k: int, n_perm: int, seed: int
              ) -> Float[np.ndarray, "m m"]:
    """Mean MI between every head and every head of a row-permuted copy:
    independence with both marginals kept, over n_perm draws. Symmetrized,
    since one draw gives (a, perm b) and (b, perm a) different values."""
    rng = np.random.default_rng(seed)
    n, m = A.shape
    H = np.stack([-_plogp(np.bincount(a, minlength=k) / n).sum()
                  for a in A.T])  # a permutation keeps every marginal
    out = np.zeros((m, m))
    for _ in range(n_perm):
        C = contingency(A, k, A[rng.permutation(n)])
        Hj = np.stack([-_plogp(C[a].astype(np.float64) / n).sum((-2, -1))
                       for a in range(m)])
        out += np.maximum(H[:, None] + H[None, :] - Hj, 0.0)
    out /= n_perm
    return 0.5 * (out + out.T)


def normalized(I: Float[np.ndarray, "m m"], H: Float[np.ndarray, "m"],
               E: Float[np.ndarray, "m m"], eps: float = 1e-9
               ) -> tuple[Float[np.ndarray, "m m"], Float[np.ndarray, "m m"]]:
    """(raw NMI, adjusted NMI). raw = I / mean(H); adjusted =
    (I - E) / (mean(H) - E). Either is 0 where its denominator vanishes
    (a pair with a constant partition, whose I is 0 too), never NaN."""
    Hm = 0.5 * (H[:, None] + H[None, :])
    nmi = np.where(Hm > eps, I / np.where(Hm > eps, Hm, 1.0), 0.0)
    D = Hm - E
    adj = np.where(D > eps, (I - E) / np.where(D > eps, D, 1.0), 0.0)
    return nmi, adj


def head_nmi(A: Int[np.ndarray, "n m"], k: int, null: str = "analytic",
             n_perm: int = 4, seed: int = 0) -> dict:
    """Every statistic the figure and summary use, for argmax labels A."""
    I, H, r, N = mutual_info(contingency(A, k))
    E = analytic_null(r, N) if null == "analytic" else \
        perm_null(A, k, n_perm, seed)
    nmi, adj = normalized(I, H, E)
    return dict(I=I, H=H, r=r, N=N, E=E, nmi=nmi, adj=adj)


# ---------- ordering ----------

def hclust_order(S: Float[np.ndarray, "m m"]) -> Int[np.ndarray, "m"]:
    """Leaf order of average-linkage clustering on 1 - clip(S, 0, 1), with
    optimal leaf ordering. Fewer than three items keep index order."""
    m = S.shape[0]
    if m < 3:
        return np.arange(m)
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    D = 1.0 - np.clip(0.5 * (S + S.T), 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), "average", optimal_ordering=True)
    return leaves_list(Z)


def head_order(S: Float[np.ndarray, "m m"], layer: Int[np.ndarray, "m"],
               dead: Bool[np.ndarray, "m"], mode: str) -> Int[np.ndarray, "m"]:
    """Display order of the m heads (see --order). Dead heads go last
    within their layer block, or last overall under "global"."""
    if mode == "global":
        live = np.flatnonzero(~dead)
        return np.concatenate([live[hclust_order(S[np.ix_(live, live)])],
                               np.flatnonzero(dead)])
    out = []
    for L in np.unique(layer):
        live = np.flatnonzero((layer == L) & ~dead)
        if mode == "hclust":
            live = live[hclust_order(S[np.ix_(live, live)])]
        out += [live, np.flatnonzero((layer == L) & dead)]
    return np.concatenate(out)


# ---------- summary ----------

def pair_means(S: Float[np.ndarray, "m m"], layer: Int[np.ndarray, "m"],
               dead: Bool[np.ndarray, "m"]) -> dict:
    """Mean of S over distinct live pairs: within a layer, between
    adjacent layers, between layers two or more apart, and the (l, l)
    matrix of block means (diagonal blocks exclude self-pairs)."""
    live = ~dead
    ok = live[:, None] & live[None, :] & ~np.eye(len(S), dtype=bool)
    gap = np.abs(layer[:, None] - layer[None, :])
    mean = lambda m: float(S[ok & m].mean()) if (ok & m).any() else np.nan
    ls = np.unique(layer)
    blocks = np.array([[mean((layer[:, None] == a) & (layer[None, :] == b))
                        for b in ls] for a in ls])
    return dict(within=mean(gap == 0), adjacent=mean(gap == 1),
                distant=mean(gap >= 2), blocks=blocks)


def top_pairs(S: Float[np.ndarray, "m m"], dead: Bool[np.ndarray, "m"],
              n: int) -> list[tuple[int, int, float]]:
    """The n largest S over distinct live pairs a < b, largest first."""
    a, b = np.triu_indices(len(S), 1)
    keep = ~dead[a] & ~dead[b]
    a, b = a[keep], b[keep]
    top = np.argsort(-S[a, b], kind="stable")[:n]
    return [(int(a[i]), int(b[i]), float(S[a[i], b[i]])) for i in top]


def summarize(st: dict, l: int, h: int, stat: str, top: int):
    """Print the null check, the within/between means, and the top pairs."""
    layer = np.repeat(np.arange(l), h)
    dead = st["dead"]
    print(f"{int(dead.sum())} dead heads (of {l * h}); per layer: "
          + " ".join(str(int(dead[layer == L].sum())) for L in range(l)))
    for name in ("nmi", "adj"):
        pm = pair_means(st[name], layer, dead)
        print(f"{name:>4}: within-layer {pm['within']:.4f}  adjacent layers "
              f"{pm['adjacent']:.4f}  layers >= 2 apart {pm['distant']:.4f}")
    pm = pair_means(st[stat], layer, dead)
    print(f"{stat} layer x layer block means:")
    for L in range(l):
        print("  " + " ".join(f"{v:7.4f}" for v in pm["blocks"][L]))
    off = ~np.eye(l * h, dtype=bool) & ~dead[:, None] & ~dead[None, :]
    Hm = 0.5 * (st["H"][:, None] + st["H"][None, :])
    print(f"null NMI (E[I]/mean H) median {np.median((st['E'] / np.maximum(Hm, 1e-12))[off]):.4f}")
    if "Eperm" in st:
        ratio = st["Eperm"][off].mean() / max(st["Eana"][off].mean(), 1e-12)
        print(f"null check: one-permutation MI mean {st['Eperm'][off].mean():.3e}"
              f" vs analytic {st['Eana'][off].mean():.3e} nats (ratio {ratio:.2f})")
    eff = np.exp(st["H"])
    print(f"top {top} pairs by {stat} (layer.head, exp H):")
    for a, b, v in top_pairs(st[stat], dead, top):
        print(f"  {layer[a]}.{a % h:<3} ({eff[a]:5.1f})  {layer[b]}.{b % h:<3}"
              f" ({eff[b]:5.1f})  {stat} {v:.4f}  nmi {st['nmi'][a, b]:.4f}")


# ---------- EI ----------

def load_ei(path, key: str, l: int, h: int) -> Float[np.ndarray, "m m"]:
    """An effinfo.py (l, h, l, h) array, flattened to (l*h, l*h) with
    index layer*h + head on both axes, matching this script's heads."""
    p = Path(path)
    if p.is_dir():
        p = p / f"{key}.npy"
    E = np.load(p)
    assert E.shape == (l, h, l, h), \
        f"{p} has shape {E.shape}, expected {(l, h, l, h)}"
    return E.reshape(l * h, l * h)


def spearman(x: Float[np.ndarray, "n"], y: Float[np.ndarray, "n"]) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


# ---------- figure ----------

def _strips(fig, ax, layer, dead, l, cmap_l):
    """Layer-membership and dead-head strips above and left of ax."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import ListedColormap
    div = make_axes_locatable(ax)
    top = div.append_axes("top", size="4%", pad=0.02, sharex=ax)
    left = div.append_axes("left", size="4%", pad=0.02, sharey=ax)
    strip = np.stack([layer, np.where(dead, l, -1)]).astype(float)
    strip[1][strip[1] < 0] = np.nan
    cm = ListedColormap([cmap_l(i) for i in range(l)] + ["black"]
                        ).with_extremes(bad="white")
    top.imshow(strip, aspect="auto", interpolation="nearest", cmap=cm,
               vmin=-0.5, vmax=l + 0.5)
    left.imshow(strip.T, aspect="auto", interpolation="nearest", cmap=cm,
                vmin=-0.5, vmax=l + 0.5)
    top.set_yticks([0, 1], ["layer", "dead"], fontsize=6)
    left.set_xticks([0, 1], ["layer", "dead"], fontsize=6, rotation=90)
    top.tick_params(labelbottom=False, bottom=False)
    left.tick_params(labelleft=False, left=False)
    return top, left


def make_norm(lim: tuple[float, float], log: bool):
    """A fixed colour norm on [lim]: logarithmic, or linear."""
    from matplotlib.colors import LogNorm, Normalize
    return LogNorm(*lim) if log else Normalize(*lim)


def draw_matrix(fig, ax, M, layer, dead, order, l, h, title, cmap, lim,
                log, cb_label, null_line=None, blocked=True):
    """One ordered heatmap with strips, separators and a colourbar. Under
    a log norm, values at or below lim[0] (chance, and negative adjusted
    values) take the lowest colour."""
    import matplotlib.pyplot as plt
    m = len(order)
    X = M[np.ix_(order, order)].astype(float)
    np.fill_diagonal(X, np.nan)
    d = dead[order]
    X[d, :] = np.nan
    X[:, d] = np.nan
    cm = plt.get_cmap(cmap).with_extremes(bad="0.82")
    vmin, vmax = lim
    shown = np.where(np.isfinite(X), np.maximum(X, vmin), np.nan) if log else X
    im = ax.imshow(shown, interpolation="nearest", cmap=cm,
                   norm=make_norm(lim, log))
    lo = layer[order]
    if blocked:
        edges = np.flatnonzero(np.diff(lo)) + 0.5
        for e in edges:
            ax.axhline(e, color="0.1", lw=0.8)
            ax.axvline(e, color="0.1", lw=0.8)
        starts = np.concatenate([[0], np.flatnonzero(np.diff(lo)) + 1, [m]])
        mids = 0.5 * (starts[:-1] + starts[1:]) - 0.5
        labs = [f"L{lo[s]}" for s in starts[:-1]]
        ax.set_xticks(mids, labs, fontsize=8)
        ax.set_yticks(mids, labs, fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if m <= 64:
        names = [f"{layer[i]}.{i % h}" for i in order]
        ax.set_xticks(np.arange(m), names, fontsize=5, rotation=90)
        ax.set_yticks(np.arange(m), names, fontsize=5)
    top, _ = _strips(fig, ax, lo, d, l, plt.get_cmap("tab10"))
    top.set_title(title, fontsize=9)
    finite = X[np.isfinite(X)]
    over = finite.size and finite.max() > vmax
    under = finite.size and finite.min() < vmin
    ext = {(1, 1): "both", (1, 0): "max", (0, 1): "min"}.get(
        (int(bool(over)), int(bool(under))), "neither")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, extend=ext)
    cb.set_label(cb_label, fontsize=8)
    if null_line is not None:
        cb.ax.axhline(null_line, color="crimson", lw=1.5)
        cb.ax.text(1.6, null_line, "null", color="crimson", fontsize=7,
                   va="center", transform=cb.ax.get_yaxis_transform())
    return im


def ei_limits(ei: Float[np.ndarray, "m m"], log: bool,
              lim: Optional[tuple[float, float]] = None
              ) -> tuple[float, float]:
    """The EI panel's scale: `lim` when given, else the 1st (log) or 0
    (linear) to the 99th percentile of its positive finite values."""
    if lim is not None:
        return lim
    fin = ei[np.isfinite(ei) & (ei > 0)]
    if not fin.size:
        return (1e-3, 1.0) if log else (0.0, 1.0)
    hi = float(np.percentile(fin, 99))
    return (float(np.percentile(fin, 1)) if log else 0.0, hi)


def draw(st: dict, l: int, h: int, order_mode: str, stat: str,
         lim: tuple[float, float], log: bool, title: str, path,
         ei: Optional[Float[np.ndarray, "m m"]] = None, ei_label: str = "EI",
         ei_lim: Optional[tuple[float, float]] = None):
    """The NMI panel, and with `ei` the EI panel and the NMI-vs-EI
    scatter, all in one head order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer = np.repeat(np.arange(l), h)
    dead = st["dead"]
    S = st[stat]
    order = head_order(st["adj"], layer, dead, order_mode)
    blocked = order_mode != "global"
    n_pan = 1 if ei is None else 3
    fig, axes = plt.subplots(
        1, n_pan, figsize=(9 if ei is None else 25, 8.4),
        gridspec_kw={"width_ratios": [1] if ei is None else [1, 1, 0.75],
                     "wspace": 0.35})
    axes = np.atleast_1d(axes)
    null = None
    if stat == "nmi":
        Hm = 0.5 * (st["H"][:, None] + st["H"][None, :])
        off = ~np.eye(l * h, dtype=bool) & ~dead[:, None] & ~dead[None, :]
        null = float(np.median((st["E"] / np.maximum(Hm, 1e-12))[off]))
    lab = {"adj": "adjusted NMI (0 = chance, 1 = same partition)",
           "nmi": "NMI (red: median null)"}[stat]
    scale = "log" if log else "linear"
    draw_matrix(fig, axes[0], S, layer, dead, order, l, h,
                f"{title}\n{stat}, order {order_mode}, {scale} scale; grey: "
                f"diagonal and dead heads", "Blues", lim, log, lab, null,
                blocked)
    if ei is not None:
        elim = ei_limits(ei, log, ei_lim)
        how = "" if ei_lim is not None else \
            (" = p1..p99" if log else " = 0..p99")
        draw_matrix(fig, axes[1], ei, layer, dead, order, l, h,
                    f"{ei_label}: rows [l, h] = first index (source, for "
                    f"EI), columns second; grey where undefined\n{scale} scale "
                    f"[{elim[0]:.3g}, {elim[1]:.3g}]{how}",
                    "Oranges", elim, log, ei_label, None, blocked)
        ax = axes[2]
        ok = np.isfinite(ei) & ~dead[:, None] & ~dead[None, :] \
            & ~np.eye(l * h, dtype=bool)
        a, b = np.nonzero(ok)
        gap = np.abs(layer[a] - layer[b])
        x, y = S[a, b], ei[a, b]
        cmg = plt.get_cmap("viridis")
        for g in np.unique(gap):
            s = gap == g
            ax.scatter(x[s], y[s], s=3, alpha=0.35, lw=0,
                       color=cmg(g / max(gap.max(), 1)),
                       label=f"layer gap {g} (n={s.sum()})")
        ax.axvline(0, color="0.5", lw=0.8, ls=":")
        ax.set_xscale("symlog", linthresh=0.01, linscale=2)
        ax.set_xlim(-0.01, 1.0)
        ax.set_xticks([-0.01, 0, 0.01, 0.1, 1],
                      ["-0.01", "0", "0.01", "0.1", "1"])
        if log and (y > 0).any():
            ax.set_yscale("log")
            ax.set_ylim(max(y[y > 0].min(), elim[0] / 10), None)
        rho = spearman(x, y) if len(x) > 2 else np.nan
        ax.set_xlabel(lab.split(" (")[0] + " (linear below 0.01, log above)")
        ax.set_ylabel(ei_label)
        ax.set_title(f"{len(x)} head pairs with {ei_label} defined, "
                     f"Spearman rho {rho:.3f}", fontsize=9)
        ax.legend(fontsize=7, markerscale=4, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
        ax.set_box_aspect(1)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_ei_arg(cfg, l, h):
    """(EI matrix, label) for --ei, or (None, None)."""
    if not cfg.ei:
        return None, None
    key = cfg.ei_key if Path(cfg.ei).is_dir() else Path(cfg.ei).stem
    return load_ei(cfg.ei, cfg.ei_key, l, h), key


# ---------- main ----------

def main():
    cfg = parse_args()
    out = Path(cfg.out) if cfg.out else Path(cfg.ckpt) / "headnmi"
    if cfg.replot:
        z = np.load(out / "headnmi.npz")
        st = {k: z[k] for k in ("I", "H", "r", "E", "nmi", "adj", "dead")}
        l, h = int(z["l"]), int(z["h"])
        if "Eperm" in z:
            st["Eperm"], st["Eana"] = z["Eperm"], z["Eana"]
        title = f"{z['run']} step {int(z['step'])}, {int(z['N'])} tail rows"
    else:
        mm = np.load(cfg.cache, mmap_mode="r")
        n = mm.shape[0]
        lo = n - cfg.eval_rows

        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        from assignmap import read_hyper, temperature_at
        from autointerp import onto_acts_fn

        T = cfg.temperature
        step = cfg.step
        if T is None:
            hyper = read_hyper(cfg.ckpt)
            step = cfg.step or ocp.CheckpointManager(
                Path(cfg.ckpt).resolve(),
                checkpointers={'state': ocp.PyTreeCheckpointer(),
                               'spec': ocp.PyTreeCheckpointer()}).latest_step()
            assert hyper is not None, \
                "no env_config in log.jsonl; pass --temperature"
            T = temperature_at(hyper, step)
        acts, _, _, meta = onto_acts_fn(cfg.ckpt, cfg.step, T)
        l, h, k = meta["l"], meta["h"], meta["k"]
        print(f"step {meta['step']}: {l} layers x {h} heads x {k} entries, "
              f"T={T:g}, {cfg.eval_rows} tail rows")
        A = []
        for s in range(lo, n, cfg.b):
            X = jnp.asarray(np.asarray(mm[s:min(s + cfg.b, n)], np.float32))
            P = acts(X).reshape(X.shape[0], l * h, k)
            A.append(np.asarray(P.argmax(-1)).astype(np.int32))
        A = np.concatenate(A)                            # (N, l*h)
        st = head_nmi(A, k, cfg.null, cfg.n_perm, cfg.seed)
        st["Eana"] = analytic_null(st["r"], st["N"])
        st["Eperm"] = st["E"] if cfg.null == "perm" else \
            perm_null(A, k, 1, cfg.seed)
        st["dead"] = np.exp(st["H"]) < cfg.min_eff
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "headnmi.npz", A=A.astype(np.int16),
                 rows=np.arange(lo, n), I=st["I"], H=st["H"], r=st["r"],
                 E=st["E"], Eana=st["Eana"], Eperm=st["Eperm"],
                 nmi=st["nmi"], adj=st["adj"], dead=st["dead"], l=l, h=h,
                 k=k, N=st["N"], null=cfg.null, min_eff=cfg.min_eff,
                 run=Path(cfg.ckpt).name, step=meta["step"], temperature=T)
        title = f"{Path(cfg.ckpt).name} step {meta['step']}, {st['N']} tail rows"

    summarize(st, l, h, cfg.stat, cfg.top)
    ei, ei_key = load_ei_arg(cfg, l, h)
    if ei is not None:
        ok = np.isfinite(ei) & ~st["dead"][:, None] & ~st["dead"][None, :] \
            & ~np.eye(l * h, dtype=bool)
        print(f"{ei_key}: Spearman rho with {cfg.stat} over {ok.sum()} "
              f"pairs {spearman(st[cfg.stat][ok], ei[ok]):.3f}; by layer "
              "pair (rows = first-index layer):")
        layer = np.repeat(np.arange(l), h)
        for a in range(l):
            cells = []
            for b in range(l):
                m = ok & (layer[:, None] == a) & (layer[None, :] == b)
                cells.append(f"{spearman(st[cfg.stat][m], ei[m]):7.3f}"
                             if m.sum() > 2 else "      -")
            print("  " + " ".join(cells))
    path = out / (f"headnmi_{cfg.order}_{cfg.stat}" + (f"_{ei_key}" if ei is not None
                                            else "") + ".png")
    log = cfg.scale == "log"
    lim = (cfg.vmin if log else 0.0, cfg.vmax)
    draw(st, l, h, cfg.order, cfg.stat, lim, log, title, path, ei, ei_key,
         tuple(cfg.ei_lim) if cfg.ei_lim else None)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
