"""Entry x label enrichment dotplots and head x label NMI heatmaps.

`assignmap.py` shows which entries each held-out row lands on; this script
asks which entries are statistically about a known label. Every held-out
row (the tail --eval-rows of the cache, the sae.py holdout) is assigned,
per (layer, head), to its argmax entry. For each (layer, head, entry e,
label value v) cell, with N rows, n_e on e, n_v labelled v and x on both:

  test      x ~ Hypergeom(N, n_v, n_e) under independence of entry and
            label (the one-sided Fisher exact test of the 2x2 table).
              greater  p = P(X >= x)       over-representation
              less     p = P(X <= x)       under-representation
              two      p = min(1, 2 P(X on x's side of the mean, x
                       inclusive)): the doubled tail, with the side chosen
                       by x vs the expected count E = n_e n_v / N
            At x = 0 the enrichment tail is P(X >= 0) = 1, so an empty
            cell is never called over-represented; its depletion tail is
            P(X = 0), small only when E is large.
  filter    a cell is tested only if max(x, E) >= --min-count, so dead
            entries and rare label values with nothing observed and
            nothing expected are not tested (and do not inflate the BH m).
  FDR       Benjamini-Hochberg across every tested cell of every layer
            and head, computed in log space so tails far below 1e-308
            keep their order.
  effect    log2 of the 2x2 odds ratio (x)(N - n_e - n_v + x) /
            ((n_e - x)(n_v - x)) with 0.5 added to each of the four
            cells (Haldane-Anscombe), so empty cells give a finite value.

--soft replaces the argmax counts with summed assignment probabilities at
the run's temperature, rounded per cell before testing (margins are taken
from the rounded table so it stays a valid contingency table). Rows'
probability mass is not a set of independent draws, so soft p-values are
descriptive only.

Figures, per layer:
  enrich_l<L>.png   dotplot. Rows are the entries of the shown heads with
                    at least one tested cell among the shown columns
                    (--heads, else the --n-heads with the largest
                    null-subtracted NMI against --by at that layer),
                    separated by head; within a head, entries are grouped
                    by the shown label they are most enriched for, then by
                    usage. Columns are --labels, else the --n-labels values
                    with the most significantly enriched cells in the shown
                    heads (--cols enriched, the default: balanced labels
                    such as the mC4 languages have no informative
                    frequency order), else the most frequent (--cols
                    freq). Colour is log2 OR clipped to
                    [-or_clip, or_clip] on a diverging map, white at 0, the
                    same limits on every page; dot area is -log10 FDR,
                    capped at --fdr-cap. Significant cells (FDR < --alpha)
                    are filled; tested, non-significant cells are hollow;
                    untested cells are blank. --drop-empty removes entries
                    with no significant cell among the shown columns.
  nmi.png           every layer as a panel of label variables x heads:
                    NMI(head argmax, label) minus the mean of --nulls
                    label-shuffled NMIs, on a fixed [0, --nmi-vmax] scale.
                    A dot marks heads within 3 null standard deviations
                    of chance. NMI here is the arithmetic-mean-normalized
                    form of experiments/ste-arm/headlang.py, whose ceiling
                    is 2 min(H_head, H_label) / (H_head + H_label) < 1
                    when the entropies differ; each row's label states
                    H_label so the ceiling for a given head can be read off.
                    The label variables are every --by available for the
                    cache (lang for a .langs.npy sidecar; tokclass, pos and
                    doc for an .index.npy), or --nmi-by.

Labels (--by) are assignmap.py's: lang, tokclass, pos, doc.

  uv run python enrich.py --ckpt data/out/gpt2_l8/ste_h76 \\
      --cache data/activations/gpt2_l8.npy --by tokclass
  uv run python enrich.py --ckpt data/out/sonar/multilingual/resid_nc \\
      --by lang --layers 0 --heads 3 7

Writes <out>/cells.csv (every tested cell: layer, head, entry, label,
count, n_entry, n_label, expected, log2OR, p, FDR, and -log10 p and FDR,
which stay finite where p underflows to 0), <out>/nmi.csv and
<out>/enrich.npz (argmax entries, labels, soft table), from which --replot
redraws everything without the model (statistics are recomputed, so
--tail/--min-count/--soft/--alpha may change). The temperature defaults to
the run's schedule at the restored step and matters only for --soft.
NOTE: Ontologizer checkpoints restore on GPU JAX only.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from jaxtyping import Bool, Float, Int

import assignmap

BYS = ("lang", "tokclass", "pos", "doc")
LN10 = np.log(10.0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Ontologizer checkpoint dir")
    p.add_argument("--step", type=int, default=0, help="default: latest")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: the run's schedule at --step")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="held-out tail rows of the cache (sae.py's default)")
    p.add_argument("--by", choices=BYS, default=None,
                   help="label for the dotplot; default: lang if the "
                   "sidecar exists, else tokclass")
    p.add_argument("--nmi-by", choices=BYS, nargs="+", default=None,
                   help="label variables for nmi.png (default: all "
                   "available for the cache)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="label values to show, in order (default: the "
                   "--n-labels most frequent in the tail)")
    p.add_argument("--n-labels", type=int, default=12)
    p.add_argument("--cols", choices=["enriched", "freq"],
                   default="enriched",
                   help="without --labels, show the label values with the "
                   "most significant enrichments in the shown heads, or "
                   "the most frequent")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="layers to draw dotplots for (default: all)")
    p.add_argument("--heads", type=int, nargs="+", default=None,
                   help="heads to show in each dotplot (default: the "
                   "--n-heads most label-informative at that layer)")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--drop-empty", action="store_true",
                   help="drop entries with no significant shown cell")
    p.add_argument("--tail", choices=["two", "greater", "less"],
                   default="two")
    p.add_argument("--min-count", type=float, default=5.0,
                   help="test a cell only if max(observed, expected) >= this")
    p.add_argument("--alpha", type=float, default=0.05, help="FDR level")
    p.add_argument("--soft", action="store_true",
                   help="count summed probabilities instead of argmaxes")
    p.add_argument("--or-clip", type=float, default=4.0,
                   help="log2 OR colour limits are +-this")
    p.add_argument("--fdr-cap", type=float, default=50.0,
                   help="-log10 FDR at which dot size saturates")
    p.add_argument("--nulls", type=int, default=5,
                   help="label-shuffled draws for the NMI chance level")
    p.add_argument("--nmi-vmax", type=float, default=0.25,
                   help="top of nmi.png's fixed scale (values above "
                   "saturate)")
    p.add_argument("--replot", action="store_true",
                   help="redraw from <out>/enrich.npz without the model")
    p.add_argument("--b", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="default <ckpt>/enrich")
    return p.parse_args(argv)


# ---------- counting ----------

def count_table(A: Int[np.ndarray, "n m"], y: Int[np.ndarray, "n"], k: int,
                V: int) -> Int[np.ndarray, "m k V"]:
    """Per head (m = l*h flattened), the entry x label contingency table
    of argmax assignments A against integer labels y."""
    n, m = A.shape
    idx = (np.arange(m)[None] * k + A) * V + y[:, None]
    return np.bincount(idx.ravel(), minlength=m * k * V).reshape(m, k, V)


def soft_table(P: Float[np.ndarray, "n m k"], y: Int[np.ndarray, "n"],
               V: int) -> Float[np.ndarray, "m k V"]:
    """Summed assignment probabilities per (head, entry, label)."""
    n, m, k = P.shape
    Y = np.zeros((n, V), np.float64)
    Y[np.arange(n), y] = 1.0
    return (P.reshape(n, m * k).astype(np.float64).T @ Y).reshape(m, k, V)


# ---------- statistics ----------

class Margins(NamedTuple):
    x: Int[np.ndarray, "m k V"]       # rows on entry e with label v
    n_e: Int[np.ndarray, "m k 1"]     # rows on entry e
    n_v: Int[np.ndarray, "m 1 V"]     # rows with label v
    N: Int[np.ndarray, "m 1 1"]       # rows in the head's table
    E: Float[np.ndarray, "m k V"]     # n_e n_v / N


def margins(C: Float[np.ndarray, "m k V"]) -> Margins:
    """Integer table (rounded if soft) and its margins, all from the
    rounded table so every 2x2 is consistent."""
    x = np.rint(C).astype(np.int64)
    n_e = x.sum(2, keepdims=True)
    n_v = x.sum(1, keepdims=True)
    N = x.sum((1, 2), keepdims=True)
    E = n_e * n_v / np.maximum(N, 1)
    return Margins(x, n_e, n_v, N, E)


def log2_odds_ratio(M: Margins) -> Float[np.ndarray, "m k V"]:
    """log2 of the 2x2 odds ratio, Haldane-Anscombe +0.5 on every cell."""
    a = M.x + 0.5
    b = M.n_e - M.x + 0.5
    c = M.n_v - M.x + 0.5
    d = M.N - M.n_e - M.n_v + M.x + 0.5
    return np.log2(a) + np.log2(d) - np.log2(b) - np.log2(c)


def hyper_log10p(x: Int[np.ndarray, "c"], N: Int[np.ndarray, "c"],
                 n_v: Int[np.ndarray, "c"], n_e: Int[np.ndarray, "c"],
                 tail: str) -> Float[np.ndarray, "c"]:
    """log10 p of x under Hypergeom(N, n_v, n_e) for the given tail (see
    the module docstring). scipy's logsf/logcdf stay finite far below
    float underflow. 'two' evaluates only the tail on x's side of the
    mean; the other tail is then >= ~1/2 and the doubled p is ~1 either
    way."""
    from scipy.stats import hypergeom
    x, N, n_v, n_e = (np.asarray(a, np.int64) for a in (x, N, n_v, n_e))
    if len(x) == 0:
        return np.zeros(0)
    if tail == "greater":
        return hypergeom.logsf(x - 1, N, n_v, n_e) / LN10
    if tail == "less":
        return hypergeom.logcdf(x, N, n_v, n_e) / LN10
    up = x * N >= n_e * n_v
    lp = np.empty(len(x))
    lp[up] = hypergeom.logsf(x[up] - 1, N[up], n_v[up], n_e[up])
    lp[~up] = hypergeom.logcdf(x[~up], N[~up], n_v[~up], n_e[~up])
    return np.minimum(0.0, lp / LN10 + np.log10(2.0))


def bh_log10(lp: Float[np.ndarray, "c"]) -> Float[np.ndarray, "c"]:
    """Benjamini-Hochberg adjusted log10 p-values (the step-up q-values:
    q_(i) = min_{j >= i} p_(j) m / j, capped at 1), in input order."""
    m = len(lp)
    if m == 0:
        return lp
    order = np.argsort(lp, kind="stable")
    q = lp[order] + np.log10(m) - np.log10(np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 0.0)
    return out


class Enrichment(NamedTuple):
    M: Margins
    log2or: Float[np.ndarray, "m k V"]
    tested: Bool[np.ndarray, "m k V"]
    log10p: Float[np.ndarray, "m k V"]    # NaN where untested
    log10q: Float[np.ndarray, "m k V"]    # BH over all tested cells


def enrichment(C: Float[np.ndarray, "m k V"], tail: str = "two",
               min_count: float = 5.0) -> Enrichment:
    """Test every (head, entry, label) cell with max(x, E) >= min_count
    and BH-adjust across all of them."""
    M = margins(C)
    tested = np.maximum(M.x, M.E) >= min_count
    b = lambda a: np.broadcast_to(a, M.x.shape)[tested]
    lp = hyper_log10p(M.x[tested], b(M.N), b(M.n_v), b(M.n_e), tail)
    log10p = np.full(M.x.shape, np.nan)
    log10q = np.full(M.x.shape, np.nan)
    log10p[tested] = lp
    log10q[tested] = bh_log10(lp)
    return Enrichment(M, log2_odds_ratio(M), tested, log10p, log10q)


# ---------- NMI ----------

def nmi_table(C: Float[np.ndarray, "ka kb"]) -> Tuple[float, float, float, float]:
    """(NMI, I bits, H_a bits, H_b bits) of a contingency table, in
    headlang.nmi's convention: I over the arithmetic mean of the two
    entropies. Computed from the table so the null can reuse bincount."""
    P = C / C.sum()
    pa, pb = P.sum(1), P.sum(0)
    H = lambda p: float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    nz = P > 0
    I = float((P[nz] * np.log2(P[nz] / np.outer(pa, pb)[nz])).sum())
    Ha, Hb = H(pa), H(pb)
    den = 0.5 * (Ha + Hb)
    return (I / den if den > 0 else 0.0), I, Ha, Hb


def nmi_heads(A: Int[np.ndarray, "n m"], y: Int[np.ndarray, "n"], k: int,
              V: int, nulls: int, seed: int
              ) -> Dict[str, Float[np.ndarray, "m"]]:
    """Per head: NMI against y, the mean and sd of NMI against `nulls`
    permutations of y, z = (NMI - mean) / sd, the head's entropy, the
    attainable ceiling 2 min(H_h, H_y) / (H_h + H_y), and I in bits."""
    m = A.shape[1]
    stats = np.array([nmi_table(T) for T in count_table(A, y, k, V)])
    rng = np.random.default_rng(seed)
    null = np.stack([[nmi_table(T)[0] for T in
                      count_table(A, y[rng.permutation(len(y))], k, V)]
                     for _ in range(nulls)])  # (nulls, m)
    mu, sd = null.mean(0), null.std(0, ddof=1) if nulls > 1 else np.zeros(m)
    Hh, Hy = stats[:, 2], stats[:, 3]
    return {"nmi": stats[:, 0], "bits": stats[:, 1], "H_head": Hh,
            "H_label": Hy, "null": mu, "null_sd": sd,
            "z": (stats[:, 0] - mu) / np.maximum(sd, 1e-12),
            "ceiling": 2 * np.minimum(Hh, Hy) / np.maximum(Hh + Hy, 1e-12)}


# ---------- dotplot layout ----------

def shown_labels(values: Sequence[str], counts: Int[np.ndarray, "V"],
                 show: Optional[Sequence[str]], n_labels: int) -> List[int]:
    """Indices into `values` of the shown columns: `show` in its order,
    else the n_labels most frequent (ties by name), as in assignmap."""
    labels = np.repeat(np.asarray(values), counts)
    _, names = assignmap.pick_rows(labels, show, n_labels, 0, 0)
    pos = {v: i for i, v in enumerate(values)}
    return [pos[v] for v in names]


def enriched_labels(R: Enrichment, rows: Sequence[int],
                    counts: Int[np.ndarray, "V"], n_labels: int, alpha: float
                    ) -> List[int]:
    """The n_labels label indices with the largest total -log10 FDR over
    their significantly enriched cells in the given flattened heads (ties
    by frequency, then index), in descending order of it. Weighting by
    strength rather than counting cells keeps a label that one entry
    captures outright ahead of labels mildly over-represented in the
    entries it vacated."""
    sig = (R.log10q[rows] < np.log10(alpha)) & (R.log2or[rows] > 0)
    hits = np.where(sig, -R.log10q[rows], 0.0).sum((0, 1))
    order = np.lexsort((np.arange(len(hits)), -counts, -hits))
    return order[:n_labels].tolist()


def entry_rows(R: Enrichment, head: int, cols: Sequence[int], alpha: float,
               drop_empty: bool) -> Int[np.ndarray, "r"]:
    """The entries of one (flattened) head to draw, in display order:
    entries with a tested cell among the shown columns, grouped by the shown column they are most significantly
    enriched for (column order), each group by usage; entries with no
    significant enrichment among the shown columns follow, by usage."""
    n_e = R.M.n_e[head, :, 0]
    live = np.flatnonzero(R.tested[head][:, cols].any(1))
    sig = (R.log10q[head][:, cols] < np.log10(alpha)) \
        & (R.log2or[head][:, cols] > 0)
    score = np.where(sig, R.log2or[head][:, cols], -np.inf)
    best = np.where(sig.any(1), score.argmax(1), len(cols))
    if drop_empty:
        anysig = (R.log10q[head][:, cols] < np.log10(alpha)).any(1)
        live = live[anysig[live]]
    return live[np.lexsort((-n_e[live], best[live]))]


def pick_heads(nmi_l: Dict[str, Float[np.ndarray, "h"]], n_heads: int
               ) -> List[int]:
    """The n_heads heads with the largest null-subtracted NMI."""
    gain = nmi_l["nmi"] - nmi_l["null"]
    return sorted(np.argsort(-gain, kind="stable")[:n_heads].tolist())


# ---------- figures ----------

def size_of(nlq: Float[np.ndarray, "..."], cap: float, smax: float = 90.0
            ) -> Float[np.ndarray, "..."]:
    """Dot area for -log10 FDR, linear up to `cap`, with a visible floor."""
    return 4.0 + (smax - 4.0) * np.clip(nlq / cap, 0, 1)


def draw_dotplot(R: Enrichment, L: int, h: int, heads: Sequence[int],
                 cols: Sequence[int], values: Sequence[str], title: str,
                 path, alpha: float = 0.05, or_clip: float = 4.0,
                 fdr_cap: float = 50.0, drop_empty: bool = False):
    """One layer's entry x label dotplot over the given heads."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    ys, yl, groups = [], [], []
    for hd in heads:
        f = L * h + hd
        es = entry_rows(R, f, cols, alpha, drop_empty)
        groups.append((hd, len(ys), len(es)))
        for e in es:
            ys.append((f, e))
            yl.append(f"e{e}  n={int(R.M.n_e[f, e, 0])}")
    nr, nc = max(len(ys), 1), len(cols)
    H = max(4.2, 1.6 + 0.16 * nr)
    fig, (ax, lax) = plt.subplots(
        1, 2, figsize=(2.6 + 0.42 * nc + 1.4, H),
        gridspec_kw={"width_ratios": [0.42 * nc, 1.4], "wspace": 0.05})
    norm = Normalize(-or_clip, or_clip)
    cmap = plt.get_cmap("RdBu_r")
    sig_level = np.log10(alpha)
    for r, (f, e) in enumerate(ys):
        for j, c in enumerate(cols):
            if not R.tested[f, e, c]:
                continue
            nlq = -R.log10q[f, e, c]
            col = cmap(norm(np.clip(R.log2or[f, e, c], -or_clip, or_clip)))
            s = size_of(nlq, fdr_cap)
            if R.log10q[f, e, c] < sig_level:
                ax.scatter(j, r, s=s, c=[col], edgecolors="0.25",
                           linewidths=0.3)
            else:
                ax.scatter(j, r, s=s, facecolors="none", edgecolors="0.75",
                           linewidths=0.5)
    # head names sit left of the entry tick labels, offset by their width
    from matplotlib.transforms import offset_copy
    wl = 3.4 * max((len(s) for s in yl), default=0) + 10  # points, at 6pt
    tr = offset_copy(ax.transAxes, fig=fig, x=-wl, units="points")
    for hd, start, cnt in groups:
        if start > 0:
            ax.axhline(start - 0.5, color="0.2", lw=0.8)
        ax.text(0, 1 - (start + cnt / 2) / nr, f"h{hd}", transform=tr,
                ha="right", va="center", rotation=90, fontsize=9,
                fontweight="bold")
    ax.set_yticks(np.arange(len(ys)), yl, fontsize=6)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_xticks(np.arange(nc), [values[c] for c in cols], rotation=60,
                  ha="right", fontsize=8)
    ax.set_xlim(-0.6, nc - 0.4)
    ax.grid(True, color="0.93", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9)

    # colourbar and size legend share a fixed-size side column, so they
    # read the same however many entry rows the page has
    lax.axis("off")
    hb = 1.6 / (0.8 * H)  # colourbar 1.6 in tall, in lax's height units
    cax = lax.inset_axes([0.05, 1 - hb, 0.12, hb])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, extend="both")
    cb.set_label("log2 OR (Haldane +0.5)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ticks = [t for t in (2, 5, 10, 20, 50, 100) if t < fdr_cap] + [fdr_cap]
    handles = [lax.scatter([], [], s=size_of(t, fdr_cap), c="0.5",
                           edgecolors="0.25", linewidths=0.3) for t in ticks]
    handles.append(lax.scatter([], [], s=size_of(1, fdr_cap),
                               facecolors="none", edgecolors="0.75"))
    names = [f"{t:g}" for t in ticks[:-1]] + [f"≥{fdr_cap:g}",
                                              f"FDR ≥ {alpha:g}"]
    lax.legend(handles, names, title="−log10 FDR", loc="upper left",
               bbox_to_anchor=(0.0, 1 - hb - 0.4 / H), fontsize=7,
               title_fontsize=7, frameon=False, labelspacing=1.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_nmi(stats: Dict[str, Dict[str, Float[np.ndarray, "m"]]], l: int,
             h: int, title: str, path, vmax: float = 0.5):
    """Every layer's label-variable x head grid of null-subtracted NMI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bys = list(stats)
    fig, axes = plt.subplots(l, 1, figsize=(2.5 + 0.28 * h,
                                            0.6 + l * (0.5 + 0.3 * len(bys))),
                             squeeze=False, sharex=True)
    for L, ax in enumerate(axes[:, 0]):
        sl = slice(L * h, (L + 1) * h)
        G = np.stack([stats[b]["nmi"][sl] - stats[b]["null"][sl] for b in bys])
        Z = np.stack([stats[b]["z"][sl] for b in bys])
        im = ax.imshow(G, aspect="auto", interpolation="nearest",
                       cmap="Blues", vmin=0.0, vmax=vmax)
        for i, j in zip(*np.nonzero(Z < 3)):
            ax.plot(j, i, ".", color="0.45", ms=3)
        ax.set_yticks(np.arange(len(bys)),
                      [f"{b} (H={stats[b]['H_label'][0]:.1f}b)" for b in bys],
                      fontsize=7)
        ax.set_ylabel(f"layer {L}", fontsize=8)
        for j in range(1, h):
            ax.axvline(j - 0.5, color="white", lw=0.4)
    axes[-1, 0].set_xticks(np.arange(h), [str(j) for j in range(h)],
                           fontsize=6)
    axes[-1, 0].set_xlabel("head  (· = within 3 null sd of chance)")
    axes[0, 0].set_title(title, fontsize=9)
    cb = fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.02, pad=0.01,
                      extend="max")
    cb.set_label("NMI − shuffled-label NMI")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- outputs ----------

def write_cells(R: Enrichment, h: int, values: Sequence[str], path):
    """Every tested cell, most significant first."""
    f, e, v = np.nonzero(R.tested)
    order = np.lexsort((-R.log2or[f, e, v], R.log10q[f, e, v]))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "head", "entry", "label", "count", "n_entry",
                    "n_label", "expected", "log2OR", "p", "FDR",
                    "neglog10_p", "neglog10_FDR"])
        for i in order:
            a, b, c = f[i], e[i], v[i]
            w.writerow([a // h, a % h, b, values[c], int(R.M.x[a, b, c]),
                        int(R.M.n_e[a, b, 0]), int(R.M.n_v[a, 0, c]),
                        f"{R.M.E[a, b, c]:.3f}", f"{R.log2or[a, b, c]:.4f}",
                        f"{10 ** R.log10p[a, b, c]:.4g}",
                        f"{10 ** R.log10q[a, b, c]:.4g}",
                        f"{-R.log10p[a, b, c]:.3f}",
                        f"{-R.log10q[a, b, c]:.3f}"])


def write_nmi(stats, h: int, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        keys = ["nmi", "null", "null_sd", "z", "ceiling", "bits", "H_head",
                "H_label"]
        w.writerow(["by", "layer", "head"] + keys)
        for b, s in stats.items():
            for f in range(len(s["nmi"])):
                w.writerow([b, f // h, f % h] + [f"{s[x][f]:.5g}" for x in keys])


def analyze(cfg, A: Int[np.ndarray, "n l h"], labels: Dict[str, np.ndarray],
            S: Optional[Float[np.ndarray, "l h k V"]], k: int, step: int,
            out: Path):
    """Statistics, CSVs and figures from the stored assignments."""
    n, l, h = A.shape
    Af = A.reshape(n, l * h)
    stats = {}
    for b, lab in labels.items():
        vals, y = np.unique(lab, return_inverse=True)
        stats[b] = nmi_heads(Af, y, k, len(vals), cfg.nulls, cfg.seed)
    write_nmi(stats, h, out / "nmi.csv")
    run = Path(cfg.ckpt).name
    draw_nmi(stats, l, h, f"{run} step {step}: head argmax vs label "
             f"(n={n})", out / "nmi.png", cfg.nmi_vmax)
    print(f"-> {out / 'nmi.png'}")

    values, y, counts = np.unique(labels[cfg.by], return_inverse=True,
                                  return_counts=True)
    if cfg.soft:
        assert S is not None, "no soft table stored; rerun without --replot"
        C = S.reshape(l * h, k, len(values))
    else:
        C = count_table(Af, y, k, len(values))
    R = enrichment(C, cfg.tail, cfg.min_count)
    write_cells(R, h, values, out / "cells.csv")
    sig = R.tested & (R.log10q < np.log10(cfg.alpha))
    print(f"{R.tested.sum()} cells tested, {sig.sum()} at FDR < {cfg.alpha}"
          f" ({(sig & (R.log2or > 0)).sum()} enriched) -> {out / 'cells.csv'}")

    for L in (cfg.layers if cfg.layers is not None else range(l)):
        heads = cfg.heads if cfg.heads is not None else pick_heads(
            {x: s[L * h:(L + 1) * h] for x, s in stats[cfg.by].items()},
            cfg.n_heads)
        if cfg.labels is None and cfg.cols == "enriched":
            cols = enriched_labels(R, [L * h + hd for hd in heads], counts,
                                   cfg.n_labels, cfg.alpha)
        else:
            cols = shown_labels(values, counts, cfg.labels, cfg.n_labels)
        mode = "soft" if cfg.soft else "argmax"
        path = out / f"enrich_l{L}.png"
        draw_dotplot(R, L, h, heads, cols, values,
                     f"{run} step {step} layer {L}: entry x {cfg.by} "
                     f"({mode}, {cfg.tail}-sided, n={n})", path,
                     cfg.alpha, cfg.or_clip, cfg.fdr_cap, cfg.drop_empty)
        print(f"-> {path}  (heads {heads})")


def available_bys(cache: Path) -> List[str]:
    side = cache.with_name(cache.name.replace(".npy", ".langs.npy"))
    index = cache.with_name(cache.name.replace(".npy", ".index.npy"))
    return (["lang"] if side.exists() else []) + \
        (["tokclass", "pos", "doc"] if index.exists() else [])


def main():
    cfg = parse_args()
    out = Path(cfg.out) if cfg.out else Path(cfg.ckpt) / "enrich"
    if cfg.replot:
        z = np.load(out / "enrich.npz")
        cfg.ckpt = str(z["run"])
        bys = [str(b) for b in z["bys"]]
        cfg.by = cfg.by or str(z["by"])
        labels = {b: z[f"labels_{b}"] for b in (cfg.nmi_by or bys)}
        labels.setdefault(cfg.by, z[f"labels_{cfg.by}"])
        S = z["S"] if str(z["by"]) == cfg.by else None
        analyze(cfg, z["A"], labels, S, int(z["k"]), int(z["step"]), out)
        return
    cache = Path(cfg.cache)
    bys = available_bys(cache)
    if cfg.by is None:
        cfg.by = "lang" if "lang" in bys else "tokclass"
    names = list(dict.fromkeys([cfg.by] + (cfg.nmi_by or bys)))

    mm = np.load(cache, mmap_mode="r")
    n = mm.shape[0]
    lo = n - cfg.eval_rows
    labels = {}
    for b in names:
        labels[b] = assignmap.tail_labels(
            argparse.Namespace(cache=cfg.cache, by=b), n, lo)
    values, y = np.unique(labels[cfg.by], return_inverse=True)
    print(f"{cfg.eval_rows} tail rows; dotplot by {cfg.by} "
          f"({len(values)} values); NMI by {names}")

    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from autointerp import onto_acts_fn

    T = cfg.temperature
    if T is None:
        hyper = assignmap.read_hyper(cfg.ckpt)
        step = cfg.step or ocp.CheckpointManager(
            Path(cfg.ckpt).resolve(),
            checkpointers={'state': ocp.PyTreeCheckpointer(),
                           'spec': ocp.PyTreeCheckpointer()}).latest_step()
        assert hyper is not None, \
            "no env_config in log.jsonl; pass --temperature"
        T = assignmap.temperature_at(hyper, step)
    acts, _, _, meta = onto_acts_fn(cfg.ckpt, cfg.step, T)
    l, h, k = meta["l"], meta["h"], meta["k"]
    print(f"step {meta['step']}: {l} layers x {h} heads x {k} entries, T={T:g}")

    A = np.empty((cfg.eval_rows, l, h), np.int16)
    S = np.zeros((l * h, k, len(values)))
    for i in range(0, cfg.eval_rows, cfg.b):
        X = np.asarray(mm[lo + i:lo + i + cfg.b], dtype=np.float32)
        P = np.asarray(acts(jnp.asarray(X))).reshape(len(X), l * h, k)
        A[i:i + len(X)] = P.argmax(-1).reshape(len(X), l, h)
        S += soft_table(P, y[i:i + len(X)], len(values))

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "enrich.npz", A=A, S=S.reshape(l, h, k, len(values)),
             by=cfg.by, bys=np.asarray(names), run=Path(cfg.ckpt).name,
             step=meta["step"], k=k, temperature=T, rows_lo=lo,
             **{f"labels_{b}": labels[b] for b in names})
    analyze(cfg, A, labels, S, k, meta["step"], out)


if __name__ == "__main__":
    main()
