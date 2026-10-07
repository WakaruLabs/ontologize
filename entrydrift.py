"""Head-grouped entry x condition heatmaps: usage and label drift.

Rows are every shown (layer, head, entry) of an Ontologizer, grouped by
head (thin separators) and layer (thick separators). Within a head,
entries are ordered by usage at the reference column, most used first,
and that order is held across columns so a row is one entry throughout.

Columns are conditions:
  steps  (--ckpt RUN [--steps S ...])  checkpoint steps of one run. Rows
         are index-aligned by construction. Without --steps, up to
         --max-cols saved steps spaced evenly in step value (first and
         last always included).
  runs   (--runs RUN[:STEP] ...)  several runs sharing an architecture
         and index-aligned, e.g. fine-tunes from a common init
         (ste_h76_ft_*). Alignment is checked: per head, the entries are
         Hungarian-matched to the reference column on decoded-direction
         cosine, and if the median over heads of the fraction of entries
         matched to themselves is below --align-min the script refuses.
         Independent seeds permute heads as well as entries, which
         within-head matching cannot undo; compare those with
         experiments/gpt2/seedstab.py or splitting.py instead.
The reference column is --ref (index, default the last).

Cell metrics (--metrics, one panel each, drawn side by side):
  usage  p_bar, the mean assignment probability over the tail --eval-rows
         of the cache (the sae.py holdout), at that column's training
         temperature (assignmap.temperature_at on the run's log.jsonl,
         unless --temperature). Sequential, white at 0, fixed [0, --vmax]
         (default 4/k; values above saturate).
  drift  cos(u_t, u_ref) between the entry's decoded direction at this
         column and at the reference column, same index. A decoded
         direction is decode(one-hot code) - decode(0): the base
         dictionary entry through the decoder, offset removed
         (autointerp.onto_acts_fn's `embed`; no router gain or fibers).
         Diverging, fixed [-1, 1], white at 0.
  match  as drift, but against the reference entry it is Hungarian-
         matched to within the same head: high match with low drift is an
         entry that moved to another index rather than changed meaning.
An entry that stops being used stops receiving gradient, so its direction
freezes and its drift reads ~1 without meaning anything. Drift and match
cells are therefore drawn grey for entries dead at the reference column
(p_bar < 1/(4k)) unless --no-mask; read the rest beside usage, since an
entry dead at an earlier column has a frozen direction there too.

A strip above the panels shows each column's temperature (log scale).
The reference column's label is red. --mark-step S draws a boundary after
the last column with step <= S and shades the strip up to it (default in steps mode: the run's
anneal_steps, when it anneals); --mark-col J does the same after column J.

Pages hold --max-heads (layer, head) pairs each, in layer-then-head
order; --layers/--heads restrict the selection.

  uv run python entrydrift.py --ckpt data/out/sonar/multilingual/resid_nc
  uv run python entrydrift.py --cache data/sonar_embeddings/mc4_4M.npy \\
      --runs data/out/sonar/multilingual/ste_h76 \\
             data/out/sonar/multilingual/ste_h76_ft_layer0 \\
             data/out/sonar/multilingual/ste_h76_ft_layer2 --ref 0

Writes <out>/drift_p<j>.png per page and drift.npz (every metric for all
l*h*k entries, the column labels, steps and temperatures, the Hungarian
permutations) so the figure can be redrawn without the model (--replot,
which honours --layers/--heads/--max-heads/--metrics/--vmax/--mark-*).
NOTE: Ontologizer checkpoints restore on GPU JAX only.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from jaxtyping import Bool, Float, Int

METRICS = ("usage", "drift", "match")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="one run: columns are its steps")
    src.add_argument("--runs", nargs="+",
                     help="RUN[:STEP] per column (index-aligned runs)")
    p.add_argument("--steps", type=int, nargs="+", default=None,
                   help="steps mode: the steps to show (default: spaced)")
    p.add_argument("--max-cols", type=int, default=16)
    p.add_argument("--ref", type=int, default=-1,
                   help="reference column index (default: the last)")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: each run's schedule at the column's step")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="held-out tail rows of the cache (sae.py's default)")
    p.add_argument("--metrics", nargs="+", choices=METRICS,
                   default=["usage", "drift"])
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--heads", type=int, nargs="+", default=None)
    p.add_argument("--max-heads", type=int, default=16,
                   help="(layer, head) pairs per page")
    p.add_argument("--vmax", type=float, default=None,
                   help="top of the usage scale (default 4/k)")
    p.add_argument("--mark-step", type=int, default=None)
    p.add_argument("--mark-col", type=int, default=None)
    p.add_argument("--no-mask", action="store_true",
                   help="draw drift/match for entries dead at the reference")
    p.add_argument("--align-min", type=float, default=0.5,
                   help="runs mode: refuse below this median per-head "
                   "self-match fraction")
    p.add_argument("--replot", action="store_true",
                   help="redraw from <out>/drift.npz without the model")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=512,
                   help="one-hot codes decoded per call")
    p.add_argument("--out", default=None,
                   help="default <ckpt>/entrydrift, or <first run>/entrydrift")
    return p.parse_args(argv)


# ---------- columns ----------

def spaced_steps(steps: Sequence[int], n: int) -> List[int]:
    """Up to n of the sorted `steps`, each nearest an evenly spaced target
    in step value (so a dense burst of saves is not oversampled), first
    and last always kept."""
    s = np.asarray(sorted(steps))
    if len(s) <= n:
        return s.tolist()
    picked = []
    for t in np.linspace(s[0], s[-1], n):
        i = int(np.abs(s - t).argmin())
        if s[i] not in picked:
            picked.append(int(s[i]))
    return sorted(picked)


def saved_steps(ckpt) -> List[int]:
    return sorted(int(p.name) for p in Path(ckpt).iterdir()
                  if p.is_dir() and p.name.isdigit())


def parse_run(spec: str) -> Tuple[str, int]:
    """RUN[:STEP] -> (run, step); step 0 means latest."""
    run, _, step = spec.rpartition(":")
    if run and step.isdigit():
        return run, int(step)
    return spec, 0


def step_label(s: int) -> str:
    return f"{s // 1000}k" if s >= 1000 and s % 1000 == 0 else str(s)


def mark_boundary(steps: Sequence[int], mark_step: Optional[int],
                  mark_col: Optional[int]) -> Optional[int]:
    """Number of columns before the mark boundary, or None: the columns
    with step <= mark_step, or columns 0..mark_col."""
    if mark_col is not None:
        return mark_col + 1
    if mark_step is None:
        return None
    n = int((np.asarray(steps) <= mark_step).sum())
    return n if 0 < n < len(steps) else None


# ---------- metrics ----------

def entry_dirs(embed: Callable, l: int, h: int, k: int, d: int,
               chunk: int = 512) -> Float[np.ndarray, "l h k d"]:
    """Decoded direction of every entry, decode(one-hot) - decode(0)
    (`autointerp.entry_directions`), in (layer, head, entry) layout."""
    from autointerp import entry_directions
    dirs, _ = entry_directions(embed, l, h, k, chunk)
    return dirs.reshape(l, h, k, d)


def unit(U: Float[np.ndarray, "... d"], eps: float = 1e-9) -> np.ndarray:
    return U / (np.linalg.norm(U, axis=-1, keepdims=True) + eps)


def drift_cos(U: Float[np.ndarray, "l h k d"],
              U_ref: Float[np.ndarray, "l h k d"]
              ) -> Float[np.ndarray, "l h k"]:
    """cos(u, u_ref) entry by entry, same index."""
    return (unit(U) * unit(U_ref)).sum(-1)


def head_match(U: Float[np.ndarray, "l h k d"],
               U_ref: Float[np.ndarray, "l h k d"]
               ) -> Tuple[Int[np.ndarray, "l h k"], Float[np.ndarray, "l h k"]]:
    """Per head, the Hungarian matching of this column's entries to the
    reference's on decoded-direction cosine: (perm, cos) where reference
    entry j is matched to this column's entry perm[..., j] with cosine
    cos[..., j], so both are indexed like the reference rows."""
    from scipy.optimize import linear_sum_assignment
    C = np.einsum("lhjd,lhid->lhji", unit(U_ref), unit(U))  # (l, h, k_ref, k)
    l, h, k, _ = C.shape
    perm = np.empty((l, h, k), int)
    cos = np.empty((l, h, k))
    for a in range(l):
        for b in range(h):
            r, c = linear_sum_assignment(-C[a, b])
            perm[a, b, r] = c
            cos[a, b, r] = C[a, b, r, c]
    return perm, cos


def self_match(perm: Int[np.ndarray, "l h k"]) -> Float[np.ndarray, "l h"]:
    """Per head, the fraction of entries matched to their own index."""
    return (perm == np.arange(perm.shape[-1])).mean(-1)


# ---------- rows ----------

def select_heads(l: int, h: int, layers: Optional[Sequence[int]],
                 heads: Optional[Sequence[int]]) -> List[Tuple[int, int]]:
    """(layer, head) pairs in layer-then-head order."""
    return [(a, b) for a in (layers if layers is not None else range(l))
            for b in (heads if heads is not None else range(h))]


def row_layout(pairs: Sequence[Tuple[int, int]],
               usage_ref: Float[np.ndarray, "l h k"]
               ) -> Tuple[Int[np.ndarray, "r 3"], List[float], List[float]]:
    """Rows (layer, head, entry) for the given heads, each head's entries
    by reference usage, most used first (stable); and the row positions of
    head and layer separators (between consecutive heads, and where the
    layer changes)."""
    k = usage_ref.shape[-1]
    rows, head_sep, layer_sep = [], [], []
    for i, (a, b) in enumerate(pairs):
        if i:
            (layer_sep if pairs[i - 1][0] != a else head_sep).append(i * k - 0.5)
        order = np.argsort(-usage_ref[a, b], kind="stable")
        rows.extend((a, b, e) for e in order)
    return np.asarray(rows, int).reshape(-1, 3), head_sep, layer_sep


def gather(V: Float[np.ndarray, "c l h k"],
           rows: Int[np.ndarray, "r 3"]) -> Float[np.ndarray, "r c"]:
    return V[:, rows[:, 0], rows[:, 1], rows[:, 2]].T


# ---------- figure ----------

CMAPS = {"usage": "Blues", "drift": "RdBu_r", "match": "RdBu_r"}
CLABEL = {"usage": "p̄ (tail mean assignment probability)",
          "drift": "cos(u_t, u_ref), same index (grey: dead at ref)",
          "match": "cos(u_t, u_ref), Hungarian-matched in head (grey: dead "
                   "at ref)"}


def cmap_bad(name: str):
    """`name` with NaN cells drawn mid-grey."""
    import matplotlib
    return matplotlib.colormaps[name].with_extremes(bad="0.6")


def draw_page(panels: dict, rows: Int[np.ndarray, "r 3"],
              head_sep: Sequence[float], layer_sep: Sequence[float],
              cols: Sequence[str], temps: Float[np.ndarray, "c"], ref: int,
              boundary: Optional[int], title: str, path, vmax: float):
    """One page: a panel per metric (rows x columns, already gathered),
    sharing rows; a temperature strip over each panel. Usage is on
    [0, vmax], cosines on [-1, 1]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(panels)
    r, c = next(iter(panels.values())).shape
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(rows[:, :2], axis=0),
                                            axis=1)) + 1]
    k = r // len(starts)
    fig, axes = plt.subplots(
        2, len(names), figsize=(len(names) * (2.2 + 0.32 * c), 2.0 + r / 70),
        squeeze=False, sharex="col",
        gridspec_kw={"height_ratios": [1, max(6, r / 70)], "hspace": 0.03,
                     "wspace": 0.25})
    x = np.arange(c)
    for j, name in enumerate(names):
        ax_t, ax = axes[0, j], axes[1, j]
        ax_t.plot(x, temps, "o-", color="0.3", ms=3, lw=1)
        if np.all(temps > 0) and temps.max() / temps.min() > 10:
            ax_t.set_yscale("log")
        ax_t.minorticks_off()
        ticks = sorted({float(temps.min()), float(temps.max())})
        ax_t.set_yticks(ticks, [f"{t:.2g}" for t in ticks])
        ax_t.set_ylabel("T", rotation=0, labelpad=8, fontsize=8)
        ax_t.tick_params(labelsize=6, labelbottom=False)
        if name == "usage":
            im = ax.imshow(panels[name], aspect="auto", interpolation="nearest",
                           cmap=CMAPS[name], vmin=0.0, vmax=vmax)
            ext = "max"
        else:
            im = ax.imshow(panels[name], aspect="auto", interpolation="nearest",
                           cmap=cmap_bad(CMAPS[name]), vmin=-1.0, vmax=1.0)
            ext = "neither"
        for y in head_sep:
            ax.axhline(y, color="0.55", lw=0.3)
        for y in layer_sep:
            ax.axhline(y, color="k", lw=1.6)
        if boundary is not None:
            # shade only the strip: a tint over the map would read as data
            ax_t.axvspan(-0.5, boundary - 0.5, color="goldenrod", alpha=0.25,
                         lw=0)
            for a in (ax_t, ax):
                a.axvline(boundary - 0.5, color="goldenrod", lw=1.6)
        ax.set_xticks(x, cols, rotation=90, fontsize=7)
        ax.get_xticklabels()[ref].set_color("crimson")
        ax.set_xlim(-0.5, c - 0.5)
        if j == 0:
            ax.set_yticks(starts + k / 2 - 0.5,
                          [f"L{rows[s, 0]} h{rows[s, 1]}" for s in starts],
                          fontsize=6)
        else:
            ax.set_yticks([])
        ax_t.set_title(name, fontsize=9)
        cb = fig.colorbar(im, ax=[ax_t, ax], fraction=0.05, pad=0.02,
                          extend=ext, aspect=40)
        cb.set_label(CLABEL[name], fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle(title + f"\n(entries by usage at the reference column "
                 f"{cols[ref]}; thin lines = heads, thick = layers)",
                 fontsize=9, y=1.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_all(cfg, Z: dict, out: Path) -> List[Path]:
    """Every page for the metrics in cfg.metrics; prints a per-layer
    summary of drift among used entries."""
    usage = Z["usage"]
    _, l, h, k = usage.shape
    ref = int(Z["ref"])
    cols = [str(s) for s in Z["cols"]]
    vmax = cfg.vmax if cfg.vmax is not None else 4.0 / k
    mark_step = cfg.mark_step
    if mark_step is None and cfg.mark_col is None and int(Z["mark_step"]) >= 0:
        mark_step = int(Z["mark_step"])
    boundary = mark_boundary(Z["steps"], mark_step, cfg.mark_col)
    used = usage[ref] > 1.0 / (4 * k)
    for L in range(l):
        u = used[L]
        print(f"layer {L}: {u.sum()}/{h * k} entries used at {cols[ref]}; "
              "median drift cos over used, by column: " +
              " ".join(f"{np.median(Z['drift'][j, L][u]):.2f}" if u.any()
                       else "nan" for j in range(len(cols))))
    pairs = select_heads(l, h, cfg.layers, cfg.heads)
    pages = [pairs[i:i + cfg.max_heads]
             for i in range(0, len(pairs), cfg.max_heads)]
    paths = []
    for j, page in enumerate(pages):
        rows, hs, ls = row_layout(page, usage[ref])
        live = used[rows[:, 0], rows[:, 1], rows[:, 2]][:, None]
        panels = {m: gather(Z[m], rows) if m == "usage" or cfg.no_mask
                  else np.where(live, gather(Z[m], rows), np.nan)
                  for m in cfg.metrics}
        path = out / f"drift_p{j}.png"
        draw_page(panels, rows, hs, ls, cols, np.asarray(Z["temps"]), ref,
                  boundary, f"{Z['title']}  page {j + 1}/{len(pages)}",
                  path, vmax)
        print(f"-> {path}")
        paths.append(path)
    return paths


# ---------- main ----------

def main():
    cfg = parse_args()
    first = cfg.ckpt or parse_run(cfg.runs[0])[0]
    out = Path(cfg.out) if cfg.out else Path(first) / "entrydrift"
    if cfg.replot:
        Z = dict(np.load(out / "drift.npz"))
        missing = [m for m in cfg.metrics if m not in Z]
        assert not missing, f"drift.npz lacks {missing}"
        draw_all(cfg, Z, out)
        return

    from assignmap import read_hyper, temperature_at
    if cfg.ckpt:
        steps = cfg.steps or spaced_steps(saved_steps(cfg.ckpt), cfg.max_cols)
        columns = [(cfg.ckpt, s) for s in steps]
        labels = [step_label(s) for s in steps]
        hyper = read_hyper(cfg.ckpt)
        if (cfg.mark_step is None and cfg.mark_col is None and hyper
                and hyper.get("anneal_steps")):
            cfg.mark_step = hyper["anneal_steps"]
        title = f"{Path(cfg.ckpt).name}: entry drift over steps"
    else:
        columns = [parse_run(s) for s in cfg.runs]
        columns = [(r, s or max(saved_steps(r))) for r, s in columns]
        labels = [f"{Path(r).name}:{step_label(s)}" for r, s in columns]
        title = "entry drift across runs"
    ref = cfg.ref % len(columns)
    temps = []
    for run, s in columns:
        if cfg.temperature is not None:
            temps.append(cfg.temperature)
            continue
        hy = read_hyper(run)
        assert hy is not None, f"no env_config in {run}/log.jsonl; " \
            "pass --temperature"
        temps.append(temperature_at(hy, s))

    import jax
    import jax.numpy as jnp
    from autointerp import onto_acts_fn

    mm = np.load(cfg.cache, mmap_mode="r")
    X = np.asarray(mm[mm.shape[0] - cfg.eval_rows:], dtype=np.float32)
    d = mm.shape[1]

    def measure(j):
        run, s = columns[j]
        acts, embed, F, meta = onto_acts_fn(run, s, temps[j])
        l, h, k = meta["l"], meta["h"], meta["k"]
        tot = np.zeros(F)
        for i in range(0, len(X), cfg.b):
            tot += np.asarray(acts(jnp.asarray(X[i:i + cfg.b]))).sum(0)
        U = entry_dirs(embed, l, h, k, d, cfg.chunk)
        jax.clear_caches()
        print(f"  {labels[j]}: T={temps[j]:g}")
        return (tot / len(X)).reshape(l, h, k), U, (l, h, k)

    order = [ref] + [j for j in range(len(columns)) if j != ref]
    res = {}
    U_ref = None
    for j in order:
        usage, U, shape = measure(j)
        if U_ref is None:
            U_ref, shape_ref = U, shape
        assert shape == shape_ref, \
            f"{labels[j]} is {shape}, reference is {shape_ref}"
        perm, mcos = head_match(U, U_ref)
        frac = self_match(perm)
        print(f"  {labels[j]}: median self-match {np.median(frac):.2f}")
        if cfg.runs and j != ref and np.median(frac) < cfg.align_min:
            raise SystemExit(
                f"{labels[j]} is not index-aligned with {labels[ref]}: "
                f"median per-head self-match {np.median(frac):.2f} < "
                f"--align-min {cfg.align_min}. Independent seeds permute "
                "heads as well as entries; use experiments/gpt2/seedstab.py "
                "or splitting.py for them.")
        res[j] = (usage, drift_cos(U, U_ref), mcos, perm, frac)
    stack = lambda i: np.stack([res[j][i] for j in range(len(columns))])
    Z = {"usage": stack(0), "drift": stack(1), "match": stack(2),
         "perm": stack(3), "self_match": stack(4), "ref": ref,
         "cols": np.asarray(labels), "runs": np.asarray([r for r, _ in columns]),
         "steps": np.asarray([s for _, s in columns]),
         "temps": np.asarray(temps), "title": title,
         "mark_step": -1 if cfg.mark_step is None else cfg.mark_step,
         "eval_rows": cfg.eval_rows, "cache": str(cfg.cache)}
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "drift.npz", **Z)
    draw_all(cfg, Z, out)


if __name__ == "__main__":
    main()
