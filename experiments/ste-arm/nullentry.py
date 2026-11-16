"""Does any head spend an entry on nothing?

A head writes its layer's input gain times the decoded atom of the entry a
row selects. The gain belongs to the layer, so a head cannot quiet itself
on one row while its neighbours write; its only way to write nothing there
is an entry whose atom is near zero, an "off" entry like the silent phase
of the radio head or a Potts unit's silent state
(scratch/literature/peripheral-2026-11-16.md). A row's assignment sums to
one by construction, so such an entry would have to be learned.

Atoms are decoded as `headcontrib.py` decodes them: each head isolated,
less `decode(0)`, in the whitened frame. Nothing is measured two ways:
  zero      the atom's norm, so the head adds nothing to the output;
  centered  the atom's distance from the head's usage-weighted mean
            atom, so the head adds only what it adds on average and
            nothing particular to the row.
An entry is quiet when it is used (selected on at least --min-use / k of
the rows) and its atom is below --ratio of its head's median used atom.
Measured against the head's own median, so a layer whose atoms are small
throughout (an inert last layer) does not count as abstaining; the ratio
to the layer's median is printed beside it.

For each quiet entry: its usage, both ratios, and its rows' mean gain
against the layer's, since a row with little left to explain has a small
gain and writes little through any entry. With `--index` (a harvest's
`.index.npy`, whose third column is the token), also whether the quiet
entry takes the rows the head has no category for: the share of an
entry's rows whose token is among its own ten commonest, over the same
share in random rows of the same number (1 = the corpus's mix), for the
quiet entry and, usage-weighted, for the head's other used entries.

The centered ratio is biased low for an entry that takes most of its
head's rows, since the mean it is measured from sits near it.

  JAX_PLATFORMS=cpu uv run python experiments/ste-arm/nullentry.py \\
      --a data/out/gpt2_l8/ste_h20_cat128 data/out/gpt2_l8/ste_h76
  # SONAR: --cache data/sonar_embeddings/mc4_4M.npy
  #        --mse-weights data/out/sonar/mse_weights.npy
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from jaxtyping import Bool, Float, Int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", nargs="+", required=True, help="checkpoint dirs")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="irrelevant under select='ste'")
    p.add_argument("--cache", default="data/activations/gpt2_l8.npy")
    p.add_argument("--mse-weights",
                   default="data/activations/gpt2_l8.mse_weights_matched.npy")
    p.add_argument("--rows", type=int, default=32768,
                   help="from the cache's held-out tail")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--ratio", type=float, nargs="+", default=[0.1, 0.25],
                   help="quiet: atom below this share of the head's median")
    p.add_argument("--min-use", type=float, default=0.25,
                   help="used: selected on at least this / k of the rows")
    p.add_argument("--show", type=int, default=12,
                   help="quiet entries listed per run, smallest ratio first")
    p.add_argument("--index", default=None,
                   help="the cache's .index.npy (doc, pos, token): adds each "
                        "listed quiet entry's token concentration")
    return p.parse_args()


def top_share(tokens: Int[np.ndarray, "n"], top: int = 10) -> float:
    """Share of the rows whose token is among their `top` commonest."""
    return float(np.sort(np.bincount(tokens))[::-1][:top].sum() / len(tokens))


def concentration(tokens: Int[np.ndarray, "N"], rows: Bool[np.ndarray, "N"],
                  rng: np.random.Generator, reps: int = 20) -> float:
    """`top_share` of the selected rows over its mean for as many rows
    drawn at random from all of them: 1 is the corpus's own mix."""
    n = int(rows.sum())
    base = np.mean([top_share(rng.choice(tokens, n, replace=False))
                    for _ in range(reps)])
    return top_share(tokens[rows]) / float(base)


def entry_table(D: Float[np.ndarray, "H k d"], I: Int[np.ndarray, "H n"]
                ) -> Dict[str, np.ndarray]:
    """Per entry: usage, atom norm, and distance from the head's
    usage-weighted mean atom; per head, that mean's norm."""
    H, k = D.shape[:2]
    usage = np.stack([np.bincount(r, minlength=k) for r in I]) / I.shape[1]
    mean = (usage[..., None] * D).sum(1)                       # (H, d)
    return {"usage": usage,
            "norm": np.linalg.norm(D, axis=-1),
            "cnorm": np.linalg.norm(D - mean[:, None], axis=-1),
            "mean_norm": np.linalg.norm(mean, axis=-1)}


def head_ratio(norm: Float[np.ndarray, "H k"],
               used: Bool[np.ndarray, "H k"]) -> Float[np.ndarray, "H k"]:
    """Each entry's norm over its head's median norm among used entries
    (NaN for a head with none, and centered for a head with one, whose
    only used entry is its own mean)."""
    med = np.array([np.median(n[u]) if u.any() else np.nan
                    for n, u in zip(norm, used)])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(med[:, None] > 0, norm / med[:, None], np.nan)


def quiet(norm: Float[np.ndarray, "H k"], usage: Float[np.ndarray, "H k"],
          ratio: float, min_use: float) -> Bool[np.ndarray, "H k"]:
    """Used entries whose atom is below `ratio` of their head's median."""
    k = norm.shape[1]
    used = usage >= min_use / k
    return used & (head_ratio(norm, used) < ratio)


def report(name: str, D: Float[np.ndarray, "H k d"],
           I: Int[np.ndarray, "H n"], G: Float[np.ndarray, "l n"],
           layer: Int[np.ndarray, "H"], used_step: int,
           cfg: argparse.Namespace, tokens=None) -> None:
    H, k = D.shape[:2]
    nl = int(layer.max()) + 1
    t = entry_table(D, I)
    used = t["usage"] >= cfg.min_use / k
    print(f"\n{name} step {used_step}: l={nl} h={H // nl} k={k}, "
          f"{I.shape[1]} rows; used = selected on >= {cfg.min_use:g}/k")
    cols = "  ".join(f"q<{r:g} zero/cen" for r in cfg.ratio)
    print(f" layer  |mean|/med  rare  1-entry  min used/head-med zero  cen   "
          f"{cols}")
    for i in range(nl):
        m = layer == i
        lmed = np.median(t["norm"][m][used[m]])
        rz = head_ratio(t["norm"][m], used[m])
        rc = head_ratio(t["cnorm"][m], used[m])
        counts = "  ".join(
            f"{quiet(t['norm'][m], t['usage'][m], r, cfg.min_use).any(1).sum():5d}"
            f"/{quiet(t['cnorm'][m], t['usage'][m], r, cfg.min_use).any(1).sum():<5d}"
            for r in cfg.ratio)
        print(f"  {i:>3}    {np.median(t['mean_norm'][m]) / lmed:6.3f}  "
              f"{(~used[m]).sum():5d}    {(used[m].sum(1) == 1).sum():5d}"
              f"          "
              f"{np.nanmin(np.where(used[m], rz, np.nan)):.3f}  "
              f"{np.nanmin(np.where(used[m], rc, np.nan)):.3f}   {counts}")
    print("   (q<r: heads with a quiet entry, zero/centered; rare: entries "
          "below the use threshold; 1-entry: heads with one used entry, "
          "constant on every row)")

    r0 = max(cfg.ratio)
    q = quiet(t["norm"], t["usage"], r0, cfg.min_use)
    rz = head_ratio(t["norm"], used)
    rc = head_ratio(t["cnorm"], used)
    js, es = np.nonzero(q)
    if not len(js):
        print(f"   no used entry below {r0:g} of its head's median")
        return
    print(f"   quiet at {r0:g}, smallest first: layer head entry  usage*k  "
          f"/head-med zero cen  /layer-med  gain on rows / layer"
          + ("  tokens: quiet / others" if tokens is not None else ""))
    order = np.argsort(rz[js, es])
    rng = np.random.default_rng(0)
    for j, e in zip(js[order][:cfg.show], es[order][:cfg.show]):
        i = layer[j]
        lmed = np.median(t["norm"][layer == i][used[layer == i]])
        rows = I[j] == e
        tok = ""
        if tokens is not None:
            others = [o for o in np.nonzero(used[j])[0]
                      if o != e and (I[j] == o).sum() >= 50]
            c = [concentration(tokens, I[j] == o, rng) for o in others]
            wts = [t["usage"][j, o] for o in others]
            rest = np.average(c, weights=wts) if c else np.nan
            tok = (f"   {concentration(tokens, rows, rng):.2f}x / "
                   f"{rest:.2f}x")
        print(f"     {i:>3}  {j - i * (H // nl):4d}  {e:4d}   "
              f"{t['usage'][j, e] * k:6.2f}   {rz[j, e]:.3f} {rc[j, e]:.3f}"
              f"     {t['norm'][j, e] / lmed:.3f}      "
              f"{G[i][rows].mean() / G[i].mean():.3f}{tok}")
    print(f"   {len(js)} quiet entries in {len(set(js))} heads; they take "
          f"{t['usage'][q].sum() / len(set(js)) * 100:.1f}% of their "
          f"heads' rows on average")


def main() -> None:
    cfg = parse_args()
    from headcontrib import atoms_and_codes
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    w_sqrt = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    tokens = None if cfg.index is None else np.asarray(
        np.load(cfg.index, mmap_mode="r")[-cfg.rows:, 2])
    for a in cfg.a:
        D, I, G, layer, used = atoms_and_codes(a, cfg.step, cfg.temperature,
                                               X, w_sqrt, cfg.batch)
        report(Path(a).name, D, I, G, layer, used, cfg, tokens)


if __name__ == "__main__":
    main()
