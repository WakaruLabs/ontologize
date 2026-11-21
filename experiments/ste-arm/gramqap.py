"""Do two checkpoints' heads share the relations among their atoms,
whichever directions the atoms point in?

`atomgram.py` compares two heads' atom Grams with entries paired by linear
assignment on the atoms themselves, which needs the atoms to point the
same ways. A Gram is unchanged by any rotation of a head's atoms, so two
heads could share their relational structure (which atoms lie near which)
while the atoms point elsewhere, and a match on the atoms would miss it.
Here entries are paired from the Grams alone: per pair of heads, the
permutation of B's entries that best aligns its Gram with A's, a quadratic
assignment problem, solved by scipy's FAQ (Vogelstein et al. 2015).

Only a Gram's deviations from its mean off-diagonal cosine can carry a
correspondence: a nearly one-parameter Gram is nearly invariant under
every permutation. The QAP is posed on the deviations, with the diagonal
zeroed, which leaves its optimum unchanged (the means and norms are
permutation invariant) and keeps the large shared part out of FAQ's
relaxation.

Choosing the permutation to minimize the distance it then reports
flatters every pair, so the null is matched the same way. The QAP is
solved for every pair of a layer's heads, and the ratio is
`atomgram.py`'s: mean distance to the layer's other heads over distance to
the paired head (heads paired by index within a run, by linear assignment
on the distance across runs), with the off-diagonal RMS distance between
the raw Grams. Orders:

  matched  `atomgram.py`'s: entries paired by linear assignment on the
           cosine of centered decoded atoms
  gram     the QAP from the barycenter and from --starts random starts,
           the best kept: the Grams alone, blind to any rotation of the
           atoms
  best     the better of `gram` and the QAP started from `matched`, so it
           never reads a larger distance than `matched` does

The ratio pairs heads by the assignment most favourable to it, so heads
that do not correspond read above 1. `split_gram` is that level for `gram`:
the same statistic between the two halves of A's own layer, heads with A's
statistics and no correspondence. Only `gram` is read for the halves, and
it never uses the start from an atom match.

Within one run (`--a` and `--b` the same directory) stored order is the
true correspondence, so the run against its own earlier checkpoint tests
whether the Grams alone determine it: `entries_gram` is the share of a
paired head's entries the `gram` order maps to themselves, and
`recovers_*` the share of heads the assignment on each distance pairs
with themselves.

  # one run against its own earlier checkpoint, then two seeds
  JAX_PLATFORMS=cpu uv run python experiments/ste-arm/gramqap.py \\
      --a data/out/sonar/multilingual/ste_h76_init01 --step-a 369500 \\
      --b data/out/sonar/multilingual/ste_h76_init01 --step-b 350000
  JAX_PLATFORMS=cpu uv run python experiments/ste-arm/gramqap.py \\
      --a data/out/sonar/multilingual/ste_h76_init01 --step-a 369500 \\
      --b data/out/sonar/multilingual/ste_h76_i01_s43 --step-b 369500

Writes <out>/gram.csv (space, centered, layer, the three ratios, the split
chance level, entry and head recovery within a run) and <out>/summary.json;
default out directory
data/out/sonar/gramqap/<a>_<step>_vs_<b>_<step>.
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import csv
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment, quadratic_assignment

from atomgram import (VARIANTS, centered, entry_match, gram, head_atoms,
                      matched_rms, pair_ratio)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir")
    p.add_argument("--b", required=True, help="checkpoint dir")
    p.add_argument("--step-a", type=int, default=0)
    p.add_argument("--step-b", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--rows", type=int, default=32768,
                   help="cache tail rows that set usage (centering)")
    p.add_argument("--starts", type=int, default=4,
                   help="random FAQ starts per pair of heads")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", default=None)
    return p.parse_args()


# ---------- the matching (pure) ----------

def deviation(G: Float[np.ndarray, "... k k"]) -> Float[np.ndarray, "... k k"]:
    """Each Gram less its mean off-diagonal entry, with the diagonal zeroed."""
    k = G.shape[-1]
    off = ~np.eye(k, dtype=bool)
    D = G - G[..., off].mean(-1)[..., None, None]
    D[..., ~off] = 0.0
    return D


def rms(G1: Float[np.ndarray, "k k"], G2: Float[np.ndarray, "k k"]) -> float:
    """Off-diagonal RMS difference."""
    off = ~np.eye(G1.shape[-1], dtype=bool)
    return float(np.sqrt(((G1 - G2)[off] ** 2).mean()))


def faq(D1: Float[np.ndarray, "k k"], D2: Float[np.ndarray, "k k"],
        P0, seed: int = 0) -> Int[np.ndarray, "k"]:
    """The order of D2's entries that FAQ aligns with D1's from the start
    P0 ("barycenter", "randomized" or a doubly stochastic matrix), so
    that D2[c][:, c] lines up with D1."""
    return quadratic_assignment(D1, D2, method="faq",
                                options={"maximize": True, "P0": P0,
                                         "rng": seed}).col_ind


def qap_order(G1: Float[np.ndarray, "k k"], G2: Float[np.ndarray, "k k"],
              starts: int, rng: np.random.Generator,
              init: Optional[Int[np.ndarray, "k"]] = None
              ) -> Tuple[Int[np.ndarray, "k"], Optional[Int[np.ndarray, "k"]]]:
    """(the best order of G2's entries over FAQ from the barycenter and
    from `starts` of scipy's randomized starts, the order FAQ reaches from
    `init`, or None without one), each scored by the off-diagonal RMS
    between G1 and the reordered G2. On planted relabellings the
    barycenter start finds the permutation where random starts often do
    not (`tests/test_gramqap.py`); the random starts are extra tries."""
    D1, D2 = deviation(G1), deviation(G2)
    best, best_d = None, np.inf
    for s in range(starts + 1):
        c = faq(D1, D2, "barycenter" if s == 0 else "randomized",
                int(rng.integers(2 ** 31)))
        d = rms(G1, G2[np.ix_(c, c)])
        if d < best_d:
            best, best_d = c, d
    from_init = None
    if init is not None:
        P0 = np.zeros((len(G1), len(G1)))
        P0[np.arange(len(G1)), init] = 1.0
        from_init = faq(D1, D2, P0)
    return best, from_init


def _row(args) -> Tuple[Float[np.ndarray, "hb"], Float[np.ndarray, "hb"],
                        Int[np.ndarray, "hb k"]]:
    """One head of A against every head of B: the `gram` and `best`
    distances and the `gram` orders."""
    G1, G2s, inits, starts, seed = args
    rng = np.random.default_rng(seed)
    hb, k = G2s.shape[0], G1.shape[0]
    d_gram, d_best = np.empty(hb), np.empty(hb)
    orders = np.empty((hb, k), np.int64)
    for j in range(hb):
        c, ci = qap_order(G1, G2s[j], starts, rng, inits[j])
        orders[j] = c
        d_gram[j] = rms(G1, G2s[j][np.ix_(c, c)])
        cands = [d_gram[j], rms(G1, G2s[j][np.ix_(ci, ci)]),
                 rms(G1, G2s[j][np.ix_(inits[j], inits[j])])]
        d_best[j] = min(cands)
    return d_gram, d_best, orders


def qap_distances(Ga: Float[np.ndarray, "ha k k"],
                  Gb: Float[np.ndarray, "hb k k"],
                  perm: Int[np.ndarray, "ha hb k"], starts: int, seed: int,
                  ex: Optional[ProcessPoolExecutor] = None
                  ) -> Tuple[Float[np.ndarray, "ha hb"],
                             Float[np.ndarray, "ha hb"],
                             Int[np.ndarray, "ha hb k"]]:
    """For every pair of heads, the `gram` and `best` distances and the
    `gram` order of B's entries; rows of A run on `ex` when given."""
    jobs = [(Ga[i], Gb, perm[i], starts, [seed, i]) for i in range(len(Ga))]
    rows = list(ex.map(_row, jobs)) if ex else [_row(j) for j in jobs]
    return (np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]),
            np.stack([r[2] for r in rows]))


def entries_recovered(orders: Int[np.ndarray, "h h k"]) -> float:
    """Within a run: the share of each head's entries that its own
    earlier self's `gram` order maps to themselves."""
    h, k = orders.shape[0], orders.shape[2]
    return float((orders[np.arange(h), np.arange(h)] == np.arange(k)).mean())


def heads_recovered(D: Float[np.ndarray, "h h"]) -> float:
    """Share of heads the linear assignment on D pairs with themselves."""
    return float((linear_sum_assignment(D)[1] == np.arange(len(D))).mean())


# ---------- the experiment ----------

def compare(atoms_a: Dict[str, List[np.ndarray]],
            atoms_b: Dict[str, List[np.ndarray]],
            usage_a: List[np.ndarray], usage_b: List[np.ndarray],
            same_run: bool, starts: int, seed: int,
            ex: Optional[ProcessPoolExecutor] = None) -> List[dict]:
    rows = []
    for i, (ua, ub) in enumerate(zip(usage_a, usage_b)):
        perm = entry_match(centered(atoms_a["decoded"][i], ua),
                           centered(atoms_b["decoded"][i], ub))
        for v, (space, center) in enumerate(VARIANTS):
            Aa, Ab = atoms_a[space][i], atoms_b[space][i]
            if center:
                Aa, Ab = centered(Aa, ua), centered(Ab, ub)
            Ga, Gb = gram(Aa), gram(Ab)
            D_m = matched_rms(Ga, Gb, perm)
            D_g, D_b, orders = qap_distances(Ga, Gb, perm, starts,
                                             seed * 1000 + 10 * i + v, ex)
            # chance for `gram` across runs: the same statistic between the
            # two halves of A's own layer, heads with A's statistics that
            # do not correspond, so only the favourable pairing lifts it
            hh = len(Ga) // 2
            D_s, _, _ = qap_distances(Ga[:hh], Ga[hh:2 * hh],
                                      perm[:hh, hh:2 * hh], starts,
                                      seed * 1000 + 10 * i + v + 500, ex)
            row = {"space": space, "centered": center, "layer": i,
                   "matched": pair_ratio(D_m, same_run),
                   "gram": pair_ratio(D_g, same_run),
                   "best": pair_ratio(D_b, same_run),
                   "split_gram": pair_ratio(D_s, False),
                   "entries_gram": np.nan, "recovers_matched": np.nan,
                   "recovers_gram": np.nan}
            if same_run:
                row.update(entries_gram=entries_recovered(orders),
                           recovers_matched=heads_recovered(D_m),
                           recovers_gram=heads_recovered(D_g))
            rows.append(row)
            print(f"  layer {i} {space}{' centered' if center else ''}: "
                  f"matched {row['matched']:.3f}, gram {row['gram']:.3f}, "
                  f"best {row['best']:.3f}, split {row['split_gram']:.3f}"
                  + (f"; entries {row['entries_gram']:.2f}, heads "
                     f"{row['recovers_gram']:.2f}" if same_run else ""),
                  flush=True)
    return rows


def main() -> None:
    cfg = parse_args()
    same_run = Path(cfg.a).resolve() == Path(cfg.b).resolve()
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    w_sqrt = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    atoms_a, usage_a, sa = head_atoms(cfg.a, cfg.step_a, cfg.temperature, X,
                                      w_sqrt, cfg.batch)
    atoms_b, usage_b, sb = head_atoms(cfg.b, cfg.step_b, cfg.temperature, X,
                                      w_sqrt, cfg.batch)
    if [u.shape for u in usage_a] != [u.shape for u in usage_b]:
        raise SystemExit(f"{cfg.a} and {cfg.b} differ in l, h or k")
    na, nb = Path(cfg.a).name, Path(cfg.b).name
    print(f"A {na} step {sa}, B {nb} step {sb}: "
          f"{'one run' if same_run else 'two runs'}, {len(X)} rows, "
          f"{cfg.starts} random starts per pair")
    # spawned, not forked: the parent has JAX's threads running
    with ProcessPoolExecutor(cfg.workers,
                             mp_context=mp.get_context("spawn")) as ex:
        rows = compare(atoms_a, atoms_b, usage_a, usage_b, same_run,
                       cfg.starts, cfg.seed, ex)
    out = Path(cfg.out or f"data/out/sonar/gramqap/{na}_{sa}_vs_{nb}_{sb}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gram.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "a": cfg.a, "step_a": sa, "b": cfg.b, "step_b": sb,
        "same_run": same_run, "rows": len(X), "cache": cfg.cache,
        "mse_weights": cfg.mse_weights, "temperature": cfg.temperature,
        "starts": cfg.starts, "seed": cfg.seed}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
