"""Range coverage between heads: does every pair of two heads' entries co-occur?

Dorrell et al. (ICLR 2025) give the condition under which a non-negative,
energy-efficient code gives each variable its own neurons, the writeup's
head-disjointness: the variables' joint range must be spread enough. For
categorical variables that is pairwise extreme-point independence, every
combination of two variables' values occurring. Entry pairs that never
co-occur can share coordinates at no cost to such an energy.

For every layer and pair of heads this builds the k x k table of how often
each pair of winning entries (the argmax of each head's assignment) co-occurs
on the evaluation tail, and counts its structural zeros: empty cells whose
expected count under independence, n f_ij f_i'j' with f the heads'
winning-entry frequencies, is at least --min-expected. Chance alone leaves
such a cell empty with probability about exp(-E); the null shuffles one
head's winners across rows, keeping both marginals.

It then asks whether atoms overlap more where entries never co-occur: the
mean cosine between the absolute atoms |a_ij| and |a_i'j'| of structurally
empty cells, against co-occurring cells whose expected counts fall in the
same doubling bin, as a z-score over the co-occurring cells' spread. Under
`ConcatDictBlock` cross-head atoms share no coordinates, so this is skipped.

  uv run python experiments/ste-arm/headrange.py \\
      --model data/out/sonar/multilingual/resid_nc \\
      --model data/out/gpt2_l8/ste_h76 --out data/out/headsupport/headrange.json
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from jaxtyping import Float, Int


def pair_table(a: Int[np.ndarray, "n"], b: Int[np.ndarray, "n"],
               k: int) -> Int[np.ndarray, "k k"]:
    """Co-occurrence counts of two heads' winning entries."""
    return np.bincount(a * k + b, minlength=k * k).reshape(k, k)


def expected(f_a: Float[np.ndarray, "k"], f_b: Float[np.ndarray, "k"],
             n: int) -> Float[np.ndarray, "k k"]:
    """Expected counts under independence with the observed marginals."""
    return n * np.outer(f_a, f_b)


def structural(C: Int[np.ndarray, "k k"], E: Float[np.ndarray, "k k"],
               tau: float) -> tuple:
    """Cells tested (E >= tau), and which of them are empty."""
    tested = E >= tau
    return tested, tested & (C == 0)


def overlap_stats(cells: dict) -> dict:
    """Bin-matched contrast of atom cosine, empty against co-occurring cells.

    `cells` maps a bin to sums for the empty cells (`z_n`, `z_s`) and the
    co-occurring ones (`c_n`, `c_s`, `c_ss`). The contrast is the mean over
    empty cells of (cos - their bin's co-occurring mean); its standard error
    takes each bin's co-occurring spread, as a within-bin permutation would."""
    num, var, n0 = 0.0, 0.0, 0
    for s in cells.values():
        if s["z_n"] == 0 or s["c_n"] < 2:
            continue
        mu = s["c_s"] / s["c_n"]
        sd2 = max(s["c_ss"] / s["c_n"] - mu * mu, 0.0)
        num += s["z_s"] - s["z_n"] * mu
        var += s["z_n"] * sd2
        n0 += s["z_n"]
    if n0 == 0:
        return {"empty_cells": 0, "excess_cos": None, "z": None}
    return {"empty_cells": n0, "excess_cos": num / n0,
            "z": num / np.sqrt(var) if var > 0 else None}


def winners(model, params, X: Float[np.ndarray, "n d"], T: float,
            b: int) -> Int[np.ndarray, "n l h"]:
    """Each head's winning entry per row (`Ontologizer.classify`, so
    gain-shape and the router apply, as in `headsupport.mean_assignment`)."""
    import jax
    import jax.numpy as jnp

    def assign(module, X):
        E, _ = module.encode(X, 0.0, None)
        _, Ps = module.classify(E, X_ref=X, temperature=T)
        return Ps
    run = jax.jit(lambda X: model.apply({"params": params}, X, method=assign))
    out = []
    for i in range(0, len(X) - b + 1, b):
        P = np.asarray(run(jnp.asarray(X[i:i + b])))      # l b hk
        P = P.reshape(model.l, P.shape[1], model.h, model.k)
        out.append(P.argmax(-1).transpose(1, 0, 2))        # b l h
    return np.concatenate(out).astype(np.int32)


def layer_measure(A: Int[np.ndarray, "n h"], U: Float[np.ndarray, "h k e"],
                  k: int, tau: float, draws: int,
                  rng: np.random.Generator) -> dict:
    """Range coverage of one layer's head pairs, its shuffle null, and the
    atom-overlap contrast (`U` the unit absolute atoms, None to skip)."""
    n, h = A.shape
    f = np.stack([np.bincount(A[:, i], minlength=k) / n for i in range(h)])
    perms = [rng.permutation(n) for _ in range(draws)]
    tested = empty = null_empty = 0
    chance = 0.0
    pair_frac, clean = [], 0
    bins: dict = {}
    for i in range(h):
        for j in range(i + 1, h):
            E = expected(f[i], f[j], n)
            C = pair_table(A[:, i], A[:, j], k)
            t, z = structural(C, E, tau)
            nt, nz = int(t.sum()), int(z.sum())
            if nt == 0:
                continue
            tested += nt
            empty += nz
            chance += float(np.exp(-E[t]).sum())
            pair_frac.append(nz / nt)
            clean += nz == 0
            for p in perms:
                Cn = pair_table(A[:, i], A[p, j], k)
                null_empty += int((t & (Cn == 0)).sum())
            if U is None:
                continue
            cos = U[i] @ U[j].T
            key = np.floor(np.log2(E[t] / tau)).astype(int)
            zz = z[t]
            cc = cos[t]
            for bkey in np.unique(key):
                m = key == bkey
                s = bins.setdefault(int(bkey), {"z_n": 0, "z_s": 0.0,
                                                "c_n": 0, "c_s": 0.0,
                                                "c_ss": 0.0})
                mz, mc = m & zz, m & ~zz
                s["z_n"] += int(mz.sum())
                s["z_s"] += float(cc[mz].sum())
                s["c_n"] += int(mc.sum())
                s["c_s"] += float(cc[mc].sum())
                s["c_ss"] += float((cc[mc] ** 2).sum())
    pf = np.array(pair_frac)
    out = {"pairs": len(pf), "tested_cells": tested, "empty_cells": empty,
           "empty_frac": empty / tested if tested else None,
           "chance_frac": chance / tested if tested else None,
           "null_frac": null_empty / (tested * draws) if tested and draws else None,
           "pairs_full_coverage": clean / len(pf) if len(pf) else None,
           "pair_empty_frac_quantiles": (np.quantile(pf, [0.1, 0.5, 0.9]).tolist()
                                         if len(pf) else None)}
    if U is not None:
        out["overlap"] = overlap_stats(bins)
    return out


def measure(path: str, cfg: argparse.Namespace,
            rng: np.random.Generator) -> dict:
    from headsupport import atoms, eval_tail, hyper, temperature
    from pareto import load_onto

    run = Path(path).resolve()
    model, params, step = load_onto(run, cfg.step)
    T = temperature(hyper(run), step)
    concat = bool(getattr(model, "concat", False))
    A = winners(model, params, eval_tail(run, cfg.eval_rows), T, cfg.b)
    W, _ = atoms(model, params)
    layers = []
    for i in range(model.l):
        U = None
        if not concat:
            Wa = np.abs(W[i])
            U = Wa / (np.linalg.norm(Wa, axis=-1, keepdims=True) + 1e-12)
        layers.append(layer_measure(A[:, i], U, model.k, cfg.min_expected,
                                    cfg.draws, rng))
        print(f"  {run.name} layer {i}: {summary(layers[-1])}", flush=True)
    return {"model": run.name, "path": str(run), "step": int(step), "T": T,
            "n": int(len(A)), "l": model.l, "h": model.h, "k": model.k,
            "concat": concat, "signed": bool(getattr(model, "signed", False)),
            "min_expected": cfg.min_expected, "layers": layers}


def summary(r: dict) -> str:
    s = (f"{r['empty_cells']} empty of {r['tested_cells']} tested cells "
         f"({r['empty_frac']:.2e}; null {r['null_frac']:.1e}, chance "
         f"{r['chance_frac']:.1e}); pairs with full coverage "
         f"{r['pairs_full_coverage']:.3f} of {r['pairs']}")
    o = r.get("overlap")
    if o and o["excess_cos"] is not None:
        s += f"; empty-cell excess cos {o['excess_cos']:+.4f} (z {o['z']:+.1f})"
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=[],
                   help="Ontologizer checkpoint dir; repeat for several")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--eval-rows", type=int, default=32768)
    p.add_argument("--b", type=int, default=1024)
    p.add_argument("--min-expected", type=float, default=10.0,
                   help="test only cells with at least this expected count")
    p.add_argument("--draws", type=int, default=3,
                   help="row shuffles of the second head, for the null")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)
    results = []
    for path in cfg.model:
        r = measure(path, cfg, rng)
        results.append(r)
        if cfg.out:
            Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
