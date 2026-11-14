"""Are a head's cells convex, and is it blind to the sign of its input?

A head with affine logits cuts its input space into a power diagram, whose
cells are convex (Aurenhammer 1991, §3.1): the mean of any rows a head
assigns to one entry, and the midpoint of any two of them, get that entry
again. The random-projection heads of the steering null (`steerembed.py
--model lsh`: argmax over k random directions after centering on the mean)
are of this kind. The Ontologizer's classifier is bilinear: each logit is a
product of two linear forms of the shaped input, a quadratic, so two
entries' boundary is a quadric and a cell can be non-convex or in pieces.
And when the constant coordinate's terms are small, a quadratic logit is
nearly even in the input, so a row and its negation get the same entry.

For each layer of each checkpoint, on the held-out tail and on the exact
input the layer's classifier sees (`gainshape_in`'s output: the unit shape
followed by any constant coordinates), per head:
  mean2, mean8, mean32
            the share of means of 2, 8 or 32 distinct random rows of one
            entry (--draws per entry used by >= --min-count rows), each
            rescaled to the mean norm of its rows, that the head assigns
            to that entry: convexity tested at a growing scale;
  centroid  the same for the mean of all the entry's rows;
  negation  the share of rows whose negated shape (constant coordinates
            kept) keeps its entry.
For a convex cone every mean share is 1; on the unit sphere the classifier
sees, that is geodesic convexity within a hemisphere. The random-projection
heads are scored the same way on the raw rows, centered on the mean they
are centered on, with negation as reflection through it.

    JAX_PLATFORMS=cpu uv run python experiments/ste-arm/cellshape.py \\
        --ckpt data/out/sonar/multilingual/ste_h76_init01:0.00015 \\
               data/out/sonar/multilingual/resid_nc:0.03
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from pareto import load_onto

# rows per mean, smallest first; 0 is the mean of all the entry's rows
SIZES = (2, 8, 32, 0)
KEYS = ("mean2", "mean8", "mean32", "centroid", "negation")


def cell_means(S: Float[np.ndarray, "n d"], A: Int[np.ndarray, "n h"], k: int,
               size: int, draws: int, min_count: int,
               rng: np.random.Generator
               ) -> Tuple[Float[np.ndarray, "c d"], Int[np.ndarray, "c 2"]]:
    """Means of `size` distinct random rows of one entry, `draws` per used
    entry (size 0: one mean of all the entry's rows, the centroid), in the
    shape coordinates `S`, each rescaled to the mean norm of the rows it
    averages (so unit rows give unit points). Returns the points and their
    (head, entry)."""
    norm = np.linalg.norm(S, axis=1)
    pts, ids = [], []
    for j in range(A.shape[1]):
        for e in range(k):
            rows = np.flatnonzero(A[:, j] == e)
            if len(rows) < max(min_count, size):
                continue
            if size == 0:
                pick = rows[None]
            else:
                pick = np.stack([rng.choice(rows, size, replace=False)
                                 for _ in range(draws)])
            m = S[pick].mean(1)
            scale = norm[pick].mean(1)
            pts.append(m * (scale / np.maximum(np.linalg.norm(m, axis=1),
                                               1e-30))[:, None])
            ids.extend([(j, e)] * len(pick))
    d = S.shape[1]
    return (np.concatenate(pts).astype(np.float32) if pts
            else np.zeros((0, d), np.float32),
            np.asarray(ids, np.int64).reshape(-1, 2))


def kept(assign: Callable[[np.ndarray], np.ndarray],
         P: Float[np.ndarray, "c d"], ids: Int[np.ndarray, "c 2"]
         ) -> Float[np.ndarray, "c"]:
    """Whether each point gets its own (head, entry) back."""
    if not len(P):
        return np.zeros(0)
    got = assign(P)                                       # (c, h)
    return (got[np.arange(len(P)), ids[:, 0]] == ids[:, 1]).astype(float)


def per_head(flags: Float[np.ndarray, "c"], heads: Int[np.ndarray, "c"],
             h: int) -> Float[np.ndarray, "h"]:
    """Mean of `flags` per head; NaN for a head with no points."""
    s = np.bincount(heads, flags, minlength=h)
    n = np.bincount(heads, minlength=h)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def shape_tests(S: Float[np.ndarray, "n d"], A: Int[np.ndarray, "n h"], k: int,
                assign: Callable[[np.ndarray], np.ndarray],
                negate: Callable[[np.ndarray], np.ndarray],
                min_count: int, draws: int, rng: np.random.Generator,
                sizes: Tuple[int, ...] = SIZES
                ) -> Dict[str, Float[np.ndarray, "h"]]:
    """Per-head shares for one layer of heads: for each mean size (0 for
    the centroid), the share of means that keep their entry; and the share
    of rows whose negation keeps theirs. `assign` maps shape points to
    entries (n, h); `negate` maps rows to their negations in the same
    coordinates."""
    h = A.shape[1]
    out = {}
    for size in sizes:
        P, ids = cell_means(S, A, k, size, draws, min_count, rng)
        out[f"mean{size}" if size else "centroid"] = per_head(
            kept(assign, P, ids), ids[:, 0], h)
        if size == 0:
            out["entries"] = np.bincount(ids[:, 0], minlength=h).astype(float)
    out["negation"] = (assign(negate(S)) == A).mean(0)
    return out


def layer_inputs(ckpt: str, T: float, X: Float[np.ndarray, "n d"], batch: int
                 ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Callable],
                            List[int], Dict]:
    """Per layer: the classifier's input U on every row, its argmax code,
    a function from inputs to argmax codes, and the trailing constant
    coordinate count."""
    model, raw, step = load_onto(ckpt, 0)
    params = {"params": raw}

    def probe(module, Xb: Float[Array, "b d"]):
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        Us, As = [], []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            Us.append(U)
            As.append(jnp.argmax(P, -1))
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            if i < module.l - 1:
                Ein = module.nextinput(Xb, R, P.reshape(Xb.shape[0], -1))
        return Us, As

    f = jax.jit(lambda p, x: model.apply(p, x, method=probe))
    Us = [[] for _ in range(model.l)]
    As = [[] for _ in range(model.l)]
    for i in range(0, len(X), batch):
        U, A = f(params, jnp.asarray(X[i:i + batch]))
        for li in range(model.l):
            Us[li].append(np.asarray(U[li]))
            As[li].append(np.asarray(A[li]))
    Us = [np.concatenate(u) for u in Us]
    As = [np.concatenate(a) for a in As]

    def assigner(li: int) -> Callable[[np.ndarray], np.ndarray]:
        def cls(module, Ub):
            de = module.dictencs[li]
            return jnp.argmax(de.dict.cluster(de.classifier(Ub), T), -1)
        g = jax.jit(lambda p, u: model.apply(p, u, method=cls))
        return lambda U: np.concatenate(
            [np.asarray(g(params, jnp.asarray(U[i:i + batch])))
             for i in range(0, len(U), batch)])

    n_const = [int(de_n) for de_n in
               (model.bind(params).dictencs[li].n_const
                for li in range(model.l))]
    info = {"ckpt": ckpt, "step": step, "l": model.l, "h": model.h,
            "k": model.k, "temperature": T, "n_const": n_const}
    return Us, As, [assigner(li) for li in range(model.l)], n_const, info


def summarize(res: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Means over heads, the worst head, and for the mean tests the share
    of heads that keep every point."""
    out = {}
    for key in KEYS:
        v = res[key][np.isfinite(res[key])]
        out[key] = float(v.mean()) if len(v) else float("nan")
        out[f"{key}_min"] = float(v.min()) if len(v) else float("nan")
        if key != "negation":
            out[f"{key}_heads_all"] = float((v == 1.0).mean()) if len(v) \
                else float("nan")
    out["entries_per_head"] = float(res["entries"].mean())
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="checkpoint dirs, each as dir:temperature")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=32768,
                   help="the cache's held-out tail")
    p.add_argument("--min-count", type=int, default=32,
                   help="score only entries used by this many rows")
    p.add_argument("--draws", type=int, default=16,
                   help="random means of each size drawn per entry")
    p.add_argument("--lsh-heads", type=int, default=76)
    p.add_argument("--lsh-k", type=int, default=32)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/out/sonar/cellshape")
    cfg = p.parse_args()

    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    results, per_head_rows = [], []

    # random-projection heads, as steerembed.py builds them
    W = rng.standard_normal((cfg.lsh_heads, cfg.lsh_k, X.shape[1])
                            ).astype(np.float32)
    W /= np.linalg.norm(W, axis=-1, keepdims=True) + 1e-9
    mu = X.mean(0, keepdims=True)
    lsh = lambda Z: np.argmax(np.einsum("bd,hkd->bhk", Z - mu, W), -1)
    A = lsh(X)
    # the rows centered on mu, the apex of every cell's cone, so the means'
    # rescaling about it keeps a mean of cone points in the cone
    res = shape_tests(X - mu, A, cfg.lsh_k, lambda S: lsh(S + mu),
                      lambda S: -S, cfg.min_count, cfg.draws, rng)
    results.append({"model": "random-projection", "layer": 0,
                    **summarize(res)})
    for j in range(cfg.lsh_heads):
        per_head_rows.append(["random-projection", 0, j]
                             + [res[key][j] for key in KEYS + ("entries",)])

    for spec in cfg.ckpt:
        path, T = spec.rsplit(":", 1)
        Us, As, assign, n_const, info = layer_inputs(path, float(T), X,
                                                     cfg.batch)
        name = Path(path).name
        print(f"{name} step {info['step']}: {info['l']} x {info['h']} "
              f"heads, k {info['k']}, constant coordinates {n_const}")
        for li in range(info["l"]):
            U, nc = Us[li], n_const[li]
            d = U.shape[1] - nc
            tail = U[:, d:].mean(0) if nc else np.zeros(0, np.float32)
            full = (lambda S, tail=tail:
                    np.concatenate([S, np.broadcast_to(tail, (len(S), nc))],
                                   1).astype(np.float32))
            res = shape_tests(U[:, :d], As[li], info["k"],
                              lambda S, f=assign[li], full=full: f(full(S)),
                              lambda S: -S, cfg.min_count, cfg.draws, rng)
            results.append({"model": name, "layer": li, **summarize(res)})
            for j in range(info["h"]):
                per_head_rows.append([name, li, j]
                                     + [res[key][j]
                                        for key in KEYS + ("entries",)])

    print(f"\n{len(X)} held-out rows; entries used by >= {cfg.min_count} "
          f"rows; {cfg.draws} random means of each size per entry. Share "
          f"of means that keep their entry, mean over heads; 'all' is the "
          f"share of heads that keep every centroid")
    print(f"{'model':>20} {'layer':>5} {'entries':>8} {'mean2':>7} "
          f"{'mean8':>7} {'mean32':>7} {'centroid':>9} {'all':>7} "
          f"{'negation':>9}")
    for r in results:
        print(f"{r['model']:>20} {r['layer']:>5} {r['entries_per_head']:8.1f} "
              f"{r['mean2']:7.3f} {r['mean8']:7.3f} {r['mean32']:7.3f} "
              f"{r['centroid']:9.3f} {r['centroid_heads_all']:7.1%} "
              f"{r['negation']:9.3f}")
    (out / "summary.json").write_text(json.dumps(
        {"rows": len(X), "cache": cfg.cache, "min_count": cfg.min_count,
         "draws": cfg.draws, "results": results}, indent=1))
    with open(out / "heads.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "layer", "head", *KEYS, "entries"])
        w.writerows(per_head_rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
