"""The rates a trained hard code reaches without retraining: each layer
prefix with its last layer's heads dropped, cheapest first, and the lower
convex hull of those points, against which other runs are read.

A dropped head sends nothing and writes its mean output over the tail rows,
scaled by the row's gain (the float code: gains unpriced, as ratecurve.py's
"float"). Layer i's heads are ranked by the FVU each adds when dropped
alone, per realized bit it saves (a head that sends nothing goes first),
and the prefix that ends at layer i is scored with the first c of them
dropped, c in steps of --step. Earlier layers are kept whole and later ones
are absent, so layer i's curve runs from about prefix i-1's rate (every
head dropped) to prefix i's (none). Rates are realized bits, the plug-in
usage entropy per head (ratecurve.head_bits).

The first --ckpt is the reference. Since any two codes can be time-shared
row by row, every point on the chord between two of the reference's points
is reachable, and the reference curve is the lower convex hull of all of
them. Every point of every later run is read against it at the same rate.

Writes headdrop.csv (run, layer, dropped, bits, fvu), headbits.csv (run,
layer, head, bits, used, dfvu) and, with two or more runs, against.csv
(each later point with the hull's FVU at its rate).

  JAX_PLATFORMS=cpu uv run python experiments/rate-penalty/headdrop.py \\
      --ckpt <reference run> <run> ... --out data/out/sonar/rate_penalty/headdrop
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "rate-penalty"))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from pareto import load_onto
from ratecurve import head_bits, usage_counts


def drop_order(dfvu: Float[np.ndarray, "h"], bits: Float[np.ndarray, "h"]
               ) -> Int[np.ndarray, "h"]:
    """Heads in the order a greedy rate cut drops them: least FVU added
    per bit saved first, and a head that sends no bits before any other."""
    ratio = np.where(bits > 0, dfvu / np.maximum(bits, 1e-12), -np.inf)
    return np.argsort(ratio, kind="stable")


def drop_masks(order: Int[np.ndarray, "h"], counts: Int[np.ndarray, "m"]
               ) -> Float[np.ndarray, "m h"]:
    """Row j marks the first counts[j] heads of `order` as dropped."""
    M = np.zeros((len(counts), len(order)), np.float32)
    for j, c in enumerate(counts):
        M[j, order[:c]] = 1.0
    return M


def lower_hull(bits: Float[np.ndarray, "n"], fvu: Float[np.ndarray, "n"]
               ) -> Tuple[Float[np.ndarray, "v"], Float[np.ndarray, "v"]]:
    """Vertices of the lower convex hull of (bits, fvu), by increasing
    bits. Only its decreasing part matters for a rate--distortion curve,
    but the whole lower hull is returned."""
    pts = sorted(zip(bits.tolist(), fvu.tolist()))
    hull: List[Tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # drop the middle point if it lies on or above the chord
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hb, hf = zip(*hull)
    return np.array(hb), np.array(hf)


def hull_at(hb: Float[np.ndarray, "v"], hf: Float[np.ndarray, "v"],
            b: Float[np.ndarray, "n"]) -> Float[np.ndarray, "n"]:
    """The hull's FVU at each rate in `b`; NaN outside its range."""
    return np.where((b >= hb[0]) & (b <= hb[-1]), np.interp(b, hb, hf), np.nan)


def fvu_of(err: Float[np.ndarray, "m d"], var: Float[np.ndarray, "d"],
           w: Float[np.ndarray, "d"]) -> Float[np.ndarray, "m"]:
    """coordfvu.py's FVU_w for each row of per-coordinate mean errors."""
    return (err * w).mean(1) / (var * w).mean()


def layer_runner(ckpt: str, step: int, T: float, X: Float[np.ndarray, "n d"],
                 batch: int):
    """Returns (info, run), where run(i, Obar, M) gives, for the prefix that
    ends at layer i with the heads M marks replaced by Obar: the
    per-coordinate mean squared error for each row of M, the sum over rows
    of each head's output, and every row's argmax entries at layer i."""
    model, raw, step = load_onto(ckpt, step)
    params = {"params": raw}

    def fwd(module, Xb: Float[Array, "b d"], Obar: Float[Array, "h e"],
            M: Float[Array, "m h"], i: int
            ) -> Tuple[Float[Array, "m d"], Float[Array, "h e"], Int[Array, "b h"]]:
        b = Xb.shape[0]
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        for j, de in enumerate(module.dictencs[:i + 1]):
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            O = de.head_outputs(U, P)
            if j == i:
                break
            R = R + de.gained(de.dict.combine(O), G)
            Ein = module.nextinput(Xb, R, P.reshape(b, -1))
        R = R + de.gained(de.dict.combine(O), G)
        D = jnp.einsum("mh,bhe->bme", M, O - Obar)
        Rm = R[:, None, :] - de.gained(D, None if G is None else G[:, None, :])
        Y = module.decode(Rm)
        err = ((Y - Xb[:, None, :]) ** 2).sum(0)
        idx = P.reshape(b, module.h, module.k).argmax(-1)
        return err, O.sum(0), idx

    f = jax.jit(lambda p, x, o, m, i: model.apply(p, x, o, m, i, method=fwd),
                static_argnums=4)

    def run(i: int, Obar: np.ndarray, M: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        err, Osum, idx = 0.0, 0.0, []
        o, m = jnp.asarray(Obar, jnp.float32), jnp.asarray(M, jnp.float32)
        for s in range(0, len(X), batch):
            e, os_, I = f(params, jnp.asarray(X[s:s + batch], jnp.float32), o, m, i)
            err = err + np.asarray(e, np.float64)
            Osum = Osum + np.asarray(os_, np.float64)
            idx.append(np.asarray(I))
        return err / len(X), Osum, np.concatenate(idx)

    info = {"ckpt": ckpt, "step": step, "l": model.l, "h": model.h, "k": model.k}
    return info, run


def curves(info: Dict, run, var: np.ndarray, w: np.ndarray, n: int,
           step_heads: int) -> Tuple[List[Dict], List[Dict]]:
    """Every layer's head-drop curve and its heads' bits and lone-drop FVU."""
    l, h, k = info["l"], info["h"], info["k"]
    points, heads = [], []
    base = 0.0
    for i in range(l):
        _, Osum, idx = run(i, np.zeros((h, 1)), np.zeros((1, h)))
        Obar = Osum / n
        counts = usage_counts(idx, k)
        bits = head_bits(counts)
        err, _, _ = run(i, Obar, np.vstack([np.zeros((1, h)), np.eye(h)]))
        fvu = fvu_of(err, var, w)
        dfvu = fvu[1:] - fvu[0]
        order = drop_order(dfvu, bits)
        dropped = np.unique(np.r_[np.arange(0, h, step_heads), h])
        M = drop_masks(order, dropped)
        err, _, _ = run(i, Obar, M)
        cf = fvu_of(err, var, w)
        for c, row, v in zip(dropped, M, cf):
            points.append({"layer": i + 1, "dropped": int(c),
                           "bits": float(base + ((1 - row) * bits).sum()),
                           "fvu": float(v)})
        for j in range(h):
            heads.append({"layer": i + 1, "head": j, "bits": float(bits[j]),
                          "used": int((counts[j] > 0).sum()),
                          "dfvu": float(dfvu[j])})
        print(f"  layer {i + 1}: {bits.sum():.1f} bits, fvu {cf[0]:.4f} whole, "
              f"{cf[-1]:.4f} all dropped; heads under 1 bit "
              f"{int((bits < 1).sum())}", flush=True)
        base += bits.sum()
    return points, heads


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="run directories, the reference first; dir@step for a step")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_sent16.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/sentcache/mse_weights_no837.npy")
    p.add_argument("--rows", type=int, default=32765, help="cache tail rows")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--step", type=int, default=4, help="heads dropped per point")
    p.add_argument("--out", default="data/out/sonar/rate_penalty/headdrop")
    return p.parse_args()


def write(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def main() -> None:
    cfg = parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float64)
    var = X.var(0)
    w = np.load(cfg.mse_weights).astype(np.float64)
    allpts, allheads = [], []
    hull: Optional[Tuple[np.ndarray, np.ndarray]] = None
    against = []
    for spec in cfg.ckpt:
        path, _, s = spec.partition("@")
        info, run = layer_runner(path, int(s) if s else 0, cfg.temperature, X, cfg.batch)
        name = f"{Path(path).name}_{info['step']}"
        print(name, flush=True)
        points, heads = curves(info, run, var, w, len(X), cfg.step)
        allpts += [{"run": name, **p} for p in points]
        allheads += [{"run": name, **r} for r in heads]
        b = np.array([p["bits"] for p in points])
        v = np.array([p["fvu"] for p in points])
        if hull is None:
            hull = lower_hull(b, v)
            continue
        ref = hull_at(*hull, b)
        for p, r in zip(points, ref):
            against.append({"run": name, **p, "ref_fvu": float(r),
                            "delta": float(p["fvu"] - r)})
            if p["dropped"] == 0:
                print(f"  prefix {p['layer']}: {p['bits']:.1f} bits, fvu "
                      f"{p['fvu']:.4f}, reference hull {r:.4f}", flush=True)
    write(out / "headdrop.csv", allpts)
    write(out / "headbits.csv", allheads)
    if against:
        write(out / "against.csv", against)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
