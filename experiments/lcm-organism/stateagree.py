"""Whether two models represent the same rows alike, before any dictionary.

    uv run python experiments/lcm-organism/stateagree.py --a <harvest a>.npy --b <harvest b>.npy

The two harvests must hold the same rows (their `.index.npy` sidecars are
compared). On `--rows` rows from the end (the scored tail) or the start, both
centered, it reports:
- **linear CKA**, and the same with B's rows shuffled (its floor);
- **Procrustes**: the share of B's variance that the best orthogonal map of
  A leaves unexplained, each first scaled to unit total variance. This is
  identity up to rotation and scale, the notion a whitened JEPA is
  identified up to (Balestriero & LeCun 2026, Proposition 1);
- **Procrustes after whitening each**: identity up to any invertible linear
  map, since whitening absorbs one. It reads like a canonical-correlation
  test, not a rotation test;
- **ridge R^2**, A to B and B to A: the share of the target's variance a
  linear map explains (next-token prediction's notion, Proposition 4).

Maps are fitted on the first half of the rows and scored on the second, so
1024 dimensions cannot fit noise. Two untrained inits give the floor.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from jaxtyping import Float


def center(a: Float[np.ndarray, "n d"], mu=None) -> Float[np.ndarray, "n d"]:
    return a - (a.mean(0) if mu is None else mu)


def cka(a: Float[np.ndarray, "n d"], b: Float[np.ndarray, "n e"]) -> float:
    """Linear CKA of two centered representations."""
    ab = np.linalg.norm(a.T @ b) ** 2
    return float(ab / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def whitener(a: Float[np.ndarray, "n d"], eps: float = 1e-6) -> Float[np.ndarray, "d d"]:
    """C^(-1/2) of the centered `a`, with `eps` times the mean eigenvalue
    added so a rank-deficient representation stays finite."""
    C = a.T @ a / len(a)
    lam, U = np.linalg.eigh(C)
    lam = np.maximum(lam, 0) + eps * lam.mean()
    return (U / np.sqrt(lam)) @ U.T


def procrustes(af: np.ndarray, bf: np.ndarray, at: np.ndarray, bt: np.ndarray) -> float:
    """Fit the orthogonal R minimizing |af R - bf| (the widths must match),
    and return |at R - bt|^2 / |bt|^2 on the scored rows."""
    if af.shape[1] != bf.shape[1]:
        raise ValueError("Procrustes needs equal widths")
    U, _, Vt = np.linalg.svd(af.T @ bf)
    R = U @ Vt
    return float(np.linalg.norm(at @ R - bt) ** 2 / np.linalg.norm(bt) ** 2)


def ridge_r2(xf: np.ndarray, yf: np.ndarray, xt: np.ndarray, yt: np.ndarray,
             alpha: float) -> float:
    """R^2 on the scored rows of a ridge fitted on the fit rows (all
    centered by the fit rows), with eval_next.py's penalty scale."""
    G = xf.T @ xf / len(xf)
    lam = alpha * float(np.trace(G)) / len(G)
    W = np.linalg.solve(G + lam * np.eye(len(G)), xf.T @ yf / len(xf))
    return float(1 - ((xt @ W - yt) ** 2).sum() / (yt ** 2).sum())


def agree(a: Float[np.ndarray, "n d"], b: Float[np.ndarray, "n e"],
          alpha: float = 1e-3, seed: int = 0) -> Dict[str, float]:
    """The statistics of the docstring for two representations of the same
    rows (row i of `a` and of `b` are one input)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    h = len(a) // 2
    ma, mb = a[:h].mean(0), b[:h].mean(0)
    af, at = center(a[:h], ma), center(a[h:], ma)
    bf, bt = center(b[:h], mb), center(b[h:], mb)
    sa, sb = np.sqrt((af ** 2).sum(1).mean()), np.sqrt((bf ** 2).sum(1).mean())
    out = {"rows": int(len(a)),
           "cka": cka(center(a), center(b)),
           "cka_shuffled": cka(center(a), center(b[np.random.default_rng(seed).permutation(len(b))])),
           "r2_a_to_b": ridge_r2(af, bf, at, bt, alpha),
           "r2_b_to_a": ridge_r2(bf, af, bt, at, alpha)}
    if a.shape[1] == b.shape[1]:
        Wa, Wb = whitener(af), whitener(bf)
        out["procrustes"] = procrustes(af / sa, bf / sb, at / sa, bt / sb)
        out["procrustes_whitened"] = procrustes(af @ Wa, bf @ Wb, at @ Wa, bt @ Wb)
    return out


def rows_of(path: Path, rows: int, where: str) -> Tuple[np.ndarray, np.ndarray]:
    """`rows` rows from the end or the start, with their index rows."""
    a = np.load(path, mmap_mode="r")
    ix = np.load(path.with_name(path.stem + ".index.npy"), mmap_mode="r")
    sl = slice(len(a) - rows, len(a)) if where == "tail" else slice(0, rows)
    return np.asarray(a[sl], np.float64), np.asarray(ix[sl])


def main(argv=None) -> Dict[str, float]:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--rows", type=int, default=65536)
    p.add_argument("--where", choices=("tail", "head"), default="tail")
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON of the results")
    cfg = p.parse_args(argv)
    a, ia = rows_of(Path(cfg.a), cfg.rows, cfg.where)
    b, ib = rows_of(Path(cfg.b), cfg.rows, cfg.where)
    if not np.array_equal(ia, ib):
        raise SystemExit("the two harvests' rows differ (their .index.npy sidecars)")
    res = {"a": cfg.a, "b": cfg.b, "where": cfg.where, **agree(a, b, cfg.alpha, cfg.seed)}
    print(json.dumps(res, indent=2))
    if cfg.out:
        Path(cfg.out).write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    main()
