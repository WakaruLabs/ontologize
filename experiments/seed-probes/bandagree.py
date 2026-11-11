"""Where in the data's spectrum do two seeds' heads agree?

`headcontrib.py` matches two seeds' heads by what they write and finds
agreement near its null past layer 0. It scores the whole whitened output
space at once. This splits that space by the data's own principal
directions and scores each band alone, to see whether the agreement
there is concentrates in the directions the data vary along most, and
whether the low-variance directions, which the whitened objective weighs
up, are where seeds part.

The frame is `headcontrib.py`'s: per-head output contributions, whitened
by the objective's weights and centered over rows. The bands are ranges
of the eigenvectors of the whitened data covariance (from the cache
tail), ordered by eigenvalue. Within a band, heads are matched one to one
by centered cosine, as `headcontrib.py` matches them over the whole
space, against the same null (B's rows shuffled). `--earlier` scores A
against its own earlier checkpoint in every band, since a band where one
run does not agree with itself cannot show two seeds agreeing.

Each band also reports the share of the data's whitened variance it
holds and the share of the heads' centered contribution energy.

  uv run python experiments/seed-probes/bandagree.py \\
      --a data/out/sonar/multilingual/ste_h76_init01 \\
      --b data/out/sonar/multilingual/ste_h76_i01_s43 --earlier 350000
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))

import numpy as np
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment

from headcontrib import contributions, cosine_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir")
    p.add_argument("--b", required=True, help="checkpoint dir")
    p.add_argument("--step-a", type=int, default=0)
    p.add_argument("--step-b", type=int, default=0)
    p.add_argument("--earlier", type=int, default=0,
                   help="an earlier step of A, as the within-run control "
                        "(0 = skip)")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--rows", type=int, default=4096,
                   help="scored rows, the cache's last")
    p.add_argument("--cov-rows", type=int, default=32768,
                   help="cache-tail rows the principal directions come from")
    p.add_argument("--edges", type=int, nargs="+",
                   default=[0, 8, 32, 128, 512, 1024],
                   help="band edges, as ranks in the data's spectrum")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def principal_basis(Xw: Float[np.ndarray, "n d"]
                    ) -> Tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d d"]]:
    """Eigenvalues (descending) and eigenvectors (columns) of `Xw`'s
    covariance."""
    C = np.cov(Xw, rowvar=False)
    lam, Q = np.linalg.eigh(C)
    return lam[::-1].copy(), Q[:, ::-1].copy()


def rotate(C: Float[np.ndarray, "heads n d"], Q: Float[np.ndarray, "d d"]
           ) -> Float[np.ndarray, "heads n d"]:
    """`C` centered over rows and expressed in `Q`'s columns, in place."""
    Qf = Q.astype(C.dtype)
    for i in range(len(C)):
        C[i] = (C[i] - C[i].mean(0, keepdims=True)) @ Qf
    return C


def band_scores(A: Float[np.ndarray, "ha n d"], B: Float[np.ndarray, "hb n d"],
                la: Int[np.ndarray, "ha"], edges: Sequence[int],
                rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Per band of coordinates `edges[i]:edges[i + 1]`: the mean matched
    cosine between A's and B's heads, the same with B's rows shuffled, the
    matched cosine by A's layer, and the band's share of A's energy. `A`
    and `B` are already centered over rows."""
    perm = rng.permutation(B.shape[1])
    total = float((A.astype(np.float64) ** 2).sum())
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a, b = A[..., lo:hi], B[..., lo:hi]
        M = cosine_matrix(a, b, False)
        r, c = linear_sum_assignment(-M)
        v = np.empty(len(a))
        v[r] = M[r, c]
        Mn = cosine_matrix(a, b[:, perm], False)
        rn, cn = linear_sum_assignment(-Mn)
        out.append({
            "lo": lo, "hi": hi,
            "matched": float(v.mean()),
            "null": float(Mn[rn, cn].mean()),
            "by_layer": [float(v[la == i].mean())
                         for i in range(int(la.max()) + 1)],
            "energy_share": float((a.astype(np.float64) ** 2).sum()) / total})
    return out


def main() -> None:
    cfg = parse_args()
    mm = np.load(cfg.cache, mmap_mode="r")
    w_sqrt = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    lam, Q = principal_basis(
        np.asarray(mm[-cfg.cov_rows:], np.float64) * w_sqrt)
    X = np.asarray(mm[-cfg.rows:], np.float32)
    edges = [e for e in cfg.edges if e <= len(lam)]
    rng = np.random.default_rng(cfg.seed)

    A, la, sa = contributions(cfg.a, cfg.step_a, cfg.temperature, X, w_sqrt,
                              cfg.batch)
    A = rotate(A, Q)
    B, _, sb = contributions(cfg.b, cfg.step_b, cfg.temperature, X, w_sqrt,
                             cfg.batch)
    B = rotate(B, Q)
    print(f"A {Path(cfg.a).name} step {sa}; B {Path(cfg.b).name} step {sb}: "
          f"{len(A)} heads, {cfg.rows} rows")
    res = {"seed": band_scores(A, B, la, edges, rng)}
    del B
    if cfg.earlier:
        E, _, _ = contributions(cfg.a, cfg.earlier, cfg.temperature, X,
                                w_sqrt, cfg.batch)
        res["earlier"] = band_scores(A, rotate(E, Q), la, edges, rng)
        del E

    var = [float(lam[lo:hi].sum() / lam.sum())
           for lo, hi in zip(edges[:-1], edges[1:])]
    nl = int(la.max()) + 1
    for name, rows in res.items():
        print(f"\nA against {name}: matched centered cosine by band")
        print(f"  {'ranks':<11}{'data var':>9}{'energy':>8}{'matched':>9}"
              f"{'null':>8}{'ratio':>7}  " + " ".join(
                  f"{'L' + str(i):>7}" for i in range(nl)))
        for r, v in zip(rows, var):
            r["data_var_share"] = v
            print(f"  {r['lo']:>4}-{r['hi']:<6}{v:>9.3f}"
                  f"{r['energy_share']:>8.3f}{r['matched']:>9.4f}"
                  f"{r['null']:>8.4f}{r['matched'] / r['null']:>7.1f}  "
                  + " ".join(f"{x:>7.4f}" for x in r["by_layer"]))

    out = Path(cfg.out or f"data/out/sonar/seedprobes/bands_"
                          f"{Path(cfg.a).name}__{Path(cfg.b).name}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {"a": str(cfg.a), "step_a": int(sa), "b": str(cfg.b),
         "step_b": int(sb), "earlier": cfg.earlier, "rows": cfg.rows,
         "cov_rows": cfg.cov_rows, "edges": edges, "bands": res}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
