"""Do two seeds' codes carry the same information about the input, or
does each carry some the other lacks?

Two seeds' heads disagree almost entirely at equal fit (`partition.py`,
`translate.py`). Xuan, Souvenir & Pless (ECCV 2018) train each member of an
ensemble of embeddings to a different random partition of the class labels,
and the concatenated ensemble beats its members: disagreement read as
complementary. The question here is which reading fits the seeds.

A ridge reads the input, in the objective's whitened frame, from a one-hot
code on rows no model trained on, and again from A's code concatenated with
another code X. With EV = 1 - FVU on the test rows,

  overlap(X) = (EV(A) + EV(X) - EV(A+X)) / EV(X)

is the share of what X explains that A already explains: 1 when X adds
nothing to A (a relabelling of A's information), 0 when what X explains is
orthogonal to what A explains. Any added code raises the fit, so B's
overlap is read against codes of B's shape:

  earlier    A at an earlier checkpoint (--earlier): the same model and
             nearly the same partition, the redundant end
  lsh        random-projection heads (`translate.lsh_codes`): arbitrary
             partitions of the input
  untrained  B's architecture at its initialization, as `translate.py`
             builds it

The code is the entries alone; a gain-shape model's gains are not in it,
which handicaps every code alike. Scopes are each layer's heads alone and
the whole stack. Features are the centered one-hot code with no constant
column, and the target is centered on the fit rows. The ridge penalty is
relative to the Gram's mean diagonal (`translate.ridge`), chosen per
condition by FVU on the validation rows, and the fit rows' solution is
scored on the test rows. Gram blocks are accumulated once per scope and
shared between conditions. The whole stack's pairs have 2*l*h*k one-hot
columns (24,320 for 5x76x32), about 5 GB per Gram in float64.

  JAX_PLATFORMS=cpu uv run python experiments/seed-probes/complement.py \\
      --a data/out/sonar/multilingual/ste_h76_init01 \\
      --b data/out/sonar/multilingual/ste_h76_i01_s43 --earlier 350000

Writes <out>/fvu.csv (scope, condition, lambda, validation and test FVU),
<out>/overlap.csv (scope, X, EV(A), EV(X), EV(A+X), overlap, gain) and
<out>/summary.json; default out directory
data/out/sonar/seedprobes/complement_<a>__<b>.
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from jaxtyping import Float, Int

from translate import onehot, ridge

CONTROLS = ("earlier", "lsh", "untrained")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir: the base code")
    p.add_argument("--b", required=True, help="checkpoint dir: the other seed")
    p.add_argument("--step-a", type=int, default=0)
    p.add_argument("--step-b", type=int, default=0)
    p.add_argument("--earlier", type=int, required=True,
                   help="an earlier step of A, the redundant control")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_fresh.npy",
                   help="rows no model trained on")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy",
                   help="per-dimension weights defining the whitened frame")
    p.add_argument("--val", type=int, default=16384)
    p.add_argument("--test", type=int, default=32768,
                   help="the last TEST rows; VAL rows before them choose "
                        "the penalty; the rest fit")
    p.add_argument("--lams", type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                   help="ridge penalties, as shares of the Gram's mean "
                        "diagonal")
    p.add_argument("--no-stack", action="store_true",
                   help="skip the whole-stack scope (layers only)")
    p.add_argument("--dict-init-scale", type=float, default=0.1)
    p.add_argument("--init-cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--out", default=None)
    return p.parse_args()


# ---------- the readout (pure) ----------

def moments(codes: Dict[str, Int[torch.Tensor, "n H"]],
            pairs: Sequence[Tuple[str, str]], Y: Float[torch.Tensor, "n d"],
            k: int, lo: int, hi: int, chunk: int
            ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor],
                       Dict[Tuple[str, str], torch.Tensor], torch.Tensor, int]:
    """Over rows lo:hi, in float64: each code's one-hot column sums and
    F'Y, F_p'F_q for each pair (p, q), the target's sum, and the row
    count."""
    S: Dict[str, torch.Tensor] = {}
    FY: Dict[str, torch.Tensor] = {}
    G: Dict[Tuple[str, str], torch.Tensor] = {}
    sy = torch.zeros(Y.shape[1], dtype=torch.float64)
    for i in range(lo, hi, chunk):
        j = min(i + chunk, hi)
        F = {c: onehot(T[i:j], k, False) for c, T in codes.items()}
        y = Y[i:j]
        sy += y.double().sum(0)
        for c, f in F.items():
            s, fy = f.sum(0).double(), (f.T @ y).double()
            S[c], FY[c] = ((s, fy) if c not in S
                           else (S[c] + s, FY[c] + fy))
        for p, q in pairs:
            g = (F[p].T @ F[q]).double()
            G[(p, q)] = g if (p, q) not in G else G[(p, q)] + g
    return S, FY, G, sy, hi - lo


def block(G: Dict[Tuple[str, str], torch.Tensor], p: str, q: str
          ) -> torch.Tensor:
    return G[(p, q)] if (p, q) in G else G[(q, p)].T


def system(cond: Sequence[str], S, FY, G, sy, n: int
           ) -> Tuple[Float[torch.Tensor, "f f"], Float[torch.Tensor, "f d"],
                      Float[torch.Tensor, "f"]]:
    """The centered normal equations of a ridge on the codes in `cond`
    concatenated: (G_c, B_c, the fit rows' feature mean)."""
    mu = torch.cat([S[c] / n for c in cond])
    Gc = torch.cat([torch.cat([block(G, p, q) for q in cond], 1)
                    for p in cond], 0) - n * torch.outer(mu, mu)
    Bc = torch.cat([FY[c] for c in cond]) - torch.outer(mu, sy)
    return Gc, Bc, mu


def sse(codes: Dict[str, Int[torch.Tensor, "n H"]], cond: Sequence[str],
        W: Float[torch.Tensor, "f d"], mu: Float[torch.Tensor, "f"],
        Y: Float[torch.Tensor, "n d"], k: int, lo: int, hi: int, chunk: int
        ) -> Tuple[float, float]:
    """(squared error of the readout, squared deviation of the target
    about its own mean) over rows lo:hi."""
    W32, off = W.float(), (mu @ W).float()
    err = 0.0
    for i in range(lo, hi, chunk):
        j = min(i + chunk, hi)
        F = torch.cat([onehot(codes[c][i:j], k, False) for c in cond], 1)
        err += float(((F @ W32 - off - Y[i:j]) ** 2).double().sum())
    y = Y[lo:hi].double()
    return err, float(((y - y.mean(0)) ** 2).sum())


def readout(codes: Dict[str, Int[torch.Tensor, "n H"]], cond: Sequence[str],
            S, FY, G, sy, n_fit: int, Y: Float[torch.Tensor, "n d"], k: int,
            n_val: int, n: int, lams: Sequence[float], chunk: int
            ) -> Dict[str, float]:
    """The ridge from the codes in `cond` to Y, fit on rows :n_fit with the
    penalty chosen on the next n_val rows, scored on the rest: its
    penalty and validation and test FVU."""
    Gc, Bc, mu = system(cond, S, FY, G, sy, n_fit)
    best = None
    for lam in lams:
        W = ridge(Gc, Bc, lam)
        e, t = sse(codes, cond, W, mu, Y, k, n_fit, n_fit + n_val, chunk)
        if best is None or e / t < best[1]:
            best = (lam, e / t, W)
    lam, val, W = best
    e, t = sse(codes, cond, W, mu, Y, k, n_fit + n_val, n, chunk)
    return {"lambda": lam, "val_fvu": val, "test_fvu": e / t}


def overlap(ev_a: float, ev_x: float, ev_ax: float) -> float:
    """Share of what X explains that A explains already."""
    return (ev_a + ev_x - ev_ax) / ev_x


# ---------- the experiment ----------

def main() -> None:
    cfg = parse_args()
    torch.set_num_threads(cfg.threads)
    from pareto import load_onto
    from steerembed import init_onto
    from translate import codes_of, lsh_codes

    X = np.asarray(np.load(cfg.cache, mmap_mode="r"), np.float32)
    n = len(X)
    n_fit = n - cfg.val - cfg.test
    w_sqrt = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    Y = torch.as_tensor((X - X[:n_fit].mean(0)) * w_sqrt)
    T = cfg.temperature

    ma, pa, sa = load_onto(cfg.a, cfg.step_a)
    mb, pb, sb = load_onto(cfg.b, cfg.step_b)
    l, h, k = ma.l, ma.h, ma.k
    if (mb.l, mb.h, mb.k) != (l, h, k):
        raise ValueError("A and B differ in shape")
    me, pe, se = load_onto(cfg.a, cfg.earlier)
    mi, pi = init_onto(cfg.b, cfg.seed + 1000, T, cfg.dict_init_scale,
                       cfg.init_cache)
    raw = {"A": codes_of(ma, {"params": pa}, X, T, cfg.batch),
           "B": codes_of(mb, {"params": pb}, X, T, cfg.batch),
           "earlier": codes_of(me, {"params": pe}, X, T, cfg.batch),
           "lsh": lsh_codes(X, l * h, k, cfg.seed, cfg.batch),
           "untrained": codes_of(mi, pi, X, T, cfg.batch)}
    na, nb = Path(cfg.a).name, Path(cfg.b).name
    print(f"A {na} step {sa}; B {nb} step {sb}; earlier step {se}: "
          f"{l}x{h} heads, k={k}")
    print(f"{n} rows of {Path(cfg.cache).name}: fit {n_fit}, validate "
          f"{cfg.val}, test {cfg.test}\n", flush=True)

    scopes = [(f"layer{i}", slice(i * h, (i + 1) * h)) for i in range(l)]
    if not cfg.no_stack:
        scopes.append(("stack", slice(0, l * h)))
    others = ("B",) + CONTROLS
    pairs = [("A", "A")] + [(x, x) for x in others] + [("A", x) for x in others]
    conds = [("A",)] + [(x,) for x in others] + [("A", x) for x in others]
    chunk = 4096
    fvu_rows: List[dict] = []
    ov_rows: List[dict] = []
    for scope, sl in scopes:
        codes = {c: torch.as_tensor(np.ascontiguousarray(v[:, sl]))
                 for c, v in raw.items()}
        S, FY, G, sy, nf = moments(codes, pairs, Y, k, 0, n_fit, chunk)
        res = {}
        for cond in conds:
            r = readout(codes, cond, S, FY, G, sy, nf, Y, k, cfg.val, n,
                        cfg.lams, chunk)
            res["+".join(cond)] = r
            fvu_rows.append({"scope": scope, "condition": "+".join(cond), **r})
        del G
        ev = {c: 1.0 - r["test_fvu"] for c, r in res.items()}
        line = []
        for x in others:
            o = overlap(ev["A"], ev[x], ev[f"A+{x}"])
            ov_rows.append({"scope": scope, "x": x, "ev_a": ev["A"],
                            "ev_x": ev[x], "ev_ax": ev[f"A+{x}"],
                            "overlap": o,
                            "gain": res["A"]["test_fvu"]
                            - res[f"A+{x}"]["test_fvu"]})
            line.append(f"{x} {o:.3f}")
        print(f"{scope}: FVU A {res['A']['test_fvu']:.4f}, B "
              f"{res['B']['test_fvu']:.4f}, A+B {res['A+B']['test_fvu']:.4f}; "
              f"overlap " + ", ".join(line), flush=True)

    out = Path(cfg.out or f"data/out/sonar/seedprobes/complement_{na}__{nb}")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fvu.csv", fvu_rows), ("overlap.csv", ov_rows)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "a": cfg.a, "step_a": int(sa), "b": cfg.b, "step_b": int(sb),
        "earlier": int(se), "cache": cfg.cache, "mse_weights": cfg.mse_weights,
        "rows": n, "fit": n_fit, "val": cfg.val, "test": cfg.test,
        "lams": cfg.lams, "temperature": T, "seed": cfg.seed}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
