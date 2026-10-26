"""A coordinate's error over an Ontologizer run's checkpoints, and its share
of what each layer classifies.

coordfvu.py scores a model's last kept checkpoint against the one before.
This follows one run through its checkpoints on the cache tail (the
held-out split), to see when a coordinate's error starts to grow and how
much of the residual the coordinate comes to hold. Per checkpoint:

  fvu                  whitened FVU under --mse-weights, pareto.py's
                       convention
and per --coord coordinate c:
  fvu_over_median      c's own FVU, mean (x_c - xhat_c)^2 / var_c, over the
                       median coordinate's
  err_raw_over_median  c's root-mean-square error over the median
                       coordinate's, in raw units
  resid_share_i        for each layer i, the mean over rows of c's share of
                       the squared norm of the residual layer i classifies:
                       x itself for layer 0, x - xhat_{<i} above it, with
                       xhat_{<i} the reconstruction of layers 0 to i-1
  param_spread_over_median  the spread of the parameters that write c (a
                       decoder-free model's atoms, else the decoder kernel's
                       row for c; coordfvu.py's output_params) over the
                       median coordinate's
  param_change_over_median  their root-mean-square change since the
                       previous step traced, over the median coordinate's;
                       blank at the first

resid_const's constant coordinate is left out of the residual. resid_norm
and resid_gain rescale each row, which leaves the shares unchanged. A model
with an encoder classifies the encoded input at layer 0, so its
resid_share_0 is the input's rather than the layer's. The shares above
layer 0 need the per-prefix decodes a model returns under deepsup; without
it only resid_share_0 is reported.

  uv run python experiments/ste-arm/coordtrace.py --ckpt CKPT --coord 837 \\
      --steps 25000 75000 344250

Writes <out>/trace.csv (step, coord, fvu, fvu_over_median,
err_raw_over_median, param_spread_over_median, param_change_over_median,
resid_share_0, ...) and <out>/summary.json; default out
data/out/sonar/coordtrace/<run>. Matmuls run at full float32 precision, so a
GPU run matches a CPU one.
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
from jaxtyping import Float

from coordfvu import coord_stats, kept_steps, output_params, param_moves
from pareto import load_onto
from reconseeds import is_sae, onto_recon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Ontologizer checkpoint directory")
    p.add_argument("--coord", type=int, nargs="+", required=True,
                   help="coordinates to follow")
    p.add_argument("--steps", type=int, nargs="*", default=[],
                   help="checkpoint steps (default: every kept one)")
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="classification temperature (irrelevant under select='ste')")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--rows", type=int, default=32768, help="cache tail rows")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--out", default="", help="default data/out/sonar/coordtrace/<run>")
    return p.parse_args()


def residual_shares(X: Float[np.ndarray, "b d"], Y: Float[np.ndarray, "p b d"],
                    coords: List[int]) -> Float[np.ndarray, "p c"]:
    """Summed over rows, each of `coords`' share of the squared norm of the
    residual each layer classifies: X for layer 0, X - Y[i-1] for layer i,
    with Y the per-prefix reconstructions."""
    R = np.concatenate([X[None], X[None] - Y[:-1]])
    return (R[..., coords] ** 2 / (R ** 2).sum(-1, keepdims=True)).sum(1)


def trace_sums(recon: Callable, X: Float[np.ndarray, "n d"], batch: int,
               coords: List[int]) -> Tuple[Float[np.ndarray, "d"], Float[np.ndarray, "p c"]]:
    """Mean squared error per coordinate of the final reconstruction, and
    `residual_shares` averaged over rows."""
    err = np.zeros(X.shape[1])
    share = None
    for s in range(0, len(X), batch):
        xb = X[s:s + batch]
        Y = np.asarray(recon(xb)[0], np.float64)
        x = xb.astype(np.float64)
        err += ((Y[-1] - x) ** 2).sum(0)
        sh = residual_shares(x, Y, coords)
        share = sh if share is None else share + sh
    return err / len(X), share / len(X)


def main() -> None:
    cfg = parse_args()
    if is_sae(cfg.ckpt):
        raise SystemExit("coordtrace.py follows an Ontologizer's layers; "
                         "use coordfvu.py for an sae.py run")
    jax.config.update("jax_default_matmul_precision", "highest")
    out = Path(cfg.out) if cfg.out else Path("data/out/sonar/coordtrace") / Path(cfg.ckpt).name
    out.mkdir(parents=True, exist_ok=True)
    mm = np.load(cfg.cache, mmap_mode="r")
    X = np.asarray(mm[-cfg.rows:], np.float32)
    var = X.astype(np.float64).var(0)
    w = np.load(cfg.mse_weights).astype(np.float64)
    steps = cfg.steps or kept_steps(cfg.ckpt)
    summary = {"ckpt": cfg.ckpt, "cache": cfg.cache, "rows": cfg.rows,
               "mse_weights": cfg.mse_weights, "temperature": cfg.temperature,
               "coords": cfg.coord, "steps": {}}
    P_prev = None
    with open(out / "trace.csv", "w", newline="") as f:
        wr = csv.writer(f)
        for i, step in enumerate(steps):
            recon, used, name = onto_recon(cfg.ckpt, step, cfg.temperature)
            err, share = trace_sums(recon, X, cfg.batch, cfg.coord)
            s = coord_stats(err, var, w, cfg.coord)
            P = output_params(load_onto(cfg.ckpt, used)[1], X.shape[1])
            spread = P.std(0)
            moves = param_moves(P_prev, P, cfg.coord)["coords"] if P_prev is not None else None
            P_prev = P
            if i == 0:
                wr.writerow(["step", "coord", "fvu", "fvu_over_median", "err_raw_over_median",
                             "param_spread_over_median", "param_change_over_median"]
                            + [f"resid_share_{l}" for l in range(len(share))])
            rec = {"fvu": s["fvu"], "coords": {}}
            for j, c in enumerate(cfg.coord):
                r = s["coords"][str(c)]
                ps = float(spread[c] / np.median(spread))
                pc = moves[str(c)]["change_over_median"] if moves is not None else None
                rec["coords"][str(c)] = {"fvu_over_median": r["fvu_over_median"],
                                         "err_raw_over_median": r["err_raw_over_median"],
                                         "param_spread_over_median": ps,
                                         "param_change_over_median": pc,
                                         "resid_share": [float(v) for v in share[:, j]]}
                wr.writerow([used, c, s["fvu"], r["fvu_over_median"], r["err_raw_over_median"],
                             ps, "" if pc is None else pc] + [float(v) for v in share[:, j]])
                print(f"{name}: FVU {s['fvu']:.4f}  coord {c}: err/median "
                      f"{r['err_raw_over_median']:.1f}, FVU/median {r['fvu_over_median']:.1f}, "
                      f"params spread/median {ps:.2f}, change/median "
                      + ("-" if pc is None else f"{pc:.2f}") + ", "
                      "share of residual by layer " + " ".join(f"{v:.3f}" for v in share[:, j]),
                      flush=True)
            summary["steps"][str(used)] = rec
            f.flush()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out / 'trace.csv'}, {out / 'summary.json'}")


if __name__ == "__main__":
    main()
