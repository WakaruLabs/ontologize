"""Per-coordinate error of trained models: does it concentrate on the
coordinates the inverse-variance MSE weights most?

The weights are 1 / var_d (mean 1), so a coordinate with a small variance
takes a large weight. The models train in raw coordinates with Adam, which
moves each parameter by up to about a learning rate per step whatever the
coordinate's scale, so error or movement that does not shrink with a
coordinate's scale counts most, against its variance, on the smallest
coordinate. On the cache tail (the held-out split), per model:

  fvu       whitened FVU, pareto.py's convention
  fvu_d     per coordinate, mean (x_d - xhat_d)^2 / var_d, the coordinate's
            own FVU; the whitened FVU is close to their mean when the
            weights are close to 1 / var on these rows
  share_d   the coordinate's share of the whitened error,
            w_d e_d / sum_d w_d e_d
  drift_d   for a checkpoint directory, mean (xhat_d - xhat'_d)^2 / var_d
            with xhat' the previous kept checkpoint's reconstruction: how
            far the output on the coordinate still moves, on the
            coordinate's own scale
  raw       the error and drift in raw units (root mean square), each over
            its median across coordinates, beside the coordinate's standard
            deviation over the median one: when a model's error scales with
            each coordinate's spread, the ratios are equal
  params    for a checkpoint directory, the root-mean-square change between
            the two checkpoints of the parameters that write each output
            coordinate (a decoder-free model's atoms, else the decoder
            kernel's row for the coordinate), and those parameters' spread,
            each over its median across coordinates

each for the --coords coordinates of largest weight (and any named with
--coord) and as a median over coordinates, and the rank correlation over
coordinates of fvu_d with the weight. A model is an sae.py run's params.npz or an Ontologizer checkpoint
directory, CKPT@T for a classification temperature other than --temperature
(irrelevant under select='ste').

  uv run python experiments/ste-arm/coordfvu.py
  uv run python experiments/ste-arm/coordfvu.py --models P.npz CKPT@0.03

Writes <out>/coords.csv (model, coord, weight, sd, fvu, drift: every
coordinate) and <out>/summary.json; default out data/out/sonar/coordfvu.
Matmuls run at full float32 precision (jax_default_matmul_precision
"highest"), so a GPU run matches a CPU one.
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
from jaxtyping import Float
from scipy.stats import spearmanr

from pareto import load_onto
from reconseeds import is_sae, onto_recon, sae_recon

CONV = "data/out/sonar/sae_conv/"
MULTI = "data/out/sonar/multilingual/"
DEFAULT_MODELS = (
    [CONV + f"{r}/params.npz" for r in [
        "m5120_k32", "m5120_k32_s43", "m11264_k32", "m11264_k160", "m11264_k5120",
        "m5120_l13e-06", "m5120_l11e-05", "m5120_g160top1", "m5120_g160softmax"]]
    + [MULTI + r for r in [
        "ste_h76_init01", "ste_h76_i01_s43", "ste_l1_h380_i01", "ste_l1_h380_i01_s43",
        "ste_k128_h54", "ste_k128_h54_direct", "ste_k128_h54_direct_s43"]]
    + [MULTI + "resid_nc@0.03"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="params.npz files and checkpoint directories (CKPT@T)")
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="classification temperature for a checkpoint without @T")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--rows", type=int, default=32768, help="cache tail rows")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--coords", type=int, default=3,
                   help="largest-weight coordinates to report")
    p.add_argument("--coord", type=int, nargs="*", default=[],
                   help="coordinates to report as well, whatever their weight "
                        "(one a mask set to 0, say)")
    p.add_argument("--out", default="data/out/sonar/coordfvu")
    return p.parse_args()


def report_coords(w: Float[np.ndarray, "d"], n: int, extra: List[int]) -> List[int]:
    """The n largest-weight coordinates, then those of `extra` not among them."""
    top = [int(c) for c in np.argsort(w)[::-1][:n]]
    return top + [int(c) for c in dict.fromkeys(extra) if int(c) not in top]


def kept_steps(ckpt: str) -> List[int]:
    """The steps an orbax checkpoint directory holds, ascending."""
    return sorted(int(p.name) for p in Path(ckpt).iterdir() if p.name.isdigit())


def coord_sums(recon: Callable, X: Float[np.ndarray, "n d"], batch: int,
               prev: Optional[Callable] = None
               ) -> Tuple[Float[np.ndarray, "d"], Optional[Float[np.ndarray, "d"]]]:
    """Mean squared error per coordinate of the final reconstruction, and the
    mean squared change per coordinate against `prev`'s when given."""
    err = np.zeros(X.shape[1])
    drift = np.zeros(X.shape[1]) if prev is not None else None
    for s in range(0, len(X), batch):
        xb = X[s:s + batch]
        Y = np.asarray(recon(xb)[0][-1], np.float64)
        err += ((Y - xb) ** 2).sum(0)
        if prev is not None:
            drift += ((Y - np.asarray(prev(xb)[0][-1], np.float64)) ** 2).sum(0)
    return err / len(X), (drift / len(X) if drift is not None else None)


def coord_stats(err: Float[np.ndarray, "d"], var: Float[np.ndarray, "d"],
                w: Float[np.ndarray, "d"], coords: List[int],
                drift: Optional[Float[np.ndarray, "d"]] = None) -> Dict:
    """The whitened FVU, the per-coordinate FVU's median, mean and rank
    correlation with the weight, and for each of `coords` its FVU, that over
    the median, its share of the whitened error, its raw error over the
    median raw error and, given `drift`, the same three for the drift."""
    fvu_d = err / var
    share = err * w / (err * w).sum()
    med = float(np.median(fvu_d))
    raw = np.sqrt(err)
    out = {"fvu": float((err * w).mean() / (var * w).mean()), "fvu_d_median": med,
           "fvu_d_mean": float(fvu_d.mean()),
           "spearman_fvu_d_weight": float(spearmanr(fvu_d, w)[0]), "coords": {}}
    if drift is not None:
        drift_d = drift / var
        drift_raw = np.sqrt(drift)
        out["drift_d_median"] = float(np.median(drift_d))
    for c in coords:
        r = {"fvu": float(fvu_d[c]), "fvu_over_median": float(fvu_d[c] / med),
             "share": float(share[c]), "err_raw_over_median": float(raw[c] / np.median(raw))}
        if drift is not None:
            r["drift"] = float(drift_d[c])
            r["drift_over_median"] = float(drift_d[c] / out["drift_d_median"])
            r["drift_raw_over_median"] = float(drift_raw[c] / np.median(drift_raw))
        out["coords"][str(int(c))] = r
    return out


def output_params(params: Dict, d: int) -> Float[np.ndarray, "r d"]:
    """The parameters that write the output coordinates, one column per
    coordinate: the decoder kernel's rows (its weights are (out, in)), or a
    decoder-free model's atoms, its dictionaries in embedding space."""
    if "decoder" in params:
        W = np.asarray(params["decoder"]["weights"], np.float64)
        if W.shape[0] != d:
            raise ValueError(f"decoder weights {W.shape} do not map onto {d} coordinates")
        return W.T
    return np.concatenate([np.asarray(params[k]["dict"]["weights"], np.float64).reshape(-1, d)
                           for k in sorted(params) if k.startswith("dictencs_")])


def param_moves(A: Float[np.ndarray, "r d"], B: Float[np.ndarray, "r d"],
                coords: List[int]) -> Dict:
    """The root-mean-square change from A to B per output coordinate and B's
    spread per coordinate, their medians across coordinates, and each over
    its median for `coords`."""
    change = np.sqrt(((B - A) ** 2).mean(0))
    spread = B.std(0)
    cm, sm = float(np.median(change)), float(np.median(spread))
    return {"change_median": cm, "spread_median": sm,
            "coords": {str(int(c)): {"change_over_median": float(change[c] / cm),
                                     "spread_over_median": float(spread[c] / sm)}
                       for c in coords}}


def load(spec: str, T: float) -> Tuple[Callable, Optional[Callable], str, Dict]:
    """A model's reconstruction, the previous kept checkpoint's for a
    checkpoint directory, a name, and what was loaded."""
    if is_sae(spec):
        recon, _, name = sae_recon(spec)
        return recon, None, name, {"path": spec}
    ckpt, t = spec.rsplit("@", 1) if "@" in spec else (spec, T)
    steps = kept_steps(ckpt)
    recon, step, name = onto_recon(ckpt, steps[-1], float(t))
    prev = onto_recon(ckpt, steps[-2], float(t))[0] if len(steps) > 1 else None
    return recon, prev, name, {"path": ckpt, "temperature": float(t), "step": step,
                               "prev_step": steps[-2] if len(steps) > 1 else None}


def main() -> None:
    cfg = parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    mm = np.load(cfg.cache, mmap_mode="r")
    X = np.asarray(mm[-cfg.rows:], np.float64)
    var = X.var(0)
    w = np.load(cfg.mse_weights).astype(np.float64)
    coords = report_coords(w, cfg.coords, cfg.coord)
    sd = np.sqrt(var)
    summary = {"cache": cfg.cache, "rows": cfg.rows, "mse_weights": cfg.mse_weights,
               "coords": {str(c): {"weight": float(w[c]), "sd": float(sd[c]),
                                   "sd_over_median": float(sd[c] / np.median(sd))}
                          for c in coords},
               "sd_median": float(np.median(sd)),
               "weights_vs_inverse_tail_variance_pearson": float(np.corrcoef(w, 1 / var)[0, 1]),
               "models": {}}
    with open(out / "coords.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "coord", "weight", "sd", "fvu", "drift"])
        for spec in cfg.models:
            recon, prev, name, info = load(spec, cfg.temperature)
            err, drift = coord_sums(recon, X.astype(np.float32), cfg.batch, prev)
            s = {**info, **coord_stats(err, var, w, coords, drift)}
            if info.get("prev_step") is not None:
                A = output_params(load_onto(info["path"], info["prev_step"])[1], len(w))
                B = output_params(load_onto(info["path"], info["step"])[1], len(w))
                s["params"] = param_moves(A, B, coords)
            summary["models"][name] = s
            print(name, json.dumps({k: v for k, v in s.items() if k != "path"}), flush=True)
            for c in range(len(w)):
                wr.writerow([name, c, w[c], np.sqrt(var[c]), err[c] / var[c],
                             drift[c] / var[c] if drift is not None else ""])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out / 'summary.json'}, {out / 'coords.csv'}")


if __name__ == "__main__":
    main()
