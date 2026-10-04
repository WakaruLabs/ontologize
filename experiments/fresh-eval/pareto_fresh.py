"""Reconstruction Pareto on genuinely held-out rows, for both architectures.

pareto.py's own caveat: the cache tail is held out only for the SAEs (the
Ontologizer trained on the full cache), so its curve favors the
Ontologizer end. This harness re-runs the identical comparison --
pareto.sae_points and pareto.onto_points, unmodified -- but scores on the
fresh cache written by encode_fresh.py (stream rows past the training
corpus), which neither architecture has seen. The E[p] origin is still
measured on TRAINING rows (--train-cache head), preserving pareto.py's
train/eval separation for the origin rather than fitting it to eval data.

  uv run python experiments/fresh-eval/pareto_fresh.py
  uv run python experiments/fresh-eval/pareto_fresh.py \\
      --ckpt data/out/sonar/multilingual/resid_nc --code both --plot

Writes <out>/pareto_fresh.csv (label, coeffs, index_bits, fvu_w), and
optionally pareto_fresh.png. Compare row-for-row against the original
data/out/sonar/pareto/pareto.csv: any gap that opens on the onto points
(and doesn't on the sae points) was the train/eval-tail asymmetry.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))

import argparse
import csv
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fresh-cache",
                   default="experiments/fresh-eval/out/mc4_fresh.npy",
                   help="held-out cache written by encode_fresh.py")
    p.add_argument("--train-cache",
                   default="data/sonar_embeddings/mc4_4M.npy",
                   help="training cache; used ONLY for the E[p] origin")
    p.add_argument("--ckpt", default="data/out/sonar/multilingual/resid_nc_hm",
                   help="Ontologizer checkpoint dir (resid_nc_hm / resid_nc)")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--ms", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 16])
    p.add_argument("--code", choices=["dev", "head", "both"], default="dev")
    p.add_argument("--origin", choices=["meanp", "uniform"], default="meanp")
    p.add_argument("--origin-rows", type=int, default=65536)
    p.add_argument("--sae", nargs="*", default=None,
                   help="params.npz paths (default: data/out/sonar/sae/*/)")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--eval-rows", type=int, default=0,
                   help="fresh rows to score (0 = the whole fresh cache)")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--out", default="experiments/fresh-eval/out")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main():
    import jax.numpy as jnp
    import pareto

    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.fresh_cache, mmap_mode="r")
    n_eval = min(cfg.eval_rows or mm.shape[0], mm.shape[0])
    X_eval = np.asarray(mm[-n_eval:], dtype=np.float32)
    w = np.load(cfg.mse_weights) if cfg.mse_weights else np.ones(mm.shape[1])
    base_w = (X_eval.var(0) * w).mean()
    print(f"fresh eval: {n_eval} rows from {cfg.fresh_cache} "
          f"(origin from {cfg.train_cache} head)")

    # pareto.{sae,onto}_points read cfg.cache only for the E[p] origin;
    # eval rows are passed explicitly, so pointing cache at the TRAINING
    # cache keeps the origin measured on training data while every FVU is
    # scored on the fresh slice.
    ns = argparse.Namespace(
        ckpt=cfg.ckpt, step=cfg.step, temperature=cfg.temperature,
        ms=cfg.ms, code=cfg.code, origin=cfg.origin,
        origin_rows=cfg.origin_rows, sae=cfg.sae, cache=cfg.train_cache,
        b=min(cfg.b, len(X_eval)))

    points = pareto.sae_points(ns, X_eval,
                               jnp.asarray(np.sqrt(w), jnp.float32), base_w)
    points += pareto.onto_points(ns, X_eval, w, base_w)
    points.sort(key=lambda r: (r[1], r[2]))

    print(f"\n{'point':<28} {'coeffs':>7} {'idx bits':>9} {'FVU_w':>8}")
    for label, coeffs, bits, fvu in points:
        print(f"{label:<28} {coeffs:>7} {bits:>9} {fvu:>8.4f}")

    with open(out / "pareto_fresh.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "coeffs", "index_bits", "fvu_w"])
        writer.writerows(points)
    print(f"\n-> {out / 'pareto_fresh.csv'}")

    if cfg.plot:  # same rendering as pareto.py --plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, coeffs, _, fvu in points:
            if coeffs == 0:
                continue
            c, mk = (("tab:orange", "s") if label.startswith("sae") else
                     ("tab:green", "^") if label.startswith("onto heads") else
                     ("tab:blue", "o"))
            ax.scatter(coeffs, fvu, c=c, marker=mk)
            ax.annotate(label, (coeffs, fvu), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("continuous coefficients / sample")
        ax.set_ylabel("FVU (whitened, fresh rows)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "pareto_fresh.png", dpi=150)
        print(f"-> {out / 'pareto_fresh.png'}")


if __name__ == "__main__":
    main()
