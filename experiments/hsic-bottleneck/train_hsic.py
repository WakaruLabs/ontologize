"""Train the SONAR Ontologizer with HSIC penalties replacing (or joining)
the four live auxiliary loss terms -- the ablation harness for the
hsic-bottleneck experiment.

Arms (--arm):
  aux    the four stock auxiliary terms exactly as sonar.py ships them
         (s_L1F=1e-9, s_bcossim=1e-5, s_hcossim=1e-6, s_Hm=1e-6); no
         HSIC. Control arm -- should reproduce a resid_nc_hm-like run
         through the HSICHyperparams code path.
  hsic   all four stock aux terms zeroed; the pairwise-head CKA penalty
         (and optionally the residual-dependence term) carries all the
         regularization pressure.
  both   stock aux terms AND the HSIC penalties together.

Everything not named above mirrors the live sonar.py resid_nc_hm
configuration (resid forwarding, deepsup, resid_norm/const, schedules,
whitened MSE). ghost is always False here: the probe that exposes the
per-layer codes reuses the ghost slot (see hsic_hyperparams.py), and
the bilinear config makes the ghost path inert anyway (sonar.py).

loss.csv note: column 3 (labeled MSE_ghost by the stock pipeline)
records the raw HSIC penalty for these runs (0 in the aux arm).

Run (from the repo root; each arm is a FULL training run):

  uv run python experiments/hsic-bottleneck/train_hsic.py --arm aux
  uv run python experiments/hsic-bottleneck/train_hsic.py --arm hsic \\
      --s-hsic-heads 1e-4
  uv run python experiments/hsic-bottleneck/train_hsic.py --arm both \\
      --s-hsic-heads 1e-4 --s-hsic-res 1e-4
"""
# disable preallocation so jax and torch can share VRAM (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

# the four live stock aux scales (sonar.py)
AUX = dict(s_L1F=1e-9, s_bcossim=1e-5, s_hcossim=1e-6, s_Hm=1e-6)
ZERO = dict(s_L1F=0.0, s_bcossim=0.0, s_hcossim=0.0, s_Hm=0.0)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=["aux", "hsic", "both"])
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy",
                   help="'' disables target whitening")
    p.add_argument("--out-base", default=str(EXP_DIR / "runs"),
                   help="run dir parent; the run lands in <out-base>/<arm>")
    p.add_argument("--s-hsic-heads", type=float, default=1e-4,
                   help="pairwise-head CKA scale (hsic/both arms). The "
                        "per-layer CKA is O(0.01-0.1) for healthy heads, "
                        "so 1e-4 puts the term at ~1e-6..1e-5 -- the same "
                        "order as the stock aux contributions; sweep "
                        "{1e-5, 1e-4, 1e-3} before quoting a result")
    p.add_argument("--s-hsic-res", type=float, default=0.0,
                   help="HSIC(residual, target) scale (0 = off)")
    p.add_argument("--hsic-sigma2", type=float, default=0.0,
                   help="fixed RBF bandwidth^2 for head grams; 0 = median "
                        "heuristic per head per batch")
    p.add_argument("--hsic-estimator", default="unbiased",
                   choices=["biased", "unbiased"],
                   help="estimator for the residual term")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    cfg = parse_args()
    out = Path(cfg.out_base) / cfg.arm

    from ontologize.training.config import Metadata, TrainingEnv
    from ontologize.data.loaders import NpyDataSource
    from hsic_hyperparams import HSICHyperparams

    # ---- sonar.py resid_nc_hm baseline ----
    d, e_dec, k, h, l = 1024, 2048, 32, 32, 5
    b = 256

    aux = AUX if cfg.arm in ("aux", "both") else ZERO
    s_hsic_heads = cfg.s_hsic_heads if cfg.arm in ("hsic", "both") else 0.0
    s_hsic_res = cfg.s_hsic_res if cfg.arm in ("hsic", "both") else 0.0

    hyper = HSICHyperparams(
        d, d, b, cfg.epochs, 5e-5, 0.0, 1.0,
        "batchnorm", "normal", "featvar", 0.0, 0.02, 0.1,
        s_g=0.0, s_L1K=0.0, s_H=0.0, **aux,
        p_drop=0.1, p_drop_start=0.0,
        mse_weights=cfg.mse_weights or None,
        temperature_end=0.03, anneal_steps=50000,
        sd_K_end=0.2 * 0.03,
        ghost=False, seed=cfg.seed,
        s_hsic_heads=s_hsic_heads, s_hsic_res=s_hsic_res,
        hsic_h=h, hsic_sigma2=cfg.hsic_sigma2,
        hsic_estimator=cfg.hsic_estimator)

    model = hyper.ontologizer(
        d, d, e_dec, k, h, l, n=2, gate="none",
        scaled=False, encoded=False, e_enc=2048,
        forward="resid", deepsup=True, deepsup_sg=False,
        resid_norm=True, resid_const=True,
        dtype_str="float32", dtype_p_str="float32")

    meta = Metadata(
        "embedding", "cointegrated/SONAR_200_text_encoder", out_path=out,
        threads=0, save_each=100, checkpoint_each=10000)

    src = NpyDataSource(cfg.cache)
    env = TrainingEnv(model, hyper, meta,
                      kwargs_loader={"d": d, "shuffle": True})

    print(f"arm {cfg.arm} -> {out}")
    print(f"  aux={aux}  s_hsic_heads={s_hsic_heads} "
          f"s_hsic_res={s_hsic_res}")
    env.train(src, encoder=None)
    print("Training Complete!")

    from ontologize.visualize.loss import plot_loss
    plot_loss(out, base=2)


if __name__ == "__main__":
    main()
