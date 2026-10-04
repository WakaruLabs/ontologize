"""Retrain configs targeting the layer-4 head freeze, one mitigation per
variant.

The reference failure (sonar.py, s_Hm calibration note): resid_nc froze
22/32 layer-4 heads after step ~211k. Mechanism, per the comments in
sonar.py / ontologize/training/config.py: once the temperature anneal
lands at its floor, a head whose softmax has saturated stops receiving
classifier gradient ("pressure applied before its softmax saturates and
gradients vanish" is exactly what the s_Hm term tries to provide), and
the only remaining mechanisms that reach entries behind a large logit
margin are winner dropout and classifier logit noise.

Each variant changes exactly ONE knob against the live resid_nc_hm
baseline (all other settings copied from sonar.py):

  baseline     the live sonar.py config, for a controlled comparison
               (includes s_Hm=1e-6, the mean-entropy bonus that already
               opposes freezing).
  slow_anneal  anneal_steps 50k -> 150k. The freeze happened AFTER the
               50k anneal finished, so this stretches the hardening
               window in which labels are still soft and every entry
               still receives gradient through the soft mixture
               (config.py: dropout "is redundant while labels are soft;
               its value is in the hardening window" -- the same is true
               of plain softmax gradient). NOTE: p_drop and sd_K share
               the anneal_steps horizon (ontostate.schedules), so their
               ramps stretch with it; that is deliberate here.
  drop_ramp    p_drop 0.1 -> 0.25 (endpoint of the linear ramp; start
               stays 0.0). Winner dropout is the one mechanism that
               "reaches entries behind arbitrarily large margins"
               (sonar.py) -- a saturated head still has its argmax
               masked p_drop of the time, handing the runner-up the win
               and its gradient.
  sdk_floor    sd_K_end floored at sd_K (= 0.02, i.e. no anneal of the
               classifier logit noise). sd_K enters the softmax as
               sd_K / T, so a constant sd_K under the falling
               temperature means effective exploration GROWING to
               0.02 / 0.03 = 0.67 logit units at the anneal floor --
               3x the historically calibrated hard-phase level the
               baseline deliberately anneals down to (sonar.py). That
               excess exploration is the point of this arm: late-phase
               noise keeps sampling runner-ups after margins widen.

Run (from the repo root; each variant is a FULL training run):

  uv run python experiments/layer4-freeze/retrain_variants.py \\
      --variant slow_anneal
  uv run python experiments/layer4-freeze/retrain_variants.py \\
      --variant drop_ramp --out-base data/out/sonar/multilingual

Then diagnose each run with freeze_diag.py and compare freeze onset /
final frozen counts across variants.
"""
# disable preallocation so jax and torch can share VRAM (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent

# ---- baseline: the live resid_nc_hm configuration (sonar.py) ----
BASE = dict(
    d=1024, e_dec=2048, k=32, h=32, l=5,
    n=2, gate="none", scaled=False, encoded=False, e_enc=2048,
    fwd_mode="resid", deepsup=True, deepsup_sg=False,
    resid_norm=True, resid_const=True,
    b=256, epochs=24, lr=5e-5, wd=0.0,
    temperature=1.0, temperature_end=0.03, anneal_steps=50000,
    noise_in="batchnorm", sd_in=0.0,
    noise_K="normal", noise_F="featvar", sd_K=0.02, sd_F=0.1,
    p_drop=0.1, p_drop_start=0.0,
    sd_K_end=0.2 * 0.03,          # 0.2 * temperature_end (sonar.py)
    ghost=False,
    s_g=1e-4, s_L1K=0.0, s_L1F=1e-9, s_H=0.0,
    s_bcossim=1e-5, s_hcossim=1e-6, s_Hm=1e-6,
    dtype_str="float32", dtype_p_str="float32",
    save_each=100, checkpoint_each=10000,
)

# ---- one-knob deltas; rationale in the module docstring ----
VARIANTS = {
    "baseline": {},
    "slow_anneal": {"anneal_steps": 150000},
    "drop_ramp": {"p_drop": 0.25},
    "sdk_floor": {"sd_K_end": BASE["sd_K"]},  # geometric anneal to itself
                                              # = constant sd_K
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True, choices=sorted(VARIANTS),
                   help="which mitigation to train")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy",
                   help="precomputed-embedding cache (encode_corpus.py)")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy",
                   help="per-dim inverse-variance weights; '' disables")
    p.add_argument("--out-base", default=str(EXP_DIR / "runs"),
                   help="run dir parent; the run lands in "
                        "<out-base>/l4fix_<variant>")
    p.add_argument("--epochs", type=int, default=0,
                   help="override epochs (0 = baseline value)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature-end", type=float, default=0.0,
                   help="optional extra override of the anneal floor "
                        "(0 = baseline 0.03); if set, sd_K_end is "
                        "recomputed as 0.2 * floor unless the variant "
                        "pins it")
    return p.parse_args()


def build_cfg(cfg):
    c = dict(BASE)
    c.update(VARIANTS[cfg.variant])
    if cfg.epochs:
        c["epochs"] = cfg.epochs
    if cfg.temperature_end:
        c["temperature_end"] = cfg.temperature_end
        if "sd_K_end" not in VARIANTS[cfg.variant]:
            c["sd_K_end"] = 0.2 * cfg.temperature_end
    return c


def main():
    cfg = parse_args()
    c = build_cfg(cfg)
    out = Path(cfg.out_base) / f"l4fix_{cfg.variant}"

    from ontologize.training.config import Hyperparams, Metadata, TrainingEnv
    from ontologize.data.loaders import NpyDataSource

    d = c["d"]
    hyper = Hyperparams(
        d, d, c["b"], c["epochs"], c["lr"], c["wd"], c["temperature"],
        c["noise_in"], c["noise_K"], c["noise_F"],
        c["sd_in"], c["sd_K"], c["sd_F"],
        s_g=c["s_g"], s_L1K=c["s_L1K"], s_L1F=c["s_L1F"], s_H=c["s_H"],
        s_bcossim=c["s_bcossim"], s_hcossim=c["s_hcossim"], s_Hm=c["s_Hm"],
        p_drop=c["p_drop"],
        mse_weights=cfg.mse_weights or None,
        temperature_end=c["temperature_end"], anneal_steps=c["anneal_steps"],
        p_drop_start=c["p_drop_start"], sd_K_end=c["sd_K_end"],
        ghost=c["ghost"], seed=cfg.seed)

    model = hyper.ontologizer(
        d, d, c["e_dec"], c["k"], c["h"], c["l"],
        n=c["n"], gate=c["gate"], scaled=c["scaled"],
        encoded=c["encoded"], e_enc=c["e_enc"],
        forward=c["fwd_mode"], deepsup=c["deepsup"],
        deepsup_sg=c["deepsup_sg"],
        resid_norm=c["resid_norm"], resid_const=c["resid_const"],
        dtype_str=c["dtype_str"], dtype_p_str=c["dtype_p_str"])

    meta = Metadata(
        "embedding", "cointegrated/SONAR_200_text_encoder", out_path=out,
        threads=0, save_each=c["save_each"],
        checkpoint_each=c["checkpoint_each"])

    src = NpyDataSource(cfg.cache)
    env = TrainingEnv(model, hyper, meta,
                      kwargs_loader={"d": d, "shuffle": True})

    print(f"variant {cfg.variant} -> {out}")
    print(f"  anneal_steps={c['anneal_steps']} "
          f"temperature_end={c['temperature_end']} "
          f"p_drop={c['p_drop']} sd_K_end={c['sd_K_end']}")
    env.train(src, encoder=None)
    print("Training Complete!")

    from ontologize.visualize.loss import plot_loss
    plot_loss(out, base=2)


if __name__ == "__main__":
    main()
