"""Second-seed Ontologizer replica: sonar.py's exact live config, seed
changed.

The SAE side already has a seed-stability protocol (sae.py --seed 43 ->
splitting.py match fractions: 0.31/0.24/0.15/0.03 at tau 0.3/0.5/0.7/0.9
for k32 vs k32_s43). The Ontologizer has no replica to run that protocol
on. This script trains one.

It does NOT copy sonar.py's constants -- it imports sonar.py as a module
and reads every hyperparameter from it, so the replica tracks the live
config by construction (if sonar.py's constants change, so does this).
Only three things differ from the primary run:

  seed        Hyperparams.seed (--seed, default 43): parameter init and
              all training noise/dropout draws
  data seed   the cached-corpus shuffle seed (--data-seed, default =
              --seed; sonar.py leaves the loader at its default 42) --
              matching sae.py's replica, where --seed changes both init
              and the data order
  out         a fresh checkpoint dir (--out, default <sonar out>_s<seed>)

Refuses to write into sonar.py's own output directory.

  uv run python experiments/seed-stability/train_seed43.py
  uv run python experiments/seed-stability/train_seed43.py --epochs 2  # pilot

Then run match_heads.py on the pair (and/or splitting.py for the
unconstrained entry-level frame).
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))

import argparse

# the live training config: every constant below comes from here, so the
# replica cannot drift from the primary run's architecture/schedule.
# (importing sonar also preloads CuDNN and picks the torch/jax device,
# exactly as running it directly would)
import sonar as base

from ontologize.training.config import Hyperparams, Metadata, TrainingEnv
from ontologize.data.loaders import NpyDataSource
from ontologize.visualize.loss import plot_loss


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=43,
                   help="Hyperparams seed (init + noise/dropout draws)")
    p.add_argument("--data-seed", type=int, default=None,
                   help="cache shuffle seed (default: --seed)")
    p.add_argument("--out", default=None,
                   help="checkpoint dir (default: <sonar.py out>_s<seed>)")
    p.add_argument("--cache", default=None,
                   help="embedding cache (default: sonar.py's, "
                        "data/sonar_embeddings/mc4_4M.npy)")
    p.add_argument("--epochs", type=int, default=0,
                   help="override sonar.py's epoch count (0 = same; use a "
                        "small value for a pilot run)")
    return p.parse_args()


def main():
    cfg = parse_args()
    cache = Path(cfg.cache) if cfg.cache else base.cache
    assert cache and Path(cache).exists(), \
        f"embedding cache {cache} missing (encode_corpus.py builds it); " \
        "the replica only supports the cached-embedding path"
    out = Path(cfg.out) if cfg.out else \
        base.out.with_name(base.out.name + f"_s{cfg.seed}")
    assert out.resolve() != Path(base.out).resolve(), \
        "refusing to write into sonar.py's own output directory"
    data_seed = cfg.data_seed if cfg.data_seed is not None else cfg.seed
    epochs = cfg.epochs or base.epochs
    print(f"replica of {base.out.name}: seed {cfg.seed} "
          f"(data seed {data_seed}) -> {out}")

    src = NpyDataSource(cache)

    # mirror sonar.main()'s construction argument-for-argument, reading
    # every value from the sonar module; only seed/epochs are overridden
    hyper = Hyperparams(
        base.d, base.d, base.b, epochs, base.lr, base.wd, base.temperature,
        base.noise_in, base.noise_K, base.noise_F,
        base.sd_in, base.sd_K, base.sd_F,
        s_g=base.s_g, s_L1K=base.s_L1K, s_L1F=base.s_L1F, s_H=base.s_H,
        s_bcossim=base.s_bcossim, s_hcossim=base.s_hcossim, s_Hm=base.s_Hm,
        p_drop=base.p_drop, mse_weights=base.mse_weights,
        temperature_end=base.temperature_end, anneal_steps=base.anneal_steps,
        p_drop_start=base.p_drop_start, sd_K_end=base.sd_K_end,
        ghost=base.ghost,
        seed=cfg.seed,
    )

    model = hyper.ontologizer(
        base.d, base.d, base.e_dec, base.k, base.h, base.l,
        n=base.n, gate=base.gate, scaled=base.scaled,
        encoded=base.encoded, e_enc=base.e_enc,
        forward=base.fwd_mode, deepsup=base.deepsup,
        deepsup_sg=base.deepsup_sg,
        resid_norm=base.resid_norm, resid_const=base.resid_const,
        dtype_str=base.dtype_str, dtype_p_str=base.dtype_p_str)

    meta = Metadata(
        "embedding", base.encoder, out_path=out,
        threads=base.threads,
        save_each=base.save_each, checkpoint_each=base.checkpoint_each)

    # sonar.py's cached path: no pretrained encoder in the loop; the extra
    # seed kwarg reshuffles the cached corpus differently from the primary
    kwargs_loader = {"d": base.d, "shuffle": base.shuffle, "seed": data_seed}

    env = TrainingEnv(model, hyper, meta, kwargs_loader=kwargs_loader)
    env.train(src, encoder=None)
    print("Training Complete!")

    plot_loss(out, base=2)


if __name__ == "__main__":
    main()
