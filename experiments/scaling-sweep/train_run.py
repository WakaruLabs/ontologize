"""Train one scaling-sweep config: a faithful port of sonar.py's main()
that reads its constants from a gen_configs.py JSON instead of module
globals. Every field maps 1:1 onto a sonar.py name; nothing about the
training path itself is changed (same Hyperparams/Metadata/TrainingEnv
wiring, same cache-backed EmbeddingLoader, same checkpoint cadence).

  uv run python experiments/scaling-sweep/train_run.py \\
      experiments/scaling-sweep/configs/h32_k32_l5.json

Resumable: TrainingEnv picks up the latest checkpoint in the config's out
dir automatically. Requires the embedding cache (encode_corpus.py) and the
mse_weights npy; GPU strongly recommended (the cache path never touches
the frozen encoder, so this is pure JAX training).
"""
# disable preallocation so jax and torch can share VRAM (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import ctypes
from pathlib import Path

# Force load the pip-installed CuDNN library before JAX initializes
# (verbatim from sonar.py; harmless where the wheel is absent)
try:
    import nvidia.cudnn
    _cudnn = Path(list(nvidia.cudnn.__path__)[0]) / "lib" / "libcudnn.so"
    if _cudnn.exists():
        ctypes.CDLL(str(_cudnn), mode=os.RTLD_GLOBAL)
except (ImportError, IndexError):
    pass

import argparse
import json
import sys

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

from ontologize.training.config import Hyperparams, Metadata, TrainingEnv
from ontologize.data.loaders import NpyDataSource
from ontologize.visualize.loss import plot_loss


def build_env(cfg):
    """sonar.py main()'s wiring, verbatim, from a config dict."""
    out = Path(cfg["out"])
    d = cfg["d"]

    hyper = Hyperparams(
        d, d, cfg["b"], cfg["epochs"], cfg["lr"], cfg["wd"],
        cfg["temperature"],
        cfg["noise_in"], cfg["noise_K"], cfg["noise_F"],
        cfg["sd_in"], cfg["sd_K"], cfg["sd_F"],
        s_g=cfg["s_g"], s_L1K=cfg["s_L1K"], s_L1F=cfg["s_L1F"],
        s_H=cfg["s_H"], s_bcossim=cfg["s_bcossim"],
        s_hcossim=cfg["s_hcossim"], s_Hm=cfg["s_Hm"],
        p_drop=cfg["p_drop"], mse_weights=cfg["mse_weights"],
        temperature_end=cfg["temperature_end"],
        anneal_steps=cfg["anneal_steps"],
        p_drop_start=cfg["p_drop_start"], sd_K_end=cfg["sd_K_end"],
        ghost=cfg["ghost"], seed=cfg["seed"])

    model = hyper.ontologizer(
        d, d, cfg["e_dec"], cfg["k"], cfg["h"], cfg["l"],
        n=cfg["n"], gate=cfg["gate"], scaled=cfg["scaled"],
        encoded=cfg["encoded"], e_enc=cfg["e_enc"],
        forward=cfg["fwd_mode"], deepsup=cfg["deepsup"],
        deepsup_sg=cfg["deepsup_sg"], resid_norm=cfg["resid_norm"],
        resid_const=cfg["resid_const"],
        dtype_str=cfg["dtype_str"], dtype_p_str=cfg["dtype_p_str"])

    meta = Metadata(
        "embedding", cfg["encoder"], out_path=out, threads=0,
        save_each=cfg["save_each"], checkpoint_each=cfg["checkpoint_each"])

    kwargs_loader = {"d": d, "shuffle": cfg["shuffle"]}
    return TrainingEnv(model, hyper, meta, kwargs_loader=kwargs_loader), out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="gen_configs.py JSON")
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    cache = Path(cfg["cache"])
    if not cache.exists():
        raise SystemExit(
            f"embedding cache {cache} missing; run encode_corpus.py first "
            "(the sweep only supports the cache-backed training path)")
    if cfg["mse_weights"] and not Path(cfg["mse_weights"]).exists():
        raise SystemExit(f"mse_weights {cfg['mse_weights']} missing "
                         "(make_mse_weights.py)")

    src = NpyDataSource(cache)
    env, out = build_env(cfg)
    print(f"training {cfg['name']}: h={cfg['h']} k={cfg['k']} l={cfg['l']} "
          f"epochs={cfg['epochs']} -> {out}")
    env.train(src, encoder=None)
    print("Training Complete!")
    try:
        plot_loss(out, base=2)
    except Exception as e:  # the plot is a convenience, not the deliverable
        print(f"warning: plot_loss failed ({e!r}); loss.csv is intact")


if __name__ == "__main__":
    main()
