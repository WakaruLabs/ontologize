"""Retrain `resid_nc`'s configuration on a copy of the codebase that
trained it, with at most one setting changed.

`resid_nc` and `resid_nc_hm` were trained on the project's earlier
codebase, before this repository existed, and differ in three settings:
width, the batch mean-entropy bonus and the between-head cosine penalty.
This script replays that codebase's `sonar.py` with `resid_nc`'s values
(width 4096, neither term), and its flags change one of the three. It
trains from a copy of that codebase's training path, the `ontologize/`
package beside it (commit 4d32e33; see its `__init__.py`), because this
repository's code differs from it in more than those settings. Some known
differences can be switched off in this repository: the L1 gradient,
which that codebase never applied, `resid_gain`, `const0` and the cache
holdout. One cannot: `cossim_h`, the between-head output cosine that
`s_hcossim` weights, is retired, and setting the weight raises. And these
may not be all the differences.

The copy is later than the code that trained `resid_nc`, and it runs in
this repository's newer environment. A run with no flags is the control
that checks neither matters.

Run it from the repository root:

  uv run python experiments/coherence-sign/train_old.py --run resid_nc_ctl

Python puts this script's directory first on `sys.path`, so `import
ontologize` finds the copy rather than this repository's package.

Checkpoints land under this repository's `data/out/sonar/multilingual/`,
next to the runs they are compared with, and the loaders here read them
through `migrate_spec` as they read `resid_nc`. A run resumes from its
latest checkpoint when its directory already holds one, so a crashed run
continues by rerunning its command."""

import argparse
import ctypes
import dataclasses
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True,
                   help="output directory name under --out-root")
    p.add_argument("--s-hcossim", type=float, default=0.0,
                   help="between-head output cosine weight "
                        "(resid_nc_hm: 1e-5)")
    p.add_argument("--s-hm", type=float, default=0.0,
                   help="batch mean-entropy bonus weight (resid_nc_hm: 1e-6)")
    p.add_argument("--e", type=int, default=4096,
                   help="encoder and dictionary width (resid_nc_hm: 2048)")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--out-root",
                   default=str(REPO / "data/out/sonar/multilingual"))
    p.add_argument("--dry-run", action="store_true",
                   help="build the model, print its spec as JSON and exit")
    return p.parse_args()


def main():
    cfg = parse()

    # as the old sonar.py: allocate on demand, and load the pip CuDNN
    # before JAX initializes
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    try:
        import nvidia.cudnn
        lib = Path(list(nvidia.cudnn.__path__)[0]) / "lib" / "libcudnn.so"
        if lib.exists():
            ctypes.CDLL(str(lib), mode=os.RTLD_GLOBAL)
    except (ImportError, IndexError):
        pass

    import ontologize
    if Path(ontologize.__file__).resolve().parent != HERE / "ontologize":
        raise SystemExit(f"imported {ontologize.__file__}, not the copy "
                         f"in {HERE}")
    from ontologize.training.config import Hyperparams, Metadata, TrainingEnv
    from ontologize.data.loaders import NpyDataSource
    from ontologize.visualize.loss import plot_loss

    # byte-identical to the cache and weights resid_nc trained on
    data = REPO / "data"
    cache = data / "sonar_embeddings/mc4_4M.npy"
    out = Path(cfg.out_root).resolve() / cfg.run
    d, k, h, l = 1024, 32, 32, 5
    temperature_end = 0.03

    # resid_nc's log.jsonl env_config, value for value, except the flags
    hyper = Hyperparams(
            d, d, 256, cfg.epochs, 5e-5, 0.0, 1.0,
            "batchnorm", "normal", "featvar", 0.0, 0.02, 0.1,
            s_g=1e-4, s_L1K=0.0, s_L1F=1e-9, s_H=0.0,
            s_bcossim=1e-5, s_hcossim=cfg.s_hcossim, s_Hm=cfg.s_hm,
            p_drop=0.1, mse_weights=str(data / "out/sonar/mse_weights.npy"),
            temperature_end=temperature_end, anneal_steps=50000,
            p_drop_start=0.0, sd_K_end=0.2 * temperature_end,
            ghost=False)
    model = hyper.ontologizer(
            d, d, cfg.e, k, h, l, n=2, gate="none", scaled=False,
            encoded=False, e_enc=cfg.e, forward="resid", deepsup=True,
            deepsup_sg=False, resid_norm=True, resid_const=True,
            dtype_str="float32", dtype_p_str="float32")
    if cfg.dry_run:
        print(json.dumps(dataclasses.asdict(model), default=str))
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(
        {"code": "experiments/coherence-sign/ontologize, a copy of the "
                 "earlier codebase at commit 4d32e33",
         "args": vars(cfg)}, indent=1))
    meta = Metadata("embedding", "cointegrated/SONAR_200_text_encoder",
                    out_path=out, threads=0, save_each=100,
                    checkpoint_each=10000)
    env = TrainingEnv(model, hyper, meta,
                      kwargs_loader={"d": d, "shuffle": True})
    env.train(NpyDataSource(cache), encoder=None)
    print("Training Complete!")
    plot_loss(out, base=2)


if __name__ == "__main__":
    main()
