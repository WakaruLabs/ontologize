"""Train an arm of `experiments/ste-arm` on a Gaussian copy of its cache.

Every flag is `experiments/adam-eps/train_eps.py`'s (so `train_ste.py`'s
plus `--adam-eps`), with `gausscopy.GaussianSource` in place of the
cache's rows: `--rows` rows from the moments `gausscopy.py prepare`
wrote. `--cache` names the held-out Gaussian file, which the
initialization probe reads; pass `--epochs` rather than `--steps`.

    uv run python experiments/seed-stability/train_gauss.py \\
        --cache data/out/sonar/gausscopy/eval.npy --epochs 1 ... \\
        --out data/out/sonar/gausscopy/<arm>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "adam-eps"))
sys.path.insert(0, str(HERE))

import train_eps  # noqa: E402  (puts experiments/ste-arm on the path)
import train_ste  # noqa: E402
from gausscopy import OUT, GaussianSource  # noqa: E402


def parser():
    p = train_eps.parser()
    p.description = __doc__
    p.add_argument("--moments", default=str(OUT / "moments.npz"))
    p.add_argument("--rows", type=int, default=12_800_000,
                   help="rows per epoch: the ladder's 50,000 steps of 256")
    p.add_argument("--data-seed", type=int, default=0,
                   help="the Gaussian stream (the held-out file uses 1)")
    return p


def main() -> None:
    cfg = train_ste.finish(parser().parse_args())
    assert Path(cfg.moments).exists(), \
        f"{cfg.moments} missing: run gausscopy.py prepare first"
    src = GaussianSource(cfg.moments, cfg.rows, cfg.data_seed)
    run = train_ste.setup(cfg, src=src, hp=train_eps.with_eps(train_eps.flags_class(cfg)),
                          hp_extra=dict(adam_eps=cfg.adam_eps))
    # read in order: the rows are independent draws, so a shuffle changes
    # nothing about the data, and it would make each batch of 256 touch
    # about 256 different blocks, every one generated afresh
    train_ste.train(*run, shuffle=False)


if __name__ == "__main__":
    main()
