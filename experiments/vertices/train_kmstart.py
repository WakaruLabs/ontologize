"""Train the one-head recipe from a spherical k-means start (`kmstart.py fit`).

Every flag is `experiments/adam-eps/train_eps.py`'s, so a run and the stock
one-head pair (`experiments/seed-stability/width_seeds.sh`) differ only in
their start. `--km-start` names the start's npz, which `init` installs over
the initialization those flags build, checking that it took.

    uv run python experiments/vertices/train_kmstart.py <width_seeds.sh's
        recipe> --h 1 --seed 43 \\
        --km-start data/out/sonar/vertices/kmstart/fit/start_s42.npz \\
        --out data/out/sonar/vertices/kmstart/km42_s43
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "adam-eps"))
sys.path.insert(0, str(HERE.parent / "ste-arm"))
sys.path.insert(0, str(HERE))
import train_eps  # noqa: E402
import train_ste  # noqa: E402
from kmstart import with_kmstart  # noqa: E402


def parser():
    p = train_eps.parser()
    p.add_argument("--km-start", required=True,
                   help="the start's npz from kmstart.py fit")
    return p


def main() -> None:
    cfg = train_ste.finish(parser().parse_args())
    hp = with_kmstart(train_eps.with_eps(train_eps.flags_class(cfg)))
    run = train_ste.setup(cfg, hp=hp, hp_extra=dict(adam_eps=cfg.adam_eps,
                                                    km_start=cfg.km_start))
    train_ste.train(*run)


if __name__ == "__main__":
    main()
