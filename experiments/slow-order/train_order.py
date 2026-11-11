"""Train the Fano pair objective with the slow heads decorrelated, in order
or symmetrically (`orderloss.py`).

`experiments/next-sentence/train_pair.py` with one more term over the slow
heads, weighed by `--dec` per unit of `--s-pair`. Every flag of
`train_pair.py` applies, with the Fano term at s_pair 1e-3 as the default
(the long runs' setting, so a run here differs from them in `--dec`
alone). `--dec 0` is `train_pair.py`'s Fano arm.

    uv run python experiments/slow-order/train_order.py --dec 1
    uv run python experiments/slow-order/train_order.py --dec 1 --dec-symmetric
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "next-sentence"))
sys.path.insert(0, str(HERE))

import train_pair  # noqa: E402  (sets the XLA environment through train_ste)
import train_ste  # noqa: E402
from orderloss import OrderedPairHyperparams  # noqa: E402
from pairs import StackedPairSource  # noqa: E402

OUT = ROOT / "data" / "out" / "sonar" / "slow-order"


def parser():
    p = train_pair.parser()
    p.description = __doc__
    p.add_argument("--dec", type=float, default=1.0,
                   help="weight of the dependence between slow heads, per "
                        "unit of --s-pair (0: train_pair.py's Fano arm)")
    p.add_argument("--dec-symmetric", action="store_true",
                   help="let the gradient reach both heads of a pair: "
                        "decorrelation without the order")
    p.set_defaults(pair_term="fano", s_pair=1e-3)
    return p


def tag(cfg) -> str:
    kind = "sym" if cfg.dec_symmetric else "ord"
    return (f"{kind}_dec{cfg.dec:g}_sp{cfg.s_pair:g}_ph{cfg.pair_heads}"
            f"_s{cfg.seed or 42}")


def main() -> None:
    cfg = train_ste.finish(parser().parse_args())
    seed = cfg.seed or 42
    cfg.out = cfg.out or str(OUT / tag(cfg))
    rows, partners = train_pair.pairs_for(cfg, train_ste.base.holdout, seed)
    src = StackedPairSource(cfg.cache, rows, partners)
    print(f"{tag(cfg)}: {len(src)} pairs in the first {cfg.rows} rows of "
          f"{Path(cfg.cache).name}; slow heads {cfg.pair_heads or cfg.h} of "
          f"{cfg.h} per layer")
    run = train_ste.setup(
        cfg, src=src, hp=OrderedPairHyperparams,
        hp_extra=dict(s_pair=cfg.s_pair, pair_term=cfg.pair_term,
                      pair_h=cfg.h, pair_heads=cfg.pair_heads, dec=cfg.dec,
                      dec_ordered=not cfg.dec_symmetric))
    if cfg.keep_every:
        meta = run[2]
        assert cfg.keep_every % meta.save_each == 0, \
            f"--keep-every {cfg.keep_every} is not a multiple of the save " \
            f"cadence {meta.save_each}"
        meta.checkpoint_each = cfg.keep_every
    train_ste.train(*run)


if __name__ == "__main__":
    main()
