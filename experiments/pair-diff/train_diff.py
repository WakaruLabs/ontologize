"""Train a code on differences of two sentences from one document.

The stock straight-through objective, with x_a - x_b as the row
(`diffs.DiffSource`) in place of a sentence, under the differences' own
whitening (`diffs.py` writes it). Every flag of
`experiments/next-sentence/train_pair.py` applies, with its defaults: one
layer of 380 heads, the shipped recipe, pairs drawn at least two segments
apart from the cache's first 4M rows. The pair-term flags are unused: a
difference has no partner.

    uv run python experiments/pair-diff/diffs.py        # once
    uv run python experiments/pair-diff/train_diff.py --epochs 1
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "next-sentence"))
sys.path.insert(0, str(HERE))

import train_pair  # noqa: E402  (sets the XLA environment through train_ste)
import train_ste  # noqa: E402
from diffs import OUT, DiffSource  # noqa: E402


def parser():
    p = train_pair.parser()
    p.description = __doc__
    p.set_defaults(mse_weights=str(OUT / "mse_weights_diff.npy"), s_pair=0.0)
    return p


def main() -> None:
    cfg = train_ste.finish(parser().parse_args())
    if cfg.s_pair:
        raise SystemExit("a difference has no partner: --s-pair must be 0")
    if not Path(cfg.mse_weights).exists():
        raise SystemExit(f"{cfg.mse_weights} is missing: run diffs.py first")
    seed = cfg.seed or 42
    cfg.out = cfg.out or str(OUT / f"diff_h{cfg.h}_l{cfg.l}_s{seed}")
    rows, partners = train_pair.pairs_for(cfg, train_ste.base.holdout, seed)
    src = DiffSource(cfg.cache, rows, partners)
    print(f"{Path(cfg.out).name}: {len(src)} differences from the first "
          f"{cfg.rows} rows of {Path(cfg.cache).name}")
    run = train_ste.setup(cfg, src=src)
    if cfg.keep_every:
        meta = run[2]
        assert cfg.keep_every % meta.save_each == 0, \
            f"--keep-every {cfg.keep_every} is not a multiple of the save " \
            f"cadence {meta.save_each}"
        meta.checkpoint_each = cfg.keep_every
    train_ste.train(*run)


if __name__ == "__main__":
    main()
