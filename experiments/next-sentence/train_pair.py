"""Train a code on pairs of sentences from one document, with slow heads
that pay for every switch between the two (`pair_objective.md`).

Each record is two sentences of one document, and both are reconstructed
from their own codes. The first `--pair-heads` heads of each layer are the
slow heads: `--s-pair` times the share of them whose entry differs between
the two sentences is added to the loss (`pairloss.py`). `--s-pair 0` is the
control, trained on the same pairs, which are drawn with the run's seed.

Pairs:
- `--pairs doc` (default): every row of the training prefix that has one,
  with a partner drawn from its own document at segment distance at least
  `--min-d`;
- `--pairs adjacent`: each row and the next segment of its document.
`--sentences-only` keeps the pairs whose two rows are both full sentences.

Every flag of `experiments/ste-arm/train_ste.py` applies, with
`train_next.py`'s defaults (`RECIPE`): the shipped STE recipe on one layer of
380 heads, under the masked MSE weights. Pairs come from the cache's first
`--rows` rows; the tail is held out, as for every model trained on the cache.

    uv run python experiments/next-sentence/train_pair.py --s-pair 0.1
    uv run python experiments/next-sentence/train_pair.py --s-pair 0

`--calibrate N` trains nothing. It prints the norms of the stock loss's
gradient and the pair term's at initialization, on the classifier and on
every parameter, averaged over N batches. Their ratio on the classifier is
the s_pair at which the two pull on it equally, the centre of a sweep.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# importing train_next puts experiments/ste-arm on sys.path and imports
# train_ste, which sets the XLA environment as running it would
import train_next  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Int  # noqa: E402

from ontologize.data.loaders import doc_holdout  # noqa: E402
from pairloss import PairHyperparams  # noqa: E402
from pairs import StackedPairSource, adjacent, doc_partners  # noqa: E402

train_ste = train_next.train_ste


def parser() -> argparse.ArgumentParser:
    p = train_ste.parser()
    p.description = __doc__
    p.add_argument("--pairs", choices=("doc", "adjacent"), default="doc",
                   help="a random other sentence of the document, at least "
                        "--min-d segments away; or the next segment")
    p.add_argument("--min-d", type=int, default=2,
                   help="--pairs doc: the smallest segment distance")
    p.add_argument("--sentences-only", action="store_true",
                   help="keep pairs whose two rows are both full sentences, "
                        "not lines or pieces the length cap cut")
    p.add_argument("--s-pair", type=float, default=0.0,
                   help="weight of the slow heads' switch share (0: the "
                        "control)")
    p.add_argument("--pair-heads", type=int, default=38,
                   help="slow heads per layer, the first that many (0 = all)")
    p.add_argument("--rows", type=int, default=4_000_000,
                   help="draw pairs from the cache's first ROWS rows")
    p.add_argument("--calibrate", type=int, default=0, metavar="N",
                   help="print the two gradient norms over N batches at "
                        "initialization, and train nothing")
    p.set_defaults(**train_next.RECIPE)
    return p


def calibrate(hyper: PairHyperparams, model, src: StackedPairSource,
              n_batches: int, seed: int) -> Dict[str, float]:
    """Mean gradient norms at initialization of the stock loss and of the
    pair term, over every parameter and over the classifiers' (the only
    parameters the pair term reaches), on `n_batches` random batches with
    the step's own noise, dropout and temperature."""
    import jax
    from ontologize.training.config import Hyperparams
    from pairloss import switch_share, unstack

    state = hyper.init(model)
    kw = dict(temperature=hyper.temperature, p_drop=hyper.p_drop,
              sd_K=hyper.sd_K, sd_in=hyper.sd_in, sd_F=hyper.sd_F,
              p_revive=hyper.p_revive, revive_frac=hyper.revive_frac,
              pwak_s=hyper.pwak_s, pwak_tau=hyper.pwak_tau)

    def part(which: int):
        def f(params, X, rng):
            Y, P_all, stats, _ = state.apply_fn(params, X, rng=rng, **kw)
            if which == 0:
                return Hyperparams.loss(hyper, Y, unstack(X), None, stats)[0]
            return switch_share(P_all, hyper.pair_h, hyper.pair_heads)
        return jax.jit(jax.grad(f))

    def norms(g) -> Tuple[float, float]:
        leaves = jax.tree_util.tree_flatten_with_path(g)[0]
        sq = [(any("classifier" in str(k) for k in path),
               float((v.astype(np.float64) ** 2).sum())) for path, v in leaves]
        return (np.sqrt(sum(s for _, s in sq)),
                np.sqrt(sum(s for c, s in sq if c)))

    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    grads = (part(0), part(1))
    acc = np.zeros((2, 2))
    for _ in range(n_batches):
        X = np.stack([src[m] for m in rng.integers(len(src), size=hyper.b)])
        key, sub = jax.random.split(key)
        for i, g in enumerate(grads):
            acc[i] += norms(g(state.params, X, sub))
    acc /= n_batches
    out = {"stock_all": acc[0, 0], "stock_classifier": acc[0, 1],
           "pair_all": acc[1, 0], "pair_classifier": acc[1, 1],
           "s_pair_equal_pull": acc[0, 1] / max(acc[1, 1], 1e-30)}
    for k, v in out.items():
        print(f"  {k}: {v:.4g}")
    return out


def pairs_for(cfg: argparse.Namespace, holdout: int, seed: int
              ) -> Tuple[Int[np.ndarray, "p"], Int[np.ndarray, "p"]]:
    """(rows, partners) within the training prefix: the cache's first
    `cfg.rows` rows, short of the document-snapped held-out tail."""
    cache = Path(cfg.cache)
    n = len(np.load(cache, mmap_mode="r"))
    hi = min(cfg.rows, n - doc_holdout(cache, holdout))
    index = np.load(cache.with_name(cache.stem + ".index.npy"), mmap_mode="r")
    if cfg.pairs == "doc":
        rows, partners = doc_partners(index, 0, hi, cfg.min_d,
                                      np.random.default_rng(seed))
    else:
        rows = adjacent(index, 0, hi)
        partners = rows + 1
    if cfg.sentences_only:
        kind = np.load(cache.with_name(cache.stem + ".kind.npy"),
                       mmap_mode="r")
        keep = ((np.asarray(kind[rows]) == "sentence")
                & (np.asarray(kind[partners]) == "sentence"))
        rows, partners = rows[keep], partners[keep]
    return rows, partners


def tag(cfg: argparse.Namespace) -> str:
    """The run's name: pair type, slow-head weight and count, shape, seed."""
    pairs = f"doc{cfg.min_d}" if cfg.pairs == "doc" else "adj"
    sent = "_sent" if cfg.sentences_only else ""
    return (f"{pairs}{sent}_sp{cfg.s_pair:g}_ph{cfg.pair_heads}"
            f"_h{cfg.h}_l{cfg.l}_s{cfg.seed or 42}")


def main() -> None:
    cfg = train_ste.finish(parser().parse_args())
    seed = cfg.seed or 42
    cfg.out = cfg.out or str(train_next.OUT / "pair" / tag(cfg))
    rows, partners = pairs_for(cfg, train_ste.base.holdout, seed)
    src = StackedPairSource(cfg.cache, rows, partners)
    print(f"{tag(cfg)}: {len(src)} pairs in the first {cfg.rows} rows of "
          f"{Path(cfg.cache).name}; slow heads {cfg.pair_heads or cfg.h} of "
          f"{cfg.h} per layer")
    run = train_ste.setup(
        cfg, src=src, hp=PairHyperparams,
        hp_extra=dict(s_pair=cfg.s_pair, pair_h=cfg.h,
                      pair_heads=cfg.pair_heads))
    if cfg.calibrate:
        calibrate(run[0], run[1], src, cfg.calibrate, seed)
        return
    train_ste.train(*run)


if __name__ == "__main__":
    main()
