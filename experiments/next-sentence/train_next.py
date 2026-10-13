"""Train a code on SONAR sentence pairs, with the next sentence as its target.

The model reads a sentence's embedding x_t and is trained to output the
embedding of the sentence that follows it in the same document, x_{t+1}
(`--objective next`). The control (`--objective self`) trains the same
architecture on the same input rows to reconstruct x_t itself. Comparing the
two isolates what the objective does to the code. The question it was built
for is identifiability: unsupervised codes that see only the data cannot
recover its factors without matching inductive biases (Locatello et al.
2019), and temporal structure is one source of identifiability that
literature points to. Does a next-sentence code reproduce across seeds
better than a reconstruction code? See README.md.

Every flag of `experiments/ste-arm/train_ste.py` applies, with this
experiment's defaults:
- one layer of 380 heads, k from sonar.py (32): 1900 bits;
- `--dict-init-scale 0.1 --s-hm 1e-4`, the best flat arm in the ste-arm notes;
- the `mc4_sent16` cache and its own MSE weights.

Training pairs come from the cache's first `--rows` rows (4M by default,
about 16 GB, which stays in the page cache). The cache tail, snapped to whole
documents, is held out for `eval_next.py`.

    uv run python experiments/next-sentence/train_next.py --objective next
    uv run python experiments/next-sentence/train_next.py --objective self
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
sys.path.insert(0, str(HERE))

# importing train_ste sets the XLA environment and puts the repo root on
# sys.path, as running it directly would
import train_ste  # noqa: E402
from pairs import PairSource, pair_rows  # noqa: E402

CACHE = ROOT / "data" / "sonar_embeddings" / "mc4_sent16.npy"
OUT = ROOT / "data" / "out" / "sonar" / "next-sentence"


def parser():
    p = train_ste.parser()
    p.description = __doc__
    p.add_argument("--objective", choices=("next", "self"), default="next",
                   help="target: the next sentence's embedding, or the "
                        "input's own (the reconstruction control)")
    p.add_argument("--rows", type=int, default=4_000_000,
                   help="train on the pairs within the cache's first ROWS "
                        "rows")
    p.set_defaults(l=1, h=380, dict_init_scale=0.1, s_hm=1e-4,
                   cache=str(CACHE),
                   mse_weights=str(CACHE.with_name("mc4_sent16.mse_weights.npy")))
    return p


def check(cfg) -> None:
    """Residual forwarding hands layer i+1 `X - decode(R)`, which is the
    part of the input still unexplained only when the target IS the input;
    against another sentence it is not a residual of anything."""
    if cfg.objective == "next" and cfg.l > 1 and cfg.forward != "labels":
        raise SystemExit(
            "--objective next with --l > 1 needs --forward labels: residual "
            "forwarding reads X - decode(R), a residual only when the target "
            "is the input")


def main() -> None:
    cfg = train_ste.finish(parser().parse_args())
    check(cfg)
    seed = cfg.seed or 42
    cfg.out = cfg.out or str(
        OUT / f"{cfg.objective}_h{cfg.h}_l{cfg.l}_s{seed}")
    train, _ = pair_rows(cfg.cache, cfg.rows, train_ste.base.holdout)
    src = PairSource(cfg.cache, train, cfg.objective)
    print(f"{cfg.objective}: {len(src)} adjacent pairs in the first "
          f"{cfg.rows} rows of {Path(cfg.cache).name}")
    train_ste.train(*train_ste.setup(cfg, src=src))


if __name__ == "__main__":
    main()
