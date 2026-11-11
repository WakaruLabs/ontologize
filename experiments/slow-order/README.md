# slow-order: the slow heads told apart

`experiments/next-sentence`'s Fano term rewards each of a layer's first
38 heads for carrying information between two sentences of one document.
The block it trains reproduces across seeds, but as a block: its heads
agree with one another about as well as with another seed's, so a random
pairing of two seeds' slow heads scores nearly what the best one does, and
no single head is identified (`experiments/next-sentence/notes.md`).

Linear slow feature analysis gets distinct features from the same
objective by two constraints the Fano term lacks. Its outputs are
decorrelated, and they are ordered: each is the slowest function
uncorrelated with those before it. This experiment adds both to the Fano
arm and asks whether the slow heads then reproduce one by one.

## The term (`orderloss.py`)

`OrderedPairHyperparams` subclasses `pairloss.PairHyperparams` and adds

    dec * mean over pairs j < i of CKA(head i, head j)

to the pair term, over the slow heads of each layer. CKA is the unbiased
HSIC (Song et al.) between two heads' partitions of the batch under the
kernel "same entry", normalized by each head's own. The unbiased
estimator is zero in expectation for independent heads at any batch size;
the biased one has a floor that only a collapsed head escapes.

- **Ordered** (default). Head j's code is stop-gradiented in every pair
  j < i. Head 0 is trained by the Fano term alone, head 1 must also be
  independent of head 0, and so on: the block gets a canonical order.
- **Symmetric** (`--dec-symmetric`). The gradient reaches both heads of a
  pair. The heads are told apart but nothing orders them; this is the
  control for what the order adds beyond decorrelation.

`dec` is per unit of `s_pair`, as the Fano term is, and the raw Fano
value stays in `loss.csv` column 2.

## Run

```bash
# one epoch (15,420 steps of 256 pairs); two fit on the card at once
uv run python experiments/slow-order/train_order.py --dec 1 --epochs 1 \
    --keep-every 2500
uv run python experiments/slow-order/train_order.py --dec 1 --dec-symmetric \
    --epochs 1 --keep-every 2500 --seed 43
```

`train_order.py` is `experiments/next-sentence/train_pair.py` with
`--dec` and `--dec-symmetric`; every other flag is that script's, with the
Fano term at `s_pair` 1e-3 as the default, so a run differs from the long
Fano runs in the new term alone. Runs go to
`data/out/sonar/slow-order/{ord,sym}_dec<dec>_sp<s_pair>_ph<heads>_s<seed>`.

## Scoring (`order.py`)

```bash
JAX_PLATFORMS=cpu uv run python experiments/slow-order/order.py \
    --a <run, seed 42> --b <run, seed 43> --step 15000
```

On the cache's held-out tail, for two seeds of one arm:

| statistic | reads |
|---|---|
| `matched`, `all_pairs`, `within_a/b`, `margin` | `seedmatch.py`'s specificity of the slow block: the matched NMI beside any pairing's, the NMI among one seed's own slow heads, and each head's best partner over its runner-up |
| `same_index` | the NMI between head i of one seed and head i of the other, with no matching |
| `on_diagonal`, `best_is_own` | the share of the match, and of best partners, that keep the index; chance is 1/38 |
| `index_rank` | rank correlation of a head's index with its matched partner's |
| `offset` | mean NMI between two heads by how far apart their indices are, across the seeds and within one |
| `neighbour_gap` | a head's NMI with the other seed's head of its index minus the mean of the four entries one index away, as a mean over heads with its standard error |
| `kappa_trend`, `lang_trend` | rank correlation of the index with a head's kappa between sentences `--d` apart, and with its NMI with the language label |
| `prefix` | mean NMI of a head with the heads before it, for each half of the block |
| `fvu` | held-out whitened FVU of the whole model |

The comparison is three arms at one step: the Fano runs
(`data/out/sonar/next-sentence/long/doc2_fano_sp1e-3_ph38{,_s43}`),
symmetric, and ordered. Decorrelation alone shows as a low `within` with
`matched` no further above `all_pairs` than before. An order shows as
`same_index` near `matched`, both above `all_pairs`, and an `index_rank`
near 1; but so does a block in which what heads share merely falls with
the index. `offset` tells the two apart: an identified head is closer to
the other seed's head of its index (`cross` at 0) than to that seed's
neighbouring heads (`cross` at 1) or its own seed's (`within` at 1).

`experiments/next-sentence/seedmatch.py` and `transitions.py` take these
runs unchanged for the block's other statistics.

Tests: `tests/test_slow_order.py`.
