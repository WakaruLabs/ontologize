# pair-diff: a code of the difference of two sentences

A sentence's embedding carries what its document holds fixed (language,
topic, register) and what the sentence itself says. The pair objective
(`experiments/next-sentence`) trains slow heads to keep the first. This
experiment goes the other way: it trains the stock code on the
difference x_a - x_b of two sentences from one document, which cancels
what the two share, to the extent that is additive, and leaves what
varies.

The reason to try it is seed agreement. Joshi et al.'s identifiability
result for sparse shift autoencoders covers this setting: from paired
samples, the concepts that vary within a pair are identified up to
permutation and scale, and those every pair holds fixed drop out. Two
seeds' sentence codes share little head by head (findings, `sec:seeds`);
the question is whether two seeds' difference codes share more.

## Pieces

| file | does |
|---|---|
| `diffs.py` | `DiffSource`, a grain source of x[row] - x[partner] for the pairs `train_pair.py` draws (one document, at least `--min-d` segments apart). Run as a script it writes the differences' whitening (`mse_weights_diff.npy`: inverse variance per coordinate, zero where the sentence weights are, scaled to their sum) and the held-out tail's differences with their rows and languages |
| `train_diff.py` | the stock straight-through trainer on `DiffSource`: `train_pair.py`'s flags and defaults (one layer of 380 heads, k 32), `s_pair` 0 |
| `evaldiff.py` | two seeds of the difference model on the held-out differences, beside two seeds of a sentence model of the same shape on the first sentence of each pair |

A difference is not a unit vector (median norm 0.26 on this cache, from
sentences of norm about 0.2), and its sign is arbitrary: the pair
(b, a) is as valid as (a, b). The model sees one order per pair.
`evaldiff.py`'s `antipodal` reads what each head does with the other.

## Run

```bash
uv run python experiments/pair-diff/diffs.py                 # once, CPU
uv run python experiments/pair-diff/train_diff.py --epochs 1 --keep-every 2500
uv run python experiments/pair-diff/train_diff.py --epochs 1 --keep-every 2500 --seed 43
JAX_PLATFORMS=cpu uv run python experiments/pair-diff/evaldiff.py \
    --a data/out/sonar/pair-diff/diff_h380_l1_s42 \
    --b data/out/sonar/pair-diff/diff_h380_l1_s43 --step 15000
```

## What `evaldiff.py` reports

- **Seed match**, for the difference pair and the sentence pair: matched
  NMI under the one-to-one head match; the same with one seed's rows
  shuffled (the null); the mean NMI of any head with any head
  (`all_pairs`), since a code whose heads all tell one thing matches under
  any pairing; each head's best partner over its runner-up; and the
  matched pairs' NMI inside each language, beside a within-language
  shuffle. Within-language NMI is biased upward on a few hundred rows a
  language, so it is read as its margin over that shuffle.
- **Per model**: held-out whitened FVU (the sentence models' on
  sentences; the difference models' on differences), head entropy, and
  the mean and largest NMI of a head with the document's language,
  beside the mean with the labels shuffled. For a
  sentence model also the FVU of x_hat_a - x_hat_b as a reconstruction of
  the difference under the differences' weights: what a sentence code
  already gives, at twice the heads.
- **Antipodal**, per difference model: the share of heads that keep
  their entry when the difference is negated, beside chance, and the NMI
  between the two codes.

The sentence models default to the pair objective's `s_pair = 0`
controls, read at the same `--step`.

Tests: `tests/test_pair_diff.py`.
