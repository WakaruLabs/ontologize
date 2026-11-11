# seed-probes: two weaker kinds of agreement between seeds

The seed comparisons in `experiments/ste-arm` (`partition.py`,
`headcontrib.py`) match heads one to one and score the whole output space
at once. Past layer 0 they find agreement near its null. Two weaker kinds
of agreement would not show up there, and each has a script here.

## `translate.py`: the same information, split differently

Seed B's layer could determine each head of seed A without any head of B
matching it. The script predicts every head of A from a code and scores
the prediction on rows no model trained on (`mc4_fresh`).

A score needs a control: both codes are functions of the same input, so
any code of enough capacity predicts A's heads to a degree. The same probe
is therefore run from codes that training did not shape, and the result is
the margin over them, as a probe is read against its control task.

| source | what it is |
|---|---|
| `seed` | B's code |
| `lsh` | random-projection heads with the shape of B's code |
| `untrained` | B's architecture at its initialization |
| `earlier` | A's own code at an earlier checkpoint, a ceiling |
| `input` | the input itself, by a linear probe |

The probe is one multinomial ridge per source on the code's one-hot
features, solved in closed form for all of A's heads at once (float64
Cholesky in torch, on the GPU when there is one), with a relative penalty
chosen on validation rows. `same` uses the source's heads of the target's
layer and `all` the whole stack's.

A ridge on one-hot features reads only what is additive across the
source's heads. A head of A that is a joint function of two heads of B,
with neither informative alone, is invisible to it
(`tests/test_seed_probes.py` pins this). `--mlp` adds a probe with one
hidden layer on the same features, which can read such a head.

## `bandagree.py`: agreement in part of the output space

Seeds could agree on what heads write along the data's main directions
and differ elsewhere. The script takes `headcontrib.py`'s per-head
contributions, whitened and centered, expresses them in the eigenvectors
of the whitened data covariance, and repeats the matched-cosine
comparison within bands of that spectrum, each against its own
shuffled-row null. `--earlier` scores a run against its own earlier
checkpoint in every band, since a band where one run does not agree with
itself cannot show two seeds agreeing.

## Running

```bash
uv run python experiments/seed-probes/translate.py \
    --a data/out/sonar/multilingual/ste_h76_init01 \
    --b data/out/sonar/multilingual/ste_h76_i01_s43 --earlier 350000
uv run python experiments/seed-probes/bandagree.py \
    --a data/out/sonar/multilingual/ste_h76_init01 \
    --b data/out/sonar/multilingual/ste_h76_i01_s43 --earlier 350000
```

Outputs go to `data/out/sonar/seedprobes/`. `bandagree.py` holds two
models' contributions in memory, about 6 GB each at 380 heads and 4,096
rows.
