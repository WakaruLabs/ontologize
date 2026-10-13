# next-sentence: codes trained to predict the next sentence

Every model in this project has been trained to reconstruct its own input. This experiment trains the
same architecture to predict the **next** sentence's SONAR embedding from the current one, x_t →
x_{t+1}. It uses the sentence cache `mc4_sent16`, which stores up to 16 consecutive sentences per mC4
document.

## Why

Two runs of the hard code that differ only in seed reach the same held-out error, but their heads don't
replicate. The mean matched-head NMI is 0.012 for the stack and 0.018 for the one-layer flat arm,
against a 0.005 null. The only head that recurs is a topic head in the stack's layer 0 (findings,
`sec:seeds`). Locatello et al. (2019) explain why that is expected: no method that sees
only the data can recover its factors without inductive biases, on the model and on the data, that
match. One source of identifiability that
literature points to is temporal structure, which a corpus of documents has and a bag of sentences
does not.

A next-sentence target rewards a code only for what adjacent sentences share: language, topic,
register, the slow, document-level factors. The question is whether codes trained this way **reproduce
across seeds** better than reconstruction codes do. Reconstruction quality isn't the point here; the
next sentence is mostly unpredictable (`notes.md`). See `scratch/literature/disentanglement.md` for the
reasoning.

## Design

- **Pairs.** A pair is a cache row and the row after it, when both belong to one document and the second
  is that document's next segment (`pairs.adjacent`).
  - The sentence builder skips unusable segments inside a run, so neighbouring rows of a document aren't
    always neighbouring sentences; those pairs are dropped.
  - Training pairs come from the cache's first `--rows` rows: 4M by default, which is 3.2M pairs and
    about 16 GB, small enough to stay in the page cache.
  - Evaluation pairs come from the 32,768-row tail, snapped to whole documents as for every model trained
    on a cache (`pairs.pair_rows`), so no document is in both.
- **Objectives.** `--objective next` targets x_{t+1}. `--objective self` reconstructs x_t, over the same
  input rows. The two differ only in the target.
- **Architecture: one layer.** Residual forwarding hands layer i+1 `X − decode(R)`, which is the part of
  the input still unexplained only when the target *is* the input. For a next-sentence target it isn't
  a residual of anything.
  - So the default is the flat arm: one layer of 380 heads at k = 32 (1900 bits), with
    `--dict-init-scale 0.1 --s-hm 1e-4`, the best flat arm in `experiments/ste-arm/notes.md` (FVU_w
    0.2750 on mC4 documents).
  - `train_next.py` refuses `--l > 1` under `next` unless `--forward labels`.
  - A deeper version needs a forwarding mode that doesn't read the target, such as the transcoder design
    (ii), `forward='cond'`, in the project's earlier codebase (commit `14f7dba`).
- **Inherited bias.** The layer still applies gain-shape: it classifies the input's direction and
  scales its output by the input's norm. For x_{t+1} that's an assumption that adjacent sentences have
  similar norms.
- **Loss.** Whitened MSE with the cache's own weights (`mc4_sent16.mse_weights.npy`). Both sentences of a
  pair live in the same space, so the weights apply to either target.

## Files

| File | What it does |
|---|---|
| `pairs.py` | `adjacent`, `pair_rows`, and `PairSource`, a grain source of `(x_t, target)` pairs |
| `train_next.py` | the trainer: `experiments/ste-arm/train_ste.py`'s flags, checks and setup (`parser`, `finish`, `setup`), with this experiment's defaults and a pair source |
| `eval_next.py` | held-out FVU_w of each checkpoint against the next sentence and against the input, beside three baselines: the training mean, copying x_t, and ridge regression from x_t |

`ontostate.train` accepts `(input, target)` tuple batches; any other batch is its own target. The
tests are in `tests/test_next_sentence.py`.

## Run

The comparison needs two seeds per objective, at matched steps:

```bash
uv run python experiments/next-sentence/train_next.py --objective next
uv run python experiments/next-sentence/train_next.py --objective next --seed 43
uv run python experiments/next-sentence/train_next.py --objective self
uv run python experiments/next-sentence/train_next.py --objective self --seed 43
```

Runs land in `data/out/sonar/next-sentence/<objective>_h380_l1_s<seed>`. `--epochs` defaults to sonar.py's
24, which is about 300k steps over 3.2M pairs at batch 256; pass a smaller value for a pilot.

Then score fidelity and seed agreement:

```bash
D=data/out/sonar/next-sentence
uv run python experiments/next-sentence/eval_next.py \
    --ckpt $D/next_h380_l1_s42 $D/self_h380_l1_s42 --out $D/eval.json
uv run python experiments/ste-arm/partition.py --cache data/sonar_embeddings/mc4_sent16.npy \
    --a $D/next_h380_l1_s42 --b $D/next_h380_l1_s43
uv run python experiments/ste-arm/partition.py --cache data/sonar_embeddings/mc4_sent16.npy \
    --a $D/self_h380_l1_s42 --b $D/self_h380_l1_s43
```

`partition.py` matches heads across the two seeds by NMI (Hungarian) on the cache's last 32,768 rows, and
reports the result against shuffled nulls. The prediction is that the `next` pair matches better than
the `self` pair, which should land near the flat arm's 0.018 on mC4 documents. Language labels (`mc4_sent16.langs.npy`) allow a head × language check of which factors
each objective's heads carry.
