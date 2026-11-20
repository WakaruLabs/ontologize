# Forcing one entry: does it move decodes into a script?

`force.py` tests a claim of the form "forcing entry j of head i in layer ℓ
turns decoded text into language X while keeping its meaning". The claim
this was written for came from an interactive `decode.py` session: in
`resid_nc`, entry 17 of head 8 in layer 1 "consistently translates decoded
text into Burmese while preserving semantic content". It had no script, no
row count and no null, and it was made before the decode fixes that
`textfid.SonarDecoder` now applies (each output rescaled to SONAR's norm
before decoding).

## What it measures

For every entry of the head, over the same held-out rows, with the head set
to that entry through `withArgs`:

- **script share**: the fraction of a decode's letters in the target
  script (`SCRIPTS`; Myanmar is U+1000–109F and its two extension blocks),
  and the fraction of rows where it is at least half;
- **meaning kept**: the cosine between the decode's SONAR re-encoding and
  the source row, beside the cosine to a different row (a seeded
  derangement), which is what any decode shares with an unrelated row.

The head's other entries are the null, so the target entry is read by its
rank among them, not by an absolute threshold. The input row itself (`x`)
and the unforced reconstruction (`recon`) are decoded as references.

**The positive control** (`controls.json`) decodes rows of the script's own
language, input and reconstruction, through the same decoder. A script share
of 0 for the forced entries means something only if these rows come out in
the script. Forcing English as the output language makes the decoder
translate, so under the default decode even Burmese rows come out in
English. `--free-lang` instead lets the decoder pick the language token it
starts with, as M2M100 does when the first token is not forced, and records
how often that token is the script's (`lang_tok`). `--controls-only` runs
the usage pass and the control alone.

Rows come from the cache's held-out tail (the last 32,768 rows), excluding
the target script's own language. `usage.csv` gives, per language, how
often the tail's rows choose the entry on their own, from the model's
argmax at its trained temperature.

The re-encoding uses the English source tag, as `mc4_4M` was encoded for 80
of its 86 configurations, so the cosine compares like with like. By
default the decoder is forced to English, as in every text-space evaluation
here, which the positive control shows cannot reveal a language.

Rows go to the model as the cache stores them. `--unit` scales them to unit
norm, which a model trained on `mc4_4M` (unit rows) needs when it reads
`mc4_sent16` (SONAR's own embeddings).

## The scan: every entry, read by a language probe

`scan.py` asks the question without a target entry: does any entry of any
head move outputs toward a language? Decoding every entry is too slow
(12,160 entries in the recipe ladder's hard stacks), so it reads outputs
with a class-balanced logistic probe for the language against the rest,
fitted on the rows just before the held-out tail and checked on the tail's
inputs and reconstructions (AUROC). The probe is affine in a row's unit
direction, so it runs inside the jitted forward.

Every entry is forced over the same tail rows of other languages, split into
halves A and B. An entry's effect is the shift in mean probe logit from the
unforced reconstruction. Entries are ranked on A and read on B, which keeps
the largest of 12,160 effects from passing for a real one. The A–B
correlation over all entries says how much of the spread is signal. The
size of an effect is read against the gap between the language's
reconstructions and the rest's.

`usage.csv` gives each entry's usage among the language's tail rows and
among the rest, from one forward over the tail. The entries with the
largest shifts, the entries the language's rows choose most, and as many
random ones are then decoded with the free decoder, beside the positive
control.

## Run

```bash
JAX_PLATFORMS=cpu uv run python experiments/entry-force/force.py \
    --model data/out/sonar/multilingual/resid_nc --layer 1 --head 8 \
    --entry 17 --device cuda --free-lang
uv run python experiments/entry-force/scan.py \
    --model data/out/sonar/recipe_ladder/ste_h76_init01_med837_ctrl \
    --cache data/sonar_embeddings/mc4_sent16.npy --lang my --device cuda
```

`force.py` runs JAX on the CPU and M2M100 and the SONAR encoder on
`--device`. `scan.py` runs JAX on the GPU when there is one, for its 12,160
forced forwards, and reads the temperature from the run's `log.jsonl`
unless `--temperature` is given. Outputs go to
`data/out/sonar/entry_force/<run>_<step>_L<layer>H<head>/` and
`data/out/sonar/entry_force/scan_<run>_<step>_<lang>/`.
`tests/test_entry_force.py` checks the script share, the null pairing, the
rank, the row choice, the AUROC, reading the probe back as an affine map,
and the usage log-odds.

`notes.md` has the results.
