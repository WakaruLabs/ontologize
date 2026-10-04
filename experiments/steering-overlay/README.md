# steering-overlay

The decisive steering-quality test: does the Ontologizer's
classification-forcing beat directions a supervisor could have computed
directly from labels? steerfid.py only compares native directions against
random controls; this overlay adds the supervised baselines (difference of
means, logistic-probe direction) on one shared task — language steering,
using the free labels in the cache's `.langs.npy` sidecar — with all arms
applied at the same matched magnitudes and scored in steerfid's
effect/collateral frame plus one common yardstick (the dm-direction score).

## Run

From the repo root (paths below are relative to it):

```bash
# Ontologizer (classification-forcing arm = withArgs h_set/k_set deltas)
uv run python experiments/steering-overlay/steer_overlay.py \
    --model data/out/sonar/multilingual/resid_nc_hm --device cuda

# the earlier run
uv run python experiments/steering-overlay/steer_overlay.py \
    --model data/out/sonar/multilingual/resid_nc --device cuda

# bonus: the same overlay for an SAE (native arm = t-stat-selected W_dec row)
uv run python experiments/steering-overlay/steer_overlay.py \
    --model data/out/sonar/sae/m5120_k32/params.npz --device cuda

# add the unforced-decode language-token readout (2x generation cost)
uv run python experiments/steering-overlay/steer_overlay.py \
    --model data/out/sonar/multilingual/resid_nc_hm --device cuda --langid
```

## Inputs

- `--model`: Ontologizer checkpoint dir (known runs:
  `data/out/sonar/multilingual/resid_nc_hm`, `data/out/sonar/multilingual/resid_nc`)
  or an sae.py `params.npz`. Loaded via `steerfid.load_steerable`, so
  everything steerfid accepts works here.
- `--cache`: embedding cache (default `data/sonar_embeddings/mc4_4M.npy`);
  its `.langs.npy` sidecar must exist (encode_corpus.py writes it).
- `--languages` / `--n-langs`: target set (default: 8 most frequent mC4
  codes in the labeled slice). `--train-rows` (default 65536) is the
  labeled head slice used for t-statistics, means, and probes.
- Steered rows come from the cache tail (`--n-pool`, per-language
  `--n-samples` rows not labeled with the target language). For rows the
  Ontologizer has *never* trained on, point `--cache` at the fresh cache
  from `experiments/fresh-eval/` (`experiments/fresh-eval/out/mc4_fresh.npy`,
  whose `.langs.npy` sidecar this harness picks up automatically).

## Outputs (in `experiments/steering-overlay/out/<model name>/`)

- `steer_overlay.csv` — per (arm, language, magnitude): `eff_native`,
  `hit_native` (each arm's own steerfid-normalized frame), `eff_dm`,
  `hit_dm` (the shared dm-direction frame), `collateral` (1 − chrF),
  optional `langid_base`/`langid_steer`.
- `steers.jsonl` — every base/steered decode pair for inspection (and as
  input to `experiments/task-naturalness/`).
- `meta.json` — selected onto tags per language with their t-statistics.
- A summary table (arm × magnitude, averaged over languages) is printed.

Reading it: the claim "classification-forcing is a competitive steering
primitive" needs the onto row to sit at comparable `eff_dm`/`hit_dm` for
comparable `collateral` against the dm and probe rows; the random row is
the collateral floor with no intended effect.

## Cost

- Ontologizer checkpoint restore requires GPU JAX (orbax sharding
  metadata); SAE `.npz` models work on CPU JAX.
- Activation/statistics passes: one pass over `--train-rows` (65k) — ~1–2
  min on GPU. Probe fits: seconds.
- Generation dominates: ≈ (n_langs × arms + n_random) × mags × n_samples
  decodes + as many re-encodes ≈ 1,000–1,500 M2M100 generations at the
  defaults. **~15–30 min on a single GPU** (SONAR encoder + M2M100 decoder
  ≈ 3 GB VRAM alongside the JAX model); several hours on CPU. `--langid`
  roughly doubles generation time.

## Assumptions / caveats

- Language is used as the steering concept because it is the only labeled
  concept the pipeline ships; the sidecar's mC4 codes (e.g. `en`,
  `hi-Latn`) are compared as raw strings.
- The onto arm's tag per language is chosen by the largest two-sample
  t-statistic of tag activation vs the language label (langprobe's
  statistic) among tags with fire rate > `--min-rate`. If no tag encodes
  the language, the onto arm honestly loses — that is part of the result,
  not a bug; `meta.json` records the selected tag and its t.
- The cycle re-encodes forced-English decodes (steerfid's convention), so
  all `eff_*` numbers measure language signal that survives translation;
  this attenuates every arm identically. `--langid` decodes without
  forcing English and checks the decoder's own language token (assumed to
  be sequence position 1, after the decoder-start token — M2M100's
  generation layout).
- Dense scores (dm/probe) have no firing threshold, so their native frame
  normalizes by q90(L-rows) − median(non-L rows) instead of
  firing-conditional quantiles; `dense_frame` documents this deviation
  from `steerfid.firing_quantiles`.
