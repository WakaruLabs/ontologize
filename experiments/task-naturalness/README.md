# task-naturalness

Scores steered decodes for fluency and content preservation. steerfid.py
measures effect vs collateral; nothing in the battery yet asks whether
the steered text is *natural* — a steer that produces high-effect
gibberish and a steer that produces a fluent paraphrase with the concept
added can score identically on collateral. Inputs are the per-steer
decode logs (`steers.jsonl`) that steerfid.py and
`experiments/steering-overlay/` already write.

Readouts:

- **chrF(base, steered)** (textfid's chrF): content preservation, the
  complement of steerfid's collateral, recomputed here per pair.
- **len_ratio**: steered/base character length; repetition loops and
  truncation collapse show up here first.
- **NLL_generic**: mean per-token M2M100 NLL of each text conditioned on
  the corpus-mean embedding — the same generic conditioning textfid.py
  uses as its floor. A reference-free fluency proxy: rising
  `nll_steered − nll_base` means the steered text is drifting off the
  decoder's manifold. (Caveat: text that *is* generic scores best, so
  read it jointly with chrF, never alone.)
- **Blinded human sheet**: steered/unsteered pairs with sides A/B
  seed-randomized, blank rating columns (fluency 1–5 per side, meaning
  preservation 0–2), and the unblinding key in a separate file.

## Run

From the repo root:

```bash
# full readout on a steerfid run (NLL pass on GPU)
uv run python experiments/task-naturalness/naturalness.py \
    --steers data/out/sonar/steerfid/<run>/steers.jsonl --device cuda

# CPU-only, no decoder needed (chrF + lengths + blind sheet)
uv run python experiments/task-naturalness/naturalness.py \
    --steers experiments/steering-overlay/out/<run>/steers.jsonl --skip-nll

# several runs on one sheet (e.g. onto vs sae steers, blind across models)
uv run python experiments/task-naturalness/naturalness.py \
    --steers data/out/sonar/steerfid/runA/steers.jsonl \
             data/out/sonar/steerfid/runB/steers.jsonl --device cuda
```

## Inputs

- `--steers`: one or more `steers.jsonl` files (fields used: `kind`,
  `feature`, `mag`, `sample`, `base`, `steered`; steering-overlay's
  `lang` is carried through when present). Typical producers:
  `steerfid.py --model data/out/sonar/multilingual/resid_nc_hm` (or
  `resid_nc`, or an SAE `params.npz`), and the steering-overlay harness.
- `--cache` (default `data/sonar_embeddings/mc4_4M.npy`): read only for
  the corpus-mean embedding of the NLL pass.

## Outputs (in `experiments/task-naturalness/out/<run>/`)

- `naturalness.csv` — per pair: chrf, exact, len_ratio, nll_base,
  nll_steered, nll_delta.
- `rating.csv` — the blind sheet (`--n-pairs`, default 120, sampled
  round-robin over (kind, mag) strata among non-identical pairs; row
  order shuffled). Hand this file to the rater.
- `key.csv` — pair_id → which side is steered + full metadata. **Do not
  open before rating.**
- `summary.json` + a printed (kind × mag) table.

## Cost

- chrF / lengths / blind sheet: CPU, seconds.
- NLL pass: 2 decoder forwards per pair. For a default steerfid run
  (~1,000–1,500 pairs): **~2–5 min on GPU**, ~30–60 min on CPU
  (`--skip-nll` to omit). M2M100 decoder ≈ 1.2 GB VRAM. No JAX and no
  checkpoint needed at all.

## Assumptions / caveats

- NLL labels are built as `[eng_Latn, tokens..., eos]` — the exact layout
  textfid.py scores generated sequences in (its decodes are forced
  eng_Latn); the decoder-start token is added by the model's internal
  label shift. Tokens are capped at `--max-tokens` (64) per text.
- Random-control rows from steers.jsonl are kept: they calibrate what
  collateral-without-intended-effect reads like on every readout, and
  they appear on the blind sheet like any other arm.
- Pairs where steered == base are scored (chrF 1.0) but excluded from the
  rating sheet — there is nothing to rate.
