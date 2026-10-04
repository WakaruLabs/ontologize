# layer4-freeze: diagnose and mitigate the layer-4 head freeze

The resid_nc reference run froze 22/32 layer-4 heads after step ~211k
(documented in `sonar.py`, the `s_Hm` calibration note). A frozen head
answers with the same dictionary entry for essentially every sample: its
softmax has saturated, classifier gradient has vanished, and the head's
capacity is dead weight. This experiment has two parts: a
checkpoint-sweep **diagnostic** that localizes the freeze in time and
per head, and a **retrain harness** with the three candidate mitigations
that the training code's own comments identify, one knob per variant.

Nothing here modifies the `ontologize/` package or the root scripts;
both harnesses import from the repo.

## Files

- `freeze_diag.py` -- walks every retained checkpoint of a run, runs a
  fixed batch of cache-tail rows through a clean forward probe (the
  `autointerp.onto_acts_fn` convention), and reports per (layer, head):
  argmax `top_share` / live-entry count / usage entropy (temperature-free),
  argmax **churn** between consecutive checkpoints, mean per-sample soft
  entropy and per-head `KL_m` at `--temperature`, and a `frozen` flag
  (`top_share >= 0.95` and `churn <= 0.02` by default).
- `retrain_variants.py` -- full training runs against the resid_nc_hm
  baseline config with exactly one mitigation changed per variant:
  `baseline`, `slow_anneal` (anneal_steps 50k -> 150k), `drop_ramp`
  (p_drop 0.1 -> 0.25), `sdk_floor` (sd_K held at 0.02 instead of
  annealing to 0.2 * temperature_end). Rationale for each is in the
  module docstring, quoting the source comments it comes from.

## Run

From the repo root (`uv run` so the project venv resolves):

```bash
# 1. diagnose the reference freeze (writes into this dir by default)
uv run python experiments/layer4-freeze/freeze_diag.py \
    --ckpt data/out/sonar/multilingual/resid_nc \
    --out experiments/layer4-freeze/diag_resid_nc

# 2. diagnose the s_Hm run for comparison
uv run python experiments/layer4-freeze/freeze_diag.py \
    --ckpt data/out/sonar/multilingual/resid_nc_hm \
    --out experiments/layer4-freeze/diag_resid_nc_hm

# 3. train the mitigation variants (each is a FULL run)
uv run python experiments/layer4-freeze/retrain_variants.py --variant baseline
uv run python experiments/layer4-freeze/retrain_variants.py --variant slow_anneal
uv run python experiments/layer4-freeze/retrain_variants.py --variant drop_ramp
uv run python experiments/layer4-freeze/retrain_variants.py --variant sdk_floor

# 4. diagnose each variant run
uv run python experiments/layer4-freeze/freeze_diag.py \
    --ckpt experiments/layer4-freeze/runs/l4fix_drop_ramp \
    --out experiments/layer4-freeze/diag_drop_ramp
```

## Inputs

| input | default | notes |
|---|---|---|
| `--ckpt` | `data/out/sonar/multilingual/resid_nc` | orbax run dir; the other known run is `.../resid_nc_hm` |
| `--cache` | `data/sonar_embeddings/mc4_4M.npy` | `encode_corpus.py` output; the diagnostic uses the **tail** rows (sae.py eval-split convention) |
| `--mse-weights` (retrain) | `data/out/sonar/mse_weights.npy` | pass `''` to disable target whitening |
| `--out-base` (retrain) | `experiments/layer4-freeze/runs` | variant runs land in `<out-base>/l4fix_<variant>` |

## Outputs

`freeze_diag.py` writes into `--out`:

- `heads.csv` -- one row per (checkpoint, layer, head): `top_entry`,
  `top_share`, `n_live`, `usage_bits`, `churn`, `H_mean`, `KL_m`, `frozen`.
- `summary.csv` -- per (checkpoint, layer): frozen count and means.
- `freeze_report.md` -- final-state per-layer table, per-head freeze
  onset steps (first retained checkpoint from which the head stays
  frozen through the end), and a layer-4 note against the 22/32
  reference number.

`retrain_variants.py` writes a normal run dir (orbax checkpoints,
`loss.csv`, `log.jsonl`, loss plots) under `--out-base`.

## Interpreting the sweep

- Freezing is detected on **argmax** statistics, so it does not depend
  on reconstructing the run's temperature schedule; `--temperature`
  (default 0.03, the anneal floor) only affects the two soft-entropy
  columns.
- Cross-reference onset steps in `freeze_report.md` against the run's
  schedule endpoints in its `log.jsonl` (`anneal_steps`,
  `temperature_end`): the reference freeze started well after the
  anneal floor was reached, which is what motivates all three variants.
- Note the first retained checkpoint has no churn; its `frozen` flag is
  share-only. Checkpoint retention is `save_each=100` for the most
  recent 10 plus every `checkpoint_each=10000` kept forever, so the
  sweep's time resolution is 10k steps for most of the run.

## Cost (honest estimate)

- `freeze_diag.py`: per checkpoint, one orbax restore plus a forward
  pass over 4096 rows of a small model. **GPU: seconds per checkpoint,
  a few minutes for a ~30-60 checkpoint run. CPU: works (the model is
  ~10M params), expect ~1-2 min per checkpoint, so up to ~1-2 h for a
  full sweep.** Memory: ~1 GB. Restore of GPU-written checkpoints on
  CPU JAX is the same caveat refit.py notes; if restore complains, run
  on the GPU box.
- `retrain_variants.py`: each variant is a **full training run** of the
  sonar.py scale (4M cached rows, b=256, 24 epochs = ~375k steps).
  On the GPU that trained resid_nc_hm this is on the order of **many
  hours to ~a day per variant**. Not CPU-feasible. Disk: same as a
  normal run dir (checkpoints every 10k steps kept indefinitely).
