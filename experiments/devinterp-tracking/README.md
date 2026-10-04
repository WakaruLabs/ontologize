# devinterp-tracking

Developmental-interpretability sweep over a run's saved checkpoints: do
per-head classifications sharpen at stage boundaries (temperature-anneal
milestones, the hardening window, late-training head freezing), and does
head-level semantic coherence (headcoh's z-scores) emerge gradually or in
jumps?

Per retained checkpoint it computes:

- headcoh geometry: `group_zscores` over the exact end-to-end direction
  matrix `G` (`refit.onto_linear_model`, i.e. the same probe every
  post-hoc analysis uses), per-head `z_cos` / `z_sv` vs size-matched
  within-layer nulls;
- code shape on a fixed cache slice: per-head mean per-sample entropy
  (bits), sharpness (mean max p), usage KL(E[p] || uniform) in bits;
  `KL_m_eval` (per-layer head-mean, summed over layers) reproduces the
  loss.csv `KL_m` stat's definition on a larger, noise-free sample;
- the matching loss.csv block means (`train_loss`, `train_MSE`,
  `train_KL_m`) so training-time and eval-time curves line up per step.

## Run (from the repository root; GPU JAX required for checkpoint restore)

```bash
# fixed eval temperature (post-hoc convention, T = 0.03)
uv run python experiments/devinterp-tracking/track.py \
    --ckpt data/out/sonar/multilingual/resid_nc_hm

# same sweep at each checkpoint's own training-time annealed temperature
# (schedule read from the run's log.jsonl env_config record)
uv run python experiments/devinterp-tracking/track.py \
    --ckpt data/out/sonar/multilingual/resid_nc_hm --schedule \
    --out experiments/devinterp-tracking/out/resid_nc_hm_sched
```

Run both. Sharpening visible at FIXED temperature is learning; the
difference between the two sweeps is what the schedule alone contributes.
`--ckpt data/out/sonar/multilingual/resid_nc` sweeps the other live run
(pre-KL_m: its loss.csv has 8 columns, handled automatically, and
`train_KL_m` stays empty).

## Inputs

- `--ckpt`: run dir in the orbax CheckpointManager layout
  (`Metadata.manager`: saves every `save_each=100` steps keeping the last
  10, plus every `checkpoint_each=10000` kept indefinitely, so the sweep
  sees ~every-10k snapshots over the whole run plus a fine-grained recent
  window). Defaults documented above.
- `--cache`: embedding cache (default `data/sonar_embeddings/mc4_4M.npy`);
  the activation stats use a fixed slice from the cache head
  (`--rows 16384`), identical at every step.
- `--save-each`: the run's loss-flush cadence (default 100, sonar.py's
  live value); only used for the loss.csv join.

## Outputs (into `experiments/devinterp-tracking/out/<run name>/`)

- `steps.csv`: one row per checkpoint: `step, T_eval, mean/median z_cos,
  frac_z_cos_gt2, frac_z_cos_lt_neg2, mean_z_sv, mean_H_sample,
  mean_sharpness, KL_m_eval`, per-layer `H_sample_l*/sharpness_l*/
  KL_usage_l*`, and `train_loss/train_MSE/train_KL_m`. Ready to plot
  metric-vs-step directly.
- `heads.csv`: one row per (checkpoint, head): layer, head, z_cos, z_sv,
  per-head entropy/sharpness/usage-KL. For finding individual heads that
  snap (a stagewise story predicts per-head step functions at different
  times; a gradual story predicts synchronized slow drift).

Both files are append-only and resumable (steps already in steps.csv are
skipped), so the sweep can be re-run as training deposits new checkpoints.
If a run crashes between the heads.csv and steps.csv writes for a step,
that step's heads rows can appear twice; dedupe on (step, head_global)
when plotting.

## Cost

Per checkpoint: one restore (~23M params), one decode of 5120 constant
codes, ~4 activation batches of 4096 rows, and the numpy nulls
(5 pools x 200 draws). The dominant cost is jit recompilation of the acts
probe per checkpoint (a fresh closure per restore), a few seconds each.
Expect roughly 20-40 s/checkpoint on the training GPU; a run with ~40
retained checkpoints finishes in ~15-30 min. Not CPU-feasible: Ontologizer
checkpoints restore on GPU JAX only (same constraint pareto.py/textfid.py
note). `--stride`/`--min-step` cut the sweep; `--n-null 500` matches
headcoh.py's default resolution at ~2x null cost.

## Assumptions

- Checkpoint steps are read from `CheckpointManager.all_steps()`; the
  spec item is restored per step (it is saved alongside state at every
  step by `OntoState.save`), so architecture changes mid-run would be
  picked up rather than assumed.
- loss.csv rows within each `save_each` block are rotated
  (`stats_insert` writes row `step % save_each`), so the join uses block
  MEANS over `(step - save_each, step]` instead of trying to index exact
  rows.
- Eval activation stats are noise-free full-soft-forward tag
  probabilities; training-time `KL_m` was measured on b=256 noisy batches,
  so `KL_m_eval` and `train_KL_m` agree in definition but not exactly in
  value.
