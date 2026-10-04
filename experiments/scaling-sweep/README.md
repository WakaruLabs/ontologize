# scaling-sweep

Tests the paper's scaling claim: "polysemanticity decreases with parameter
count", with parameters grown along the three axes the architecture
offers: width (h, heads per layer), head size (k, entries per head), and
depth (l, DictEnc layers). A config-grid runner around the live values
(h=32, k=32, l=5), plus analysis over each run's `loss.csv` and
`headcoh.py` output.

Operationalization: a head's polysemanticity is read off headcoh's
within-head decode-direction coherence (`z_cos` / `z_sv` vs size-matched
within-layer nulls). Coherent heads = alternative values of one shared
dimension; null-level heads = arbitrary partition cells. The claim
predicts mean `z_cos` and `frac(z_cos > 2)` rise with parameter count
(and usage imbalance `KL_m` falls).

## Run (from the repository root)

```bash
# 1. generate the config grid + queue (writes configs/*.json, queue.sh)
uv run python experiments/scaling-sweep/gen_configs.py

# 2. run the queue (GPU; sequential; resumable per run)
bash experiments/scaling-sweep/queue.sh

# 3. analysis (CPU-only; rerunnable while the queue is still going,
#    unfinished runs just carry NaN headcoh columns)
uv run python experiments/scaling-sweep/analyze.py
```

Useful variants:

```bash
uv run python experiments/scaling-sweep/gen_configs.py --h 16 32 64 --k 32 --l 5
uv run python experiments/scaling-sweep/gen_configs.py --seed 43   # replicas, _s43 dirs
uv run python experiments/scaling-sweep/gen_configs.py --full-cross  # 27 runs; weeks
```

## Inputs

- Embedding cache `data/sonar_embeddings/mc4_4M.npy` (encode_corpus.py)
  and `data/out/sonar/mse_weights.npy` (make_mse_weights.py): identical to
  the live sonar.py training inputs. train_run.py refuses to start
  without them.
- Configs mirror sonar.py's live `resid_nc_hm` settings field for field
  (resid forwarding, deepsup joint, resid_norm/const, bilinear n=2
  classifier, KL_m bonus, whitened MSE, annealing schedules). Two
  deliberate deviations, both CLI-exposed: `epochs` defaults to 4 (not
  24) so a 7-run sweep is affordable, and out dirs live under
  `data/out/sonar/multilingual/scaling/<name>` (never the live run dirs
  `resid_nc` / `resid_nc_hm`, which stay untouched as the h32_k32_l5
  reference points if you prefer to reuse them: point analyze at a
  hand-written config whose `out` names one of them).

## Outputs

- Per run (under `data/out/sonar/multilingual/scaling/<name>/`): the
  standard training artifacts (orbax checkpoints, `loss.csv`,
  `log.jsonl`) plus `headcoh/heads.csv` + `headcoh/summary.json` from the
  queue's post-training headcoh step.
- Into this experiment dir: `results.csv` and `results.md`: one row per
  config with (h, k, l), analytic `n_params`, dictionary size `n_tags`,
  tail-mean training MSE and KL_m, and the headcoh coherence stats, plus
  Spearman correlations of each against log parameter count.

## Cost (honest)

Each run is real training: 4 epochs x 15625 steps/epoch at b=256 on the
4M cache = 62.5k steps, chosen to clear the 50k-step anneal horizon.
Throughput on the cache path is pure-JAX and undocumented; measure one
run before committing (at 5 it/s a run is ~3.5 h, at 1 it/s ~17 h). The
default 7-run star is therefore roughly 1 to 5 GPU-days total; the full
cross is ~4x that. The headcoh step per run is minutes. analyze.py is
seconds on CPU. Larger grid points (h=64, k=64, l=7) also cost more per
step and more VRAM (dictionary and classifier scale with h*k and l).

## Assumptions / caveats

- `n_params` is analytic (bilinear unbiased classifier, abs()'d dict,
  unbiased decoder, encoded=False, resid input d+1 for upper layers);
  it reproduces the ~23M of the live center config.
- loss.csv's MSE column under deepsup is the mean over l prefix
  reconstructions, so it is NOT comparable across different l; the
  results.md flags this, and `pareto.py --ckpt <run>` gives a
  final-reconstruction FVU where the l axis matters.
- Coherence z-scores are computed against each model's OWN entry-direction
  pool (headcoh's design), so they measure head structure relative to that
  model's dictionary, which is the right polysemanticity reading but not
  an absolute cross-model reconstruction comparison.
- 7 star points is a directional experiment, not an inferential one; the
  full cross plus seed replicas (`--seed 43`) is the publishable version.
- A deeper (l=7) or wider run at only 62.5k steps may be undertrained
  relative to the center; the `steps` and `train_mse_tail` columns are in
  results.csv precisely so undertraining is visible rather than silent.
