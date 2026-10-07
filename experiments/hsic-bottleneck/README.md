# hsic-bottleneck: HSIC penalties vs the four auxiliary loss terms

The live sonar.py run regularizes the code with four auxiliary scalars:
`s_L1F` (embedding sparsity), `s_bcossim` (sample-similarity penalty),
`s_hcossim` (head-similarity penalty), `s_Hm` (batch mean-entropy
bonus). All four are hand-scaled proxies for one underlying wish: heads
that carry **independent, non-redundant information**. HSIC (the
Hilbert-Schmidt Independence Criterion) measures that wish directly.
This experiment prototypes an RBF-kernel HSIC estimator and wires it
into training as a drop-in loss replacement, without touching the
`ontologize/` package.

## Files

- `hsic.py` -- standalone JAX module: `rbf_gram` (median-heuristic
  bandwidth under `stop_gradient`), `hsic_biased` (Gretton et al.
  2005), `hsic_unbiased` (Song et al. 2012), `cka`, and
  `pairwise_head_cka` -- mean pairwise CKA over all head pairs of one
  layer at O(h b^2) via the sum-of-grams identity. Self-test under
  `__main__` (CPU, seconds).
- `hsic_hyperparams.py` -- `HSICHyperparams(Hyperparams)`: composes
  with the stock training stack by (1) overriding `init` to build the
  train-step `apply_fn` from a probe that mirrors
  `Ontologizer.withStats` but returns the per-layer classifications in
  the (unused, `ghost=False`) ghost slot, and (2) overriding `loss` to
  add `s_hsic_heads` / `s_hsic_res` on top of the whitened MSE and any
  inherited aux terms. `TrainingEnv` and resume work unchanged.
- `train_hsic.py` -- the ablation harness (`--arm aux|hsic|both`),
  otherwise mirroring the sonar.py resid_nc_hm configuration.

## The penalties

- **`s_hsic_heads`** (the replacement candidate): mean pairwise CKA
  between the h heads of each layer, treating each head's
  classification p over the batch as a random variable; summed over
  layers like the stock aux stats. Independent heads score ~0,
  redundant heads ~1. This subsumes what `s_hcossim` (head
  decorrelation) and `s_bcossim` (sample-similarity structure, which
  enters through the batch grams) approximate linearly.
- **`s_hsic_res`** (optional, off by default): HSIC between the
  reconstruction residual and the target -- the bottleneck-style term:
  drive the residual to be statistically independent of the target,
  i.e. leave no predictable structure unexplained.

**Honest limitation, by construction:** a *frozen* head emits a
constant code; its centered Gram is ~0, so `pairwise_head_cka` neither
rewards nor punishes it. HSIC replaces the *decorrelation* terms, not
the *load-balancing* one -- if the hsic-only arm re-freezes layer 4
where the `both` arm does not, that is the expected signature that
`s_Hm` is doing irreplaceable work. Do not zero `s_Hm` expecting HSIC
to cover it.

## Ablation design

| arm | stock aux (`s_L1F, s_bcossim, s_hcossim, s_Hm`) | HSIC |
|---|---|---|
| `aux` | live sonar.py values (1e-9, 1e-5, 1e-6, 1e-6) | off |
| `hsic` | all zero | `s_hsic_heads` (default 1e-4) |
| `both` | live values | `s_hsic_heads` (+ optional `s_hsic_res`) |

All arms run through the same `HSICHyperparams` code path (the `aux`
arm just has zero HSIC scales), so the comparison isolates the loss
terms, not the plumbing. Compare across arms:

- reconstruction: `loss.csv` MSE column, `pareto.py` / `refit.py` FVU;
- head health: `experiments/layer4-freeze/freeze_diag.py` on each run
  dir (frozen counts, usage entropy);
- head quality: `headcoh.py` (semantic coherence), `headstruct.py`;
- scale sweep: `--s-hsic-heads {1e-5, 1e-4, 1e-3}` -- healthy per-layer
  CKA is O(0.01-0.1), so 1e-4 lands the term at the same order as the
  stock aux contributions relative to the final whitened MSE (~3e-6).

## Run

```bash
# 0. sanity: estimator self-test (CPU, seconds)
uv run python experiments/hsic-bottleneck/hsic.py

# 1. the three arms (each a FULL training run)
uv run python experiments/hsic-bottleneck/train_hsic.py --arm aux
uv run python experiments/hsic-bottleneck/train_hsic.py --arm hsic --s-hsic-heads 1e-4
uv run python experiments/hsic-bottleneck/train_hsic.py --arm both --s-hsic-heads 1e-4
```

## Inputs

| input | default | notes |
|---|---|---|
| `--cache` | `data/sonar_embeddings/mc4_4M.npy` | `encode_corpus.py` output |
| `--mse-weights` | `data/out/sonar/mse_weights.npy` | `''` disables whitening |
| `--out-base` | `experiments/hsic-bottleneck/runs` | runs land in `<out-base>/<arm>` |
| `--resid-gain` | `1` | gain-shape forwarding; `resid_nc_hm` has it off (`0`) |
| `--const0` | `1` | constant coordinate at layer 0; `resid_nc_hm` predates it (`0`) |

`--resid-gain` and `--const0` are passed explicitly because the model
otherwise takes the dataclass defaults, and `resid_gain`'s changed
(False -> True on 2026-09-10): which side of that the original arms
trained on is not recorded (`experiments/layer4-freeze/notes.md`). The
`aux` and `both` arms set `s_hcossim`, which is retired and now raises;
only the `hsic` arm runs as specified.

Reference run dirs for comparison:
`data/out/sonar/multilingual/resid_nc_hm` (the live aux config) and
`data/out/sonar/multilingual/resid_nc` (pre-`s_Hm`).

## Outputs

Each arm writes a standard run dir (orbax checkpoints, `loss.csv`,
`log.jsonl`, loss plots). **`loss.csv` caveat:** the third column
(index 2), labeled `MSE_ghost` by the stock pipeline, records the raw
(unweighted) HSIC penalty in these runs (0 in the `aux` arm). The rest of
the row is the stock layout (`ontologize/visualize/loss.py:COLUMNS`) --
the internal aux stats are still computed (and `stop_gradient`-ed when
their scale is 0), so `KL_m`, `entropy` etc. remain comparable across
arms.

## Cost (honest estimate)

- `hsic.py` self-test: CPU, seconds.
- Training arms: each is a **full sonar.py-scale run** (~375k steps at
  b=256): many hours to ~a day of GPU per arm; not CPU-feasible. The
  HSIC overhead is l x h Gram matrices of b^2 = 5x32x256^2 (~10M
  entries) plus a per-head median per step: expect **~5-15% step-time
  overhead** over the aux arm at b=256. Memory overhead ~100 MB.
- `s_hsic_res` adds one b^2 x d-dominated Gram pair per step:
  negligible at b=256.
