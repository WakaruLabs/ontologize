# rl-classifications

Keira's own open question (paper/description, introduction, "Steering
could push it too far off manifold"): *"can we RL a model for high/low
scores on a classification and have it converge faster than paired
steering or PCA vectors?"* This is the minimal honest first experiment,
built entirely from machinery already in the repo:

- **reward**: a chosen (layer, head, entry)'s tag probability under the
  frozen Ontologizer (the exact soft-forward probe steerfid.py and
  autointerp.py use), quantile-normalized (steerfid's `firing_quantiles`),
  minus a collateral penalty.
- **policy**: a small MLP adapter on the SONAR embedding, producing a
  steering DIRECTION applied at fixed magnitude `--mag` and rescaled to
  |x|, i.e. exactly steerfid's matched-budget protocol, so the comparison
  against paired steering is at equal intervention budget.
- **loop**: REINFORCE (Gaussian policy over directions, per-state
  mean-reward baseline) in jax, with the M2M100 decode / SONAR re-encode
  cycle providing the non-differentiable reward; plus a differentiable
  in-embedding sanity tier.
- **comparison**: `eval` scores policy vs the model's own paired-steering
  delta (withArgs set-intervention, steerfid's onto mode) vs random
  directions on held-out cache-tail rows, in steerfid's effect / hit-rate
  / collateral columns.

"Converge faster than paired steering" is operationalized honestly:
paired steering needs zero training, so the real questions are (a) how
many reward queries the policy needs to reach paired steering's effect at
matched collateral (read `rl_log.csv` against the `paired_steering` row of
`compare.csv`), and (b) whether it ends up beating it.

## What is runnable vs design (be honest with yourself here)

| piece | status |
|---|---|
| `train --reward embed` (differentiable in-embed reward, gradient ascent) | **runnable** on GPU JAX alone; the debug/ceiling tier, minutes |
| `train --reward cycle` (REINFORCE, decode+re-encode reward) | **runnable but expensive**; implemented end to end, not yet validated on a real run (no checkpoints in this tree), and REINFORCE at b=32, sigma=0.3 is a first guess, expect hyperparameter iteration |
| `eval` (policy vs paired steering vs random) | **runnable** once a policy exists |
| `--direction low` | runnable only with `--rows-file` (rows where the entry fires, e.g. harvested top rows); random corpus rows give a zero-gradient floor |
| PCA-vector comparison arm | **design only**: needs a per-classification PCA direction (top PC of embeddings grouped by the head's argmax entry vs rest, over a cache slice); slot it in as a fourth `arms` entry in `evaluate` |
| multi-entry / multi-head policies, KL-to-decoder naturalness regularizers, off-manifold detection in the reward | **design only**; noted so nobody mistakes this skeleton for that experiment |

## Run (from the repository root)

```bash
# pick a live entry first (autointerp harvest freq, or steerfid fire rates)

# 1. sanity tier: differentiable in-embed reward (GPU jax, ~minutes)
uv run python experiments/rl-classifications/rl_reinforce.py train \
    --ckpt data/out/sonar/multilingual/resid_nc \
    --layer 0 --head 3 --entry 5 --reward embed --steps 500

# 2. the actual experiment: cycle-consistency reward, REINFORCE
uv run python experiments/rl-classifications/rl_reinforce.py train \
    --ckpt data/out/sonar/multilingual/resid_nc \
    --layer 0 --head 3 --entry 5 --reward cycle --steps 300 \
    --b 16 --n-samples 4 --device cuda

# 3. compare vs paired steering + random at matched magnitude
uv run python experiments/rl-classifications/rl_reinforce.py eval \
    --ckpt data/out/sonar/multilingual/resid_nc \
    --layer 0 --head 3 --entry 5 --reward cycle --n-eval 16 --device cuda
```

## Inputs

- `--ckpt`: frozen Ontologizer run dir; default
  `data/out/sonar/multilingual/resid_nc`
  (`data/out/sonar/multilingual/resid_nc_hm` is the other live run).
  Restores on GPU JAX only.
- `--cache`: `data/sonar_embeddings/mc4_4M.npy`. Training rows come from
  the cache head; the eval stage uses the tail rows sae.py holds out.
- The M2M100 decoder / SONAR encoder pair (`raxtemur/SONAR_200_text_decoder`,
  `cointegrated/SONAR_200_text_encoder`) download from HuggingFace on
  first use.

## Outputs (into `experiments/rl-classifications/out/<run>_l..h..k.._<reward>/`)

- `meta.json`: target entry, fire rate, quantiles, hyperparameters.
- `rl_log.csv`: per-step `reward, eff, coll, hit` (the convergence curve
  the "faster than paired steering" question is asked of).
- `policy.npz`: latest policy weights (+ step; resume warm-starts weights
  with a fresh optimizer state).
- `compare.csv`: `arm, mag, effect, hit_rate, collateral, n` for policy /
  paired_steering / random.
- `eval_texts.jsonl`: per-row base and steered decodes for qualitative
  inspection (the "did it just learn to print Burmese" check).

## Cost (honest)

- `embed` tier: one jitted step per update; 500 steps in single-digit
  minutes on the training GPU.
- `cycle` tier: per step decodes `b * (n_samples + 1)` texts (M2M100
  greedy, max_length 48) plus re-encodes them. At b=16, n_samples=4 that
  is 80 generations/step: roughly 5-15 s/step on a GPU, minutes/step on
  CPU. 300 steps is therefore ~0.5-1.5 GPU-hours (and ~24k reward
  queries; log this against paired steering's zero).
- `eval`: 4 arms x n_eval decodes + cycles; minutes.

## Assumptions / design choices

- Reward normalization q_hi/q_hit is measured once on a cache-head
  reference slice, conditional on firing (steerfid's convention); a dead
  entry aborts loudly rather than training against a zero.
- The policy steers at FIXED magnitude and renormalizes to |x|: it cannot
  win by norm inflation, only by direction choice, keeping the comparison
  with paired steering at matched budget (steerfid's design).
- The paired-steering arm re-derives its per-sample direction from the
  frozen model (withArgs set minus no-op, unit-normalized), so eval never
  reuses policy machinery for the baseline.
- `hit_rate` keeps steerfid's "cycled activation clears q_hit" semantics;
  under `--direction low` read it as the complement (the eval stage
  flips effect's sign but not hit's definition).
- REINFORCE variance control is just per-state mean baselines + global
  advantage normalization. If the curve is flat noise, the next steps in
  order: raise n_samples, lower sigma, switch the surrogate to two-sided
  antithetic sampling (eps and -eps share a state), none of which change
  the interfaces here.
