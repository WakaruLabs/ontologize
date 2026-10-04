# seed-stability

Partition-level stability for the Ontologizer — the follow-up the paper's
frame note names explicitly: for a steering generator the unit that must
reproduce across runs is the head's task (partition), not the individual
feature vector. The SAE side has its numbers (k32 vs k32_s43 matched
fractions **0.31 / 0.24 / 0.15 / 0.03** at τ = 0.3/0.5/0.7/0.9, via
splitting.py); this experiment produces the Ontologizer's, at both the
head and the entry level.

Two pieces:

1. `train_seed43.py` — a second-seed replica of the live sonar.py config.
   It **imports sonar.py and reads every constant from it** (no copied
   hyperparameters to drift), changing only: `Hyperparams.seed`
   (default 43), the cached-corpus shuffle seed (default = seed, matching
   how `sae.py --seed 43` changes both init and data order), and the
   output directory.
2. `match_heads.py` — matches the two runs' HEADS by Hungarian assignment
   on decode-direction similarity (each entry's exact SONAR-space decode
   direction = its row of G from `refit.onto_linear_model`; head-pair
   similarity = mean entry cosine under the optimal entry bijection,
   itself a Hungarian), then matches entries within matched heads, then
   re-scores the matched entry pairs in splitting.py's own co-fire
   containment frame at the same τ sweep — the number that lands directly
   next to 0.31/0.24/0.15/0.03.

## Run

From the repo root:

```bash
# 1. train the replica (full run; --epochs 2 for a cheap pilot)
uv run python experiments/seed-stability/train_seed43.py

# 2. match heads + entries between the two runs
uv run python experiments/seed-stability/match_heads.py \
    --a data/out/sonar/multilingual/resid_nc_hm \
    --b data/out/sonar/multilingual/resid_nc_hm_s43

# variations: whitened metric, layer-common structure removed, geometry only
uv run python experiments/seed-stability/match_heads.py --metric whitened
uv run python experiments/seed-stability/match_heads.py --center
uv run python experiments/seed-stability/match_heads.py --rows 0

# complementary, unconstrained entry-level frame (existing tooling):
uv run python splitting.py --a data/out/sonar/multilingual/resid_nc_hm \
    --b data/out/sonar/multilingual/resid_nc_hm_s43
```

## Inputs

- Primary checkpoint: `data/out/sonar/multilingual/resid_nc_hm` (default;
  `resid_nc` also works for cross-config sanity checks — match_heads
  accepts any two checkpoint dirs with the same l/h/k).
- Replica checkpoint: written by `train_seed43.py` to
  `<sonar out>_s<seed>` (default
  `data/out/sonar/multilingual/resid_nc_hm_s43`).
- Cache `data/sonar_embeddings/mc4_4M.npy` (training for the replica;
  shared rows for the activation cross-check).
- `data/out/sonar/mse_weights.npy` for `--metric whitened`.

## Outputs (in `experiments/seed-stability/out/<a>__<b>/`)

- `heads.csv` — per head: matched partner, head_sim (mean matched-entry
  cosine), entry-cosine fractions, containment fraction at τ=0.7.
- `entries.csv` — every matched entry pair: cosine, fire counts, co-fire,
  both containment directions.
- `summary.json` + printed tables:
  - matched-head and matched-entry fractions over the cosine sweep, with
    the shuffled-regrouping null (`--nulls`) for the head statistic;
  - **the headline**: matched-entry containment fractions at
    τ = 0.3/0.5/0.7/0.9 over Hungarian-matched pairs, printed next to
    the SAE reference numbers.

Reading it: head-level stability with entry-level churn (high head_sim,
low entry containment) supports "the head's task reproduces, its entry
vectors do not" — the partition-level claim. Low head_sim near the null
would say the partition itself is seed-noise.

## Cost

- `train_seed43.py`: **a full training run** — identical cost to the
  original resid_nc_hm run (24 epochs × ~15.6k steps/epoch ≈ 375k steps
  from the cache; order of several GPU-hours to ~a day on one GPU
  depending on hardware; it checkpoints every 100 steps and resumes).
  There is no cheap substitute; `--epochs 2` gives a pilot whose
  stability numbers are not directly comparable (less annealing).
- `match_heads.py`: checkpoint restores need GPU JAX; G extraction is one
  `decodeEntries`-style pass (seconds). The Hungarian stage is ≈ h²
  assignments of size k per layer (+ nulls) — **seconds with scipy, ~1–2
  min with the pure-python fallback**. The activation cross-check at the
  default 262k rows is **~2–5 min on GPU**.

## Assumptions / caveats

- Head matching is within-layer by default: with `forward="resid"` the
  layers are ordered residual stages, so layer i's heads can only
  meaningfully correspond to layer i's. `--cross-layer` lifts this if you
  want to check leakage.
- Decode-direction cosines can be inflated by layer-common structure
  (e.g. a shared corpus-mean component); `--center` subtracts each
  layer's mean direction first, and the shuffled-regrouping null
  quantifies what "random heads" score either way. `--metric whitened`
  re-runs everything under the objective's inner product (headcoh's
  convention).
- Hungarian assignment uses `scipy.optimize.linear_sum_assignment`
  (scipy is already in the environment as a jax transitive dependency —
  see uv.lock); an exact pure-python O(n³) fallback is included, so no
  new dependency either way.
- The containment cross-check scores the *matched* pairs only — it is a
  lower bound on unconstrained entry stability by construction; run
  splitting.py on the same pair for the unconstrained number.
- `train_seed43.py` reshuffles the data by default (matching the SAE
  replica protocol). Pass `--data-seed 42` to isolate pure
  init-seed sensitivity at fixed data order.
