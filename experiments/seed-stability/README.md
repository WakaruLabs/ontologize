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

## Subspaces (`subspace.py`)

Seed agreement one level up from heads and latents: do two seeds span
the same subspaces, and do they agree with each other more than each
agrees with the data? Everything is in the loss's whitened frame, with
the whitened data's top-r and bottom-r principal subspaces and chance
(r/d) as the references.

- `--mode sae` is Gerasimov et al.'s subspace transfer. It splits each
  SAE's live latents into those with a match at decoder cosine 0.7 in
  the other seed and those without, size-matched. Per set it gives the
  energy share in its own top-r subspace (the ceiling), in the other
  seed's subspace for the same kind of set, and in the references.
- `--mode stack` is LoRA's normalized projection similarity
  φ(i, j) = ||A_iᵀ B_j||²_F / min(i, j) between the eigenbases of each
  layer's contribution x̂_{ℓ+1} − x̂_ℓ in the two seeds, and between each
  seed and the data. Chance is r/d on the diagonal, so rank 8 in 1,024
  dimensions tops out at 128× chance. A layer past the first is fit to
  the residual x − x̂_ℓ, so each seed's contribution is also compared with
  the principal subspace of its own layer input and of the other seed's,
  and the two inputs with each other.

```bash
uv run python experiments/seed-stability/subspace.py --mode sae \
    --a data/out/sonar/sae_conv/m5120_k32/params.npz \
    --b data/out/sonar/sae_conv/m5120_k32_s43/params.npz \
    --out data/out/sonar/seed_subspace/sae_k32
uv run python experiments/seed-stability/subspace.py --mode stack \
    --a data/out/sonar/multilingual/ste_h76_init01 \
    --b data/out/sonar/multilingual/ste_h76_i01_s43 --step 369500 \
    --out data/out/sonar/seed_subspace/ste_h76_init01
```

`tests/test_subspace.py` checks φ on nested, orthogonal and random
subspaces, the energy share, and the transfer curves on planted sets.

`gpt2_sae_seeds.sh` trains a seed pair (42, 43) of the GPT-2 k32 SAE at
`sae.py`'s defaults into `data/out/gpt2_l8/sae_seeds/`, the
language-model counterpart of the SONAR pair. It waits for unit
`recipe-ladder-nonoise` and an idle GPU, and runs as a systemd unit:

```bash
systemd-run --user --unit gpt2-sae-seeds -p Restart=on-failure -p RestartSec=120 \
    --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/gpt2_sae_seeds.log \
    -p StandardError=append:$PWD/experiments/logs/gpt2_sae_seeds.log \
    /usr/bin/bash experiments/seed-stability/gpt2_sae_seeds.sh
```

## A Gaussian copy (`gausscopy.py`, `train_gauss.py`, `gauss_seeds.sh`)

The baseline for every seed statistic at once. A model trained on rows
from the Gaussian with the training rows' mean and covariance has nothing
to agree on beyond the covariance, so whatever the cache pair agrees on
beyond this pair is structure beyond the data's second moments.

- `gausscopy.py prepare` reads the first 12.8M rows of `mc4_sent16` (the
  recipe ladder's 50,000 steps of 256) and saves their mean, covariance
  and its square root (from the eigendecomposition, which tolerates a
  covariance too ill-conditioned for Cholesky) to
  `data/out/sonar/gausscopy/moments.npz`. It writes 65,536 held-out rows
  from another stream to `eval.npy`, in the cache layout, with what the
  sample moments miss by in `eval.meta.json`. The scorers read its last
  rows with `--cache`; the trainer's initialization probe reads its first
  64.
- `GaussianSource` generates row i as block i // 4096 of a stream keyed by
  (seed, block). A resumed run sees the rows it would have seen, and two
  seeds of an arm see the same rows in the same order, as they do on the
  cache (the trainer does not shuffle).
- `train_gauss.py` is `experiments/adam-eps/train_eps.py` with that source
  in place of the cache's rows; pass `--epochs` rather than `--steps`.
- `gauss_seeds.sh` trains the ladder's all-off arm (with its training
  noise) at seeds 42 and 43, then scores the pair with `partition.py`,
  `headcontrib.py`, `subspace.py --mode stack` and `ratecurve.py` on
  `eval.npy`, into `data/out/sonar/gausscopy/scores/`. It waits for an idle
  GPU and runs as a systemd unit:

```bash
systemd-run --user --unit gauss-seeds -p Restart=on-failure -p RestartSec=120 \
    --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/gauss_seeds.log \
    -p StandardError=append:$PWD/experiments/logs/gauss_seeds.log \
    /usr/bin/bash experiments/seed-stability/gauss_seeds.sh
```

`tests/test_gausscopy.py` checks the chunked moments, the square root of a
rank-deficient covariance, and the source: its rows' moments, its
determinism across a pickle and a block boundary, and its streams.

## How far a partition of the data reproduces (`kmeans_seeds.py`)

A layer-0 head partitions the data itself, so the question of what a
reproducible partition of these rows looks like has a direct reference:
k-means. `kmeans_seeds.py` runs it in the loss's whitened frame
(k-means++ seeding, Lloyd's iterations to a fixed point, an emptied cell
reseeded at the farthest row) at several seeds and values of k, fits on
131,072 rows drawn from the training prefix, and assigns the 32,768-row
tail. Between every pair of seeds it reports the tail partitions' NMI
(`partition.py`'s `nmi_matrix`, so the numbers read beside the heads'), a
null with one partition's rows shuffled, and each seed's FVU.

- `--gauss` runs the same on the Gaussian copy: training rows from its
  training stream, the tail from `eval.npy`.
- `--mixture` runs it on a copy that keeps each language's mean and
  covariance (`gausscopy.py mixture` writes `mixture.npz` from the
  `.langs.npy` sidecar), with every row drawn under the real row's
  language. Its pooled covariance is the cache's, so it differs from the
  Gaussian copy by the language structure alone.
- `--within-language` (the same `mixture.npz`) runs it on the cache's own
  rows less their language's mean.
- For every data set but the Gaussian copy, each seed's cells are also
  scored against the rows' languages (NMI, `language_nmi`).
- `--ckpt` adds, per layer of a stack, each head's best NMI against the
  k-means partitions at the model's k.

```bash
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py \
    --gauss data/out/sonar/gausscopy \
    --ckpt data/out/sonar/recipe_ladder/ste_h76_init01_med837_nobcos_nosupp_nol1f_nohm
uv run python experiments/seed-stability/gausscopy.py mixture
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py \
    --mixture data/out/sonar/gausscopy/mixture.npz \
    --within-language data/out/sonar/gausscopy/mixture.npz \
    --out data/out/sonar/kmeans_seeds_language
```

Output: `data/out/sonar/kmeans_seeds/summary.json`.
`tests/test_kmeans_seeds.py` checks the assignment against brute force,
agreement on separated clusters and along a dominant direction, its
absence on an isotropic Gaussian, and that every cell ends up used.
`tests/test_gausscopy.py` checks the per-language moments against numpy's,
their total covariance against the pooled one, and that mixture rows
follow their language's moments and depend only on its stream.

## Summed heads against a cascade (`kmeans_structure.py`)

The stack differs from one k-means in two ways: each layer sums many
heads, and the layers form a residual cascade. `kmeans_structure.py`
separates them with quantizers that need no training.

- `--mode aq` sums h codebooks at one stage (additive quantization), fit by
  FAISS's LSQ++, `--restarts` times per h. Heads are matched between runs
  by NMI with a Hungarian assignment, beside a shuffled null, and the runs
  are read against their training error: agreement among the `--best`
  lowest-error runs, and its rank correlation with each pair's worse
  error. FAISS is not a project dependency, so this mode runs under
  `uv run --with faiss-cpu`; `unpack` turns its packed codes into one entry
  per head.
- `--mode rvq` is a cascade of single k-means stages, each on the previous
  stages' residual, run independently per seed and with stage 0 shared
  (taken from the first seed), so that stage 1 sees identical inputs.

```bash
JAX_PLATFORMS=cpu uv run --with faiss-cpu python experiments/seed-stability/kmeans_structure.py \
    --mode aq --train-rows 32768 --gauss data/out/sonar/gausscopy
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_structure.py --mode rvq
```

Output: `data/out/sonar/kmeans_structure/aq.json` and `rvq.json`.
`tests/test_kmeans_structure.py` checks the cascade's falling prefix error
and a shared first stage, the head matching, and the unpacking of FAISS's
codes.

## Agreement against width in the trained stack (`width_seeds.sh`)

LSQ's agreement halves with each doubling of its codebooks. The trained
counterpart is a one-layer stack of h heads of 32. `width_seeds.sh` trains
one with the recipe ladder's all-off flags on `mc4_sent16` at seeds 42 and
43 for each h in `WIDTHS` (default 1, 4, 16, 76), 50,000 steps each. It
scores each pair with `partition.py` and `ratecurve.py` in
`data/out/sonar/width_seeds/scores/`. It waits for an idle GPU and trains
one run at a time, so it runs as a systemd unit, as `gauss_seeds.sh` does.
A one-head stack needs `DictBlock.support_overlap` to read 0 at h = 1,
which it now does (`tests/test_support.py`).

```bash
systemd-run --user --unit width-seeds -p Restart=on-failure -p RestartSec=120 \
    --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/width_seeds.log \
    -p StandardError=append:$PWD/experiments/logs/width_seeds.log \
    /usr/bin/bash experiments/seed-stability/width_seeds.sh
```

## Cost

- `train_seed43.py`: **a full training run** — identical cost to the
  original resid_nc_hm run (24 epochs × ~15.6k steps/epoch ≈ 375k steps
  from the cache; order of several GPU-hours to ~a day on one GPU
  depending on hardware; it checkpoints every 100 steps and resumes).
  There is no cheap substitute; `--epochs 2` gives a pilot whose
  stability numbers are not directly comparable (less annealing).
- `match_heads.py`: G extraction is one
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
