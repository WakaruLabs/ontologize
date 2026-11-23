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
`gausscopy.py head` writes the training stream's first rows as `head.npy`,
for a scorer that reads a cache's first rows (below, the SAE pair).

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
- `--marginals pca ica` runs it on copies of the cache's own rows (the
  same training and tail rows, whitened) that keep each coordinate's
  distribution in a frame fitted on the training rows and break every tie
  between coordinates: each coordinate's column is permuted on its own
  (`marginal_copy`). In `pca` the coordinates are the principal
  directions, so the copy keeps the covariance and each direction's
  marginal (its skew, tails and modes). In `ica` they are the sources of a
  symmetric FastICA (log-cosh contrast, in numpy) in the sphered principal
  frame, so the copy is exactly an independent-components model with the
  rows' own source marginals. Between the Gaussian copy and the cache,
  they say whether the reproducible cells come from oddly shaped
  directions or from how directions combine. Each frame's per-coordinate
  table (variance share, skew, excess kurtosis, Sarle's bimodality
  coefficient) goes to `coordinates_<frame>.csv`, with a summary and
  FastICA's convergence in `summary.json`. FastICA at d = 1024 on 131,072
  rows takes a few seconds an iteration on the CPU, up to `--ica-iters`.
  Every 25 iterations it prints the largest and median change, how many
  rows still change by the tolerance or more, its steps, and the largest
  change over two iterations, which stays small while rows flip between
  two states. Symmetric FastICA can cycle that way instead of converging.
  `--ica-patience N` damps its steps: each row of W moves part of the way
  to its update, and a row's share halves whenever that row, still above
  the tolerance, goes N iterations without a new low. The fixed points are
  the same, and convergence is read on the full update.
  `--ica-shared-step` uses one share for all rows instead, halved whenever
  the largest change goes N iterations without a new low. The
  iteration-800 frame in `notes.md` was fitted that way. From a random
  start the shared step stalls: on the same rows and start it left 751
  rows above the tolerance after 800 iterations, against 127 for per-row
  steps. Each fitted frame is saved as `frame_<frame>.npz`
  (mean, eigenvalues, eigenvectors, and for `ica` the unmixing), the ICA
  frame also every 100 iterations as `frame_ica.partial.npz` with its
  step, and `--ica-init` continues FastICA from either, fit on the same
  rows, at that step (`--ica-step` sets another) and counting the earlier
  iterations. With `--ica-iters 0` the saved frame is used as it is, once
  its principal frame is checked against the rows.
- `--groups TAU ...` runs it on copies between the ICA copy and the cache.
  The sources' magnitudes |s_i| are correlated over the training rows
  (Spearman, `energy_dependence`): uncorrelated sources that tend to be
  large together are the dependence ICA leaves. Average-linkage clustering
  on 1 − that correlation, cut at mean correlation TAU, groups them
  (`energy_groups`), and the copy permutes each group's sources together
  (`group_copy`), keeping each group's joint distribution and nothing that
  ties groups. All singletons is the ICA copy, one group the cache.
  `energy_ica.npy` holds the correlations and `groups_ica.npz` each
  grouping. `--no-cache` leaves the cache's own rows unscored.
- `--scale-copy` runs it on a copy that keeps a scale shared by all of a
  row's sources and nothing else that ties them. The row's scale is the
  median of its sources' magnitudes (`row_scale`). The rows are sorted by
  it into 64 bins of equal count, and within each bin every source's
  column is permuted on its own (`scale_copy`). Every source keeps its
  values exactly and, up to the bin's width, how they depend on the scale.
  Because the scale is estimated from the same sources, the copy adds a
  magnitude correlation of about 0.75/p between them. Dividing by the scale
  and multiplying it back after permuting would instead assume each source
  is the scale times something independent of it. On `mc4_sent16` that
  raised the sources' median kurtosis from 4.8 to 6.1. `summary.json`'s
  `ica_scale_stats` compares the rows' and the copy's scale spread, median
  magnitude correlation and median kurtosis.
- `--kmeans spherical` clusters directions instead: the rows less the
  training mean, scaled to unit length, with each row in the unit centre it
  is most aligned with (`spherical_kmeans`). FVU is still Euclidean, against
  each cell's mean training row (`cell_means`). On rows whose lengths vary,
  Euclidean k-means lumps the short rows into one oversized cell and stops
  at worse optima. `--kmeans spherical_lloyd` runs Euclidean Lloyd's
  iterations from the spherical solution's cell means, which asks whether
  Euclidean k-means can reach that basin from a better start. `--trim-seed Q` draws Euclidean k-means++'s centres only
  from the training rows within the Q quantile of distance from the mean
  (`trimmed_rows`). Every table also gives the share of the tail in its
  largest cell (`largest_cell`).
- The cells are also scored against the rows' languages (NMI,
  `language_nmi`) for the data sets whose tail row i stands for the
  cache's tail row i: the cache, the mixture and within.
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
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py \
    --marginals pca ica --out data/out/sonar/kmeans_seeds_marginals
```

Output: `data/out/sonar/kmeans_seeds/summary.json`.
`tests/test_kmeans_seeds.py` checks the assignment against brute force,
agreement on separated clusters and along a dominant direction, its
absence on an isotropic Gaussian, and that every cell ends up used. On rows
planted along four directions at widely varying lengths, spherical k-means
finds the directions with near-equal cells where Euclidean k-means puts most
rows in one cell, and trimmed seeding keeps the farthest rows out. For
the marginal copies: the `pca` copy keeps every principal coordinate's
values and the covariance and loses a planted dependence between
uncorrelated coordinates; FastICA recovers planted sources, with damped
steps too, per row or shared, and, started at its own fixed point, stays
there, and a shared step refuses per-row starting steps; a saved
`ica` fit continues on its own rows, or with no iterations is used as it
is, and is refused on others; the `ica`
copy keeps two uniform sources mixed along non-orthogonal directions
inside their parallelogram, where the `pca` copy spills rows outside it;
the shape statistics read a Gaussian, a uniform and two modes; the
magnitude correlations find sources planted to share a scale and group
them apart from independent ones; and the group copy keeps a group's
magnitude correlation while breaking it between groups, which the ICA
copy breaks within the group too. With a scale shared by every source,
the scale copy keeps the sources' magnitude correlation, which the ICA
copy breaks; without one, the scale copy adds next to none. A second
draw of training rows avoids the first.

## Whether FastICA's sources reproduce (`ica_seeds.py`)

FastICA, as `kmeans_seeds.py` fits it, on its training rows (A, continued
from a saved `frame_ica.npz` with `--frame-a`) and on as many rows from the
same prefix and disjoint from them (B, another random start). On the
held-out tail, each of A's sources is matched one to one with B's
(Hungarian) by the |correlation| of the two sources and by the |cosine|
of their directions in the whitened frame. Beside them: B's sources rotated
at random, which no data pin, and the two principal frames' coordinates.
`sources.csv` gives each of A's sources its kurtosis, variance share,
partner, both matched values and its runner-up; `summary.json` bins them by
kurtosis, since ICA cannot pin a source near Gaussian. Both fits are damped
(`--patience`, default 50; per row unless `--shared-step`) and
checkpointed (`frame_a.partial.npz`, `frame_b.partial.npz`; `--frame-b`
continues B). `--iters` caps each fit's
iterations, and `--iters-a` A's alone: `--iters-a 0` uses `--frame-a` as it
is, so B can be given the iterations A had.

```bash
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/ica_seeds.py \
    --frame-a data/out/sonar/kmeans_seeds_ica/frame_ica.npz --iters-a 0 --iters 800 \
    --out data/out/sonar/ica_seeds
```

`tests/test_ica_seeds.py` checks that a relabelled, sign-flipped copy
matches exactly, and on planted sources that non-Gaussian ones reproduce
across three disjoint samples, a random rotation does not, and two
Gaussian sources land in a different rotation of their plane.

## Whether a stack's atoms point along the sources (`atomsources.py`)

The stack's atoms are set against FastICA's source directions: the rows of
the mixing matrix in the loss's whitened frame. Each decoded atom
(`headcontrib.atoms_and_codes`) is centered on its head's usage-weighted
mean atom, and only entries used on `--rows` tail rows count.
- **Per run and layer:** each source's best |cosine| with any atom, by
  class from `ica_seeds.py`'s `sources.csv`. The classes are skewed
  (kurtosis ≥ 5, |skew| ≥ 1), symmetric (kurtosis < 3, |skew| < 0.3) and
  the rest.
- **References:** the same for the sources rotated at random (the same
  spectrum, nothing ICA found) and for the principal directions.
- **Per atom:** its best |cosine| with any source, and the class of its
  nearest source.
- **Per source:** the Spearman correlation of its best cosine with its
  correlation across samples and with its variance share.
- **With `--b`:** the share of each class that both runs have an atom
  along, beside the product of the two runs' shares.

It reads only the checkpoints, the frame and the tail rows. Output:
`summary.json` and `sources.csv` (each source's best cosine per run and
layer).

```bash
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/atomsources.py \
    --frame data/out/sonar/kmeans_seeds_ica/frame_ica.iter800.npz \
    --sources data/out/sonar/ica_seeds/sources.csv \
    --a <run> --b <run>_s43 --out <dir>
```

`tests/test_atomsources.py` checks that copies of some directions are
found along them and not along the rest, that the classes follow kurtosis
and skew, that the rotated reference keeps the spectrum, that centering
drops unused entries, and that "both runs" is read against independence.

## Which heads reproduce as heads (`headscreen.py`)

A multitrait–multimethod screen of a seed pair. Each head of A is matched
to one of B's over the whole models (Hungarian on NMI, as `partition.py`
matches), on the first third of the tail rows; on the same third it finds
each side's runner-up partner in the other seed and its closest other head
in its own seed. On the other two thirds, the matched NMI must exceed zero,
both runner-ups and both within-seed neighbours. Every NMI is corrected by
the Miller–Madow bias of the plug-in mutual information,
(r_a − 1)(r_b − 1)/2n nats for r occupied cells, over the mean entropy;
`null_check` gives the corrected NMI of a rows-shuffled pairing. Each
comparison's noise is the SD, over a layer's heads, of how much the two
scoring thirds disagree, halved; a head passes when every comparison is
more than `--z` (default 2) such SDs above zero. `heads.csv` has every
matched pair, and `summary.json` the passing heads per layer and how many
fail each comparison.

```bash
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/headscreen.py \
    --a data/out/sonar/recipe_ladder/ste_h76_init01_med837_ctrl \
    --b data/out/sonar/recipe_ladder/ste_h76_init01_med837_ctrl_s43 \
    --rows 65536 --out data/out/sonar/headscreen/ctrl
```

`tests/test_headscreen.py` checks that independent partitions read zero
once corrected while the plug-in NMI does not, and on planted heads that a
shared head passes, a shared head whose own seed holds a copy of it fails
its within-seed comparison, and the rest fail.
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

## The SAE pair on a Gaussian copy (`sae_gauss.py`, `sae_gauss_seeds.sh`)

The k32 SAE pair (`sae_conv/m5120_k32`, `_s43`, on `mc4_4M`) shares the
subspace of its unmatched latents at 96–97% of the optimum, against the
data's top principal directions' roughly half (`subspace.py --mode sae`).
The same pair trained on a Gaussian copy of `mc4_4M` says how much of that
the covariance alone gives.

- The copy: `gausscopy.py prepare --cache data/sonar_embeddings/mc4_4M.npy
  --rows 3941632 --eval-rows 557056 --out data/out/sonar/gausscopy_4M`, the
  moments of every row the real pair trained on, and held-out rows enough
  for `splitting.py`'s 524,288 at the head and the 32,768-row tail.
  `gausscopy.py head` writes the first 524,288 rows of the training stream
  as `head.npy`, since `splitting.py` read the real pair's training rows.
- `sae_gauss.py` is `sae.py` (all its flags) with `--gauss DIR`: it
  generates the first n rows of the copy's training stream in memory (n
  defaults to the cache's own training split, 3,941,632 rows, 16 GB; under
  3 minutes) and hands them, with the last `--eval-rows` of `eval.npy`, to
  `sae.train(cfg, rows=...)`, which then reads no cache. It writes
  `gauss.json` beside the run.
- `sae_gauss_seeds.sh` trains the real pair's settings (`--m 5120 --topk
  32 --epochs 150`, 144,300 steps of 4,096) at seeds 42 and 43 into
  `data/out/sonar/sae_gauss/`, then scores the pair with `subspace.py
  --mode sae` on the copy's tail and `splitting.py` on `head.npy`, into
  `scores/`. It waits for the unit named by `AFTER` (default `width-seeds`)
  to be down at three checks five minutes apart, then for an idle GPU:

```bash
systemd-run --user --unit sae-gauss-seeds -p Restart=on-failure -p RestartSec=120 \
    --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/sae_gauss_seeds.log \
    -p StandardError=append:$PWD/experiments/logs/sae_gauss_seeds.log \
    /usr/bin/bash experiments/seed-stability/sae_gauss_seeds.sh
```

`tests/test_sae.py` checks that rows passed in memory train exactly as the
same rows in a cache do, and that `--block-rows` refuses them.
`tests/test_gausscopy.py` checks that `sae_gauss.py`'s rows are the
training stream's and its held-out rows the end of `eval.npy`.

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
