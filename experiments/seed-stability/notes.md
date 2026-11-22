# seed-stability: notes

## 2026-11-19: the SAE seed baseline, remeasured, and beside the field's own statistics

The SAE side of every seed comparison here is the `k32` pair, `sae_conv/m5120_k32` against
`m5120_k32_s43`. Its containment fractions (0.31 / 0.24 / 0.15 / 0.03) were typed into the
findings, `match_heads.py` and `make_figs.py` from a run that no longer had an output on disk.
`splitting.py` was rerun on the pair, and now also prints three statistics the literature reports
seed agreement in, so the containment fractions can be read against them.

**Setup.** `splitting.py`'s defaults: the cache's first 524,288 rows (training rows for both SAEs),
latents firing at least 50 times live (5,114 of 5,120 in A); the agreement statistics both ways,
A->B / B->A. Output in `data/out/sonar/splitting/m5120_k32__m5120_k32_s43/` (`agreement.json`).

**Containment reproduces.**

| τ | A matched | B matched |
|---|---|---|
| 0.3 | 0.307 | 0.313 |
| 0.5 | 0.235 | 0.234 |
| 0.7 | 0.147 | 0.147 |
| 0.9 | 0.030 | 0.030 |

These round to the numbers in the findings, the figure and the scripts, so none of them changes.

**The field's statistics are low too.**

| statistic | this pair | reported for language-model SAEs |
|---|---|---|
| decoder share, cosine > 0.7 (Leask et al.) | 0.331 / 0.334 | 0.94, same-size GPT-2 SAEs from two seeds |
| PW-MCC (Song et al.) | 0.511 (random directions 0.117) | 0.818 TopK on Pythia-160M, 0.790 on Gemma-2-2B |
| mean MPPC (Wang et al.) | 0.384 / 0.385 | 0.81 across SAE seeds on Pythia |

- **Decoder share over thresholds:** 0.489 / 0.500 above cosine 0.4, 0.419 / 0.430 above 0.5,
  0.241 / 0.241 above 0.9.
- **PW-MCC's matched pairs** above 0.4, 0.5, 0.7 and 0.9: 0.445, 0.400, 0.328 and 0.241.
- **MPPC above** 0.4, 0.5, 0.7 and 0.9: 0.392, 0.356, 0.298 and 0.178 (A->B; B->A within 0.004).
- **What it settles.** The findings explained this pair's 3-31% by the containment metric's
  strictness: under the decoder-cosine and activation-correlation metrics, language-model SAEs
  report agreement near 0.8. Under those metrics this pair reads 0.33-0.51. So the gap to
  language-model SAEs is not the metric. Under the same containment metric the pair still agrees
  30 to 300 times as often as the softmax Ontologizer's seed pair (at most 0.1%), which is the
  comparison the findings rest on, and that stands.
- **What it does not settle.** This SAE leaves half the variance unexplained (FVU_w 0.499), and
  the papers' SAEs were trained on language-model activations and, in Song et al., configured to
  maximize PW-MCC. Whether the low agreement comes from SONAR or from the fit is open. A seed pair
  of the GPT-2 `m12160_k32` SAE (master has one seed) would separate them, at one GPU run.
- **PW-MCC matches an earlier reading of the same checkpoints** (0.511, random 0.116), which used
  its own matching code.

```bash
JAX_PLATFORMS=cpu uv run python splitting.py \
    --a data/out/sonar/sae_conv/m5120_k32/params.npz \
    --b data/out/sonar/sae_conv/m5120_k32_s43/params.npz \
    --out data/out/sonar/splitting/m5120_k32__m5120_k32_s43
```

## 2026-11-20: subspaces, against the data's own (`subspace.py`)

**Short version.** The SAE pair's unmatched latents span a subspace that the other seed
reproduces almost to the optimum and about twice as well as the data's principal directions. The
hard stack's layers 0 and 3 agree with each other no more than each agrees with the data, and
layers 1–2 only slightly more in their first few directions. At layer 4 the seeds pick different
data-aligned subspaces. The old notes' rank-8 table (ste-arm, "Subspaces, and why their agreement
is not the architecture's") is φ(8)/chance and reproduces here within 20%, on 32,768 rows. Its
text's "25x to 33x" did not match it, and nor did the appendix's.

**SAE pair** (`sae_conv/m5120_k32`, `_s43`; whitened frame, 32,768 tail rows of `mc4_4M`). Of
5,120 and 5,118 live latents, 33.0% and 33.3% have a match at decoder cosine ≥ 0.7 in the other
seed (Gerasimov et al.: 0.746 on GPT-2). Energy share of seed 42's set inside each subspace, mean
over 10 size-matched draws (`seed_subspace/sae_k32/`):

| set | r | own (ceiling) | other seed's | data's top-r | data's bottom-r | chance |
|---|---|---|---|---|---|---|
| unmatched (3,413) | 8 | 0.078 | 0.075 | 0.031 | 0.0002 | 0.008 |
| | 64 | 0.355 | 0.342 | 0.172 | 0.002 | 0.063 |
| | 128 | 0.564 | 0.547 | 0.306 | 0.007 | 0.125 |
| | 512 | 0.951 | 0.931 | 0.877 | 0.123 | 0.500 |
| matched (1,690) | 8 | 0.081 | 0.079 | 0.061 | 0.0003 | 0.008 |
| | 64 | 0.301 | 0.291 | 0.254 | 0.007 | 0.063 |
| | 128 | 0.457 | 0.439 | 0.383 | 0.019 | 0.125 |

- **The other seed captures 96–97% of the optimum at every rank**, for both sets.
- **The data's own top-r captures about half as much for the unmatched set**, and nearly as much
  for the matched set: matched latents lie more along the data's principal directions.
- So Gerasimov et al.'s "reproducible subspaces" survives the control that removed the stack's
  layer agreement. The SAE's unstable latents re-express a subspace both seeds share beyond the
  data's second moments.

**Hard stack** (`ste_h76_init01`, `ste_h76_i01_s43`, step 369500; each layer's contribution
x̂_{ℓ+1} − x̂_ℓ, whitened and centered; `seed_subspace/ste_h76_init01/`). φ(r, r) over chance
(r/1024), seeds against each other | seed 42 against the data | seed 43 against the data:

| layer | r = 1 | 8 | 32 | 128 |
|---|---|---|---|---|
| 0 | 1023 \| 1021 \| 1021 | 126.9 \| 123.2 \| 120.5 | 31.4 \| 30.2 \| 30.1 | 7.5 \| 7.2 \| 7.2 |
| 1 | 979 \| 683 \| 553 | 121.8 \| 103.7 \| 104.5 | 27.5 \| 26.2 \| 26.0 | 6.2 \| 6.3 \| 6.4 |
| 2 | 842 \| 636 \| 463 | 106.9 \| 97.6 \| 97.1 | 22.6 \| 23.0 \| 23.5 | 4.8 \| 5.5 \| 5.5 |
| 3 | 779 \| 670 \| 645 | 82.1 \| 85.5 \| 86.0 | 13.0 \| 17.3 \| 17.3 | 3.2 \| 4.2 \| 4.2 |
| 4 | 204 \| 266 \| 306 | 10.6 \| 26.9 \| 25.7 | 3.1 \| 6.7 \| 6.8 | 1.7 \| 2.6 \| 2.6 |

The ceiling is d/r: 1024 at r = 1, 128 at 8, 32 at 32, 8 at 128. In raw φ, the top direction is
shared at 0.999 in layer 0, and at 0.96 and 0.82 between seeds in layers 1–2 against 0.45–0.67 with
the data.
- **Layer 0** is the data's principal subspace in both seeds (φ ≥ 0.93 to r = 128).
- **Layers 1–2** agree a little beyond the data in their first 1–16 directions, and not past rank
  32. Their own inputs account for it (next section).
- **Layer 3** agrees with the other seed less than with the data from rank 8 on.
- **Layer 4's seeds share under half of what each shares with the data** from rank 2 to rank 32
  (0.39–0.47 of it), and 0.67–0.77 of it at rank 1.
- The full prefix grid φ(i, j) per layer is in `subspace_stack.npz`.

The two statistics differ (energy of a set of directions, against overlap of covariance
eigenbases), so the SAE and the stack are compared by reading, not number to number. The
findings' appendix now gives the stack's φ with its formula, and the SAE's transfer beside it.

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

## 2026-11-20: the stack against each layer's own input

**Short version.** Layers 1–2's agreement beyond the data is the cascade's. A layer past the first
is fit to the residual y_ℓ = x − x̂_ℓ that earlier layers leave. Each seed's contribution lies
along the principal subspace of its own y_ℓ about as closely as along the other seed's
contribution, and the two seeds' residuals share their leading directions. Against the larger of
the two single-run references (the data's principal subspace, or the layer's own input's), the
seeds agree at most 4.1% more, at any layer and rank from 1 to 512, and by more than 0.4% only
at layer 0, whose input is the data.

**Setup.** As above, same rows and checkpoints; `subspace.py --mode stack` now also takes each
seed's layer input y_ℓ (whitened and centered) and its covariance eigenbasis. At layer 0 that is
the data's, and the two seeds' layer-0 inputs read NPS 1.000 at every rank, as they should.

**NPS(r)** (raw; between seeds | seed 42 against its own input | seed 43 against its own | the two
inputs against each other):

| layer | r = 1 | 2 | 8 | 32 |
|---|---|---|---|---|
| 1 | 0.956 \| 0.958 \| 0.948 \| 0.962 | 0.972 \| 0.977 \| 0.979 \| 0.966 | 0.951 \| 0.947 \| 0.934 \| 0.922 | 0.860 \| 0.874 \| 0.869 \| 0.834 |
| 2 | 0.822 \| 0.898 \| 0.683 \| 0.743 | 0.910 \| 0.928 \| 0.933 \| 0.883 | 0.835 \| 0.873 \| 0.860 \| 0.745 | 0.705 \| 0.699 \| 0.716 \| 0.590 |
| 3 | 0.761 \| 0.799 \| 0.764 \| 0.651 | 0.741 \| 0.771 \| 0.772 \| 0.651 | 0.642 \| 0.581 \| 0.592 \| 0.399 | 0.408 \| 0.428 \| 0.429 \| 0.279 |
| 4 | 0.199 \| 0.164 \| 0.020 \| 0.000 | 0.154 \| 0.131 \| 0.086 \| 0.017 | 0.083 \| 0.109 \| 0.088 \| 0.044 | 0.097 \| 0.157 \| 0.149 \| 0.077 |

- **Layers 1–2 follow their inputs.** The between-seed NPS is within 0.05 of the two seeds' mean
  NPS against their own inputs at every rank, and the data's leading-direction gap (NPS(1) 0.96
  and 0.82 between seeds, 0.45–0.67 with the data) is gone. Seed 43's layer-2 NPS(1) against its
  own input (0.683) is the one low value; at rank 2 it reads 0.933, so its input's first two
  directions are near a tie.
- **The residuals part with depth.** The two seeds' inputs agree at NPS(8) 0.92, 0.75, 0.40 and
  0.04 at layers 1–4, and at layer 4 their leading directions are orthogonal (NPS(1) 0.000).
- **Layer 3** agrees across seeds a little more than with its own input at ranks 4–8, but less than
  with the data, which is the larger reference there from rank 8 on.
- **Layer 4's contributions** lie along neither their own inputs (NPS(8) 0.09–0.11) nor each
  other (0.08), and along the data about twice as closely (0.20–0.21).
- **Ratio to the larger reference** (between-seed NPS over the larger of the data's and own
  input's, both seeds): at rank 8, 1.030, 1.004, 0.956, 0.955 and 0.395 for layers 0–4. Over all
  ranks, layer 0 reads 0.995–1.041 and layers 1–4 at most 1.004.
- **For the writeup.** The findings' results and discussion now say the seeds agree no more than
  each agrees with the data or with its own layer's input, at most 3% more at rank 8. The
  appendix gives the comparison under the name NPS(d′), since the writeup reserves φ, A and r.

Same command as above; the input comparison is in `subspace_stack.json` under `inputs`.

## 2026-11-20: queued, a GPT-2 SAE seed pair

The 11-19 section left open whether the SONAR k32 pair's low agreement is SONAR's or the fit's.
GPT-2's `gpt2_l8/sae/m12160_k32` has one seed, and it is older (2026-09-29) than the current
`gpt2_l8` harvest files (10-01), so a seed-43 replica of it might not share its data.
`gpt2_sae_seeds.sh` instead trains a fresh pair at sae.py's defaults, seeds 42 and 43, into
`data/out/gpt2_l8/sae_seeds/`, as unit `gpt2-sae-seeds`, once `recipe-ladder-nonoise` has
finished (about 20 minutes a seed, from the old run's timestamps). A three-step CPU smoke run
trained. Then `splitting.py` and `subspace.py --mode sae` on it, beside the SONAR pair.

## 2026-11-20: the GPT-2 SAE pair agrees more than SONAR's, and less than the papers'

**Short version.** The same k32 recipe on GPT-2's `gpt2_l8` harvest fits far better (FVU_w 0.188
against SONAR's 0.499) and agrees more across seeds on every lenient statistic, but stays below
the agreement the papers report for language-model SAEs. It trained on far less data than those
SAEs (4M distinct tokens against 300–500M), and their comparisons use other widths and sparsity
rules, so the pair does not apportion the SONAR pair's gap among the substrate, the fit and the
training. The subspace result carries over: the other seed reproduces the unmatched latents'
subspace almost to the optimum, well beyond the data's.

**Where.** Unit `gpt2-sae-seeds` trained `m12160_k32` and `m12160_k32_s43` at sae.py's defaults
(20 epochs, 19,360 steps of 4,096) into `data/out/gpt2_l8/sae_seeds/`: FVU_w 0.1875 and 0.1876 on
sae.py's held-out tail, no dead latents. Seed 42 took 1 h 35 min, against seed 43's 24 min,
because it shared the CPU with scoring jobs. `splitting.py` read the cache's first 524,288 rows,
as for SONAR (12,124 of 12,160 latents fire at least 50 times); `subspace.py` read 32,768 tail
rows in the loss's whitened frame. Outputs in `data/out/gpt2_l8/splitting/m12160_k32__m12160_k32_s43/`
and `data/out/gpt2_l8/seed_subspace/sae_k32/`.

**The field's statistics** (A->B / B->A where they differ):

| statistic | SONAR pair | GPT-2 pair | reported |
|---|---|---|---|
| decoder share, cosine > 0.7 (Leask et al.) | 0.331 / 0.334 | 0.434 / 0.434 | 0.94, two 3,072-latent L1 GPT-2 SAEs |
| PW-MCC (Song et al.) | 0.511 (random 0.117) | 0.627 (random 0.142) | 0.818 Pythia-160M, 0.790 Gemma-2-2B |
| mean MPPC (Wang et al.) | 0.384 / 0.385 | 0.688 / 0.688 | 0.81 on Pythia |

- **The GPT-2 pair sits between SONAR's and the papers' on each.** It closes most of the gap in
  activation correlation, about two fifths in PW-MCC and a sixth in decoder share.
- **Decoder share over thresholds:** 0.929 above cosine 0.4, 0.80 above 0.5, 0.099 above 0.9 (both
  ways). SONAR's: 0.49–0.50, 0.42–0.43, 0.24.
- **MPPC above** 0.4, 0.5, 0.7 and 0.9: 0.867, 0.779, 0.547 and 0.196 (A->B).
- **Containment reverses at strict thresholds.** A matched at τ = 0.3/0.5/0.7/0.9: 0.634 / 0.284 /
  0.078 / 0.005, against SONAR's 0.307 / 0.235 / 0.147 / 0.030. The GPT-2 latents correlate better
  but are contained in each other less tightly. The findings' comparison of the softmax
  Ontologizer (at most 0.1%) with the SONAR pair (3–31%) uses the SONAR numbers.
- **What it does not separate.** On GPT-2 the fit is better too, so the pair cannot tell SONAR's
  lower agreement from its poorer fit. Nor does it show that the recipe falls short, since the
  published SAEs differ in data, training length, width and sparsity:
  - Leask et al.'s 0.94: two 3,072-latent (4×) L1 SAEs on GPT-2 small's layer-8 residual, 300M
    OpenWebText tokens in one pass, 128-token contexts, lr 4e-4.
  - Song et al.'s 0.818 / 0.790: TopK at k = 20 (Pythia-160M, 16,384 latents) and k = 80
    (Gemma-2-2B, 65,536 latents), each the highest PW-MCC of a sweep over k. Pythia's trained on
    500M Pile tokens in 244k steps, and PW-MCC rose over training (BatchTopK's had not saturated
    at 2.5 × 10^5 steps).
  - Wang et al.'s 0.81: L1 SAEs of 24,576 latents (32×) on Pythia-160M; no token count given.
  - This pair: 3.97M distinct tokens (34,846 C4 documents, 128-token contexts), 79M rows seen in
    20 passes, 12,160 latents (16×), TopK 32, lr 1e-4. The SONAR pair: 3.97M distinct rows, about
    590M seen in some 150 passes (`sae_ladder.sh`), 5,120 latents, lr 1e-4 and then 4e-4.
- **The test that would separate data from method** is this pair on a larger harvest. At 300M
  tokens a float32 cache is about 920 GB, so it would stream activations or use a smaller
  harvest. A cheaper first look is whether agreement still rises with steps on the current
  harvest, which needs intermediate checkpoints.

**Subspaces** (`--mode sae`). 43.1% and 43.2% of the live latents have a match at decoder cosine
0.7 (SONAR 33%; Gerasimov et al. 0.746 on GPT-2). Energy share of seed 42's set, mean over 10
size-matched draws; d = 768, so chance at r = 64 is 0.083:

| set | r | own (ceiling) | other seed's | data's top-r | data's bottom-r | chance |
|---|---|---|---|---|---|---|
| unmatched (6,902) | 8 | 0.047 | 0.044 | 0.026 | 0.001 | 0.010 |
| | 64 | 0.244 | 0.230 | 0.150 | 0.027 | 0.083 |
| | 128 | 0.405 | 0.383 | 0.276 | 0.067 | 0.167 |
| | 512 | 0.892 | 0.873 | 0.826 | 0.499 | 0.667 |
| matched (5,238) | 8 | 0.048 | 0.045 | 0.027 | 0.001 | 0.010 |
| | 64 | 0.247 | 0.236 | 0.159 | 0.025 | 0.083 |

- **The other seed captures 94–98% of the optimum** at every rank from 8 to 512, as on SONAR
  (96–97%).
- **The data's top-r captures 56–72% of it for r from 8 to 128, in both sets.** On SONAR the
  unmatched set's was 40–54% and the matched set's 75–84%, so there the matched latents lie more
  along the data's principal directions; on GPT-2 the two sets look alike.

```bash
systemd-run --user --unit gpt2-sae-seeds -p Restart=on-failure -p RestartSec=120 \
    --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/gpt2_sae_seeds.log \
    -p StandardError=append:$PWD/experiments/logs/gpt2_sae_seeds.log \
    /usr/bin/bash experiments/seed-stability/gpt2_sae_seeds.sh
uv run python splitting.py --cache data/activations/gpt2_l8.npy \
    --a data/out/gpt2_l8/sae_seeds/m12160_k32/params.npz \
    --b data/out/gpt2_l8/sae_seeds/m12160_k32_s43/params.npz \
    --out data/out/gpt2_l8/splitting/m12160_k32__m12160_k32_s43
uv run python experiments/seed-stability/subspace.py --mode sae \
    --a data/out/gpt2_l8/sae_seeds/m12160_k32/params.npz \
    --b data/out/gpt2_l8/sae_seeds/m12160_k32_s43/params.npz \
    --cache data/activations/gpt2_l8.npy --mse-weights data/activations/gpt2_l8.mse_weights.npy \
    --out data/out/gpt2_l8/seed_subspace/sae_k32
```

## 2026-11-20: queued, the all-off arm on a Gaussian copy of its data

**Why.** Three results point at the data's second moments:
- the hard stack loses only a sixth of its fit on Gaussian rows with the tail's covariance
  (`ste-arm/notes.md`, the random-vector health check);
- its seeds share the data's and their inputs' principal subspaces and nothing beyond them;
- its layer-0 partitions barely agree though both seeds see the same input there
  (`recipe-ladder/notes.md`, without training noise).

A Gaussian has nothing beyond its covariance to agree on, and where directions share a variance a
quantizer's cells can rotate among them at no cost. So the all-off arm trained on a Gaussian copy
of `mc4_sent16`, at two seeds, is the baseline for every seed statistic at once. If it agrees as
much as the cache pair, the seeds agree on the spectrum and nothing else; whatever the cache pair
agrees on beyond it is structure beyond the second moments.

**The copy** (`gausscopy.py prepare`, 4 min 53 s on CPU). The mean and covariance of the first
12.8M rows of `mc4_sent16`, the rows the ladder's arms train on, are in
`data/out/sonar/gausscopy/moments.npz`. The covariance has one zero eigenvalue (−4e-20 after
rounding; the largest is 1.3e-3), so the copy keeps that null direction. The 65,536 held-out rows
(`eval.npy`, stream 1) miss the mean by at most 1.1e-4 per coordinate and the covariance by 6.6%
in Frobenius norm, about what 65,536 samples in 1,024 dimensions allow. On the last 32,768 rows,
under the `med837` weights, the weighted variance is 0.0072 for both the copy and the cache's tail;
the mean's norm is 0.0535 against 0.0536, a row's 0.217 against 0.213. A CPU smoke run of eight
steps starts at loss 1.03e-5, against the cache arm's 1.04e-5.

**The run.** Unit `gauss-seeds` (`gauss_seeds.sh`), from 21:30, waits for an idle GPU and then
trains the arm (the ladder's `nobcos_nosupp_nol1f_nohm` flags, with its training noise) at seeds
42 and 43, about 1.2 GPU-hours each, before scoring the pair as the cache pair was scored. The
comparison is the noisy cache pair in `recipe-ladder/notes.md`.

**Restarted 2026-11-21 02:24.** The first start, at 01:41, ran at 7 s a step (about 100 hours
for the arm): grain reads a source from several threads, and `GaussianSource`'s single-block cache,
unguarded, had each reader thread regenerate its block with a BLAS call, 16 times per block,
which took about 19 cores. The cache now holds eight blocks under a lock, each generated once
(`tests/test_gausscopy.py`), with the same rows as before. The 200-step partial output was
removed and the unit restarted; it waits for an idle GPU.

**Restarted again 2026-11-21 06:27; the 02:24 diagnosis was wrong.** Training began at 06:20 and
ran at 18 s a step (about 250 hours). The cause was the shuffle, not the threads. `train_ste.py`
takes `sonar.py`'s `shuffle = True`, so each batch of 256 drew rows from about 256 different
blocks, and the lock made them generate one after another. Measured on the loader alone: 28.5 s a
batch shuffled, 45 ms in order. The rows are independent draws, so reading them in order changes
nothing about the data. `train_gauss.py` now passes `shuffle=False` through a new optional
argument of `train_ste.train`, and the rows and `eval.npy` are unchanged. The partial output
(`log.jsonl` only, no checkpoint) was removed and the unit restarted with the command below.

**Done when** the scores in `data/out/sonar/gausscopy/scores/` (written once
`scores/done.log` exists) have been read beside the noisy cache pair, layer by layer, for partition
NMI, contribution cosine, NPS(8) and prefix FVU, and the reading is a dated section here that
says which agreement the cache pair has beyond the copy's. The k-means baseline below gives the
expected direction. *(Done 2026-11-21: "on a Gaussian copy of its data, the stack's seeds agree as
they do on SONAR, except at layer 0", below.)*

```bash
uv run python experiments/seed-stability/gausscopy.py prepare
systemd-run --user --unit gauss-seeds -p Restart=on-failure -p RestartSec=120 \
    --working-directory=$PWD \
    -p StandardOutput=append:$PWD/experiments/logs/gauss_seeds.log \
    -p StandardError=append:$PWD/experiments/logs/gauss_seeds.log \
    /usr/bin/bash experiments/seed-stability/gauss_seeds.sh
```

## 2026-11-20: k-means partitions of SONAR reproduce, beyond its Gaussian copy's, and the heads are not near them

**Short version.** A 32-cell k-means of `mc4_sent16`'s whitened rows reproduces across seeds at
partition NMI 0.61, against 0.40 for its Gaussian copy, and the gap widens at 128 cells (0.74
against 0.39). At 8 cells the two are alike (0.53 and 0.56), so coarse cells follow the spectrum
and finer ones find structure beyond it. So SONAR has a reproducible 32-way partition beyond its
second moments. The stack's heads neither reproduce one (matched layer-0 heads 0.025 between
seeds, `recipe-ladder/notes.md`) nor sit near the k-means cells: a layer-0 head's best NMI with
any seed's k-means partition has median 0.054 (at most 0.144), and layers 1–4 are at 0.004–0.005.
The heads' disagreement across seeds is therefore not what the data force on any 32-cell
partition.

**Setup** (`kmeans_seeds.py`, CPU, 23:55). k-means++ and Lloyd's iterations to a fixed point in
the `med837` whitened frame, five seeds (42–46, so ten pairs), fit on 131,072 rows drawn from
the first 12.8M (the ladder's training rows) and assigned on the last 32,768. The copy's training
rows are its training stream and its tail is `gausscopy/eval.npy`. The heads are the noisy
all-off arm's seed 42 (`ste_h76_init01_med837_nobcos_nosupp_nol1f_nohm`), on the same tail. Output
in `data/out/sonar/kmeans_seeds/summary.json`.

| k | cache: seed NMI [range] | Gaussian copy | null, cache / copy | FVU, cache / copy |
|---|---|---|---|---|
| 8 | 0.533 [0.366, 0.988] | 0.555 [0.470, 0.636] | 0.0002 / 0.0004 | 0.969 / 0.971 |
| 32 | 0.607 [0.495, 0.761] | 0.396 [0.383, 0.416] | 0.0010 / 0.0042 | 0.948 / 0.958 |
| 128 | 0.737 [0.679, 0.756] | 0.391 [0.384, 0.393] | 0.0092 / 0.0528 | 0.922 / 0.948 |

- **The copy's null rises with k** (0.053 at 128 cells), because its cells are nearly equal in
  size and the plug-in NMI's bias grows with the number of cells. Over the null the gap at 128 is
  0.73 against 0.34.
- **SONAR's k-means also fits better than the copy's** at 32 and 128 cells (0.948 and 0.922,
  against 0.958 and 0.948), as cluster structure beyond the covariance would.
- **At 8 cells SONAR's seeds split into near-identical and quite different solutions** (0.37 to
  0.99), the copy's do not (0.47 to 0.64): two or more distinct local optima on the real data.
- **Against the planted circle** (`experiments/planted-circle/notes.md`). Seeds that share a
  factor and differ only in where they cut it still read NMI about 0.7, and spare heads stop
  aligning with factors altogether. A k-means is one partition with no spare heads; a layer is 76
  heads summed. The reproducible structure is in the data, and the multi-head code does not
  settle on it.
- **What it predicts for the Gaussian seed pair** (queued above). If its heads agree across
  seeds as much as the cache pair's do, the heads' agreement does not reflect the structure
  k-means finds; if less, the cache pair's small excess over the null is that structure.

```bash
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py \
    --gauss data/out/sonar/gausscopy \
    --ckpt data/out/sonar/recipe_ladder/ste_h76_init01_med837_nobcos_nosupp_nol1f_nohm
```

## 2026-11-21: summed heads and the cascade each cost agreement, at equal error

**Short version.** The stack differs from one k-means in two ways, and both lower how far a code
of SONAR reproduces across seeds, with k-means alone and no training. Summing h codebooks at one
stage (additive quantization, FAISS's LSQ++) roughly halves the matched head NMI with each
doubling of h: 0.57, 0.30, 0.17 and 0.10 at h = 1, 2, 4 and 8. Every restart reaches nearly the
same error (training FVU within 0.002 at each h), and the lowest-error runs agree no more than
the rest, so these are many equally good solutions, not a search that sometimes fails. A residual
cascade of single k-means stages falls from 0.64 at stage 0 to 0.23, 0.19 and 0.07. Sharing stage
0, so that stage 1 sees identical inputs, lifts stage 1 to 0.44: the cascade costs about half of
stage 1's agreement, and the rest is lost because the residual has less structure to agree on.
SONAR stays above its Gaussian copy at every h. At the additive rate, 76 heads would land near
0.015, the order of the stack's matched layer-0 heads (0.025); that is an extrapolation, since
h = 76 was not run. *(Measured later the same day, in the next section: 0.009 at h = 76, below the
extrapolation and below the stack's layer 0.)*

**Setup** (`kmeans_structure.py`, CPU, 00:4x–02:20). The `med837` whitened frame, k = 32, the last
32,768 rows of `mc4_sent16` as the tail, and the Gaussian copy's held-out rows for `gauss`.
- **Additive:** LSQ++ (`faiss.LocalSearchQuantizer`, 5 bits per codebook), eight restarts per h
  (seeds 0–7, so 28 pairs), each fit on 32,768 training rows. Heads are matched between two runs
  by NMI with a Hungarian assignment, as `partition.py` matches the stack's heads.
- **Residual:** k-means per stage (at most 100 Lloyd iterations), cascade seeds 42–45, fit on
  65,536 training rows. `independent`: every seed runs every stage (6 pairs). `shared`: stage 0 is
  seed 42's, and the other three seeds run stages 1–3 on its residual (3 pairs, against seed 42).
- Outputs in `data/out/sonar/kmeans_structure/` (`aq.json`, `rvq.json`, `run.log`).

**Additive, matched head NMI between runs:**

| h | SONAR, all pairs [range] | lowest-error 3 | Gaussian copy | training FVU, SONAR (8 runs) |
|---|---|---|---|---|
| 1 | 0.574 [0.543, 0.593] | 0.562 | 0.352 | 0.932–0.934 |
| 2 | 0.295 [0.269, 0.332] | 0.303 | 0.129 | 0.897–0.899 |
| 4 | 0.167 [0.156, 0.193] | 0.169 | 0.058 | 0.850–0.851 |
| 8 | 0.097 [0.087, 0.102] | 0.099 | 0.029 | 0.787–0.788 |

The rows-shuffled null is 0.004–0.005 throughout. The rank correlation between a pair's NMI and
its worse training FVU is −0.52 to +0.25 on SONAR, with no consistent sign. The ratio between
successive h runs 0.51, 0.57 and 0.58 on SONAR and 0.37, 0.45 and 0.49 on the copy.

- **Why this is not a weak solver.** On two planted additive factors (d = 8, two codebooks of 4)
  LSQ recovers the factors in some restarts and mixes them in others, and the mixed ones have far
  higher error (0.67–2.34 against 0.010–0.013, ten restarts), so there best-of-N recovers them.
  On SONAR the restarts do not differ in error, so no selection by error makes them agree. The
  simple block descent first written for this mixed even the planted factors in most runs and was
  replaced (the planted runs were a scratch check, not kept).
- **The data's structure persists in the additive code** (SONAR above its copy at every h, by
  more than three times at h = 8) but is spread across heads, each carrying less of it.

**Residual, NMI between two seeds' same stage:**

| stage | independent | stage 0 shared | prefix FVU (independent) |
|---|---|---|---|
| 0 | 0.638 | 1 | 0.947 |
| 1 | 0.226 | 0.441 | 0.929 |
| 2 | 0.192 | 0.267 | 0.913 |
| 3 | 0.069 | 0.103 | 0.902 |

Stage 0 here differs from `kmeans_seeds.py`'s 0.607 by its training rows (65,536 against
131,072) and the 100-iteration cap.

- **Against the stack.** Its layer 0 is 76 heads summed on identical inputs, which the additive
  trend puts near 0.015, and its layers 1–4 add the cascade's loss on top; their matched NMI of
  0.005–0.007 against a null of 0.0048 is consistent with both. Neither training noise
  (`recipe-ladder/notes.md`) nor the absence of structure (the k-means baseline above) is needed
  to explain it.
- **Open.** h = 16 and 32 under LSQ, to test the trend before relying on the extrapolation; and
  whether a stack trained with fewer heads per layer at equal bits reproduces more, which is the
  design question this raises.

```bash
JAX_PLATFORMS=cpu uv run --with faiss-cpu python experiments/seed-stability/kmeans_structure.py \
    --mode aq --train-rows 32768 --gauss data/out/sonar/gausscopy
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_structure.py --mode rvq
```

## 2026-11-21: additive codes at the stack's width reproduce less than the stack's layer 0

**Short version.** Run out to the stack's 76 heads, LSQ's matched head NMI keeps falling: 0.057 at
16 codebooks, 0.020 at 32 and 0.0089 at 76, against a null of 0.0047. The stack's matched layer-0
heads reach 0.025 to 0.027 against 0.0048 (`recipe-ladder/notes.md`), about five times LSQ's
excess over the null. So the stack's layer-0 agreement is not low for an additive code of its
width; it is higher than that of a search-based one. The extrapolation in the last section put 76
codebooks near 0.015, too high.

**Setup.** As the last section's additive arm, SONAR only, with three restarts per h (three pairs),
32,768 training rows. Output in `data/out/sonar/kmeans_structure/aq_wide/` (`aq.json`,
`run.log`).

| h | matched head NMI [range] | held-out FVU (training) |
|---|---|---|
| 16 | 0.0574 [0.0562, 0.0587] | 0.738–0.739 (0.706–0.707) |
| 32 | 0.0196 [0.0193, 0.0199] | 0.682–0.683 (0.626–0.627) |
| 76 | 0.0089 [0.0089, 0.0089] | 0.531 (0.456) |

- **The fall is not slowing.** Successive ratios are 0.59 (8 to 16), 0.34 (16 to 32) and 0.45 over
  the step from 32 to 76.
- **These codes overfit.** At h = 76, the 2.5M codebook numbers are fit on 32,768 rows: training FVU
  0.456, held out 0.531. Still, held out, LSQ's 76 codebooks fit better than the stack's first
  layer (0.6045 at about 380 bits, `ratecurve.py`, under `no837` weights where these use `med837`).
  So the comparison is a better fit at the same width that agrees less.
- **Why the stack's layer 0 might agree more** is not tested here. It is one layer of five, trained
  jointly, so it need not carry everything a lone 76-codebook code must, and it encodes by a
  classifier rather than by search. Either could make its heads coarser and more repeatable.
- Three restarts per h make the rank correlation with error (±1.00 in the log) meaningless here;
  the spread of error across restarts is again small (at most 0.0015 held out).

```bash
JAX_PLATFORMS=cpu uv run --with faiss-cpu python experiments/seed-stability/kmeans_structure.py \
    --mode aq --hs 16 32 76 --restarts 3 --best 2 --train-rows 32768 \
    --out data/out/sonar/kmeans_structure/aq_wide
```

## 2026-11-21: on a Gaussian copy of its data, the stack's seeds agree as they do on SONAR, except at layer 0

**Short version.** Trained at two seeds on rows from a Gaussian with `mc4_sent16`'s mean and
covariance, the all-off arm's upper layers agree across seeds about as much as on SONAR. In
layers 1–4 the copy's pair gives 67–86% of the cache pair's small excess over the null in
partition NMI and 83–94% in contribution cosine, so most of what those layers share is the
covariance's. Layer 0 is the exception. Its excess is 3.6 times the copy's in NMI (0.0223 against
0.0062) and 3.0 times in cosine, so two thirds to three quarters of layer 0's agreement on SONAR
comes from structure beyond the second moments. The k-means baseline points the same way (0.61 against 0.40 at 32
cells), though the heads are far below it. The subspaces go the other way: the copy's seeds agree
more at layers 1–3, and so do their layer inputs. The subspace agreement the cache pair shows is
therefore no more than the covariance gives. On SONAR the two seeds' residuals drift further apart
than on a Gaussian.

**Where.** Unit `gauss-seeds`, restarted 06:27 (the section above). It waited for another
session's GPU job, trained seed 42 from about 10:38 to 11:43 (50,000 steps in 65 min, 12.7 steps a
second) and seed 43 until about 12:49, then scored the pair by 13:04 (`scores/done.log`). Outputs
are in `data/out/sonar/gausscopy/`: `alloff`, `alloff_s43`, and `scores/` with one log per script.
The scorers read the copy's held-out rows (`eval.npy`, stream 1, 65,536 rows) with the cache
pair's row counts and weights: `med837` for contributions and subspaces, `ratecurve.py`'s default
`no837` for the prefix FVU. The cache pair is the noisy all-off arm and its `_s43`
(`recipe-ladder/notes.md`, 2026-11-20, "without training noise"), scored on `mc4_sent16`'s tail.

**Seed agreement by layer**, cache pair / Gaussian copy's pair:

| layer | partition NMI | excess over null | contribution cosine | excess over null | NPS(8), contributions | NPS(8), layer inputs |
|---|---|---|---|---|---|---|
| 0 | 0.0271 / 0.0110 | 0.0223 / 0.0062 | 0.0346 / 0.0124 | 0.0333 / 0.0111 | 126.6 / 126.8 | 128.0 / 128.0 |
| 1 | 0.0067 / 0.0063 | 0.0019 / 0.0015 | 0.0048 / 0.0046 | 0.0035 / 0.0033 | 101.8 / 120.8 | 97.6 / 119.9 |
| 2 | 0.0059 / 0.0057 | 0.0011 / 0.0009 | 0.0037 / 0.0033 | 0.0024 / 0.0020 | 76.7 / 109.8 | 60.6 / 107.7 |
| 3 | 0.0055 / 0.0054 | 0.0007 / 0.0006 | 0.0031 / 0.0029 | 0.0018 / 0.0016 | 28.7 / 64.4 | 19.5 / 57.1 |
| 4 | 0.0054 / 0.0052 | 0.0006 / 0.0004 | 0.0027 / 0.0025 | 0.0014 / 0.0012 | 17.4 / 22.8 | 18.9 / 17.4 |
| null (rows shuffled) | 0.0048 / 0.0048 | | 0.0013 / 0.0013 | | | |

NPS(8) is over chance r/d, with a ceiling of 128.

- **Matched heads land in their own layer as often:** 76, 58, 28, 31 and 41 of 76 on the copy,
  against 76, 58, 30, 24 and 35.
- **Head sizes are alike.** Layer 4's median centered contribution is 0.066% of the target
  variance, against 0.063% and 0.067% on the cache. The layer-0 lift in cosine at equal atom spread
  is 13.0×, against 16.7× and 14.2×, so the copy keeps most of layer 0's link between head size and
  agreement.
- **Each seed's contribution still follows its own input more than the other seed's,** as on the
  cache (layer 3: 74–76 against 59, on the cache 29–31 against 21–23).

**Prefix FVU over the free coordinates, gains as computed** (seed 42 / seed 43):

| data | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| SONAR (`mc4_sent16` tail) | 0.6045 / 0.6034 | 0.4278 / 0.4270 | 0.3106 / 0.3099 | 0.2278 / 0.2274 | 0.1681 / 0.1679 |
| Gaussian copy (`eval.npy`) | 0.6382 / 0.6379 | 0.4479 / 0.4475 | 0.3240 / 0.3237 | 0.2369 / 0.2368 | 0.1744 / 0.1743 |

The copy is harder to code at every prefix, by 0.034 at one layer and 0.006 at five, as a Gaussian
is the hardest source for its covariance under squared error. The copy's seeds agree in fit to
0.0004 at every prefix, the cache's to 0.0011. Pinning the gains changes the copy's readings by at
most 0.0003.

**Reading.**
- The stack's disagreement past layer 0 does not need SONAR. A Gaussian with no structure beyond
  its covariance gives the same near-null agreement, so those layers' disagreement is a property of
  the architecture and its training on that spectrum.
- Layer 0 picks up real structure: the share of its agreement that exceeds the copy's is the part
  of the code that SONAR's non-Gaussian structure pins down. It is still small in absolute terms
  (NMI 0.027 against k-means' 0.61 at the same k).
- The subspace measure cannot see this difference. The copy reproduces subspaces better, so a
  subspace match on SONAR is a covariance statement.

**Not tested.** One pair on each data. The copy is matched in its first two moments only, so it
tells "beyond the covariance" from "within it", not which higher-order structure layer 0 uses.

**Memory.** The unit's cgroup peak was 97.6 GB over training and scoring. That count includes page
cache, and the ladder's training unit read 52.6 GB. Which step drove it was not traced. Run heavy
CPU jobs beside such a unit with care.

## 2026-11-21: SONAR's reproducible k-means cells are not language

**Short version.** k-means partitions of SONAR reproduce across seeds beyond its Gaussian copy's
(above, 2026-11-20). Language is the obvious candidate for the structure, so two copies test it.
- A mixture that keeps each language's mean and covariance and nothing else reproduces at 32
  cells as SONAR does (0.596 against 0.607).
- But SONAR's own rows less their language's mean reproduce just as well (0.673), and their cells
  barely follow language (NMI 0.038).

So the mixture's match was a match of strength, not of structure. SONAR's reproducible cells are
mostly something other than language. In a one-seed check they are not segment kind, length or
position in the document either, which leaves content (topic or genre) as the likely candidate,
unchecked. Language means add agreement only at 128 cells (0.737 against 0.659).

**Setup** (`kmeans_seeds.py`, CPU). As the 2026-11-20 section: the `med837` whitened frame, seeds
42–46 (ten pairs), fit on 131,072 rows drawn from the first 12.8M, assigned on the last 32,768.
- **The mixture copy** (`gausscopy.py mixture`, `mixture.npz`) holds each of the 86 languages'
  mean and covariance over the training prefix. The rarest language has 38,033 rows there, 37
  times the dimension, so no shrinkage was needed. Each training and tail row is drawn under the
  real row's language, so the language composition is the cache's. Its pooled covariance is the
  cache's (relative error 1e-15, `mixture.meta.json`). The language means carry 4.6% of the
  variance, in the raw frame.
- **Within** is the cache's own rows less their language's mean (`--within-language`).
- **The cells' NMI with language** is each seed's tail partition against the rows' languages (86
  values), averaged over the five seeds.

| data | seed NMI, k = 8 | 32 | 128 | cells' NMI with language, k = 8 | 32 | 128 |
|---|---|---|---|---|---|---|
| SONAR | 0.533 | 0.607 | 0.737 | 0.106 | 0.179 | 0.248 |
| SONAR less language means | 0.666 | 0.673 | 0.659 | 0.018 | 0.038 | 0.089 |
| language mixture | 0.690 | 0.596 | 0.581 | 0.188 | 0.290 | 0.355 |
| Gaussian copy (2026-11-20) | 0.555 | 0.396 | 0.391 | | | |

The shuffled nulls are at most 0.0104. The FVUs at 32 cells are 0.948 (SONAR), 0.960 (within)
and 0.951 (mixture).

- **Removing the language means raises agreement at 8 and 32 cells.** One reading, untested: with
  the means in, some cells follow language, and those cells reproduce less well than the rest.
- **The mixture's cells follow language more than SONAR's** (0.29 against 0.18 at 32). They are
  language cells that happen to reproduce about as well.
- **What the within-language cells are, exploratory** (one seed, 42, a scratch script not in the
  repository; NMI with each label on the tail):

| data, k | segment kind (3) | norm decile (10) | position in document (16) | language (86) |
|---|---|---|---|---|
| SONAR, 8 | 0.088 | 0.034 | 0.007 | 0.057 |
| SONAR, 32 | 0.067 | 0.122 | 0.007 | 0.185 |
| within, 8 | 0.096 | 0.039 | 0.009 | 0.014 |
| within, 32 | 0.090 | 0.080 | 0.009 | 0.029 |

  None comes close to explaining a partition that reproduces at 0.67. `partition.py`'s NMI
  divides by the mean of the two entropies, so against kind (0.96 nats on the tail) a 32-cell
  partition (about 3.47) reads at most about 0.43; kind's 0.09 is about a fifth of that, not
  nothing.

**For the stack.** Layer 0's agreement beyond the Gaussian copy (the last section) need not be
language either. The language mixture is a cheaper next copy for a stack pair than SONAR less
its means, but this result says it would test the wrong structure. A pair on the within-language
rows would test whether the heads' small agreement survives losing the language means.

Output: `data/out/sonar/kmeans_seeds_language/summary.json` and `run.log`, and
`data/out/sonar/gausscopy/mixture.npz` and `mixture.meta.json`.

```bash
uv run python experiments/seed-stability/gausscopy.py mixture
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py \
    --mixture data/out/sonar/gausscopy/mixture.npz \
    --within-language data/out/sonar/gausscopy/mixture.npz \
    --out data/out/sonar/kmeans_seeds_language
```

## 2026-11-21: queued, seed agreement against width in the trained stack

Unit `width-seeds` (`width_seeds.sh`, the README's command), queued 23:11 behind another session's
`lcm-readout-x-rot`, which held the GPU. It trains one-layer stacks of 1, 4, 16 and 76 heads of 32
at seeds 42 and 43 (the ladder's all-off recipe, 50,000 steps each) and scores each pair with
`partition.py` and `ratecurve.py` into `data/out/sonar/width_seeds/scores/`.

**Done when** `scores/done.log` exists and the matched head NMI at each width has been read beside
LSQ's (0.57, 0.17 and 0.057 at h = 1, 4 and 16; 0.0089 at 76; "summed heads and the cascade" and
"additive codes at the stack's width" above) and beside k-means' 0.61 at k = 32. The h = 76 pair
should also be read against the five-layer stack's layer 0 (0.025–0.027), and the section should
say whether the trained curve follows LSQ's halving.

## 2026-11-22: queued, the k32 SAE seed pair on a Gaussian copy of `mc4_4M`

Unit `sae-gauss-seeds` (`sae_gauss_seeds.sh`, the README's command), queued 00:35 to run after
`width-seeds`. It trains `sae_conv/m5120_k32`'s settings (150 epochs, 144,300 steps of 4,096) at
seeds 42 and 43 on 3,941,632 rows of a Gaussian with the moments of every row the real pair
trained on (`data/out/sonar/gausscopy_4M`), then scores the pair with `subspace.py --mode sae` on
the copy's held-out tail and `splitting.py` on the first 524,288 rows of its training stream, as
the real pair was scored. About 2.6 GPU-hours a run, from the block-sparse SAEs' 15 steps a second
at the same size.
- The copy's held-out covariance is within 1.4% (relative Frobenius) of the training rows' over
  557,056 rows (`eval.meta.json`). Its smallest eigenvalue is 9e-9, against 0.053 for the
  largest: `mc4_4M`'s rows are unit vectors, so one direction (the norm's) has almost no
  variance. The copy keeps the covariance, not the unit norm.
- A CPU smoke run at full size generated the rows in under 3 minutes (2 steps, FVU_w 0.839), and
  both scorers read its output.

**Done when** `data/out/sonar/sae_gauss/scores/done.log` exists and the Gaussian pair's table
(unmatched and matched sets at r = 8, 64, 128 and 512: own, other seed's, data's top-r) sits
beside the real pair's (2026-11-20, "Subspaces"), with its matched share at decoder cosine ≥ 0.7
beside the real pair's 33%, and `splitting.py`'s PW-MCC beside the real pair's 0.51. The section
should say whether the real pair's transfer of 96–97% of the optimum, about twice the data's
top-r, survives the copy.

## 2026-11-22: SONAR's coordinates shuffled apart keep half its k-means agreement at 32 cells, and little at 128

**Short version.** Two copies of `mc4_sent16`'s own rows keep the distribution of every
coordinate and nothing that ties coordinates together. One copy uses the principal frame, the
other FastICA's.
- At 32 cells both copies reproduce across seeds about halfway between the Gaussian copy and
  SONAR: 0.507 and 0.492, against 0.396 and 0.607.
- At 128 cells they keep little of SONAR's lead: 0.425 and 0.492, against 0.391 and 0.737. That
  is a tenth and three tenths of the excess.

So at 32 cells about half of what makes SONAR's cells reproduce lies in how single directions
are distributed. At 128 cells most of it lies in how directions combine. That joint structure is
not language either ("not language", above): SONAR's rows less their language means still
reproduce at 0.659 at 128 cells.

**Setup** (`kmeans_seeds.py --marginals pca ica`, CPU, 00:33 to 03:55). As in the 2026-11-20
section: the `med837` whitened frame, seeds 42–46 (ten pairs), fit on 131,072 rows drawn from
the first 12.8M, assigned on the last 32,768. Each copy permutes every coordinate's column on its
own, separately in the training rows and the tail (`marginal_copy`). It then maps the rows back
to the whitened frame, so k-means reads it as it reads the cache. The cache rows give the
2026-11-20 numbers again.
- **Principal frame.** The training rows' 1,023 principal directions with variance. The copy
  keeps the covariance.
- **ICA frame.** Symmetric FastICA (log-cosh) in the sphered principal frame, fit on the training
  rows. It stopped at the 200-iteration cap, short of its tolerance of 1e-4: in the last step the
  largest change over the 1,023 rows, 1 − |w_new · w|, was 0.19. The frame is therefore not
  FastICA's fixed point. The copy is still exactly an independent-coordinates model with these
  sources' marginals, and a converged frame might find sources further from Gaussian. Loading the
  rows and fitting both frames took about 17 minutes, so the README's few seconds an iteration
  holds.

| data | seed NMI, k = 8 | 32 | 128 | FVU, k = 8 | 32 | 128 |
|---|---|---|---|---|---|---|
| SONAR | 0.533 | 0.607 | 0.737 | 0.969 | 0.948 | 0.922 |
| ICA-frame copy | 0.570 [0.480, 0.637] | 0.492 [0.460, 0.519] | 0.492 [0.471, 0.505] | 0.970 | 0.955 | 0.940 |
| principal-frame copy | 0.788 [0.662, 0.986] | 0.507 [0.488, 0.538] | 0.425 [0.418, 0.434] | 0.968 | 0.955 | 0.945 |
| Gaussian copy (2026-11-20) | 0.555 | 0.396 | 0.391 | 0.971 | 0.958 | 0.948 |

The copies' shuffled nulls at 128 cells are 0.039 (ICA) and 0.053 (principal), like the Gaussian
copy's 0.053, because their cells are nearly equal in size. SONAR's is 0.009. Read over the
nulls, the shares below hardly move.

- **Each copy's share of SONAR's excess over the Gaussian copy**, (copy − Gaussian) / (SONAR −
  Gaussian), is 53% (principal) and 45% (ICA) at 32 cells, and 10% and 29% at 128.
- **The fit gives the same shares.** At 128 cells SONAR's FVU is 0.026 below the Gaussian copy's,
  and the copies recover 0.003 (principal) and 0.008 (ICA) of it. At 32 cells SONAR is 0.010
  below, and each copy recovers about a third of that.
- **The principal directions are nearly Gaussian; FastICA's sources are not.**
  - In the principal frame the median excess kurtosis is 0.89 (largest 1.6) and the median
    |skew| 0.01. The 20 largest directions hold 14% of the variance and have median kurtosis
    0.37.
  - Every FastICA source has excess kurtosis above 1 (median 4.6, largest 215), and the median
    |skew| is 0.81. The 184 sources with kurtosis above 10 hold 25% of the variance.
  - So the sources are sparse: a peak near zero and a long tail to one side. One reading, not
    tested: rows far out along one source form rays, which cells at 128 can follow from seed to
    seed. The principal copy, nearly Gaussian in each coordinate, has none. Why that copy gains so
    much at 8 and 32 cells is not clear.
- **Sarle's coefficient finds no clear second mode.**
  - It passes 5/9 for 24 ICA sources. Each of them has |skew| 2.2–3.7 and kurtosis 7–23, so
    skew alone takes the coefficient there: a one-mode exponential reads exactly 5/9.
  - In the principal frame only the first direction reaches it: 0.557, with skew 0.81 and
    kurtosis −0.03, holding 2.1% of the variance. It is the one candidate for two modes, not
    checked.
- **At 8 cells the principal copy reproduces more than SONAR does** (0.79). Like SONAR's, its
  pairs split into near-identical and quite different solutions (0.66–0.99). Possibly the first
  direction's skew fixes one coarse cut; not checked. The ICA copy at 8 cells is at the Gaussian
  copy's level.

**For the stack.** A stack pair on the ICA-frame copy would say whether layer 0's excess over the
Gaussian copy (2026-11-21) comes from the sources' sparsity or from how they combine. The copy
here is only 131,072 training rows, so that pair needs a streamed copy: the frame fitted once,
with each block's columns permuted within the block.

Output: `data/out/sonar/kmeans_seeds_marginals/` (`summary.json`, `run.log`,
`coordinates_pca.csv`, `coordinates_ica.csv`).

```bash
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py \
    --marginals pca ica --out data/out/sonar/kmeans_seeds_marginals
```
