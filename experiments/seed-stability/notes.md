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
