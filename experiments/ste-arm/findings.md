# ste-arm: findings

The straight-through (hard-code) Ontologizer's results, condensed: each
finding's headline numbers and conclusion. The measurements, eliminations,
superseded readings and commands behind them are in [`notes.md`](notes.md),
under the section named in each heading here. Code, layout and usage are
in [`README.md`](README.md).

All SONAR numbers are whitened FVU (FVU_w) on the held-out cache tail.
Unless stated otherwise the SONAR arm is 5 x 76 heads at k = 32 (1,900
bits) at the fixed initialization (`ste_h76_init01`), and its flat
counterpart is 1 x 380 heads at the same bits and parameter count.

## The discrete code is the model

*notes: Why; Result*

The softmax Ontologizer's argmax is not its code: on `sweep_softmax_shm` the
hard row is FVU_w 8.7e6 against 0.0007 for the soft forward. Under
`select="ste"` the forward is the argmax, and the hard and soft rows agree
to four decimals at every checkpoint scored. A hard code transmits exactly
`l * h * log2(k)` bits. Reverse water-filling puts the best 800-bit code at
FVU_w 0.226 *if the embeddings were Gaussian* with their covariance; real
data is no harder to compress, so that is a reference rather than a floor
(a real 800-bit code could do better), and it informed raising the head
count to 380 rather than requiring it.

## Reconstruction against SAEs and a single-layer variant

*notes: Initialization; Result*

| code | coeffs | index bits | FVU_w |
|---|---|---|---|
| **straight-through, 5 x 76** | **0** | **1900** | **0.154** |
| SAE `m11264_k160` | 160 | 2154 | 0.258 |
| SAE ReLU+L1 3e-5 | 519 | 6390 | 0.220 |
| SAE ReLU+L1 1e-5 | 1438 | 17723 | 0.048 |
| 1-layer hard variant `g160top1` | 160 | 1972 | 0.274 |

**A purely discrete code beats sparse linear ones at matched bits**: 40%
below the top-k SAE that also transmits 160 coefficients, and 30% below an
L1 SAE spending 3.4x the bits. `g160top1` and `g160softmax` are not SAE
baselines: they are `sae.py` with per-head competition, i.e. one-layer
Ontologizers with linear classifiers (hard and softmax selection), and are
reported as simplified variants. The stack is 44% below the hard one. That needs the initialization fix: the `abs()`'d
dictionary's rows sum coherently, the shipped arm starts at a residual of
1.84e8, and `--dict-init-scale 0.1` takes a third off the error (0.2293 to
0.1535) and brings every head to near-uniform use of its entries (summed
head entropy 90% to 99.7% of nominal; an upper bound on the information
the code carries, equal to it only if heads are independent). The arm sits
3.1x above the Gaussian reference for its rate (0.0499 at 1,900 bits), so at
least that far from the best 1,900-bit code. At identical bits, 380 binary heads
reach 0.1844 on 2.9x fewer parameters.

## Depth at fixed capacity

*notes: Depth at fixed capacity; What the cascade contributes*

| arm | summed head entropy | FVU_w |
|---|---|---|
| stack, 5 x 76 | 99.7% | **0.1535** (both seeds) |
| flat, 1 x 380 | 99.9% | 0.2750 |

**Depth is worth 1.79x the error at matched bits, parameters and head
usage.** Summed head entropy cannot say whether the two codes carry
different information jointly; that is not measured. The stack leads at every checkpoint (1.59x at 10k, peaking
at 1.83x near 200k); neither arm has converged, and the flat arm improves
faster over the last quarter (2.75% against 0.86%), so the gap is slowly
narrowing. It is not usage (a ZCA-whitened flat arm uses every entry
uniformly and is still 1.7x worse), not input conditioning, and not the per-layer gain
channel (pinning it costs 2.1%). Blending each layer's subtracted
reconstruction toward another row's costs 35% of FVU at a quarter blend:
the cascade's value is the pairing of each stage's input with the sample's
own earlier error. A flat layer is a product quantizer; the stack is a
residual quantizer.

## Steering

*notes: Steering, scored in embedding space; Steering at the model's own
magnitude; Why collateral is direction-blind*

Steering is scored in embedding space: step the input, re-run the model,
and record whether the target entry is now selected (realized) and the
share of other heads that changed (collateral). The text round trip is too
lossy to resolve an intervention (12% of head choices survive an unsteered
cycle).

| model | realized @0.25 | collateral @0.25 | random collateral | native realized | native collateral |
|---|---|---|---|---|---|
| hard-code stack | 0.365 | 0.657 | 0.655 | 0.077 | 0.536 |
| hard-code flat | 0.996 | 0.328 | 0.368 | 0.169 | 0.043 |
| softmax stack (`sweep_softmax_shm`) | 0.769 | 0.566 | 0.559 | 0.189 | 0.250 |
| 1-layer hard variant `g160top1` | 0.582 | **0.026** | 0.232 | **0.410** | **0.014** |
| 1-layer soft variant `g160softmax` | 1.000 | 0.301 | 0.336 | 0.536 | 0.067 |

Native is the step the model itself would apply: the forced entry's output
change, or an SAE latent's decoder row at its mean activation.

- **Collateral is set by step size, not direction**, in every Ontologizer:
  a decode step and a random step of the same length disturb the same share
  of other heads, at fixed and at native magnitude. Linearizing with the
  cascade held out gives the same: direct-path flips equal random's in
  every arm.
- **The hard-code stack barely steers at its own scale**: 7.7% realized,
  against 1.6% for random, while moving 54% of other heads.
- **The cascade doubles collateral but is not all of it.** The flat arm
  still moves a third of the other heads; its earlier 9% was an artifact of
  heads collapsed onto few entries at the shipped initialization.
- **The hard-code heads sit on their boundaries**: median winner-runner-up
  margin 1.5e-4 at layer 0, against 2.0e-2 for binary heads, which flip
  least of any Ontologizer.
- **The one clean steerer is the simplest hard-code Ontologizer.**
  `g160top1` -- one layer, linear classifiers, hard selection -- is the
  only model whose decode direction avoids other heads' margins.
  `g160softmax`, which differs only in the selection rule, has no such
  alignment, and neither does the five-layer bilinear stack.
- The softmax stack is the more steerable Ontologizer (0.77-0.82 realized
  at 0.25 across two samples, against 0.36), and `BilinearBlock.rev`'s
  eigendirection had a sign bug; fixed, it roughly doubles eigenvector
  steering on the soft model and stays dead on the hard one.

## Seed reproducibility

*notes: Seed reproducibility; The same picture on GPT-2; Is layer 0's
reproducibility a size effect?; The router on GPT-2*

Two seeds reach identical error, 0.1535 and 0.1535, while nothing that
identifies an individual head agrees:

| measure, SONAR stack | seeds | null |
|---|---|---|
| head partitions (mean matched NMI) | 0.0121 | 0.0048 |
| head contributions (centered cosine) | 0.0074 | 0.0015 |
| per-layer subspaces | equal to each model's agreement with the data | |
| per-head atom Grams | 1.05-1.11x chance | |

**What reproduces is aggregate or layer-level.** Layer 0 is the exception,
on SONAR and on GPT-2 (5 x 20 heads, k = 256, concatenated):

| GPT-2 layer | partition NMI (null 0.179) | contribution cosine (null 0.0017) |
|---|---|---|
| 0 | 0.359 | 0.171 |
| 1-4 | 0.186-0.188 | 0.004-0.017 |

- **Layer 0's larger contributions are all gain**: it receives the raw input
  (GPT-2 norm ~122 against residuals of ~60). At equal atom spread it still
  reproduces 14x better on GPT-2 and 23x on SONAR.
- **The stack costs reproducibility at matched size**: against flat-arm
  heads of the same size, which see the same input with nothing downstream,
  the stack's layer 0 reproduces ~8x worse and its deeper layers 19-49x
  worse.
- **The router changes this on GPT-2.** With the per-head router on, the
  last two layers switch off (zero contribution, both seeds), FVU improves
  2.4% (0.4108 against 0.4208), and contribution agreement doubles, rising
  7-12x at layers 1-2. Partition agreement does not follow, so what
  reproduces is what heads write, not how they carve.

## What heads encode

*notes: Heads against a known factor; The topic head recurs across seeds*

**The strongest head encodes topic and genre, and it is the one head that
recurs across seeds.**

| run | top layer-0 head | eta^2 | next head | match across runs |
|---|---|---|---|---|
| `ste_h76_init01` (seed 42) | h54 | 0.117 | 0.062 | |
| `ste_h76_i01_s43` (seed 43) | h14 | 0.122 | 0.068 | NMI 0.413 (other heads 0.010) |
| `ste_h76` (shipped init, seed 42) | h54 | 0.111 | 0.080 | NMI 0.355 (other heads 0.024) |

Its entries decode to crime news, geopolitics, encyclopedic place
description, manufacturing, marketing, legislative text and entertainment;
contrastive auto-interp scores genre and topic descriptions at F1 0.52
against a 0.27 null. It also reproduces on contributions (0.844 against
0.249 for the next head).

Language, the one factor with labels, is carried as neighbourhoods across
heads, never by one head: best-head NMI 0.18 of an attainable 0.875, as a
language-agnostic embedding should give. Partitions still carry it well
above chance, and the hard code carries four times what the softmax
reference does (32-head probe 0.42 against 0.28).

## Dictionary geometry

*notes: The decoder's null space; The mechanism: the orthant forces a
collinear dictionary; Does a softer code raise the dictionary's rank?*

| GPT-2 layer 8 | row cosine | eff. rank | FVU_w (held out) |
|---|---|---|---|
| signed, e = 1536 | +0.0009 | 27.37 | 0.1540 |
| signed, e = 768 | +0.0019 | 26.95 | 0.1543 |
| abs, e = 1536 | +0.3038 | 9.15 | 0.1934 |
| abs, e = 768 | +0.4543 | 5.41 | 0.2927 |
| concat, d_head = 32 | +0.5353 | 3.30 | 0.4155 |

- **Non-negativity forces a collinear dictionary and costs reconstruction**:
  a third of the signed effective rank and 26% more held-out error at
  e = 1536, 90% at a square decoder. Row cosine orders all five arms by
  error at convergence. Width matters only under `abs` (narrowing costs
  +0.2% signed, +51% abs). The gap at e = 1536 peaked at 50% near step
  200k and is narrowing, but has not closed in a full run.
- **Row collinearity measured in the dictionary space is mostly gauge**:
  74-78% of each atom's energy is in the decoder's null space. A head's
  decoded atoms are orthogonal (0.023 at layer 0, slightly negative
  below). Dictionary statistics belong after decoding.
- **Training duration shapes the dictionary 3.8x more than the selection
  rule**, and the last layer becomes inert once the error before it falls
  below FVU 0.005.

## Head independence was an estimator artifact

*notes: Head-independence pressure*

| layer | 0 | 1-4 | mean |
|---|---|---|---|
| unbiased CKA | 0.0041 | 0.0002-0.0005 | **0.0011** |
| biased CKA | 0.0315 | 0.096-0.106 | 0.0867 |

The biased estimator reads chance co-occurrence as dependence, with a
floor that grows as the batch shrinks. **This architecture produces
near-independent heads with no pressure to do so**, and the penalty
descended only the bias. Before trusting a statistic as a training signal,
measure it on a null it should score zero on.

## In progress: k = 128, and atoms without a decoder

*notes: k = 128, and atoms without a decoder*

`--direct` (ported from `headline`) puts signed atoms in the output space
with no decoder, so every atom is an embedding-space direction. At
5 x 54 heads, k = 128 (1,890 bits), 3,000-step pilots train cleanly; the
direct variant leads at matched step (MSE 2.37e-4 against 2.81e-4 decoded
and 2.75e-4 for the k = 32 stack), which ranks nothing yet. Full 24-epoch
runs of both, at seeds 42 and 43, are training.
