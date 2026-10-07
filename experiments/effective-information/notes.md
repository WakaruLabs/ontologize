# effective-information: notes

Settings for every run below: `effinfo.py` defaults -- 4096 contexts from
the 32768-row held-out tail of `mc4_4M.npy` (two halves of 2048 for
split-half reliability), observational statistics over the whole tail,
argmax effect variable, temperature from the run's `log.jsonl`. EI is
pairwise, source head -> one downstream head, in bits (ceiling log2 32 =
5). One checkpoint per arm, one seed.

## 2026-10-06: softmax arm (`bl_ctl_full`, step 369500)

5 layers x 32 heads x 32 entries, T = 0.03 (annealed from 1.0 over the
first 50k steps).

| src -> eff | mean EI | max EI | det | deg | live EI | obs. MI |
|---|---|---|---|---|---|---|
| 0 -> 1 | 2.736 | 3.239 | 3.290 | 0.554 | 2.664 | 0.0017 |
| 0 -> 2 | 2.669 | 3.346 | 3.498 | 0.829 | 2.605 | 0.0016 |
| 0 -> 3 | 2.438 | 3.278 | 3.376 | 0.937 | 2.396 | 0.0013 |
| 0 -> 4 | 2.044 | 2.919 | 3.068 | 1.024 | 2.022 | 0.0015 |
| 1 -> 2 | 2.132 | 2.711 | 2.860 | 0.728 | 2.132 | 0.0045 |
| 1 -> 3 | 2.174 | 3.127 | 3.049 | 0.875 | 2.174 | 0.0034 |
| 1 -> 4 | 1.868 | 2.845 | 2.853 | 0.985 | 1.870 | 0.0029 |
| 2 -> 3 | 1.236 | 2.271 | 2.052 | 0.816 | 1.138 | 0.0172 |
| 2 -> 4 | 1.076 | 2.058 | 2.033 | 0.957 | 1.007 | 0.0200 |
| 3 -> 4 | 0.619 | 1.417 | 1.648 | 1.029 | 0.495 | 0.1231 |

Every one of the 10240 descendant pairs has EI > 0.1 bits; split-half
r = 1.000. Natural sharpness (mean max assignment) by layer: 0.106,
0.130, 0.256, 0.420, 0.553 (uniform = 0.031). Live entries per head
(argmax usage >= 1e-3): 27.1, 31.0, 26.8, 24.6, 23.6, so live-entry EI
is close to full EI here: the soft arm's problem is one-hot versus soft
states, not dead entries.

Reading. Forcing one head's entry moves every downstream head's choice
by 0.6-2.7 bits on average, while the same pairs' observational MI on
the natural forward pass is 0.001-0.12 bits. That is not evidence that
heads are strong causal variables: EI falls with depth while sharpness
rises, and across heads sharpness predicts EI inversely (Spearman
rho(sharpness, max downstream EI) = -0.83 over the 128 source heads;
-0.55, -0.65, -0.80, -0.93 within layers 0-3). The flatter a head's
natural assignments, the further a one-hot entry is from anything it
does naturally, and the larger the downstream response -- the
intervention distribution, not the head, sets the number. Realized bits
also track EI within layers (rho 0.43, 0.49, 0.87, 0.91), so frozen or
rarely used heads with degenerate unused atoms are a second, compatible
contribution. On a softmax arm, uniform-entry EI is a measure of
sensitivity to off-distribution interventions; the hard-code arm, whose
natural states are the one-hot entries, is the on-distribution test.

## 2026-10-06: hard-code arm (`ste_h76`, step 369500)

The shipped straight-through arm: 5 layers x 76 heads x 32 entries, T =
1.5e-4 constant (irrelevant to the argmax forward). Natural sharpness is
1.000 in every layer, so one-hot entries are its natural states. Live
entries per head (argmax usage >= 1e-3): 10.6 in layer 0 (median 6),
32.0 in layers 1-4.

| src -> eff | EI, all k | max | det | deg | **live EI** | obs. MI |
|---|---|---|---|---|---|---|
| 0 -> 1 | 0.2232 | 0.465 | 0.315 | 0.092 | **0.0126** | 0.0019 |
| 0 -> 2 | 0.1833 | 0.390 | 0.301 | 0.117 | **0.0095** | 0.0006 |
| 0 -> 3 | 0.1404 | 0.316 | 0.286 | 0.145 | **0.0075** | 0.0004 |
| 0 -> 4 | 0.0742 | 0.176 | 0.183 | 0.109 | **0.0051** | 0.0002 |
| 1 -> 2 | 0.0150 | 0.020 | 0.083 | 0.068 | **0.0150** | 0.0010 |
| 1 -> 3 | 0.0110 | 0.013 | 0.122 | 0.111 | **0.0110** | 0.0007 |
| 1 -> 4 | 0.0065 | 0.008 | 0.101 | 0.094 | **0.0065** | 0.0004 |
| 2 -> 3 | 0.0138 | 0.018 | 0.125 | 0.112 | **0.0138** | 0.0009 |
| 2 -> 4 | 0.0066 | 0.009 | 0.102 | 0.096 | **0.0066** | 0.0005 |
| 3 -> 4 | 0.0070 | 0.011 | 0.104 | 0.097 | **0.0070** | 0.0007 |

Split-half r = 0.999 over the 57760 pairs (0.85-0.94 within source
layers 1-3, whose range is narrow; part of that can be bias shared by
both halves).

Uniform-over-all-k EI is again an off-distribution number, for a
different reason: the layer-0 heads with the largest all-k EI (0.41-0.47
bits) naturally use 3 or 5 entries (realized bits 1.58 = log2 3, 2.32 =
log2 5), so 27-29 of their 32 interventions force entries they never
take; heads using all 32 entries evenly sit at ~0.03. Restricted to live
entries, layer 0 drops to the level of every other layer.

On-distribution (live EI), per pair: median 0.0062, 0.0110, 0.0099,
0.0069 bits for source layers 0-3; 99th percentile <= 0.040; the largest
edge anywhere is 0.066 bits (layer 0 head 54, 30 live entries). Summed
over every downstream head -- a summary that counts shared influence
repeatedly, not an information quantity, since a source's joint EI is at
most log2 of its live entries -- the median source head reaches 1.98,
2.47, 1.55, 0.53 bits for layers 0-3 (layer-0 max 10.45, the large-head
outliers).

## Reading

1. **Uniform-entry EI, the textbook definition, measures off-distribution
   interventions in both arms.** In the softmax arm a one-hot entry is far
   from any natural (near-flat) assignment, and EI tracks how flat the
   source head is (rho -0.83). In the hard arm, heads that use few entries
   are forced onto dead ones. Live-entry EI on a hard code is the number
   that answers the formal reading's question.
2. **On-distribution, no single edge of the causal graph is strong.** In
   `ste_h76` a head's value changes any one downstream head's choice by
   ~0.01 bits of a possible 5, at every depth. The formal reading's graph
   (every earlier head a parent of every later one) holds structurally,
   but the influence is spread thinly over many downstream heads, or is
   joint across sources, which pairwise EI cannot see.
3. **Interventional dependence exceeds observational dependence 11-16x**
   per source layer (7-26x per layer pair; live EI 0.005-0.015 vs
   observational MI 0.0002-0.002 bits). Under the
   data distribution, pairs of hard-code heads co-vary almost not at all;
   forcing one still moves the other slightly. Shared input is not what
   links them.

One checkpoint per arm, one seed, pairwise EI only, argmax effect variable.
