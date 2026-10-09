# The direct + gain-shape + STE Ontologizer (design note, 2026-09-29)

Status: one-epoch preliminary result on GPT-2 small, layer 8. Worth a careful follow-up.

## The design

A residual stack of `l` layers; each layer has `h` heads; each head is a `k`-way classifier
whose output is one **dictionary entry living directly in activation space**.

- **Direct entries** (`direct=True, signed_dict=True`): entries `g_hk ∈ R^d`, signed, with
  no latent space and no decoder. Every class is literally a direction in activation space,
  i.e. a steering vector. (The original design put non-negative entries in a shared
  latent space read out by one linear decoder.)
- **Gain-shape residual coding** (`resid_first, resid_norm, resid_const, resid_gain, gain_clip`):
  every layer, including layer 0, classifies the *unit direction* of the residual it is
  correcting, `[n̂; 1]` with `n = X − Σ_{j<i} c_j`, and its output is scaled by the observed
  residual norm `‖n‖` (clipped at `‖X‖`). The model is positively homogeneous:
  `f(cX) = c·f(X)`. Shape (which entry) and gain (how much) are separated, and the constant
  coordinate breaks the bilinear classifier's sign-blindness.
- **Hard selection with a straight-through estimator** (`select="ste"`): the forward pass uses
  the argmax entry, the backward pass the softmax gradient, with the temperature annealed
  1 → 0.03 over 30k steps. The code is discrete by construction: `l·h·log2(k)` bits per
  token (5 × 32 × 5 = 800 here), plus the five layers' gains, its only continuous numbers.
- Bilinear classifiers (`n=2`), joint deep supervision, winner dropout 0 → 0.1, logit noise
  0.02 → 0.006, featvar output noise 0.1, the small regularizers of `sonar.py`.

Command:

```
uv run python experiments/gpt2/train_onto.py --name V_direct_gain_ste --epochs 1 --lr 2e-4 \
    --anneal-steps 30000 --resid-first --direct --signed-dict --select ste --resid-gain --gain-clip
```

## Result (held-out FineWeb tokens; 1 epoch = 10M tokens, 6.2 min)

| model | code per token | FVU | GPT-2 loss recovered | ΔCE (nats) |
|---|---|---|---|---|
| **direct + gain + STE** | 800 bits + 5 gains | **0.217** | **96.95%** | 0.136 |
| STE + gain (latent + decoder) | 800 bits + 5 gains | 0.290 | 94.3% | 0.257 |
| STE alone | 800 bits | 0.327 | 93.3% | 0.299 |
| STE + gain, private heads | 800 bits + 5 gains | 0.288 | 94.2% | 0.261 |
| top-k SAE, k=32, m=5120 (10 epochs) | 32 coefficients + ~276 index bits | 0.189 | 96.2% | 0.171 |
| top-k SAE, k=128, m=5120 (10 epochs) | 128 coefficients + ~859 index bits | 0.116 | 98.6% | 0.063 |
| soft Ontologizer (reference) | 4960 continuous mixture weights | 0.016 | 99.85% | 0.007 |

So an 800-bit code recovers more of GPT-2's loss than a k=32 SAE that was
trained on 10× the data (about 5× the compute, since an Ontologizer step is ~2 SAE steps),
while transmitting only its five per-layer gains as continuous numbers. The SAE still has
the lower FVU.

Per-layer FVU: 0.48 → 0.37 → 0.30 → 0.26 → 0.22. Every layer contributes. Label usage is
close to uniform in every layer (imbalance 0.24, 0.04, 0.01, 0.01, 0.02 bits of 5); the
other STE runs froze layer 4 at 4.8–5.0 bits (every head constant).

## Why it seems to work (hypotheses, not yet tested)

1. **Prototypes want to live in activation space.** With non-negative entries in a latent
   space and a shared linear decoder, a single selected entry is a point in a cone that the
   decoder then maps; under soft mixing this costs nothing (the soft code got *better*
   without the constraint too), but under hard selection the reachable set of prototype
   positions is restricted. Signed entries in `R^d` can be placed anywhere.
2. **Gain-shape supplies the one continuous degree of freedom a hard code most needs.**
   Under `resid_norm` alone a hard layer adds a fixed-size vector whatever the residual's
   size; that made hard upper layers *harmful* (they raised the FVU with depth at high LR).
   With the observed gain, each layer's step scales with what is left to explain.
3. **No usage collapse.** Without the latent bottleneck, losing entries keep receiving useful
   gradient and the deepest layer never freezes. The exact reason deserves a look: it may be
   the decoder's shared rows that coupled entries' gradients before.

## Open questions and next steps

- **Fibers.** Add a small local basis per entry (`fiber_rank r`): base point + tangent
  coordinates read from the residual. This is the tangent-bundle picture; it should recover
  part of the SAE's FVU advantage while keeping the code discrete apart from `r` numbers.
  Runs `V_fiber4_ste`, `V_fiber8_ste` (private-block version) are queued; a direct-space
  fiber variant should follow.
- **More labels per head** (`--k 64/128`): 6–7 bits per head. Queued under STE + gain.
- **Longer training.** All Ontologizer runs are 1 epoch vs. the SAEs' 10; the hard code
  should improve further (3-epoch runs queued).
- **Seed stability of the head partitions** (`seedstab.py`): are the heads' partitions of
  tokens reproducible across seeds up to head permutation? Preliminary evidence from two
  *different* configs sharing an init: NMI ≈ 0.15 in layer 0 (null 0.04), so far from stable.
  Same-config seed replicas are queued.
- **Selection schedules.** Anneal-T STE (this run) beat STE at constant T=1 (0.50), the
  soft→hard mix at T=1 (0.39) and normalized top-m is queued. The backward temperature has
  to track the forward: too high and the straight-through gradient is noise; too low and
  losers stop learning.
- **Interventions/steering.** Entries are directions in activation space, so a set-intervention
  is literally adding `‖n‖·(g_new − g_old)` at that layer. Untested here.
- **Other models/layers, and SONAR.** Only GPT-2 small layer 8 so far.

## Caveats

One epoch, one seed, one model and layer. FVU and loss recovered on 65k / 65k held-out
tokens (documents disjoint from training). The SAE baselines were trained for 10 epochs
with `sae.py` (unwhitened MSE) and evaluated with the same splice code.

## Follow-ups (2026-09-30; 1 epoch each, held-out FineWeb)

| variant | code per token | FVU | loss recovered |
|---|---|---|---|
| base (above) | 800 bits | 0.217 | 96.95% |
| base, seed 43 | 800 bits | 0.219 | 96.85% |
| k = 64 labels per head | 960 bits | 0.186 | 97.6% |
| k = 128 labels per head | 1120 bits | 0.165 | 97.9% |
| + rank-4 fiber, dropout 0.5 | 800 bits + 640 coords | 0.059 (base only 0.274) | 99.4% |
| + rank-8 fiber, dropout 0.5 | 800 bits + 1280 coords | 0.014 (base only 0.304) | 99.9% |
| + rank-4 fiber, joint cap 0.2 | 800 bits + 640 coords | 0.180 (base only 0.227) | 97.6% |
| + rank-1 radial fiber (gain per entry) | 800 bits + 160 coords | 0.148 (base only 0.669) | 98.5% |
| frozen base, rank-4 fibers trained after | 800 bits + 640 coords | 0.016 (base only 0.215 = unchanged) | 99.9% |
| 640-wide top-k (k_z=16) sparse encoder per head before the classifier | 800 bits | 0.286 | 94.8% |
| 640-wide JumpReLU (~74 active) before the classifier | 800 bits | 0.282 | 95.2% |
| 4096-wide top-k (k_z=64) per head, 40M tokens, resampling | 800 bits | 0.225 | 96.65% |
| dense front-end, 40M tokens (equal-data reference) | 800 bits | 0.195 | 97.6% |
| private heads (latent block-diagonal dictionary, shared decoder), 40M tokens | 800 bits | 0.186 | 97.5% |
| private heads, k = 64, 40M tokens | 960 bits | **0.164** | **97.9%** |

- The result is reproducible across seeds (0.002 FVU). More labels help monotonically with diminishing returns; without
  direct entries + gain, the k=64/128 hard codes collapse (FVU 0.75/0.80).
- Fibers (`fiber_rank r`: a rank-r local basis per entry, coordinates read linearly from the unit residual) recover the
  SAE's FVU advantage and more, but only with an "honesty" mechanism that keeps the discrete base meaningful: fiber
  dropout keeps the base-only FVU at 0.23–0.30 whatever the rank; a joint group-lasso cap on the fiber norm gives a hard
  guarantee at the cost of most of the fiber's value; unconstrained fibers take over completely (base-only FVU 10).
- Seed stability (`seedstab.py`, Hungarian matching of heads by partition NMI): head partitions of the two seeds agree
  only in layer 0 (NMI 0.09 vs 0.05 for unrelated heads within one run) and not at all in layers 1–4. Which labels share a
  head is seed-dependent; whatever is reproducible is at most the prototype set, to be tested by matching entries.
- Sparse encoders in front of the classifier never beat the dense classifier on the unit residual: 32-wide hurt (0.265
  top-k, 0.246 JumpReLU), 640-wide plateau at ~0.28 whatever the sparsity (k_z 16 … ~129 active), and a 4096×64 per-head run
  on the full 40M tokens with dead-latent resampling reaches 0.225 vs 0.195 for the dense front-end on the same tokens (it matches
  layer 0 and loses ground in every later layer), while most of its width goes unused (layer 1 ended with 88% dead latents at
  no cost to its prefix FVU). Any value they have is interpretive
  (each label decision as a function of a few named features), not reconstructive.
- Fibers trained on a *frozen* base (the base's argmax code and entries untouched) reach the reconstruction of unconstrained
  joint fibers with the base exactly as honest as before — the honesty mechanisms in joint training were suppressing the
  fibers rather than protecting the base.
