# ste-arm

A straight-through (hard-code) Ontologizer arm, and the calibration it
needed. `train_ste.py` imports `sonar.py` and reads every constant from
it, so the arm cannot drift from the live config; only the flags below
are overridden.

## Why

Two measurements motivate it, both from the repo's own artifacts.

**The live softmax code is not discrete.** `pareto.py`'s end-to-end
argmax row for `sweep_softmax_shm` is whitened FVU 8,690,470, against
0.0007 for the same model's soft forward. So the entry assignment that
`decode.py`'s interventions and `decode_tags.py` read is not what
carries the embedding. Under `select="ste"` the forward *is* the
argmax, so no such gap can open, and the first checkpoint confirmed it:
the `hard (argmax)` and full-soft rows agree to four decimals.

**800 bits cannot reconstruct SONAR.** A hard code transmits exactly
`l*h*log2(k)` bits. Reverse water-filling on the whitened eval tail's
covariance (participation ratio 268, 667 dimensions for 90% of
variance) puts the best distortion *any* 800-bit code can reach at
FVU_w 0.226, and the live architecture spends exactly 800. So the head
count has to rise for a discrete ontology to be possible at all. Heads
are the cheap axis: bits and parameters are both linear in `h`, and
heads run in parallel where layers are sequential.

| whitened FVU target | bits | heads at k=32 |
|---|---|---|
| 0.226 | 800 | 160, the live architecture |
| 0.05 | 1,898 | 380, this arm's default `--h 76` |
| 0.0023, the live soft model | 4,167 | 833 |

## The calibration, and why the live schedule does not transport

A naive port (live config, `select="ste"`) trains but learns nothing in
the classifier. Three settings have to change, each for a measured
reason. The root cause is one property: **the argmax forward is
scale-invariant, so nothing ever pushes the classifier's logits up.**
They sit at their initialization scale, measured here at a standard
deviation of 0.0015 within a head, for the whole run.

- **Logit noise must be relative, not absolute.** `sd_K=0.02` is an
  absolute perturbation in logit units. Against a spread of 0.0015 it
  is thirteen times the signal, so the argmax is chosen by noise. But
  setting it to zero removes the only exploration and the codebook
  collapses. `--noise-k batchnorm` scales the noise by the logit RMS,
  which is the form an argmax forward needs.
- **The surrogate temperature must track the logit scale.** `ste`
  back-propagates `softmax(K/T)`, so T alone decides whether the
  gradient is informative. The live anneal floor of 0.03 leaves
  `spread/T` at 0.05 and the surrogate flat; even T=0.01 gives max p
  0.043 against a uniform 0.031. The classifier then receives
  essentially no signal, and what improves is the dictionary adapting
  to a fixed random partition.
- **`s_Hm` must be rescaled.** Its 1e-6 is calibrated against a
  whitened MSE of ~3e-6. A hard code trains at an MSE orders of
  magnitude larger, where the term is inert and heads collapse.

Measured at 4000 steps, `lr=1e-5`, `--noise-k batchnorm --sd-k 0.3`:

| T | spread/T | max p | effective entries/head | raw FVU |
|---|---|---|---|---|
| 0.01 | 0.15 | 0.043 | 23.4 | 178 |
| 0.0015 | 0.78 | 0.135 | 26.4 | 193 |
| 0.0005 | 3.38 | 0.681 | 14.0 | 59.2 |
| 0.00015 | 16.84 | 0.974 | 3.8 | 7.3 |

Sharpening the surrogate is what makes the classifier train at all
(its logit spread finally grows, 0.0015 to 0.0025) and improves
reconstruction 25-fold, but it trades directly against head usage.
That trade is what `--s-hm` exists to hold.

**Learning rate.** At sonar.py's 5e-5 every sharp-surrogate arm was
stable for ~1700 steps and then collapsed abruptly, MSE rising three
orders of magnitude while KL_m jumped from 3 to 13 bits. 1e-5 is
stable; 3e-6 is too slow to be worth it.

Note the learning-rate confound when reading any of these tables: arms
at different `lr` are not comparable step-for-step.

## Result

`pareto.py` on the held-out eval tail, whitened FVU. The arm is
`--temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3
--s-hm 1e-4`, h=76.

Final, after the full 24 epochs (369,500 steps).

| point | index bits | continuous coeffs | FVU_w |
|---|---|---|---|
| origin (input-independent) | 0 | 0 | 1.029 |
| **ste h76, hard argmax** | **1900** | **0** | **0.2293** |
| ste h76, soft forward | 1900 | 0 | 0.2293 |
| live softmax h32, hard argmax | 800 | 0 | 8,690,470 |
| top2 h32, hard argmax | 800 | 0 | 0.471 |
| SAE m11264_k160 | 2154 | 160 | 0.258 |
| SAE ReLU+L1, L0 = 519 | 6390 | 519 | 0.220 |
| 1-layer hard variant g160top1 | 1972 | 160 | 0.274 |
| rate floor at 1900 bits | 1900 | 0 | 0.05 |

(Rows from the converged frontier in `data/out/sonar/pareto_unified`: the
structural ladder plus an L1 sweep, all at the ~144k-step schedule.
`g160top1` is not an SAE baseline: `sae.py --groups` with hard per-head
competition is a one-layer Ontologizer with linear classifiers, so it is
listed as a simplified variant. The same holds for `g160softmax` with
softmax selection.)

Three things to read off it.

**The soft and hard rows are identical**, to four decimals, at every
checkpoint scored. That is the point of the arm: the discrete code is
the model rather than a lossy reading of it, so the entry assignment the
interventions and `decode_tags.py` operate on is the thing that
reconstructs. The live softmax model's two rows differ by nine orders
of magnitude.

**At matched bits, a purely discrete code beats a sparse linear one.**
At 1900 bits and no continuous coefficients the arm scores 0.2293,
against 0.258-0.274 for SAEs that spend about the same index bits plus
160 continuous coefficients. That is the comparison raising the head
count was for: the earlier 800-bit hard codes could not get near it. An
L1 SAE spending 3.4x the bits plus 519 coefficients edges past it
(0.220); the fixed initialization below takes the arm to 0.1535, under
every one of these.

Note the `dev m` truncation rows are *worse* than the hard row here
(0.343 at m=1, 0.321 at m=4), the reverse of their behaviour on a
softmax model. They are not a capacity curve for this arm: `cluster`
returns a one-hot under `ste`, so pinning entries toward the corpus-mean
origin only adds mass the model was never trained to decode. The hard
row is the whole code, and the honest capacity axis for a hard model is
`l*h*log2(k)` itself.

**It is within five times its own rate floor**, where the live model's
discrete reading was seven orders of magnitude away. The remaining gap
is what a product-residual VQ with non-negative atoms gives up against
an optimal quantizer.

Training curve: reconstruction converges early and the dictionary
keeps improving for the rest of the run, which is the argument for
training it out rather than stopping at the plateau.

| step | MSE | KL_m | row collinearity / layer |
|---|---|---|---|
| 44,500 | 3.50e-4 | 3.19 | 0.812 |
| 137,500 | 3.37e-4 | 2.78 | 0.779 |
| 237,500 | 3.23e-4 | 2.73 | 0.738 |
| 369,500 | 3.31e-4 | 2.71 | 0.686 |

Head usage imbalance peaks at 14.8 bits around step 5k and falls to
2.71; within-head row collinearity falls the whole way to 0.686,
against the ~0.64 floor random non-negative rows give -- so by the end
the dictionary rows are close to as spread as non-negativity permits.
The `s_Hm` rescale is what turns the usage peak around; at the live
1e-6 the collapse stands. The last stretch buys no reconstruction (MSE
is flat from 237k) but takes collinearity 0.738 to 0.686.

Caveat: `pareto.py`'s disclosed train/eval asymmetry now runs the other
way. This arm holds the eval tail out, so its score is out-of-sample,
while the older Ontologizer rows it sits beside are not.

## Downstream: textfid and steerfid

Remeasured 2026-10-07 with the eval-mode encoder and SONAR_NORM (see "The
SONAR encoder ran with dropout" below); the first runs' values are in
git history.

**Text fidelity adds nothing beyond FVU.** Over the eleven Ontologizer
runs with both numbers, "loss recovered" is a monotone function of
soft-forward FVU_w (rank correlation -1). The arm sits exactly where its
reconstruction predicts, between `top2_shm` and `top2`.

| run | FVU_w | chrF | loss recovered |
|---|---|---|---|
| sweep_softmax_shm | 0.0007 | 0.706 | 0.997 |
| sweep_top8_shm | 0.0287 | 0.437 | 0.954 |
| sweep_top4_shm | 0.0761 | 0.375 | 0.892 |
| sweep_top2_shm | 0.1911 | 0.301 | 0.742 |
| **ste_h76** | **0.2293** | **0.295** | **0.695** |
| sweep_top2 | 0.3516 | 0.246 | 0.521 |
| sweep_top1 | 0.7190 | 0.170 | 0.126 |

So textfid is a check that the whitened objective transports to text,
not an independent axis. It does.

**Steerability does not track FVU, and this arm is weak on it.** Net
effect is feature effect minus the random-direction control, at
`--n-features 64`.

| run | FVU_w | net @0.25 | @0.5 | @1.0 | hit @1.0 | random hit |
|---|---|---|---|---|---|---|
| sweep_top8 | 0.0370 | 0.203 | 0.404 | 0.652 | 0.734 | 0.359 |
| topk4_rev | 0.164 | 0.135 | 0.350 | 0.636 | 0.705 | 0.312 |
| sweep_top2 | 0.3516 | 0.072 | 0.277 | 0.509 | 0.578 | 0.109 |
| sweep_top16 | 0.0052 | 0.027 | 0.110 | 0.363 | 0.381 | 0.156 |
| sweep_top4_shm | 0.0761 | 0.112 | 0.140 | 0.248 | 0.242 | 0.062 |
| sweep_softmax_shm | 0.0007 | 0.058 | 0.128 | 0.242 | 0.211 | 0.094 |
| sweep_top2_shm | 0.1911 | 0.018 | 0.071 | 0.196 | 0.150 | 0.016 |
| sweep_top8_shm | 0.0287 | -0.015 | 0.074 | 0.176 | 0.330 | 0.094 |
| sweep_top1 | 0.7190 | 0.047 | 0.104 | 0.064 | 0.102 | 0.109 |
| **ste_h76** | **0.2293** | **0.033** | **0.045** | **-0.020** | **0.055** | **0.125** |
| SAE m5120_k32 | 0.4994 | 0.048 | 0.187 | 0.402 | 0.260 | 0.016 |

The arm's raw effect is flat in magnitude (0.018, 0.014, 0.027) and no
larger than a random direction's, and its hit rate (0.055) does not
clear its own random control (0.125): interventions change the text as
much as a random direction of the same size without landing on the
intended feature. It is now the weakest arm, below even `top1`, whose
classifier never trains. The top-k SAE (`sae_conv/m5120_k32`, same 64
features and protocol) steers in text: net 0.402 at 1.0, hit 0.260
against 0.016 for random.

**The `s_Hm` confound, tested and rejected.** Every `_shm` variant
steers about three times worse than its twin (`top2` 0.509 vs
`top2_shm` 0.196 at `s_Hm` 1e-6 vs 3e-5; `top8` 0.652 vs 0.176), and
this arm carries `s_Hm` at 1e-4, so the weak steering might have been
the mean-entropy bonus rather than the argmax. It is not. A second full
arm trained identically at `--s-hm 1e-6` steers no better:

| arm | s_Hm | FVU_w | raw effect @1.0 | net @1.0 | loss recovered |
|---|---|---|---|---|---|
| ste_h76 | 1e-4 | 0.2293 | 0.027 | -0.020 | 0.695 |
| ste_h76_hm1e6 | 1e-6 | 0.2095 | 0.043 | 0.059 | 0.731 |
| sweep_top2 | 1e-6 | 0.3516 | 0.515 | 0.509 | 0.521 |
| sweep_top8 | 1e-6 | 0.0370 | 0.623 | 0.652 | 0.941 |

Read the raw effect, not the net: the two arms' random controls differ
(+0.047 vs -0.016) on only eight directions, and that noise drives most
of the net gap. On raw effect the two are indistinguishable, 0.027 and
0.043, and both sit an order of magnitude under `top2` and `top8` at
*the same* `s_Hm` of 1e-6. So the entropy bonus is not what costs this
arm its steerability; hard argmax selection is.

Removing the bonus does cost what it was added for. Head usage
imbalance ends at 5.00 bits against 2.71, about 16 effective entries
per head against 22, and row collinearity rises to 0.727 from 0.686. It
buys a little reconstruction back (FVU_w 0.2095, loss recovered 0.731),
which is again exactly the monotone FVU relationship above.

What remains untested is whether the two effects cancel: the 1e-6 arm
has more head collapse, and if collapse independently hurts steering it
could be masking a gain from dropping the bonus. Separating that needs
an arm that suppresses collapse by some route other than `s_Hm`.

Two measurement caveats. `--n-features 16` is too noisy to quote, and
even at 64 this arm's net effect changes sign with its random control:
negative at every magnitude at 16 features, positive throughout at 64
in the first run, and +0.033, +0.045, -0.020 in the rerun. Its text-
cycle effect is within the instrument's noise. And the hit rate is not comparable across hard and soft codes: under
`ste` an activation is exactly 0 or 1, so q90 and q99 are both 1 and a
"hit" demands the argmax land exactly on the target entry, where a soft
code need only cross a fractional threshold.

## Forwarding the code alongside the residual

`forward="resid_labels"` hands each upper layer `[residual | const |
code]`, keeping the residual direction and adding the previous layer's
classification as symbolic context. The pilot, otherwise identical to
the arm above and trained three epochs, is a clear negative:

| arm | steps | training MSE | held-out FVU_w (hard) |
|---|---|---|---|
| resid | 44,700 | 3.55e-4 | 0.249 |
| resid_labels | 41,700 | 5.97e-4 | 0.826 |

It led early, by a factor of fifty at step 2,500, and was overtaken by
step 8,000: the signature of a parameter advantage (its upper
classifiers are 3.4 times wider) without an information advantage,
which is what the code-predictability measurement in the section above
implies, since the residual is by construction what the code failed to
explain. The held-out number is worse than its training loss predicts
by a further factor of two, so the extra width is also being spent on
something that does not transfer; a one-hot block is a discrete key a
bilinear classifier can memorize cell by cell. That is a hypothesis
from one pilot, not a measurement. Not worth a full run.

## Steering, scored in embedding space

`steerfid.py` reads an intervention's effect off a re-encoding of the
generated text, and that instrument is too weak here to support any
conclusion. Measured on an UNSTEERED round trip, the last 512 cache rows
(`roundtrip.py`, remeasured 2026-10-07 with the eval-mode encoder and
SONAR_NORM; see the encoder section below):

| | ste_h76 | softmax (`sweep_softmax_shm`) | reference |
|---|---|---|---|
| cos(x, cycle(x)) | 0.382 (the cycle does not involve the model) | | 0.318 between random pairs |
| head argmax survives | 12.9% | 10.8% | 3.1% chance |

The cycled embedding retains barely more than the shared corpus
direction (rows sit at 0.561 to the corpus mean, and 0.561^2 is the
random-pair cosine), so a perfect intervention could be observed at most
~13% of the time -- the same order as the hit rates that metric reports.
For the hard code it is worse with depth: survival is 36.2% at layer 0
and 4.8% by layer 4, against 3.1% chance. The softmax model's runs
8.5--14.5% across layers.
Reconstruction into embedding space is fine; generation is where it
goes.

`steerembed.py` drops the round trip and asks the causal question
directly: move the input, re-run the model, is the intended entry now
selected (realization) and how many OTHER heads moved (collateral).
64 entries, 32 held-out rows each, steered only where the entry was not
already selected.

| direction | ste_h76 @0.25 / 1.0 | softmax @0.25 / 1.0 |
|---|---|---|
| decode | 0.289 / 0.591 | 0.824 / 0.569 |
| grad | 0.236 / 0.113 | 0.453 / 0.477 |
| adjoint, before the sign fix | 0.007 / 0.016 | 0.224 / 0.329 |
| adjoint, with `orient` | 0.008 / 0.021 | 0.427 / 0.501 |
| adjoint, oriented per sample | 0.026 / 0.026 | 0.446 / 0.512 |
| random | 0.023 / 0.030 | 0.020 / 0.037 |

Collateral is within 0.01 across directions at a given strength, so
these compare at equal budget.

**Steering works, and the text cycle was hiding it.** The decode
direction reaches 0.60 and 0.82 against a 0.031 chance rate, where the
cycle put every model within noise of random.

**The hard code is still the less steerable of the two, so the earlier
conclusion survives a better instrument.** At matched collateral
(~0.58) softmax realizes 0.82 against the hard code's 0.28, and it does
so at the lowest strength while the hard code needs the largest. A
hard argmax has margins a small nudge cannot cross.

**Where an entry decodes to beats the classifier's own gradient.** That is
not obvious and is worth a second look: the first-order direction is
exactly right only for infinitesimal steps, and on the hard code it
decays with strength (0.220 to 0.090) while the decode direction grows.

**`BilinearBlock.rev` has a sign bug, and fixing it does not rescue
steering.** `eigh` fixes eigenvectors only up to sign, so `rev` was
returning a direction whose sign came from LAPACK rather than from the
layer. `nlinear.orient` now settles it by the eigenvalue's sign with a
leading-coordinate tie-break, which makes `rev` well-defined; a test
pins that planting `e` and `-e`, which give the same bilinear form,
now return the same direction.

**The fix roughly doubles eigenvector steering on the soft model**,
0.224 to 0.427 at strength 0.25, which lands within noise of orienting
per sample (0.446). So the convention captures nearly all of the
available benefit.

That is more than the theory predicts, and the gap is instructive.
Moving from `x` along `v` changes `x'Bx` by `2e*lam*(x.v)`, so the
ascent direction depends on `x.v`, a property of the SAMPLE, and a
weight-only convention should not be able to supply it. It can here
because SONAR embeddings are strongly anisotropic: they sit at cosine
0.561 to the corpus mean, so `x.v` is dominated by `mean.v` and has a
consistent sign across samples. On isotropic inputs the argument holds
and the fix would buy nothing, which is what the unit test measures --
it uses Gaussian samples and finds a weight-only direction raising the
logit on about half of them. Both statements are true of their own
input distribution.

**The hard code is unmoved either way**, 0.008 to 0.021 against
random's 0.023 to 0.030, and per-sample orientation does not rescue it
either. The eigendirection carries no steering signal there, so the
decode direction remains the one to use.

### Where the decode direction gets the hard code to

Sweeping strength on `ste_h76`, all six directions at matched
collateral (which is set by the step size, not the direction -- all
six agree to 0.005 at each strength):

| strength | decode | margin | grad | adjoint | random | collateral |
|---|---|---|---|---|---|---|
| 0.25 | 0.277 | 0.118 | 0.220 | 0.008 | 0.022 | 0.58 |
| 1.00 | 0.597 | 0.201 | 0.104 | 0.021 | 0.035 | 0.81 |
| 2.00 | 0.663 | 0.223 | 0.090 | 0.021 | 0.029 | 0.87 |
| 8.00 | 0.657 | 0.226 | 0.064 | 0.024 | 0.039 | 0.92 |

**Decode is far the best and saturates at ~0.66.** A third of
interventions are unrealizable at any strength: the target entry never
wins its head no matter how hard the input is pushed.

**Against the soft model it wins only past collateral 0.85.** The two
curves cross there -- below it softmax leads (0.824 against 0.277 at
collateral 0.57), above it the hard code does (0.663 against 0.520 at
0.87). But 0.87 collateral means most other heads moved too, which is
not a regime any steering claim wants. In the clean regime the soft
model is several times better.

### On the fixed initialization

The tables above are `ste_h76`, the shipped initialization. Rerun on
`ste_h76_init01` and its seed-43 twin (60 entries, 32 rows each):

| direction | shipped @0.25 / 1.0 | init01 @0.25 / 1.0 | i01_s43 @0.25 / 1.0 |
|---|---|---|---|
| decode | 0.289 / 0.591 | 0.364 / 0.607 | 0.372 / 0.598 |
| grad | 0.236 / 0.113 | 0.304 / 0.103 | 0.308 / 0.106 |
| margin | 0.118 / 0.201 | 0.158 / 0.275 | 0.108 / 0.231 |
| adjoint, oriented | 0.008 / 0.021 | 0.001 / 0.001 | 0.000 / 0.000 |
| random | 0.023 / 0.030 | 0.018 / 0.024 | 0.022 / 0.032 |
| collateral | 0.58 / 0.81 | 0.657 / 0.877 | 0.658 / 0.877 |

**The fixed initialization realizes more and disturbs more, and the
conclusion stands.** Decode realization at 0.25 rises from 0.289 to
0.36-0.37, but collateral at the same strength rises from 0.58 to 0.66,
and the decode direction still saturates near 0.65 by strength 4. At
matched collateral the hard code remains several times less steerable
than softmax's 0.824. The adjoint direction is now dead outright, below
random at every strength in both seeds.

**The two seeds agree to within 0.015 at every strength for decode,
grad, adjoint and random, and to 0.002 in collateral** -- another aggregate that reproduces where individual
heads do not.

```bash
uv run python experiments/ste-arm/steerembed.py \
    --model data/out/sonar/multilingual/ste_h76_init01 --temperature 0.00015
```

### The aggregate hides a reversal: it is all about depth

Steering is applied to the INPUT, so a direction derived from layer i's
classifier has to survive i layers of residual computation before it
arrives. Splitting the same run by target layer, 12 entries per layer:

| layer | decode | grad | margin | adjoint | random |
|---|---|---|---|---|---|
| 0 | 0.216 | **0.583** | 0.547 | 0.237 | 0.034 |
| 1 | 0.464 | 0.089 | 0.367 | 0.000 | 0.023 |
| 2 | 0.742 | 0.130 | 0.216 | 0.000 | 0.055 |
| 3 | **0.966** | 0.130 | 0.091 | 0.000 | 0.034 |
| 4 | 0.362 | 0.047 | 0.036 | 0.005 | 0.047 |

(strength 1.0; at 0.25 layer 0 reads grad 0.688, margin 0.513, decode
0.068, so the gap there is wider still.)

**The classifier's geometry is exactly right where the input reaches it
directly.** At layer 0 `grad` and `margin` beat `decode` by three to
ten times. The earlier aggregate claim that decode wins was an artifact
of averaging over layers, where four of five are deep.

**And it fails with depth, which is the cascade.** `grad` falls from
0.583 to 0.047 across the stack while `decode` rises to 0.966 by layer
3. A direction computed against layer i's classifier input is only
valid there, and the perturbation reaches layer i only after the
earlier layers have re-encoded it. The decode direction does not have
that problem: it moves the input toward what the model itself emits
with the entry active, which is self-consistent down the whole stack.

**Layer 4 is hard for everything.** By then the residual is nearly
exhausted, so its decode delta is small and no direction does well.

The practical reading: steer layer 0 with `grad`, steer the middle
layers with `decode`, and expect little from layer 4. The layer-0 slice
is also the single-layer experiment in miniature -- with no cascade
ahead of it, the analytic directions work as the theory says.

## Row collinearity (`--kcos-target`): the one lever that worked

**The statistic is mostly gauge, so read the mechanism below with
suspicion even though the effect is real.** `rowcos` is computed on
`dicts()` in `e_dec`, and the decoder is 1024x2048 at full rank, so half
of `e_dec` is its null space and the atoms put 74-78% of their energy
there, where nothing they do reaches the output and no gradient reaches
them. Decoded into output space the same statistic reads 0.023 at layer
0 and -0.015 to -0.020 below it: a head's entries are already
essentially orthogonal in the space that matters.

| layer | rowcos in `e_dec` | rowcos decoded |
|---|---|---|
| 0 | 0.2880 | 0.0230 |
| 1 | 0.2217 | -0.0152 |
| 2 | 0.1910 | -0.0154 |
| 3 | 0.1605 | -0.0165 |
| 4 | 0.1226 | -0.0201 |

So the lever is not making entries geometrically distinguishable --
they already are -- and what it does instead is unexplained. It may act
through the 22-26% of each atom that does reach the output, or as a
regularizer on optimization rather than on the represented function.
The steering gain below was measured and holds; the account of why does
not.

`s_Hm` scores head USAGE; `s_kcossim` scores the row cosine described
above. The hard-code arms sit at 0.774, far above the 0.48 sonar.py's
0.2 target was calibrated against, so the target was set against this
arm at 0.5 with an explicit 50k ramp (the loop borrows its ramp from
`anneal_steps`, which is 0 under a constant temperature).

Unlike the HSIC penalty, the setpoint holds:

| step | uncontrolled | controlled | applied `s_kcossim` (200-step mean) |
|---|---|---|---|
| 10,000 | 0.790 | 0.790 | 4.0e-6 |
| 50,000 | 0.821 | 0.500 | 7.2e-4 |
| 200,000 | 0.804 | 0.494 | 0 |
| 377,300 | 0.774 | 0.473 | 0 |

The controller acts once (`writeup/figures/fig-kcos.pdf`): the
multiplier rises to 4.9e-3 near step 35k and pulls the max row cosine
from 0.82 to the setpoint by 50k, and from about 60k on `dual_apply`
projects it to zero while the constraint is satisfied. No drift ever
re-engages it: the statistic stays at or below 0.5 without pressure,
ending at 0.473.

**It improves steering in every direction, at lower collateral, for no
reconstruction cost.** Held-out FVU_w is 0.2157 against 0.2293
uncontrolled -- slightly better, despite a 14% worse training MSE,
which is the deepsup prefix mean rather than the full model.

| direction | ste_h76 @0.25 | + kcos @0.25 | change |
|---|---|---|---|
| decode | 0.280 | 0.322 | +15% |
| grad | 0.315 | 0.370 | +17% |
| margin | 0.174 | 0.264 | +52% |
| adjoint | 0.032 | 0.061 | +91% |
| adjoint, oriented | 0.051 | 0.113 | +122% |
| random | 0.023 | 0.023 | -- |
| collateral | 0.591 | 0.552 | lower |

So head collapse does independently hurt steerability, which is the
question left open when dropping `s_Hm` failed to help. The gain is
largest for the directions derived from the classifier's geometry
(adjoint, margin) and smallest for `decode`. The obvious reading --
that spreading a head's rows makes its atoms geometrically
distinguishable -- is ruled out by the decoded figures above: they are
already orthogonal. That the classifier-geometry directions gain most
is at least consistent with the effect being on the classifier rather
than on the dictionary, but nothing here establishes that.

Three head-health levers, three outcomes: `s_Hm` holds usage balance
but its removal does not change steering; `s_hsic_heads` moves its own
statistic cheaply but the heads were already independent, so there is
nothing there to take away; `s_kcossim` holds its setpoint precisely
and is the only one that buys steerability, by a mechanism its own
statistic does not explain.

## Initialization: a third of the error, and a warning about the rest

**Every arm in this file was initialized badly, and one scalar fixes
it.** The dictionary is `abs()`'d, so a layer's `h` rows sum coherently
and its output scales with `h`; `gained` compounds that across layers,
so the initial residual grows as roughly `(c*h)^l`. The live arm starts
at 1.84e8 where it should start near 1. `--dict-init-scale 0.1`
rescales each dictionary so a layer contributes about a tenth of its
input, which divides `h` out entirely and starts the cascade as a
near-identity:

| arm | shape | `s_Hm` | realized bits | FVU_w | params |
|---|---|---|---|---|---|
| `ste_h76` as shipped | 5x76 | 1e-4 | 1714 (90.2%) | 0.2293 | 51.93M |
| `ste_h76_init01` | 5x76 | 1e-4 | 1895 (99.7%) | **0.1535** | 51.93M |
| `ste_h76_i01_s43` (seed 43) | 5x76 | 1e-4 | 1895 (99.7%) | **0.1535** | 51.93M |
| `ste_k2_h380` | 5x380, k=2 | 5e-4 | 1887 (99.3%) | 0.1844 | 17.67M |
| `ste_l1_h380_i01_hm1e4` | 1x380 | 1e-4 | 1898 (99.9%) | 0.2750 | 51.93M |
| `ste_l1_h380_i01` | 1x380 | 5e-4 | 1898 (99.9%) | 0.2870 | 51.93M |
| `ste_l1_h380_i01_s43` | 1x380 | 5e-4 | 1898 (99.9%) | 0.2871 | 51.93M |

A third off the held-out error and the code essentially full, from the
initialization alone. It is the largest single effect in this file.

**It subsumes two of the results below rather than adding to them.**
Every arm at the fixed init fills 99.3% to 99.9% of its code, so the
utilization gap that the entropy-pressure correction and the input
conditioning were each recovering part of was the initialization.
Specifically, the entropy lever now does nothing: the flat arm reads
0.2750 at `s_Hm` 1e-4 and 0.2870 at 5e-4, both at 99.9%, so the
shipped weight is marginally *better* and the 484-to-1199-bit gain
that section reports was the initialization being partly unmasked. The
arithmetic there still holds -- per-head pressure does scale as
`s_Hm/h` per layer -- it simply has no consequence once the cascade
starts contracting.

**The depth claim survives at very nearly its reported size.** 0.1535
against the best flat arm's 0.2750 is 1.79x, where the section below
says 1.7x, and for the first time both sides fill their codes, so the
comparison is between two models each doing what they can rather than
two that were both starting 1.84e8 away from where they should.

**Read every other magnitude below as measured on the shipped
initialization.** Steering, the language probe, the gain ablation and
the blend ablation have not been remeasured, and the arms they ran on
were all badly initialized.

On the rate-distortion table, 1900 index bits and no continuous
coefficients at 0.1535 beats every converged SAE near it: 0.258-0.274 at
matched bits plus 160 coefficients, and 0.220 for the L1 SAE at 6390
index bits plus 519 coefficients. Against each code's own reverse
water-filling floor, though, depth is the less efficient use of bits:

| code | floor | achieved | ratio |
|---|---|---|---|
| 380 bits (one layer's prefix) | 0.4208 | 0.6427 | 1.53x |
| 1900 bits, 5x76 | 0.0499 | 0.1535 | 3.08x |
| 1900 bits, 5x380 k=2 | 0.0499 | 0.1844 | 3.70x |

So the cascade buys absolute accuracy and loses rate efficiency: a
single layer sits half again above its bound where the five-layer stack
sits three times above its.

**`k=2` answers its own question, in the negative for this file's
premise.** At identical bits and identical samples, 380 binary heads
reach 0.1844 against 76 thirty-two-way heads at 0.1535, so a hard code
does not transmit "exactly `l*h*log2(k)` bits and nothing else". Bits
go as `h*log2(k)` but the codebook goes as `h*k`, so the thirty-two-way
arm holds 12160 atoms against the binary one's 3800 -- same bits, three
times the dictionary. The binary arm reaches within 17% of it on 2.9x
fewer parameters, and its features are one bit each rather than one of
thirty-two, which is the form that is actually nameable.

## Depth at fixed capacity: the flat arm (`--l 1 --h 380`)

Five layers of 76 heads and one layer of 380 have the same nominal
capacity of 1900 bits and exactly the same parameter count -- 51.93M,
matching component by component, since both the dictionary (`h*k*e_dec`) and
the classifier (`h*k*d*n`) depend on the layer count and the heads per
layer only through their product. `ste_l1_h380` is trained at the same
batch and step count as `ste_h76` with an otherwise identical
configuration.

**The capacity was not matched on paper or in fact, and fixing both
does not close the gap.** Two things the original comparison got wrong
were fixed one at a time, each recovering part of it:

| arm | what changed | per-head entropy, median | live entries | realized bits | FVU_w |
|---|---|---|---|---|---|
| `ste_h76` | the stack | 4.92 of 5 | 32 of 32 | 1714 (90.2%) | **0.2293** |
| `ste_l1_h380_zca` | + whitened input | 5.00 of 5 | 32 of 32 | **1900 (100.0%)** | 0.3842 |
| `ste_l1_h380_hm5` | + matched pressure | 2.59 of 5 | 6 of 32 | 1199 (63.1%) | 0.4307 |
| `ste_l1_h380` | as first run | 1.00 of 5 | 2 of 32 | 484 (25.5%) | 0.5822 |
| `ste_l1_h380_enc` | + learned encoder | 1.00 of 5 | 2 of 32 | 380 (20.0%) | 0.6853 |

**Utilization is not the reason depth wins, and this is the cleanest
result in the file.** Conditioning the classifier's input fills the
flat arm's code completely -- every head, every entry, 1900 of 1900
bits, more than the stack itself realizes -- and it still reconstructs
1.7x worse. Whatever the residual cascade buys, it is not code
capacity and it is not code usage. Both of those can be handed to a
flat model outright, and the gap survives.

The two fixes together take the flat arm from 0.5822 to 0.3842, which
is 56% of the distance to the stack. So most of what the first version
of this section attributed to architecture was configuration, and the
headline ratio is 1.7x rather than the 2.5x first reported. What is
left is 1.7x at *more* realized bits on the flat side, which is a
narrower claim resting on nothing that has since been shown to be an
artifact.

`loss.csv` has carried this all along. `KL_m` averages
`log2(k) - entropy(E_batch[p])` over a layer's heads and the loss sums
over layers, and a one-hot code makes `E_batch[p]` the usage
distribution itself, so `realized bits = h * (l*log2(k) - KL_m)`. That
reproduces both measurements to within 0.3%, and it says utilization is
settled early: the flat arm is at 449 of its final 483 bits by step
37k, and the stack at 1643 of 1695.

| arm | structure | FVU_w | decode @0.25 | collateral | top-32 probe |
|---|---|---|---|---|---|
| `ste_h76_init01` | 5x76, fixed init | **0.1535** | -- | -- | -- |
| `ste_l1_h380_i01_hm1e4` | 1x380, fixed init | 0.2750 | -- | -- | -- |
| `ste_h76` | 5x76 residual | 0.2293 | 0.280 | 0.591 | **0.378** |
| `ste_l1_h380_zca` | 1x380 flat, conditioned | 0.3842 | -- | -- | -- |
| `ste_l1_h380_hm5` | 1x380 flat, matched pressure | 0.4307 | -- | -- | -- |
| `ste_l1_h380` | 1x380 flat | 0.5822 | **0.966** | **0.086** | 0.257 |

The top two rows are the comparison to quote: both fill 99.9% of their
code and the ratio is 1.79x. The four below them are the same pair at
the shipped initialization, where neither did, and they are what the
steering and language columns were measured on.

`ste_l1_h380` is the arm the steering and language columns were
measured on, and it was both under-regularized and unconditioned
relative to the stack. `ste_l1_h380_zca` is the corrected flat arm and
is the one to compare on reconstruction; the steering and language
numbers have not been remeasured on it.

Neither arm has converged. Held out at every 10k checkpoint
(`writeup/figures/fig-depth-curves.pdf`), the stack leads throughout: the
ratio to the compared flat arm is 1.59x at step 10k, peaks at 1.83x near
200k and ends at 1.79x. Over the last quarter the stack improves 0.86% and
the flat arms 2.3-2.75%, so the gap is narrowing slowly; where it ends
needs longer runs. (At the shipped initialization, `ste_l1_h380`'s kcos
sibling, run 2.5x longer, ended no better held out; that is a different
flat arm from the one compared here.)

**Depth is worth 1.79x the reconstruction error at equal nominal
capacity, equal parameters and 99.9% code utilization on both sides,
and none of it is utilization.** A flat layer is a
product quantizer: all 380 heads see the same input and their
contributions are summed, so no head can see, let alone correct,
another's error. The residual stack is a residual quantizer -- layer
i+1 classifies `X - decode(R_i)`, so each stage codes what the earlier
stages missed.

That it is the residual rather than depth as such rests on the blend
ablation below, which severs the pairing between a layer's input and
this sample's own earlier reconstruction and costs 20x the error
against a matched-magnitude control. `forward="labels"` is the mode
that would test it from the other side by removing the residual
entirely, and it has not been run at the fixed init.

But the per-layer breakdown says the error-allocation story is not the
main effect:

| layer | prefix FVU_w | added | entropy | live entries | realized bits |
|---|---|---|---|---|---|
| 0 | 0.6427 | 0.3573 | 2.85 | 10.6 | 216 |
| 1 | 0.4507 | 0.1920 | 4.98 | 32.0 | 378 |
| 2 | 0.3309 | 0.1198 | 4.94 | 32.0 | 375 |
| 3 | 0.2512 | 0.0797 | 4.89 | 32.0 | 372 |
| 4 | 0.2293 | 0.0219 | 4.91 | 32.0 | 373 |

Every layer that classifies a residual fills its code almost
completely. The one layer that classifies the raw embedding does not,
and it is the layer whose input distribution the flat arm shares. So
the cascade's first effect may be conditioning the classifier's input
rather than dividing the error, and the flat arm's 484 bits look like
layer 0's problem at five times the width. Within `ste_h76` this
contrast is already controlled for entropy pressure, since every layer
carries the same `s_Hm` weight.

`inputgeom.py` measures the premise directly, on the exact vector
`gainshape_in` hands each classifier:

| layer | mean direction | mean pairwise cos | effective dim of 1024 |
|---|---|---|---|
| 0 | 0.5638 | 0.3146 | 107.5 |
| 1 | 0.0932 | 0.0087 | 611.3 |
| 2 | 0.1522 | 0.0233 | 596.3 |
| 3 | 0.1983 | 0.0396 | 548.8 |
| 4 | 0.2348 | 0.0553 | 507.1 |

The raw embedding cloud spans a tenth of the space and its samples
average a cosine of 0.31 with each other; a residual spans over half
and averages under 0.06. Layer 0 and every flat arm classify the first,
layers 1-4 the second, at the same width, the same `k` and the same
per-head pressure. That is a within-model control on everything except
the input, and the utilization tracks the input.

With `encoded` off the encoder is a passthrough, so `E` is `X` exactly
and every arm sharing that setting reports an identical layer-0 row.
The encoder is therefore the only place a one-layer model can change
what its classifier sees, since the reconstruction target stays `X`
regardless.

### Conditioning the input: it is all of the utilization and none of the gap

Two arms supply that conditioning, and they answer different halves.

**Learning it jointly does not work.** `ste_l1_h380_enc` gives the flat
arm a randomly initialized `Linear` encoder trained alongside the
dictionary. Every one of its 380 heads ends on exactly two entries at
exactly 1.000 bits, for 380 realized bits and FVU_w 0.6853 -- worse
than having no encoder at all. The transform it found also moves the
wrong way on the axis that matters, raising mean pairwise cosine from
0.31 to 0.46 while raising effective dimension to 294. One optimizer
cannot find an input transform and a discrete dictionary at once.

**Largely superseded by the initialization section.** The utilization
this recovers, 1199 bits to 1900, is what the fixed initialization
gives the same arm for free, and the properly initialized flat arm
reaches 0.2750 where the whitened one reached 0.3842. So conditioning
was fixing the initialization by another route. What survives is the
input-geometry measurement itself, which is the evidence that the
cascade's conditioning and its pairing are one property, and the
finding that a learned encoder collapses where a supplied one does not.

**Handing it over works completely, for utilization.**
`ste_l1_h380_zca` installs `make_zca.py`'s whitener as a frozen
encoder, so the classifier sees a cloud of effective dimension 901
against the raw 107, while the reconstruction target stays `X`:

| arm | input eff. dim | realized bits | FVU_w | with the gain pinned |
|---|---|---|---|---|
| `ste_l1_h380_hm5` | 107.5 | 1199 (63.1%) | 0.4307 | 0.4307 |
| `ste_l1_h380_zca` | 901.3 | 1900 (100.0%) | 0.3842 | 0.4013 |

Every head, every entry: per-head entropy 4.999 of 5, zero heads below
1.5 bits, the full 1900. **The conditioning hypothesis is right about
utilization and nearly irrelevant to reconstruction.** It buys 58% more
realized bits and about 7% of FVU, and leaves a flat arm that fills
more of its code than the stack does while reconstructing 1.7x worse.

The fourth column is why 7% and not the 11% the third column shows.
A whitener rescales each sample by a different factor, so it hands
layer 0 a gain that varies where the raw arm's is identically 1, and
that gain reaches the decoder outside the code (see the next section).
Pinning it to its mean puts `ste_l1_h380_zca` at 0.4013, so roughly
40% of the apparent conditioning benefit was the side-channel the
whitener created rather than the geometry it was built for. The arm
changed two things and only one was controlled. A clean version would
rescale per sample so the gain stays exactly 1, which is not a linear
map and so cannot live in the encoder.

That is what closes the question this section opened. Under-filled
codes were a real defect with two real causes, and neither was the
reason for the gap. Whatever depth contributes is downstream of code
capacity and code usage alike -- it has to be the sequential dependence
between stages, since that is what is left once both are equalized.

### The code is not the whole channel: what the gain carries

`gainshape_in` splits a layer's input into a unit direction and its
norm. The classifier only sees the direction, so the code is discrete,
but `gained` multiplies the layer's whole OUTPUT by that norm. One real
number per layer per sample reaches the decoder without passing through
the bottleneck, and "1900 bits" does not count it.

It is not symmetric between the arms, and not in the direction a bit
count would suggest. SONAR embeddings are exactly unit-norm, so with no
encoder a layer-0 gain is identically 1 and carries nothing. Pinning
each gain to its held-out mean:

| arm | layer | gain sd/mean | cost of pinning |
|---|---|---|---|
| `ste_h76` | 0 | 0.0% | -0.0% |
| | 1 | 13.9% | +0.6% |
| | 2 | 14.4% | +0.7% |
| | 3 | 14.6% | +0.6% |
| | 4 | 14.7% | +0.2% |
| | all | | **+2.1%** |
| `ste_l1_h380_zca` | 0 | 14.5% | **+4.5%** |
| `ste_l1_h380_hm5` | 0 | 0.0% | -0.0% |

So the honest capacity statement is 1900 bits plus four real scalars
for the stack against 1900 bits plus none for an unconditioned flat
arm. **The channel is real and it is not the explanation.** The stack's
entire gain is worth 2.1% of its error against a 68% gap, about a
thirtieth of what depth buys. Where it matters is as a confound in the
whitened arm above, which is the only place it changes a conclusion.

### What the cascade contributes: pairing, and conditioning is its shadow

Depth survives every elimination in this file, which left "sequential
dependence between stages" as a label rather than a measurement.
`blendablate.py` measures it without retraining: layer i+1 classifies
`X - decode(R_i)`, so blend that subtracted term toward a reference
carrying the same marginals and none of the pairing, and sweep.

| blend | shuffle | mean | noise | eff dim L1 (shuffle) | (noise) |
|---|---|---|---|---|---|
| 0.000 | 0.2292 | 0.2292 | 0.2292 | 499.1 | 499.1 |
| 0.125 | 0.2460 | 0.2374 | 0.2426 | 476.0 | 504.7 |
| 0.250 | 0.3084 | 0.2652 | 0.2838 | 394.2 | 518.3 |
| 0.500 | 0.6910 | 0.4100 | 0.4665 | 193.4 | 539.6 |
| 0.750 | 1.8154 | 0.7497 | 0.8502 | 103.0 | 527.6 |
| 1.000 | 4.6613 | 1.4302 | 1.5992 | 68.5 | 496.2 |

`noise` is Gaussian rescaled per sample to the shuffle's exact
magnitude, and it is what makes the rest readable: feeding a classifier
the wrong input produces wrong corrections whatever the wrong input is.

**The pairing is load-bearing and the model is sensitive to it far
below full severance.** A quarter blend already costs 35% of FVU, and
substituting another row's reconstruction beats matched-magnitude noise
at every level, so the damage is specific rather than generic
perturbation sensitivity.

**What the sweep cannot do is separate pairing from conditioning, and
that is the finding rather than a defect.** Effective dimension falls
from 499 to 68 as the blend rises, in lockstep with the error. A
residual is small and nearly isotropic *because* it is this sample's
own error, so the two are one property seen twice, and no ablation of
this kind can hold one while moving the other. The noise column is not
the missing control either: by full blend it dominates the input, so
its high effective dimension is the noise's own isotropy and not
preserved conditioning.

That resolves the whitened arm rather than conflicting with it.
`ste_l1_h380_zca` is the one configuration that gets conditioning
*without* pairing -- a fixed transform of the input, no earlier stage
to be paired with -- and it bought 100% code utilization and about 7%
of FVU. Conditioning on its own is cheap to supply and worth little.
The cascade's value is the pairing; the conditioning is its shadow.

### The decoder cannot be sparsified, and not for the reason it looks like

Magnitude-pruning `ste_h76`'s trained decoder explodes at every level:
zeroing the smallest half takes held-out FVU_w from 0.2293 into the
thousands. There is no structure to find either -- median `|w|` is 27%
of the maximum and half the total mass sits in the largest quarter,
which is what an ordinary Gaussian matrix looks like.

The obvious culprit is the non-negative dictionary. `R` is a sum of 380
`abs()`'d vectors, so its constant part dwarfs its varying one, and the
decoder has no bias, so one matrix appears to be both decoding the
variation and attenuating that constant to reach the embedding mean:

| arm | `\|\|mean R\|\|` | mean `\|\|R-mean\|\|` | ratio |
|---|---|---|---|
| `ste_h76` | 153.7 | 13.2 | 11.6 |
| `ste_l1_h380_zca` | 289.9 | 34.6 | 8.4 |

It is demonstrably doing that: the decoder maps mean `R` to norm 0.581,
against the embedding cloud's own mean direction norm of 0.564.

**That description is right and the inference from it is wrong.**
`--biased-dec` hands the constant to 1024 free parameters instead, and
at matched step it is worth -0.62% of training MSE on the stack and
-0.20% on the conditioned flat arm. Consistent, growing slowly, and
nowhere near what "the decoder is spending its capacity on this" would
predict. The arithmetic says why: pinning `W @ mean(R)` is 1024 linear
constraints on a 1024x2048 matrix, which is 0.05% of its freedom. It
was never expensive.

So the decoder is dense because a full-rank map carrying a roughly
900-dimensional signal is dense, not because of any offset artifact.
Sparsifying it means giving up reconstruction, and there is no cheap
structure waiting to be found. The bias is a small free win worth
taking and not a route to anything.

(Both arms died of an out-of-memory at 23% and 34% of their runs, from
two trainers preallocating the card at once. The comparison above is at
matched step and holds; the absolute numbers are not converged, and a
rerun is not worth it for a sub-1% effect.)

Two mechanical notes for anyone repeating this. The whitener must
preserve the input's norm: `gainshape_in` measures the gain on the
classifier input and `gained` multiplies the layer's output by it, and
because SONAR embeddings are exactly unit-norm that gain is invisibly 1
without an encoder. Unscaled ZCA makes it 32 and inflates every
contribution by that, for a training error three orders of magnitude
off with every other statistic looking ordinary. And the eigenvalue
floor belongs against the largest eigenvalue, not the mean; against the
mean it sits below the near-null tail, and the effective dimension it
appears to buy is amplified numerical noise.

### Entropy pressure across shapes

**Superseded by the initialization section.** At the fixed init both
weights give 99.9% utilization and the shipped one is marginally
better, so this lever recovers nothing; the gap it appeared to close
was the initialization. The scaling argument below is still correct and
still worth knowing when comparing differently shaped models -- it just
has no effect on any arm here. Kept because the reasoning is the
generally applicable part.

**That control does not extend across the two arms, and correcting it
takes back two fifths of the gap.** `hmean_kl` averages over a layer's
heads and `Hyperparams.loss` sums the statistic over layers, so the
pressure on any one head scales as `s_Hm / h` per layer: 1/76 in the
stack against 1/380 in the flat arm, at the same configured weight. The
flat arm had been trained at a fifth of the stack's per-head pressure.
`ste_l1_h380_hm5` is the same arm at `s_Hm` 5e-4, which matches it
exactly:

| arm | shape | `s_Hm` | per-head | bits | of nominal | FVU_w |
|---|---|---|---|---|---|---|
| `ste_h76` | 5x76 | 1e-4 | 1.32e-6 | 1695 | 89.2% | 0.2293 |
| `ste_h76_hm1e6` | 5x76 | 1e-6 | 1.32e-8 | 1519 | 80.0% | 0.2095 |
| `ste_l1_h380` | 1x380 | 1e-4 | 2.63e-7 | 483 | 25.4% | 0.5822 |
| `ste_l1_h380_hm5` | 1x380 | 5e-4 | 1.32e-6 | 1199 | 63.1% | **0.4307** |

Matching the pressure raises realized capacity 2.5x, cuts held-out
error 26%, and takes the count of heads under 1.5 bits from 234 to 16.
**So the flat arm in the comparison above was handicapped**, by this
and by its unconditioned input both; together they account for 56% of
the gap, and the headline ratio is 1.7x rather than 2.5x.

**What this lever does not explain is the gap.** At identical per-head
pressure the flat arm fills 63% of its code against the stack's 89%,
and the stack fills 80% at one hundredth of that pressure, so the
stack fills its code essentially unprompted where the flat arm does
not. But conditioning the input takes the flat arm to 100%, past the
stack, and moves FVU about 7% once its incidental gain channel is
discounted. Usage is a thing the pressure controls; it is not the thing
the depth is buying. Note also that pressure and
reconstruction do not move together across shapes: dropping `s_Hm` by
100x costs the stack 176 bits and *improves* its FVU to 0.2095, while
raising it 5x buys the flat arm 716 bits and 0.15 of FVU. Bits are not
the objective, and the two shapes sit on opposite sides of the setting
that was tuned for one of them.

**The cost is entanglement, and it is the whole steering gap.** A
single-head embedding-space intervention in the flat model realizes the
target entry on 97% of rows while disturbing 9% of the other heads; in
the stack it realizes 28% while disturbing 59%. Restricting both to
layer 0, so the target is the same distance from the input, the gap
does not close:

| direction | stack layer 0 | flat |
|---|---|---|
| decode | 0.068 / 0.616 | 0.966 / 0.086 |
| grad | 0.688 / 0.563 | 0.969 / 0.063 |
| margin | 0.513 / 0.570 | 0.979 / 0.069 |
| random | 0.005 / 0.584 | 0.064 / 0.093 |

(realized / collateral, strength 0.25.)

**The collateral is not a property of the intervention.** A random push
of the same norm disturbs 58% of the stack's heads and 9% of the flat
model's, so it is not that steering directions are blunt -- the
residual cascade turns any input perturbation into widespread
reclassification downstream, and a deliberate one is no exception. The
flat model has nothing downstream to reclassify, so its heads are
independently addressable by construction.

**The stack also localizes language, the one labelled factor, better, in
every sense that survives the null.** Its best 32 heads recover language at 0.378
against the flat arm's 0.257, on a dense-probe ceiling of 0.721.

| arm | best NMI less null | m=1 | m=2 | m=32 | m=1 as share of m=32 |
|---|---|---|---|---|---|
| `ste_h76` | 0.261 | 0.165 | 0.202 | 0.378 | 43.6% |
| `ste_l1_h380` | 0.276 | 0.102 | 0.117 | 0.257 | 39.7% |

Per-head content is a wash once the null is subtracted, held-out probe
accuracy favours the stack at every head budget, and on concentration
proper -- how much of what the top 32 heads carry comes from the best
one -- the stack is slightly ahead.

The one statistic that favours the flat arm is `info_share_best`, where
it reads 8.7% against 4.2%. That is `I.max() / I.sum()` over an
uncorrected plug-in mutual information, summed across 380 heads whose
chance level differs eightfold between the arms (null NMI 0.001 flat,
0.008 stacked). The flat arm's heads are mostly binary, so both its
numerator and its denominator are small and differently biased. Do not
read that column across architectures.

**The row-collinearity setpoint does not transport.** `ste_l1_h380_kcos`
reads FVU_w 0.5924 against 0.5822 uncontrolled, slightly worse, where
on the stack the same target was slightly better. Two confounds run the
other way and do not rescue it: the flat arm fell back to b=128, and it
ran 929,400 steps for 119M samples against 94.6M, so it saw 26% more
data and was still no better. Its steering is already saturated (0.992
against 0.966), so there is no headroom for the lever that buys
steerability on the stack.

The reading: capacity is not the architecture. Whatever this model does
that a wide flat dictionary does not, it does through the residual
cascade -- and the same cascade is what makes single-head interventions
leak.

## The decoder's null space, and what non-negativity costs (`--signed`)

The gpt2_l8 arms decode a 1536-wide dictionary into 768 dimensions, so
`ker(D)` is 768-dimensional and half of every atom is invisible to the
output. Several readings of that were wrong and are worth recording,
because each was checkable and each failed.

- **Head interference putting content in the null space.** No: head
  null components add *coherently*, 3.3-11x over the cancelling
  expectation, so they are not interference debris.
- **Noise scale driving it.** No: the fraction does not track `sd_F`.
- **A swap space, carrying content that later becomes visible.** No:
  the per-atom visible fraction is flat at 22% across training and the
  decoder's row space rotates 2.6-3.5 degrees in total. Nothing moves
  in or out.
- **`cossim_k` inflating it** to cheapen apparent row similarity by
  hiding the shared part where the penalty cannot see it. Plausible,
  and not what the numbers show.

Two baseline errors ran the other way and both reversed an
interpretation once fixed. Against `abs(N(0,1))` rather than signed
normals, the right expectations are **18.26%** visible (not 50%) and
**63.7%** alignment with **1** (not ~0): the trained dictionary is
*less* null-heavy and *less* all-ones-aligned than chance. And the
"49% of reconstruction gradient lies in `ker(D)`" measurement is a
coordinate artifact -- w.r.t. `|W|` it is 3e-8, because `sign(W) ⊙` does
not preserve a subspace.

The size of the null part depends entirely on which norm is asked for,
which is worth stating once: 2.43% of squared L2, 23.20% of L2, 56.55%
of L1.

### A fifth failed reading: the null space as a lift

Worth recording because it survived longest and is wrong. The story was
that non-negativity, being *elementwise*, is basis-dependent and is
imposed on the stored atom `a` rather than on the visible component
`P a` that reaches the output, so null content is the "lift" that lets
a mixed-sign visible component be carried by a non-negative atom.

**It does not hold.** `P a`'s entries are not constrained -- only `a`'s
are -- so a mixed-sign `P a` is not a defect that anything needs to
repair, and null content cannot repair it in any case, being invisible
to the decoder. The supporting measurement (**0.00%** of atoms have a
non-negative `P a`) is true and vacuous: it says only that atoms have a
nonzero null component, which almost any vector in a 1536-dimensional
space with a 768-dimensional null space does.

### The mechanism: the orthant forces a collinear dictionary

Non-negativity binds through the Gram matrix, not through the null
space. Two non-negative vectors have `<a, a'> >= 0` exactly, so the
dictionary is non-negatively correlated by construction. For
independent `abs(N(0,1))` coordinates the chance pairwise cosine is
`2/pi = 0.6367`, and over 4000 draws the *minimum* is `+0.5796` -- it
does not approach zero. Signed draws give `0.0000`, minimum `-0.1314`.

The effect on the dictionary is large. Converged, geometry with
`dictgeom.py`, FVU_w held out and scored from each checkpoint
(`writeup/figures/extract_curves.py`):

| arm | `c` | eff rank | FVU_w |
|---|---|---|---|
| signed, e_dec 1536 | +0.0009 | 27.37 | 0.1540 |
| signed, e_dec 768 | +0.0019 | 26.95 | 0.1543 |
| abs, e_dec 1536 | +0.3038 | 9.15 | 0.1934 |
| abs, e_dec 768 | +0.4543 | 5.41 | 0.2927 |

**And it costs reconstruction.** `abs` at e_dec 1536 runs at a third of
the signed arms' effective rank -- 9.15 against 27.37 -- and 26% more
held-out error; at e_dec 768, 90% more. Collinearity orders all five arms
(concat included) by error at convergence, Spearman +1.00, and at +0.90 at
step 90,000, where concat is the single inversion.

Width is an **interaction** on top of that: under `signed`, `e_dec` costs
+0.2% -- nothing -- and under `abs` it costs +51%. So the collinearity
account and the cone account are both live: collinearity tracks the error
across all arms, and the square decoder makes it much worse only when the
atoms are confined to the orthant.

What removing `abs` buys is therefore both a decorrelated,
near-full-effective-rank dictionary and lower error. An earlier version of
this section, scored on `dictgeom.py`'s training-log FVU, read the cost as
2.4% and called it nearly free; that log averages all five
deep-supervision prefixes and does not measure the final output.

`s_kcossim` and `KCOS_target` are 0 in all four arms, so none of this
is a collinearity penalty doing the work. Removing one `abs` buys
roughly what the `--kcos-target` controller exists to buy.

### Where concat fits, and the floor's actual form

The bound on a head's mean pairwise cosine is

    c >= (k/n - 1) / (k - 1),      n = d_head

attained by spreading the `k` rows over the `n` coordinates with disjoint
supports, which is also where non-negativity's own `c >= 0` stops being
the binding one. So the floor only rises above zero once `k > n`: it is
0 for both summing widths and for `--d-head 32` (where `n = k = 32`),
0.0194 at `--d-head 20`, and 0.0616 at `--d-head 11`, the narrowest the
`d_out / h <= D` window allows.

`ste_h76_cat32` therefore does not face a raised floor. What it faces is
a collapsed tradeoff: reaching `c = 0` at `n = k` requires one
coordinate per atom, so a decorrelated dictionary is a one-hot one with
nothing left to express. It took the correlation instead, and ends with
the highest of the three `abs` arms (per layer, at convergence):

| arm | `cossim_k` | `cossim_k_max` | FVU_w (held out) |
|---|---|---|---|
| summing, e_dec 1536 | +0.3039 | 0.666 | 0.1934 |
| summing, e_dec 768 | +0.4543 | 0.765 | 0.2927 |
| concat, d_head 32 | +0.5353 | 0.841 | 0.4155 |

Monotone in both across these three, and the signed arms extend the same
relation: at `c ~ 0` they have 20% less error than summing/1536. Within
the `abs` family the two move together because both track how much room
the orthant constraint has, which is also what the error tracks.
`cossim_flat` is not comparable here -- `ConcatDictBlock` overrides it,
and between-head pairs are exactly orthogonal by construction and swamp
the within-head ones.

Concat is not explained by `c` alone, though. At step 90,000 it has a
slightly *lower* `c` than the summing e_dec-768 arm (0.571 against 0.578)
and a much worse held-out FVU_w (0.806 against 0.463); over the five arms
Spearman is +0.90 there, and concat is the single inversion. By
convergence the ordering is exact, because concat's `c` stops falling
while the summing arm's keeps going.

Its extra cost is reachability, which is the one place that frame does
bind. At `d_head = 32` a head's rows are non-negative vectors in `R^32`
and its output is `D_block a` for a `(768, 32)` column block: 32
generators in a 32-dimensional subspace, a simplicial cone, the most
restrictive case there is. `train_ste.py` prints the per-head null space
as 0 for this config. The summing arm at e_dec 1536 spreads a head's
rows over all 1536 dimensions against the full decoder.

### Does a softer code raise the dictionary's rank? (`dictgeom.py`)

No, and rank is the wrong variable. `dictgeom.py` reads the weights
directly; the selection sweep is config-matched (k=32, h=32, l=5,
e_dec=2048) at 369,500 steps with only `select` varying.

| select | `c` | eff rank | rank99 | FVU_w | | `c` +s_Hm | eff +s_Hm | FVU_w +s_Hm |
|---|---|---|---|---|---|---|---|---|
| top1 | +0.3613 | 6.41 | 31.3 | 0.7190 | | - | - | - |
| top2 | +0.5797 | 2.68 | 30.7 | 0.3516 | | +0.2177 | 13.83 | 0.1911 |
| top4 | - | - | - | - | | +0.1505 | 19.78 | 0.0761 |
| top8 | +0.3887 | 4.72 | 31.0 | 0.0370 | | +0.1410 | 18.72 | 0.0287 |
| top16 | +0.2244 | 12.45 | 31.1 | 0.0052 | | +0.2150 | 15.45 | 0.0043 |
| softmax | +0.2056 | 13.90 | 31.8 | 0.0023 | | +0.3592 | 9.05 | 0.0007 |

FVU_w is the final output's on the cache tail, scored from the checkpoint
(`writeup/figures/extract_curves.py`); these runs trained on the tail, so
it is in-sample. `dictgeom.py`'s own FVU_w column is the training log,
which under deep supervision averages all five prefixes' error and reads
1.3 to 74 times higher across these arms, most where the final error is
smallest.

`rank99` is 30.7-31.8 of 32 in all ten arms, so the numerical rank is
full whatever the code is and carries no signal. Effective rank varies
7-fold and is **not monotone in softness**, and the two families reverse
each other: without the mean-entropy bonus softening raises it
(6.41 -> 13.90), with it the peak is mid-ladder at top4 and softmax is
the worst arm. The selection rule is not the controlling variable.

The mechanism points the way the `_shm` family went, and `rowcos`'s
docstring already says why: under a hard code a head's output *is* one of
its rows, so collinear rows make the output independent of which entry
wins and the head carries nothing. Under a soft code, continuous
coefficients give a continuum of outputs along even small differences
between rows, so collinearity costs much less. A soft code needs rank
less than a hard one does.

Caveat on the whole table: the arms span FVU_w 0.72 to 0.0007, so they sit
at very different points of fit and nothing here is matched on
reconstruction quality.

#### Matching on FVU instead reverses it, and neither matching is a control

`dictgeom.py --fvu=` picks each arm's checkpoint closest to a target, and
`--steps=` traces one arm, which is what the comparison actually needs:
checkpoints are 10k apart and a fast arm's FVU moves further between two
of them than the spread being measured, so single matched points are
interpolated onto a common grid.

Matched on held-out FVU_w (in-sample tail; `c` and step at the first
crossing, log-interpolated between 10k checkpoints):

| select | FVU 0.60 | 0.40 | 0.25 |
|---|---|---|---|
| top2 | 0.760 @23k | 0.716 @36k | **0.531** @80k |
| top4 | 0.779 @14k | 0.762 @20k | 0.730 @29k |
| top8 | 0.769 @12k | 0.757 @16k | 0.741 @21k |
| top16 | 0.732 @14k | 0.719 @18k | 0.700 @22k |
| softmax | **0.597** @25k | **0.577** @31k | 0.559 @36k |

At moderate error the soft code has the least collinear dictionary by a
wide margin, the same direction as the early step-matched reading. Only at
the deepest common level does `top2` drop below it, and it reaches that
level at step 80k against 21k-36k for the others, so its lower `c` there
is the mirror confound: decorrelation continues after FVU flattens, and
the arm that trained longest looks decorrelated for that reason. This
table replaces one matched on `dictgeom.py`'s training-log FVU, which
averages all five prefixes and put `top2` at 0.40 near step 234k; held
out it is there by 36k.

**Duration dominates the selection rule by 3.8x.** Mean `c` falls from
0.750 at 10k to 0.217 at 369.5k, a swing of 0.533, while the spread
across all five rules at a fixed step averages 0.140.

The orderings cross at around step 100k, which is what made the
converged table above look non-monotone and the two families look
contradictory -- they sample opposite sides of it:

| step | top2 | top4 | top8 | top16 | softmax | spread |
|---|---|---|---|---|---|---|
| 10,000 | 0.795 | 0.788 | 0.774 | 0.743 | 0.648 | 0.147 |
| 50,000 | 0.658 | 0.639 | 0.612 | 0.588 | 0.519 | 0.139 |
| 100,000 | 0.459 | 0.418 | 0.406 | 0.426 | 0.435 | 0.053 |
| 150,000 | 0.329 | 0.275 | 0.256 | 0.303 | 0.370 | 0.114 |
| 369,500 | 0.218 | 0.151 | 0.141 | 0.215 | 0.359 | 0.218 |

The late half is the mechanism: the soft arm's `c` plateaus at 0.359 while
its error keeps falling, and the hard arms keep paying collinearity down
to 0.14 while their error barely moves. From 150k to 369.5k, held out:
`softmax` cuts FVU_w 85% and `c` 3%; `top8` cuts FVU_w 3% and `c` 45%;
`top16`'s FVU_w does not improve while `c` falls 29%; `top2` cuts both,
12% and 34% (`writeup/figures/fig-sweep-curves.pdf`). Collinearity costs
a hard code information directly, so it keeps working on it; a soft code
has no such pressure and stops. At convergence a softer code therefore
gives the *more* collinear dictionary, inverting the naive expectation.

This is the `s_Hm` family. The family without the bonus disagrees at
convergence, but its hard arms pair high `c` with poor FVU (`top2` at
`c` 0.58 and FVU_w 0.352, against 0.22 and 0.191 with the bonus), which
reads as head collapse -- what `s_Hm` exists to
prevent -- rather than a selection-rule effect. One seed per arm either
way.

#### The last layer: its collinearity is a correlate (`lastlayer.py`)

Collinearity falls monotonically with depth in every hard arm measured
(SONAR `ste`, `sweep_top1`, and the gpt2 `ste` and `signed` arms) but not
in the soft ones, which spike at the LAST layer: `sweep_softmax_shm`
ends at `c = 0.799` with eff 1.54, nearly rank-one, and `top16_shm` at
0.433 after reaching 0.048 one layer earlier.

`lastlayer.py` reads the deepsup per-prefix decodes and ablates the final
layer's heads two ways. **Uniform** replaces its classification with
`1/k` so it emits its mean atom; **zero** removes it. The pair separates
a layer that emits nothing from one that emits a constant.

| arm | select | L4 `c` | L4 eff | p3 | L4 adds | zero-ablate |
|---|---|---|---|---|---|---|
| sweep_top1 | top1 | +0.246 | 10.60 | 0.7511 | +4.3% | +4.5% worse |
| sweep_top2_shm | top2 | +0.188 | 14.89 | 0.2378 | +19.7% | +24.5% worse |
| sweep_top8_shm | top8 | - | - | 0.0304 | +5.5% | +5.8% worse |
| sweep_top16_shm | top16 | +0.433 | 3.46 | 0.0043 | +0.0% | +0.0% |
| sweep_softmax | softmax | +0.263 | 6.00 | 0.0023 | -1.2% | -1.1% better |
| sweep_softmax_shm | softmax | +0.799 | 1.54 | 0.0007 | -1.9% | -1.8% better |

**The contribution tracks `p3` -- the error left before the last layer --
and not the collinearity.** `sweep_softmax` settles it: its L4 is not
collapsed, `c = 0.263` at eff 6.0, and is just as useless as the
rank-one one, with zero-ablation improving held-out error. Saturation is
the cause and the geometry is downstream. Below about `p3 = 0.005` the
last layer is inert or mildly harmful; above it, it earns its place.

The two collapses are different endings, which the ablation pair
separates. `top16_shm`'s L4 shrank its norms to 4% of the other layers'
(within-head norm CV 3.03) and zero-ablating it is a literal no-op: it
emits nothing. `softmax_shm`'s L4 keeps a norm of 0.668 at CV 0.076 --
collinear AND near-equal in magnitude, so it emits a near-constant
vector -- and removing that *improves* held-out error, making it a small
overfit rather than a useful bias. Neither is a "usable scalar-gain
correction", which is what the collinearity alone suggested.

So the depth trend is not a hard-code property, as the first version of
this claim had it. Soft codes break it because they saturate before the
last layer, not because they tolerate collinearity.

Evaluating these arms needs their annealed temperature: they end at
`temperature_end = 0.03`, and scoring at the `1.0` default puts
`sweep_softmax_shm` at FVU_w 0.91 instead of 0.0007, which looks exactly
like a broken checkpoint. Their logged training MSE is far above the
clean number because it carries `sd_F = 0.1` noise and `p_drop`.

The same tool gives the gpt2 arms' exact geometry, which adds a fact the
correlation alone missed -- concat loses genuine numerical rank, not just
effective rank:

| arm | `c` | eff rank | rank99 |
|---|---|---|---|
| summing, e_dec 1536 | +0.3038 | 9.15 | 30.2 |
| summing, e_dec 768 | +0.4543 | 5.41 | 28.1 |
| concat, d_head 32 | +0.5337 | 3.30 | 17.1 |

### The reachability frame, and where it does not bind

The reachability framing that replaced the lift story is too coarse for
the `e_dec` contrast, and oversold the wide arm. `{D a : a >= 0}` is the conical hull
of `D`'s columns: a single simplicial cone when the decoder is square,
much larger at `e_dec = 2d`. But by Wendel's theorem `m` random columns
lie in some halfspace with probability `2^(1-m) sum_{i<d} C(m-1, i)`,
which at `m = 2d` is exactly `1/2` -- so `e_dec = 1536` sits on the
threshold where a random decoder's cone covers the output space at all.
Reachability is binary; the measured effect is continuous in `c`.

The cost of narrowing, under `abs`, grows as the model learns to use
the lift (held-out FVU_w at each checkpoint):

| step | e_dec 1536 | e_dec 768 | cost |
|---|---|---|---|
| 20,000 | 0.5950 | 0.6613 | +11.2% |
| 90,000 | 0.3962 | 0.4631 | +16.9% |
| 200,000 | 0.2900 | 0.3647 | +25.8% |
| 371,900 | 0.1934 | 0.2927 | +51.3% |

### The test

`--signed` drops the `abs()` in `DictBlock.dicts`. The prediction, as
written on 2026-09-30, after the signed arms were launched and before any
comparable result:

> `--signed` drops the `abs()` in `DictBlock.dicts`, which removes the
> need for any lift. The prediction is a specific *pattern* over the 2x2,
> not just a smaller gap: three cells alike and one worse. `abs` at
> e_dec=1536 is already effectively signed, because it has 768 dimensions
> of lift available; `signed` needs no lift at either width. So
> `signed/768`, `signed/1536` and `abs/1536` should agree, and `abs/768`
> alone should pay the +14.8%.
>
> It also separates lift from capacity, which the `e_dec` arm alone
> cannot. If the wide dictionary were buying representational capacity,
> `signed/1536` would still beat `signed/768`. If it were buying the
> lift, they tie.

The +14.8% it quotes is the training-log cost of narrowing, not a held-out
one. **The pattern did not happen.** Held out, `abs` is worse at both
widths:

| e_dec | abs | signed | signed vs abs | abs @90k | signed @90k |
|---|---|---|---|---|---|
| 1536 | 0.1934 | 0.1540 | -20.4% | 0.3962 | 0.2910 (-26.6%) |
| 768 | 0.2927 | 0.1543 | -47.3% | 0.4631 | 0.2942 (-36.5%) |
| narrowing | +51.3% | +0.2% | | +16.9% | +1.1% |

At e_dec 1536 the `abs` arm's excess over signed grows from 8% at step
20,000 to 50% at 200,000 and then narrows to 26% at 371,900, so part of it
may be speed, but a full run does not close it. An earlier reading of this
2x2, on `dictgeom.py`'s training-log FVU (which averages all five
deep-supervision prefixes), had the three arms agreeing within 2.4% at
convergence and called `abs` merely slower; held out, that was wrong.

The pair also settles capacity, which the `e_dec` arm alone cannot: the
two signed arms agree to +0.2%, so the wide dictionary was buying room
for the constraint, not representational capacity.

`ste_h76_sgn768` and `ste_h76_sgn` are that pair, at the reference
arms' settings and full 24 epochs:

```bash
uv run python experiments/ste-arm/train_ste.py --base gpt2_l8 \
    --dict-init-scale 1.0 --signed --e-dec 768 \
    --out data/out/gpt2_l8/ste_h76_sgn768
uv run python experiments/ste-arm/train_ste.py --base gpt2_l8 \
    --dict-init-scale 1.0 --signed --out data/out/gpt2_l8/ste_h76_sgn
```

Signed atoms give up two things the default has. Subtractive features
are harder to read as "this concept is present". And under `norm_rows`
the `|F|_1 >= h` floor disappears: it rested on non-cancellation, and
signed rows can cancel across `k` (`tests/test_signed.py` pins this).

Those costs are now weighed against a measured gain rather than against
a diagnostic, since the arms also decorrelate the dictionary by two
orders of magnitude without any collinearity pressure applied. That
survives to convergence: at 371,900 steps the signed arms sit at
`c` +0.0009 and +0.0019 with effective rank 27, against +0.30 and +0.45
at rank 9.15 and 5.41 under `abs`, and with 20% less held-out error at
e_dec 1536.

## Between-head disjoint support (`headsupport.py`): absent in every model

The disjoint-support argument -- non-negative heads that are orthogonal
own disjoint blocks of the dictionary space, as a property of the
weights -- is the premise for reading a dictionary statically, and it had
never been measured: every reported model trained before the support
overlap `sigma` existed, most with its predecessor `cossim_h` penalized
(`s_hcossim` 1e-6). `headsupport.py` computes `sigma` from the final
checkpoints: the mean off-diagonal cosine between heads' usage-weighted
coordinate profiles `s_h = sum_k pbar_hk |W_hk|`, `pbar` the mean
assignment over the eval tail at the run's final temperature. It does so
in the dictionary space and again with each atom decoded into the output
space. The null permutes each head's profile coordinates independently
(50 draws), keeping each head's own distribution. `spread` is each
head's profile participation ratio as a fraction of the dimension; a
partition into `h` blocks needs it near `1/h`.

| model | `sigma` dict (null) | spread | `sigma` decoded (null) |
|---|---|---|---|
| SONAR `resid_nc` (372.6k steps) | 0.818 (0.815) | 0.83 | 0.917 (0.910) |
| SONAR `resid_nc_hm` (8.27M steps) | 0.113 (0.118) | 0.13 | 0.892 (0.878) |
| SONAR `sweep_softmax_shm` | 0.731 (0.738) | 0.74 | 0.938 (0.930) |
| SONAR `sweep_top2_shm` | 0.862 (0.860) | 0.87 | 0.972 (0.958) |
| SONAR `ste_h76` (shipped init) | 0.942 (0.942) | 0.94 | 0.721 (0.719) |
| SONAR `ste_h76_init01` | 0.962 (0.961) | 0.96 | 0.980 (0.965) |
| SONAR `ste_l1_h380_i01_hm1e4` | 0.977 (0.977) | 0.98 | 0.974 (0.963) |
| GPT-2 abs, e_dec 1536 | 0.739 (0.643) | 0.65 | 0.948 (0.925) |
| GPT-2 abs, e_dec 768 | 0.596 (0.486) | 0.52 | 0.875 (0.852) |
| GPT-2 signed, e_dec 1536 | 0.937 (0.636) | 0.64 | 0.964 (0.943) |
| GPT-2 signed, e_dec 768 | 0.955 (0.950) | 0.95 | 0.965 (0.943) |
| GPT-2 concat, d_head 32 | 0 by construction | -- | 0.846 (0.788) |

**No model partitions its dictionary space between heads.** `sigma` is
at or above its null everywhere, and each head's profile covers 52-98%
of the coordinates where a partition needs about 3% (`h = 32`) or less.
Where `sigma` departs from the null it is *above* it: the GPT-2 arms
place their heads on the same coordinates, the signed wide arm most
(0.94 against 0.64). Decoded, heads overlap slightly more than chance in
every model. The two exceptions to dense profiles are collapsed layers
(`sweep_softmax_shm` layer 4, `sigma` 0.086 against 0.112, the inert
last layer of the selection-rule section; GPT-2 abs/768 layer 2, 0.133
at its null), not partitions.

It is not the null space. Most of each non-negative atom's energy sits in
the decoder's null space, which reconstruction never constrains, so the
dictionary-space overlap could have been gauge. Splitting each atom into
its row-space (visible) and null-space parts and scoring each against the
same shuffle null:

| model | null-space energy | `sigma` visible (null) | `sigma` null part (null) |
|---|---|---|---|
| SONAR `sweep_softmax_shm` | 0.47 | 0.899 (0.901) | 0.887 (0.889) |
| SONAR `sweep_top2_shm` | 0.89 | 0.971 (0.971) | 0.901 (0.899) |
| SONAR `ste_h76` | 0.96 | 0.771 (0.771) | 0.983 (0.982) |
| SONAR `ste_h76_init01` | 0.80 | 0.980 (0.980) | 0.981 (0.980) |
| SONAR `ste_l1_h380_i01_hm1e4` | 0.78 | 0.978 (0.977) | 0.989 (0.987) |
| GPT-2 abs, e_dec 1536 | 0.97 | 0.940 (0.934) | 0.802 (0.711) |
| GPT-2 signed, e_dec 1536 | 0.14 | 0.955 (0.706) | 0.954 (0.890) |

The visible part is at chance on SONAR and above it on GPT-2, so the
part that reaches the output does not partition either; and the e_dec 768
arms, whose square decoder has no null space, sit at or above the null in
the dictionary space itself. The row space is not coordinate-aligned, so
the visible part's support is basis-dependent; the decoded `sigma` is the
nearest basis-free check, and it agrees. The energy split differs sharply
by sign: the non-negative arms keep 78-97% of atom energy in the null
space, the signed GPT-2 arm 14%. This is not the lift reading, which the
null-space section rejects; what drives the difference is not measured
here.

The single-layer variants and the SAE's discovered groups have no
dictionary space -- their decoder rows are output-space directions -- so
only the decoded measure applies, with a head's profile
`sum_{j in g} E[z_j] |W_dec,j|` (`--sae`):

| model | heads or groups | `sigma` decoded (null) | spread |
|---|---|---|---|
| `g160top1` | 160 trained | 0.648 (0.644) | 0.64 |
| `g160softmax` | 160 trained | 0.983 (0.975) | 0.97 |
| `m11264_k32` | 359 discovered, 9025/11264 latents | 0.899 (0.892) | 0.89 |
| `m5120_k32` | 164 discovered, 4247/5120 latents | 0.917 (0.905) | 0.91 |

All at the null (ratio 1.01), as every Ontologizer's decoded `sigma` is
(1.01-1.07). That says no head or group structure, trained or discovered,
partitions the output coordinates -- but the decoded measure cannot tell
architectures apart: SONAR's output coordinates are not meaningful axes,
every model's atoms are dense across them, and all fourteen models land
within 7% of their nulls. `g160top1`'s lower absolute value only reflects
more concentrated profiles (it tiles one decoder cluster per head; see the
coherence results). The comparison `sigma` was built for exists only in
the dictionary space, which the SAE variants do not have.

So the `cossim_h` penalty at the weights used did not produce the
partition the architecture argument assumes, and the within-head
orthogonality measured earlier is the only one the trained models have.
`ConcatDictBlock` is the only way these models got disjoint support, and
by construction. `s_support` as a loss term does not produce it either:
it is satisfied in the decoder's null space (next section).

`resid_nc_hm`, the reference configuration trained 22 times longer, is
the one model whose heads do anything like avoid each other. Its
profiles are sparse (spread 0.13 against 0.52-0.98 elsewhere) and `sigma`
sits 4% below its null, in four of five layers (0.113 against 0.118).
That is still not a partition: at `h = 32` one needs spread near 0.03,
and most of the low `sigma` is the sparsity, which the null shares. Long
training sparsifies a head's coordinate profile; it barely separates
heads. `resid_nc`, the same configuration at 372.6k steps, is at its null
with dense profiles like the rest.

(Both load through `restore_spec`, which infers from the stored weights
that their layer 0 predates `resid_const`'s coordinate; the reloaded
`resid_nc` reproduces the Pareto table to four digits.)

```bash
uv run python experiments/ste-arm/headsupport.py \
    --model data/out/sonar/multilingual/ste_h76_init01 \
    --model data/out/gpt2_l8/ste_h76_sgn --out headsupport.json
```

Results: `data/out/headsupport/headsupport.json`.

## Training with the support overlap (`--s-support`): satisfied in the null space

A pilot on GPT-2 with a signed dictionary (e_dec 1536, so a 768-dimensional
decoder null space), four weights, 3 epochs (46,250 steps) each, one seed:

```bash
uv run python experiments/ste-arm/train_ste.py --base gpt2_l8 \
    --dict-init-scale 1.0 --signed --s-support W --epochs 3 \
    --out data/out/gpt2_l8/sigma_pilot/sW      # W in 0, 1e-5, 1e-4, 1e-3
```

Held-out final-output FVU at the last checkpoint; `sigma` and spread from
`headsupport.py`, averaged over layers; logged `support` (summed over
layers) at the start and end:

| `s_support` | held-out FVU | logged `support` | `sigma` dict (null) | spread | `sigma` decoded (null) |
|---|---|---|---|---|---|
| 0 | 0.423 | 4.92 → 4.77 | 0.862 (0.854) | 0.85 | 0.891 (0.826) |
| 1e-5 | 0.467 | 4.92 → 0.44 | 0.065 (0.063) | 0.06 | 0.869 (0.797) |
| 1e-4 | 0.512 | 4.92 → 0.050 | 0.0061 (0.0099) | 0.01 | 0.852 (0.778) |
| 1e-3 | 0.603 | 4.91 → 0.0057 | 0.0005 (0.0037) | 0.00 | 0.803 (0.735) |

The penalty drives the dictionary-space `sigma` to its target and, from
1e-4, below its shuffle null (ratio 0.61 and 0.15): read as stated, a
partition. It costs 10%, 21% and 43% of held-out FVU over the unpenalized
twin. Decoded, nothing changes: every arm sits 8-10% above its null, as
the unpenalized one does.

Splitting each atom into the decoder's row space (visible) and null space,
as in the previous section, shows where the partition went:

| `s_support` | null-space energy | `sigma` visible (null) | visible spread | `sigma` null part (null) | null-part spread |
|---|---|---|---|---|---|
| 0 | 0.23 | 0.880 (0.871) | 0.87 | 0.915 (0.912) | 0.91 |
| 1e-5 | 0.73 | 0.794 (0.709) | 0.71 | 0.074 (0.067) | 0.07 |
| 1e-4 | 0.90 | 0.759 (0.646) | 0.65 | 0.0084 (0.0123) | 0.01 |
| 1e-3 | 0.95 | 0.679 (0.568) | 0.57 | 0.0055 (0.0108) | 0.01 |

**The partition is in the invisible part.** The penalized arms move 73-95%
of atom energy into the null space, where each head's profile is a few
large coordinates disjoint from other heads'; those dominate `|W|`, so the
profile's cosine is small. The visible part, the only part reconstruction
sees, stays dense (spread 0.57-0.71, against about 0.03 for a partition at
`h = 32`) and overlaps *more* than its null as the weight rises (ratio
1.12-1.20 against 1.01 unpenalized). `sigma` on `|W|` is a gauge-dependent
measure whenever the decoder has a null space, and the penalty took the
gauge. The FVU cost is plausibly norm: under `norm_rows` the null spikes
take a fixed atom norm away from visible content, though that is not
measured here.

### A square decoder does not close the gauge

The same pilot with `--e-dec 768`, so the decoder is square and has no
exact null space (`data/out/gpt2_l8/sigma_pilot768`):

| `s_support` | held-out FVU | logged `support` | `sigma` dict (null) | spread | `sigma` decoded (null) |
|---|---|---|---|---|---|
| 0 | 0.432 | 4.92 → 4.79 | 0.878 (0.876) | 0.88 | 0.889 (0.822) |
| 1e-5 | 0.474 | 4.92 → 0.43 | 0.073 (0.063) | 0.06 | 0.872 (0.799) |
| 1e-4 | 0.526 | 4.91 → 0.055 | 0.0079 (0.0118) | 0.01 | 0.838 (0.782) |
| 1e-3 | 0.605 | 4.91 → 0.0068 | 0.0008 (0.0043) | 0.00 | 0.804 (0.755) |

The same table as at e_dec 1536, to within a few percent: below the
null from 1e-4 (ratio 0.67, 0.19), the same FVU cost (+10%, +22%, +40%),
decoded `sigma` 6-9% above its null throughout. The decoder grew a soft
null space in place of the missing exact one:

| `s_support` | decoder condition number | directions below 0.1 `s_max` | atom energy there | energy in a head's top 8 coordinates | per-head effective rank, dict / decoded |
|---|---|---|---|---|---|
| 0 | 1.0e4 | 472 of 768 | 0.43 | 0.02 | 24.7 / 21.8 |
| 1e-5 | 6.9e4 | 371 | 0.83 | 0.79 | 6.4 / 19.9 |
| 1e-4 | 5.9e4 | 322 | 0.93 | 0.94 | 2.0 / 17.3 |
| 1e-3 | 5.1e5 | 282 | 0.95 | 0.97 | 1.5 / 14.3 |

Each penalized head puts nearly all its atom energy on a few coordinates
of its own, in directions the decoder shrinks tenfold or more; its atoms
collapse to effective rank 1.5-2 in the dictionary space while their
decoded images keep rank 14-17. The content rides on the small
remainder, which the decoder amplifies, and that remainder overlaps
between heads as before. So a learned decoder makes `sigma` on the
dictionary weights gameable whatever its shape: removing the exact null
space only moves the exploit to the near-null one.

A penalty the gauge cannot satisfy has to be measured where the decoder
cannot rescale it: on the decoded atoms (`sigma` decoded, which no arm
has moved), or with `--direct`, where the dictionary space is the output
space and there is no decoder at all. Output coordinates are not
privileged axes, so a disjoint-coordinate partition there is a strong
and somewhat arbitrary constraint, but it cannot be faked.

The k = 128 SONAR runs already trained with the penalty, at `sonar.py`'s
default `s_support` 1e-6 (which replaced the retired `s_hcossim` 1e-6),
direct pair included. It is too weak to act: the logged `support` moves
from 4.85 to 4.82 (decoded) and 4.79 to 4.37 (direct) over 369,500
steps, and at the end all four runs sit at or above their nulls, with
dense profiles:

| run | `sigma` dict (null) | spread | `sigma` decoded (null) |
|---|---|---|---|
| `ste_k128_h54` | 0.973 (0.972) | 0.97 | 0.992 (0.954) |
| `ste_k128_h54_s43` | 0.971 (0.971) | 0.97 | 0.992 (0.954) |
| `ste_k128_h54_direct` | 0.894 (0.863) | 0.87 | = dict |
| `ste_k128_h54_direct_s43` | 0.890 (0.859) | 0.86 | = dict |

So the direct-atom test at a weight that moves `sigma` has not been run;
on GPT-2 the penalty took hold from 1e-5. Results:
`data/out/headsupport/headsupport_k128.json`.

## Per-layer decoders (`--per-layer-dec`): the shared basis is not what limits depth

Every layer's atoms are decoded by one shared `W_dec`, whose gradient comes
mostly from layer 0's error, so the deeper layers work in a basis fitted to
someone else's residual. `Ontologizer.per_layer_dec` gives each layer its
own decoder (one `Linear` over an `l * e_dec` latent in which layer i
writes only block i). Pilot on GPT-2, signed dictionary, 3 epochs
(46,250 steps), one seed, against the shared twin `sigma_pilot/s0`;
held-out prefix FVU after each layer from `codeuse.py`, with what that
layer adds in parentheses:

| decoder | layer 0 | 1 | 2 | 3 | 4 (final) | realized bits |
|---|---|---|---|---|---|---|
| shared, e 1536 | 0.721 (0.279) | 0.585 (0.136) | 0.510 (0.075) | 0.460 (0.050) | **0.423** (0.036) | 1052 (55%) |
| per-layer, e 1536 (decoder x5) | 0.728 (0.272) | 0.591 (0.138) | 0.517 (0.074) | 0.467 (0.049) | 0.429 (0.038) | 1022 (54%) |
| per-layer, e 320 (decoder parameters matched) | 0.768 (0.233) | 0.640 (0.128) | 0.573 (0.067) | 0.528 (0.045) | 0.501 (0.027) | 959 (51%) |

**Separate decoders do not help the deep layers.** At the same `e_dec`,
every layer adds what it added with the shared decoder, to within 0.007:
layers 3 and 4 add 0.049 and 0.038 against 0.050 and 0.036, and the
final FVU is 0.006 worse with five times the decoder's parameters. Matching
the decoder's parameters instead (`e_dec` 320, so each layer's dictionary
is 5x narrower) costs layer 0 most (0.233 against 0.279) and ends 0.078
worse. The small additions of the deep layers are not caused by a basis
fitted to layer 0.

Caveats: one seed, 3 epochs, codes at 51-55% of their nominal bits (the
3-epoch arms are partly trained), so differences under about 0.01 are not
read. The concat variant's coupling of head index across layers is not
tested here (these heads sum).

```bash
uv run python experiments/ste-arm/train_ste.py --base gpt2_l8 \
    --dict-init-scale 1.0 --signed --per-layer-dec --e-dec 1536 --epochs 3 \
    --out data/out/gpt2_l8/pld_pilot/pld1536     # --e-dec 320 for pld320
uv run python experiments/ste-arm/codeuse.py --temperature 1.0 \
    --model data/out/gpt2_l8/pld_pilot/pld1536 \
    --cache data/activations/gpt2_l8.npy \
    --mse-weights data/activations/gpt2_l8.mse_weights_matched.npy
```

## Head-independence pressure (`--s-hsic-heads`): a measurement artifact

**Everything in this section was chasing estimator bias.** The heads
were already independent; the statistic said otherwise because it was
biased, and the bias was larger than the quantity it was measuring.
Per-layer head CKA on `ste_h76`, the same codes scored both ways:

| layer | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| unbiased | 0.0041 | 0.0005 | 0.0005 | 0.0004 | 0.0002 | **0.0011** |
| biased | 0.0315 | 0.1062 | 0.1018 | 0.0963 | 0.0976 | 0.0867 |

The biased estimator reads chance co-occurrence between independent
categorical heads as dependence, and the floor grows as the batch
shrinks: at h=76 and k=32 it is 0.109 at b=256, 0.057 at 512 and 0.029
at 1024, against an unbiased 0.0001 at every size. At the training
batch the floor exceeded any real dependence, so the penalty had
nothing to descend but the bias. `hsic.pairwise_head_cka` now defaults
to Song et al.'s unbiased estimator, which agrees with explicit
pairwise evaluation to six digits and still reads 0.06 at 0.5 factor
correlation and 0.41 at 0.8.

That explains every result below, which are kept because the reasoning
that produced them is instructive and the failure was not where any of
it looked:

- The penalty could not reduce the biased CKA in training because most
  of that number is a floor the batch sets and the weights cannot
  reach. Under the unbiased estimator it reduces it readily, and there
  is then no dependence left worth reducing.
- It fell 56% when descended alone with MSE removed, which is the
  bias term being optimised by degrading the code in ways
  reconstruction correctly resists.
- Conditioning on language or script explains none of it (within-language
  0.0033, within-script 0.0023, against 0.0011 mixed -- noise around
  zero), so the correlated-factor hypothesis was never the binding
  question.

The finding worth keeping is the opposite of the one the arm was built
to test: **this architecture produces near-independent heads with no
pressure to do so**, at a CKA of 0.001 where 0.5 factor correlation
would read 0.06. Nothing needs to enforce it.

Before trusting any statistic as a training signal, measure it on a
null it should score zero on. `auxpull.py` answers whether a penalty
applies force; it cannot tell you the target is fictitious.

### The rerun under the unbiased estimator

`auxpull.py` now covers the penalty, and the two estimators are not
interchangeable as training signals. Same checkpoint, same batch, same
codes:

| estimator | statistic | grad at classifier | equal-pull w | applied at 1e-3 |
|---|---|---|---|---|
| biased | 0.434 | 0.221 | 7.9e-4 | 1.26 |
| unbiased | 0.006 | 0.053 | 3.3e-3 | 0.30 |

The biased estimator carries 4.2x the gradient. Its bias is not a
constant offset -- it depends on the heads' marginals -- so that excess
is force aimed at reshaping marginals rather than at reducing
dependence, and the arm trained against it spent most of its pull
there.

Finetuning the converged `ste_h76` for 5,000 steps, each at its own
equal-pull weight, against a control whose weight is low enough to
apply no force (1e-9, so 3e-7 of the objective's gradient) but nonzero,
so the statistic is still computed and logged:

| run | estimator | weight | CKA | shift | MSE |
|---|---|---|---|---|---|
| `ste_h76_hsic_ctrl` | unbiased | 1e-9 | 0.003264 -> 0.003250 | -0.6 se | +0.16% |
| `ste_h76_hsic_ftu` | unbiased | 3.31e-3 | 0.003189 -> 0.002895 | **-11.7 se** | +0.31% |
| `ste_h76_hsic_ft` | biased | 1e-3 | 0.445899 -> 0.444302 | -11.1 se | -0.17% |

(first and last 500 steps, as means of 100-step blocks so the standard
error absorbs the trace's autocorrelation. CKA is summed over the five
layers.)

**The penalty was never broken; the statistic was.** Under the unbiased
estimator it moves its own target 9.2% where an untreated control moves
0.4%, for a third of a percent of reconstruction. Under the biased one
it applied 1.26x the objective's own gradient and still moved the
statistic 0.4%, because most of that value is a floor set by the batch
rather than by the model, and no amount of force on the weights will
shift it.

**It still does not matter.** The unbiased statistic starts at 0.00064
per layer and ends at 0.00058, where 0.5 factor correlation reads 0.06.
The penalty is a working lever attached to nothing: the earlier
conclusion holds, for a sharper reason than "it cannot move it."

### What the arm measured, before that was understood

The head-level language probe found language spread thinly over layer 0
rather than held in one head, which is what a model with several
hundred heads and no reason to localize should do. `hsic_ste.py` adds
the mean pairwise CKA between heads' classifications as a penalty, so
that sharing a variable costs something. Two weights, otherwise the
configured arm:

| step | CKA at 1e-4 | CKA at 1e-3 | MSE at 1e-4 | MSE at 1e-3 |
|---|---|---|---|---|
| 5,000 | 0.083 | 0.084 | 5.20e-3 | 5.12e-3 |
| 13,000 | 0.216 | 0.195 | 6.21e-4 | 6.19e-4 |
| 17,000 | 0.382 | 0.361 | 4.55e-4 | 4.58e-4 |
| 21,000 | 0.415 | 0.405 | 4.10e-4 | 4.16e-4 |

The unpenalized arm converges at 0.434, so ten times the weight buys a
2% reduction. Two things are worth keeping from that.

**Heads acquire dependence as reconstruction converges.** CKA is near
0.08 while MSE is still falling fast and rises to its plateau between
steps 9,000 and 21,000. A penalty here has to prevent a rise, not undo
one, and no pilot shorter than ~20k steps can tell the weights apart --
everything looks fine before step 9,000.

**Loss-value share is not gradient share, and only the second selects a
weight.** At 1e-4 the penalty was 12% of the loss value, which reads
like a weak setting. Measured at the converged checkpoint, batch 256:

| gradient norm | MSE | CKA | ratio |
|---|---|---|---|
| classifier | 1.76e-4 | 2.21e-1 | 1250 |
| dictionary | 9.96e-4 | 2.21e-1 | 221 |

so the weight for equal pull on the classifier is 8e-4 and 1e-3 is
already balanced, not weak. At balanced pull the statistic does not
move, which says the penalty is a weak lever on this architecture
rather than a mis-scaled one: reducing head dependence costs more
reconstruction than the penalty saves, even one-for-one. Dominating it
needs ~1e-2, where the penalty's gradient is an order of magnitude over
the objective's; whether the resulting model is worth having is
untested.

The gradient ratio is the number to measure first for any new auxiliary
term here, and it is two `jax.grad` calls.

**Trained out to 380,200 steps at the balanced weight, it changes
nothing and costs localization.** Final CKA is 0.441 against the
unpenalized arm's 0.434 -- marginally higher, with MSE, usage balance
and row collinearity all within noise. And on the head-level language
probe it is the worst of the hard arms: best-head NMI 0.199 against
0.269, holding 3.1% of the head/language information against 4.2%.

That is the reverse of the hypothesis. The penalty was added because a
model with several hundred heads has no reason to concentrate a
variable in one of them, and making sharing costly should encourage
localization. Pairwise-CKA pressure neither reduced head dependence nor
localized anything; if anything it diffused language further. The
direction is closed unless a different dependence statistic is proposed.

## Seed reproducibility: the function is determined, the features are not

Two runs differing only in seed, both at the fixed initialization,
reach **identical** held-out error -- 0.1535 and 0.1535 for the stack,
0.2870 and 0.2871 for the flat arm. Nothing below that agrees. Four
measurements, each against its own null:

| what is compared | stack | flat | null |
|---|---|---|---|
| head partitions (NMI, `partition.py`) | 0.0121 | 0.0177 | 0.0048 |
| head contributions (cosine, centered) | 0.0074 | -- | 0.0015 |
| per-layer subspace, rank 8 | see below | -- | 1.0x chance |
| per-head atom Grams | 1.05-1.11x within-layer chance | -- | -- |

**Partitions.** The best-matched pair of heads across seeds reaches an
NMI of 0.42, but the mean is 0.012 against a 0.005 shuffled null. The
`splitting.py` entry-containment measure reads 0.000 at its standard
threshold, which is not a metric artifact: the same partition measure
scores 0.38 overall and 0.87 on layer 0 comparing one run against its
own checkpoint 19,500 steps earlier, so it detects agreement when there
is any.

**Contributions.** A head is also a vector-valued function, and two
heads could carve differently while adding the same thing. They do not:
centered cosine between matched contribution matrices is 0.0074 against
a 0.0015 null, with the best pair at 0.081. Uncentered the same
comparison reads 0.205 with 99.5% layer agreement, which is the
non-negative dictionary's shared offset and its per-layer gain, not
reproducibility.

**Subspaces, and why their agreement is not the architecture's.** Top-8
eigenspaces of the per-layer contributions agree 25x to 33x above
chance, falling to 1.1x by rank 128. But comparing each model against
the *cache's own* principal directions in the whitened output frame
gives the same numbers:

| layer | A vs B | A vs cache | B vs cache |
|---|---|---|---|
| 0 | 116.7 | 114.2 | 108.0 |
| 1 | 123.7 | 101.3 | 101.2 |
| 2 | 115.5 | 97.3 | 97.6 |
| 3 | 86.1 | 85.6 | 88.6 |
| 4 | 8.9 | 24.0 | 21.8 |

(overlap over chance at rank 8.) For layers 0 to 3 the models agree
with each other no more than each agrees with the data, so their
agreement is the data's dominant directions and nothing is left for the
architecture. Layer 4 fails the other way: each model tracks the data
about 23x above chance while agreeing with the other only 9x, so the
two seeds pick *different* data-aligned directions.

**Atom Grams reproduce, but there is almost nothing in them to
reproduce.** Per-layer row collinearity agrees to four digits across
seeds (0.2880/0.2891 at layer 0 down to 0.1226/0.1233 at layer 4), and
so do the mean spectra. But the spectrum is nearly one-parameter: the
top eigenvalue is exactly `1 + (k-1)*rowcos`, correlates with it at
1.0000 across all 380 heads, and carries 82% of the spectrum's norm.
That single dominant eigenvalue is forced by non-negativity, not
learned -- random `abs()`'d rows give rowcos 0.64 and 97% of the norm
in the top eigenvalue, where signed rows give 0.00 and 22%. So the Gram
has about one free parameter.

Comparing the Grams themselves rather than their spectra -- atoms put
in a canonical order by each one's mean cosine to the others in its own
head, then differenced off-diagonally -- puts individual agreement at
chance, against a null that pairs heads *within the same layer*:

| layer | across seeds | same run, 19.5k steps earlier |
|---|---|---|
| 0 | 1.11x | 1.15x |
| 1 | 1.05x | 1.07x |
| 2 | 1.06x | 1.07x |
| 3 | 1.06x | 1.07x |
| 4 | 1.07x | 1.11x |

The second column is the sharper result. One run against its own recent
checkpoint scores the same as a different seed, where the partition
measure separates those two comparisons thirtyfold (0.38 against
0.012). So a head's Gram barely moves across 19,500 steps of training
and barely moves across a seed: it is close to a fixed property of the
configuration rather than something learned, which is why its one free
parameter reproduces to four digits.

The table above is computed in `e_dec`, where 74-78% of each atom is in
the decoder's null space and so cannot be either learned or reproduced
(see the row-collinearity section). Decoding the atoms into output space
first removes that and changes nothing: 1.18x at layer 0 falling to
1.03x at layer 4. The conclusion survives, but it survived by luck
rather than design, and any future comparison of dictionaries should
decode before measuring.

Three figures from earlier passes were wrong and are worth recording as
traps. Against a whole-model random pairing the spectra read 35x, which
was entirely the layer signal, since layers differ in collinearity by
0.12 to 0.29 while heads within a layer differ by a standard deviation
of 0.0014. The spectral test at 1.5-3.0x was measuring a necessary
condition only -- different Grams can share a spectrum -- so it
overstated the direct figure by roughly twofold. And a first direct pass
recomputed its normalizer per layer, which produced ratios below 1 for
an assignment minimizing that same distance; impossible, and the only
reason it was caught.

What the Gram does measure is how far training drives the atoms apart
against the constraint: 0.64 at random, 0.29 at layer 0, 0.12 at layer
4. Monotone in depth, and real work.

**The one positive result is that coarse features reproduce better,
and the flat arm shows it cleanly.** With all 380 heads at one depth,
so depth cannot confound it, a head's matched agreement correlates with
its contribution size at **r = 0.963** (0.911 against the varying part
alone). In the stack the same relationship is confined to layer 0 at
0.41 and 0.53, and is absent in layers 1 to 4. Flat heads also
reproduce better overall, 0.0177 against the stack's 0.0121 on the same
null, which is what a seed-independent input predicts.

**So what reproduces is aggregate or layer-level:** the total error to
four digits, per-layer collinearity to four digits, and the data's
dominant directions. What does not reproduce is anything identifying an
individual head -- what it separates, what it contributes, the subspace
it works in beyond what the data forces, or its own internal geometry.
Feature-level claims about individual heads have no foundation in this
architecture as it currently trains, and every head-level result
elsewhere in this file inherits that.

Each of the four measures needed a control before it said anything, and
three of them reversed under one. Entry containment read 0.000 and looked
like a broken metric until the same-run checkpoint comparison showed it
detects agreement at 0.38 when there is any. Uncentered contributions
read 0.205 with 99.5% layer agreement, which was the non-negative
dictionary's shared offset. Subspaces read 25-33x above chance until the
cache's own directions turned out to score the same. And the Gram read
35x, then 1.5-3.0x, then chance. The pattern is consistent enough to
state as a rule: in this architecture, non-negativity and the
layer-dependent gain put structure into every measure's floor, so a
result without a null of the same shape is not a result.

Layer 0 is the exception in every head-level measure and in none of the
aggregate ones, which is consistent with it being the only layer whose
input does not depend on upstream choices.

### The same picture on GPT-2, concat (`partition.py`, `headcontrib.py`)

`gpt2_l8/ste_h20_cat128` and `ste_h20_cat128-43` differ only in seed:
5x20 heads, k=256, `ConcatDictBlock` at d_head 128, both at 371,900
steps, scored on the held-out cache tail. Each measure is run twice --
across seeds, and against the seed-42 run's own step-350,000 checkpoint
as the positive control. `headcontrib.py` is the head-contribution
measure from the table above, now a script: each head's decoded
output contribution in the whitened frame, centered over rows, matched
by cosine.

| layer | partition NMI, seeds | control | contribution cos, seeds | control |
|---|---|---|---|---|
| 0 | **0.359** | 0.976 | **0.171** | 0.956 |
| 1 | 0.188 | 0.766 | 0.017 | 0.758 |
| 2 | 0.186 | 0.602 | 0.007 | 0.588 |
| 3 | 0.186 | 0.538 | 0.005 | 0.516 |
| 4 | 0.186 | 0.510 | 0.004 | 0.483 |
| all | 0.221 | 0.679 | 0.041 | 0.660 |
| null | 0.179 | 0.179 | 0.0017 | 0.0017 |

(partition: 32,768 rows; contribution: 8,192 rows, null max 0.0030.)

**Only layer 0 reproduces, at under a fifth of the control.** Its heads
match into the other seed's layer 0 20 of 20 times on both measures.
Layer 1 keeps its identity -- 19/20 and 20/20 of its heads land in the
other seed's layer 1 -- while agreeing on almost nothing it separates or
writes. Layers 2-4 lose even that and scatter across the deep layers.
The two measures agree layer for layer, so it is not that heads carve
differently while writing alike, or the reverse.

The partition null is high (0.18) because plug-in NMI between two
256-way partitions of 32,768 rows is bias-dominated, so it cannot
resolve the deep layers; the contribution null is tight, and there they
sit 2-4x above it at about 1% of the control. Real, and negligible.

Against `ste_h76` on SONAR the excess over null is larger (contribution
24x null against 5x), but nearly all of it is layer 0. Coarse heads
reproduce better here too: matched cosine correlates with a head's
centered contribution size at r = 0.926 overall, mostly the layer
contrast, and 0.42 and 0.61 within layers 0 and 1.

Uncentered cosine is unusable for this, as on SONAR: the non-negative
dictionary's shared offset puts the row-shuffled null's maximum at
0.85.

```bash
uv run python experiments/ste-arm/headcontrib.py \
    --a data/out/gpt2_l8/ste_h20_cat128 --b data/out/gpt2_l8/ste_h20_cat128-43
uv run python experiments/ste-arm/headcontrib.py \
    --a data/out/gpt2_l8/ste_h20_cat128 --b data/out/gpt2_l8/ste_h20_cat128 \
    --step-b 350000
uv run python experiments/ste-arm/partition.py --cache data/activations/gpt2_l8.npy \
    --a data/out/gpt2_l8/ste_h20_cat128 --b data/out/gpt2_l8/ste_h20_cat128-43
```

### Is layer 0's reproducibility a size effect?

Layer-0 heads are the largest and the most reproducible, so "layer 0
reproduces" could just be "big heads reproduce". Size here is a head's
centered contribution energy as a share of the whitened target variance.

**Within one stack it cannot be decided.** On the GPT-2 concat pair each
layer's heads are nearly one size and the ranges do not overlap -- layer
0 at 0.62-0.94%, layer 1 at 0.28-0.32%, down to 0.07% at layer 4 -- so no
deep head is size-matched to a layer-0 one and every split is an
extrapolation. Additive on cosine, a layer-0 indicator alone gives R^2
0.958 and size adds 0.005 (about a tenth of the advantage); multiplicative
on log cosine fits better (0.982), with cosine roughly proportional to
size and layer 0 a further ~4x, which puts size at about half of the
layer-0/layer-1 gap. Within layer 0 the size slope is 13x the deep
layers'.

**The flat arm breaks the confound**, since all its heads see the raw
input with nothing downstream. SONAR, fixed init, both pairs 380 heads,
4,096 held-out rows, `headcontrib.py`'s centered cosine (null 0.002, max
0.011):

| flat head size | matched cos | n |
|---|---|---|
| 0.057-0.060% | 0.008 | 95 |
| 0.060-0.062% | 0.011 | 95 |
| 0.062-0.066% | 0.035 | 95 |
| 0.066-0.071% | 0.060 | 57 |
| 0.071-0.689% | 0.150 | 38 |

Correlation with log size is r = 0.876, and the two largest heads (0.69%,
0.44%) reach 0.78 and 0.56. Size matters a great deal on its own.

But the stack's heads fall far below that curve at every depth, layer 0
included:

| stack layer | size, median | stack cos | flat cos at that size (+-10%) | gap |
|---|---|---|---|---|
| 0 | 0.31% | 0.080 | 0.67 (n=2) | ~8x |
| 1 | 0.16% | 0.005 | 0.12 (n=1) | ~23x |
| 2 | 0.11% | 0.004 | 0.19 (n=3) | ~49x |
| 3 | 0.073% | 0.003 | **0.059 (n=115)** | **~19x** |
| 4 | 0.039% | 0.003 | below the flat range | -- |

**Size is about a third of layer 0's advantage, in log terms.** Layer 0
reproduces ~27x better than layer 3; the flat curve between their sizes
accounts for 3-4x of that. Only layer 3's match is solid -- flat heads
are almost all tiny, so layers 0-2 rest on one to three heads each, and
fitted curves extrapolate absurdly (a power fit predicts 2.4 for layer
0) -- so treat the fraction as rough and the direction as firm.

**The larger result is that the stack costs reproducibility at matched
size, at every layer.** Layer 0 receives exactly the flat arm's input
and still reproduces several times worse than an equal-size flat head:
in the stack, what a layer-0 head should carve depends on what the
layers after it do with its error, so the input alone no longer pins
it. Deeper layers lose more. This is the reproducibility face of the
entanglement result -- the cascade that makes a single-head
intervention leak into 59% of the other heads also leaves each head
underdetermined by the data.

Inside layer 0, size does most of the work. The SONAR stack's layer-0
heads are the one layer that spans a wide size range (0.27-2.3%, the
best-matched pair at cosine 0.84), and within it agreement tracks log
size at r = 0.914; layers 1-4 are near one size each and show nothing.
So the reading is: size decides which layer-0 heads reproduce, while
being in layer 0 at all -- the only layer with no upstream choices --
decides most of how far above the deep layers they sit (on log cosine,
size + layer 0 gives R^2 0.942 with the layer-0 term at 2.1, about 8x).

#### What size is made of: the layer's gain, not the head's atoms

A head's contribution on a row is its layer's input gain times the
decoded atom it selected, so its size is gain^2 times atom spread^2
(the usage-weighted RMS distance of its decoded atoms from their mean);
on both stacks that product reproduces the measured size exactly
(log correlation 1.000, median ratio 1.000).

| layer | GPT-2 gain | GPT-2 atom norm (1e-4) | SONAR gain | SONAR atom norm |
|---|---|---|---|---|
| 0 | 122.2 | 4.6 | 1.000 | 0.043 |
| 1 | 67.9 | 5.3 | 0.588 | 0.054 |
| 2 | 62.3 | 4.4 | 0.490 | 0.052 |
| 3 | 59.2 | 3.7 | 0.417 | 0.051 |
| 4 | 57.2 | 3.2 | 0.359 | 0.043 |

(decoded, whitened; medians over heads.)

**Layer 0's larger contributions are all gain.** It receives the raw
input -- GPT-2 activations of norm ~122 against residuals of ~60, unit
SONAR embeddings against residuals of 0.36-0.59 -- while its atoms are
no larger than layer 1's on either dataset, and on SONAR smaller than
layers 1-3's and equal to layer 4's. Across layers SONAR's atom norm even
correlates slightly negatively with log agreement (-0.21).

Gain is one number per layer per row, shared by every head in the
layer, so across layers "size" was the layer itself under another name.
The size paragraph above should be read that way: the part of layer 0's
lead that size accounts for is gain, which is a property of the layer
and not of any head. At equal atom spread the layer-0 lift is 14x on
GPT-2 and 23x on SONAR (log agreement on log spread plus a layer-0
indicator, R^2 0.979 and 0.941) -- larger on SONAR because its layer-0
atoms are the smaller ones.

**Within layer 0 the atoms are what matter.** SONAR's layer-0 gain is
identically 1, so its within-layer size variation is pure atom
magnitude, and log agreement tracks it at r = 0.62 (norm) and 0.64
(spread); GPT-2's layer 0 shows 0.17 and 0.42. Past layer 1 the atoms
predict nothing on either dataset. So a few genuinely large-atom
layer-0 heads are the reproducible ones; the flat arm's size curve,
whose heads all have gain 1, was an atom-magnitude curve too.

Spread and norm are one quantity here: their ratio is 0.99-1.08 in every
layer, so a head's usage-weighted mean decoded atom is near zero. The
non-negative dictionary's offset is cancelled by the decoder before it
reaches the output.

The SONAR stack repeats the GPT-2 layer pattern (0.080 at layer 0
against 0.003-0.005 below it), but the matched-size comparison is
SONAR-only; GPT-2 has no flat arm. The SONAR numbers here use the
training run's own `data/out/sonar/mse_weights.npy`. A first pass ran on
weights regenerated by `make_mse_weights.py` while that file was
missing, and the regeneration does not reproduce it -- correlation
0.992, but the largest weight is 77.8 against the original's 27.3 -- so
that script's recipe is not the one that produced the training weights.
Re-run on the originals, every conclusion here holds and the numbers
move in the third digit.

```bash
# size report, gain/atom decomposition and regressions, GPT-2
uv run python experiments/ste-arm/headcontrib.py \
    --a data/out/gpt2_l8/ste_h20_cat128 --b data/out/gpt2_l8/ste_h20_cat128-43
# against the flat arm at matched size, SONAR
uv run python experiments/ste-arm/headcontrib.py \
    --cache data/sonar_embeddings/mc4_4M.npy \
    --mse-weights data/out/sonar/mse_weights.npy --rows 4096 \
    --a data/out/sonar/multilingual/ste_h76_init01 \
    --b data/out/sonar/multilingual/ste_h76_i01_s43 \
    --ref-a data/out/sonar/multilingual/ste_l1_h380_i01 \
    --ref-b data/out/sonar/multilingual/ste_l1_h380_i01_s43
```

### Do a layer's entries contribute equally? (`entryshare.py`, `bigatoms.py`)

An entry writes only where it wins its head, and then writes its
layer's gain times its decoded atom, so its energy over the data is
`E[1{selected} * gain^2] * |atom|^2`. Per layer, over 32,768 held-out
rows:

| run | layer | gini | top 10% carry | max/median | log-var from atom norm | within heads |
|---|---|---|---|---|---|---|
| GPT-2 concat, k=256 | 0 | **0.587** | 42% | 38.7 | 76% | 60% |
| | 1 | 0.155 | 16% | 30.3 | 44% | 100% |
| | 2-4 | 0.11-0.12 | 14% | ~2 | ~40% | 100% |
| SONAR stack, k=32 | 0 | **0.326** | 32% | 46.3 | 92% | 83% |
| | 1-3 | 0.06-0.10 | 12-14% | 1.5-1.9 | 16-25% | ~100% |
| | 4 | 0.195 | 18% | 3.2 | 77% | 87% |
| SONAR flat, 1x380 | -- | 0.190 | 23% | 64.1 | 90% | 76% |

(log-var: covariance share of the variance of log entry energy; the
remainder is usage, with gain conditional on selection ~0. within
heads: the share of that variance not explained by which head an entry
is in.)

**Layers 1-3 are near-equal; layer 0 is not, and its atoms are why.**
Deep entries have an effective number of 95-99.5% of the total, the
largest about twice the median, and what inequality they have is mostly
usage, which is itself mild (CV 0.14-0.18). Layer 0's usage is just as
even -- on SONAR every entry fires 2.1-4.5% of the time against 1/k =
3.1% -- but its atom norms vary five to ten times more (CV 0.34-0.52
against 0.02-0.10), mostly inside heads. The flat arm shows the same
pattern more mildly, so it belongs to classifying the raw input. SONAR's
last layer is unequal for the other reason: some atoms have shrunk
almost to nothing (min/median 0.017), the inert final layer again. No
entry is dead anywhere.

Whole layers are unequal too, by gain: layer 0 holds 67% of entry energy
on GPT-2 and 49% on SONAR, layer 4 4-5%.

**Where the large atoms sit differs completely between the models.**

| | GPT-2 concat, 20 heads | SONAR stack, 76 heads |
|---|---|---|
| largest atom / head median | 3.7 (1.4-5.1) | 2.0 (1.5-4.4) |
| atoms > 3x layer median | 58, in **all 20** heads, at most 15 each | 11, in **4** heads |
| atoms > 5x | 1 | 5, **all in head 54** |
| largest head's energy share | 15.5% (3x uniform) | 9.2% (7x uniform) |
| Spearman(atom norm, usage) within heads | +0.81 | +0.27 |

On GPT-2 every head has a graded tail -- its 8th-largest atom is still
twice its median -- and the large atoms are its frequent outcomes (1.6x
uniform usage). Head 17 holds 15% of the layer's energy with a flat
profile, uniformly large rather than one outsized entry. Agreement is
fairly even across heads (0.135-0.235) and only loosely tied to atoms
(Spearman 0.49 with energy share, 0.34 with the largest atom).

On SONAR the large atoms are a few heads' property, and they fire at the
uniform rate: large, not frequent. Head 54 holds 8 of the 11 atoms above
3x and all 5 above 5x, carries 9% of the layer, and reproduces at
**0.844** against 0.249 for the next head. Agreement across heads tracks
the largest atom (Spearman 0.62) and how peaked the head's profile is
(0.62), not its typical atom (-0.14). That is the within-layer-0 size
effect of the section above, resolved: it is a few heads with outsized
atoms. It is not the layer-0 lift, though -- without head 54 the
layer-0 mean falls only from 0.080 to 0.070, still twenty times the deep
layers. Head 54 is also the topic/genre head: the same partition as the
shipped-init `ste_h76`'s h54, and the counterpart of seed 43's top head
(see "The topic head recurs across seeds" below).

The large atoms are distinct directions: different heads' largest atoms
are near orthogonal (median |cos| 0.022, below random atom pairs). On
SONAR they lean into the data's top 10 principal directions (9% of
their energy against 3% for a typical atom), consistent with cross-seed
agreement being the data's dominant directions; on GPT-2 they lean the
other way (4% against 13%).

One seed pair per dataset; thresholds are relative to each layer's own
median and do not compare across datasets.

```bash
uv run python experiments/ste-arm/entryshare.py --a data/out/gpt2_l8/ste_h20_cat128
uv run python experiments/ste-arm/bigatoms.py \
    --a data/out/gpt2_l8/ste_h20_cat128 --b data/out/gpt2_l8/ste_h20_cat128-43
# SONAR: add --cache data/sonar_embeddings/mc4_4M.npy
#        --mse-weights data/out/sonar/mse_weights.npy (bigatoms: --agree-rows 4096)
```

## Heads against a known factor: language (`headlang.py`)

Language is measured because it is the one factor with free labels, not
because a head was expected to encode it. SONAR is trained to be
language-agnostic -- translations land close together -- so language is a
weak residual factor of the embedding, and a reconstruction objective has
little reason to spend a head on it. This is a probe of whether
partitions line up with a known factor at all. Held-out language identity
over 86 languages, against a dense-embedding ridge ceiling of 0.721 and
chance of 0.012:

| arm | structure | best-head NMI | less null | probe m=1 | m=32 |
|---|---|---|---|---|---|
| ste_h76 | 5x76 | 0.269 | 0.261 | 0.165 | 0.378 |
| + kcos setpoint | 5x76 | 0.287 | 0.279 | 0.139 | 0.318 |
| - mean-entropy bonus | 5x76 | 0.085 | 0.078 | 0.038 | 0.322 |
| + head-independence | 5x76 | 0.199 | 0.189 | 0.097 | 0.383 |
| flat | 1x380 | 0.277 | 0.276 | 0.102 | 0.257 |
| flat + kcos | 1x380 | 0.247 | 0.246 | 0.080 | 0.271 |
| softmax reference | 5x32 | 0.055 | 0.043 | 0.036 | 0.275 |
| ste_h76_init01 (fixed init) | 5x76 | 0.180 | 0.170 | 0.103 | 0.423 |
| ste_h76_i01_s43 (fixed init, seed 43) | 5x76 | 0.165 | 0.155 | 0.089 | 0.428 |

The first seven rows are on the shipped initialization. **The fixed
initialization concentrates language less in any one head and carries
more of it across heads**: best-head NMI falls from 0.269 to 0.17-0.18
while the 32-head probe rises from 0.378 to 0.42-0.43, consistently
across both seeds. Their best language heads are L0 h46 and L0 h14 --
the latter seed 43's topic head (below), which picks up language as a
side effect of partitioning by topic.

**As expected, no head encodes language outright.** The best reaches an
NMI of 0.28 against an attainable ceiling of 0.875, and none of the
head-health levers changes that. That says little about the
architecture's claim: a language-agnostic embedding should not be
organized by language, and its strongest head is organized by topic and
genre instead (below). Testing whether heads are natural variables needs
labels for what the embedding does encode.

That ceiling is not 1.0, and an earlier version of this section said a
32-entry head "could hold all 6.43 bits", which is false: `log2(32)` is
5.00 bits against the label's 6.43, so **no single k=32 head can
represent 86 languages** and the question was partly asked of something
the architecture forbids. `nmi` normalizes by the arithmetic mean and
`I <= min(H_h, H_l)`, so the cap is `2 min / (H_h + H_l)` = 0.875 here.
A head of `k >= 128` is the smallest that could hold the label at all,
though there is no reason to expect one to. `headlang.py` had the same error in its reported ceiling
(`min(1, H_l/H_h)`, which prints 1.00 for these arms and is wrong in
both regimes); it is fixed.

**The hard code carries more of the known factor**: six times the
softmax reference on null-corrected best-head NMI on the shipped
initialization, and about four times (3.6-4.0) on the fixed one.

### Per-label cells, script, and conjunctions (`headscript.py`)

Three finer readings of the same claim on `ste_h76`, all on the cache
tail, all scored against a label-shuffled null (the maximum runs over
`l*h*k` cells, and precision is floored by a label's own share -- Latin's
raw best-cell F1 of 0.570 ranks highly at 1.02x enrichment, i.e. chance).

**Script is not the variable either.** Best-head NMI 0.191 against a
0.671 ceiling is 28.6% of attainable, against language's 30.3%, and the
same head (L0 h50) is best for both. Capacity was not the obstacle:
script is 2.52 bits inside a head's 5.00. Note the ceiling binds from the
other side here -- a 5-bit head is too FINE for a 2.52-bit label.

Most of that is not a new test: 18 of the 24 scripts are a single
language, so their cell is the language cell. The two real group labels
split -- Cyrillic (11 languages) has a genuine cell at recall 0.697 and
4.7x enrichment, Latin (53 languages, 62% of rows) sits at 1.02x.

**A single cell is a strong but permissive detector.** Best cell per
language: recall 0.74-0.98 at FPR 1.6-2.2%, 26-36x enrichment, precision
0.30-0.42. 9/86 languages reach recall 0.8 and 0/86 reach precision 0.8.
Precision 0.8 at these recalls needs FPR 0.0015 against 0.026 observed,
so the shortfall is real -- 17x -- but smaller than raw precision implies,
since 86 near-balanced classes make the negative class 85x larger.

**Conjunctions do not fix it, because the heads share their mistakes.**
Top-m cells from distinct heads (entries within a head are mutually
exclusive, so a same-head conjunction is empty by construction):

| m | AND prec | AND rec | AND F1 | FPR | if independent |
|---|---|---|---|---|---|
| 1 | 0.195 | 0.512 | 0.282 | 0.0250 | - |
| 2 | 0.433 | 0.234 | 0.303 | 0.0036 | 0.00062 (5.8x) |
| 3 | 0.584 | 0.102 | 0.174 | 0.00086 | 0.000016 (55x) |
| 4 | 0.598 | 0.048 | 0.088 | 0.00038 | 4e-7 (974x) |

Precision triples while recall falls tenfold; F1 peaks at m=2, 7% above a
single cell, and the best precision anywhere is 0.598 at 4.8% recall. OR
is strictly worse (recall 0.512 -> 0.779, precision 0.195 -> 0.041, F1
monotonically down).

Were the false positives independent, m=2 would give precision 0.83 and
m=3 0.99. Recall DOES combine nearly independently (0.234 against 0.262
predicted), so true positives are near-independent while false positives
are shared: a common set of confusable rows that many heads send to the
same wrong language, which the cell compositions show directly (`fi`'s
cell carries `et` at 0.28, `ja`'s carries `ko` at 0.29, `ja-Latn`'s
carries `ja`, `en`'s carries `en-multi`). The errors are systematic
language confusions rather than noise, so the code looks like it encodes
language NEIGHBOURHOOD rather than identity.

This is not in tension with the head-independence section below: that
concerns independence of the head codes, which says nothing about whether
their errors against an external label coincide.

### Why, from the embedding geometry (`embedgeom.py`, `headeta.py`)

SONAR embeddings are unit-normed but not centred -- `||mean|| = 0.5637`,
so the cloud is a spherical cap and 0.318 of every squared norm is a
shared direction. They are dense without being isotropic: effective
dimension 108 of 1024, yet 943 PCs for 99% of variance.

Language explains 6.51% of embedding variance and script 2.26%. The
tempting conclusion -- that an MSE objective ignores a 6.5% factor -- is
**wrong**, and `headeta.py` is the control that kills it. Corrected for
the number of classes (a random partition into `g` classes scores
`(g-1)/(n-1)`):

| partition | classes | eta^2 | x baseline |
|---|---|---|---|
| best head L0h54 | 32 | 0.1112 | 235x |
| best language head L0h50 | 32 | 0.0432 | 91x |
| script | 24 | 0.0226 | 64x |
| language | 86 | 0.0658 | 51x |
| layer-0 median head | 32 | 0.0080 | 17x |
| median head | 32 | 0.0025 | 5x |
| layer-4 median head | 32 | 0.0010 | 2x |

Language is a *stronger* partition than 90% of the heads, so neither
salience nor capacity is the obstacle. What is: languages are weakly
separated regions, not directions. Within-language mean cosine is 0.3596
against 0.3160 between, and the shared mean direction alone accounts for
0.3178 of that -- so the real cluster signal is the +0.0436 excess, 0.30
sd of the pair distribution. A partition code cuts along local residual
variance, and at 0.30 sd those cuts pass through the clusters rather than
around them, which is what produces cells that fire on a language plus its
neighbours with confusions shared across heads.

The baseline uses the nominal 32 entries for every head. Dividing by the
entries a head actually uses instead promotes shipped-init heads that use
few entries (L0 h44, 10 of 32, reads 584x against h54's 251x at 65,536
rows), so the ranking to read is raw eta^2, on which h54 leads (0.111
against h44's 0.080).

Two asides from the same table. The heads are radically unequal, 235x for
the best against 5x for the median. And the layer-4 heads are
near-random partitions at 2x, which is `lastlayer.py`'s workless final
layer arrived at from a different direction.

### What the strongest head holds (`decodehead.py`)

L0 h54, the 235x head, is not a language head: NMI 0.149 against language
where h50 reaches 0.265, 30 of 32 entries used at near-uniform load, and
each entry a broad language mixture -- 16.1 distinct languages among an
entry's top-20 activating rows, where a language feature would
concentrate.

Decoded, it reads as topic and genre: crime news (e8, e10), geopolitics
and parliament (e7, e18, e21), encyclopedic place description (e3, e19),
manufacturing (e4, e23), marketing and customer service (e9, e27),
legislative text (e25), entertainment and gaming (e13, e20). Re-decoded
at SONAR_NORM after the encoder fix (`decodehead.py --deviation`,
2026-10-07), every entry reads the same except e19, which now decodes
to generic text.

**That reading is half the head, not all of it.** `autointerp.py --mode
cacts` has an LLM describe each entry from its top-activating TEXTS, and
scores the descriptions by blinded detection, which is the only one of the
three modes that works (see below). By claim type, with the mode's 0.273
null as the floor:

| what the description claims | n | mean F1 |
|---|---|---|
| genre / topic | 16 | 0.523 |
| multilingual or code-switched passages | 8 | 0.476 |
| text artifact (duplication, mojibake) | 3 | 0.395 |
| script or region | 5 | 0.394 |

So genre/topic is the largest group and the best described, but a quarter
of the head keys on documents that MIX languages, and three entries on
encoding damage. All four beat the null, so none is a judge artifact --
the multilingual ones in particular were the suspicious case and they
score nearly as well as genre.

The topic half still explains the profile that prompted the look: the
highest variance explained of any head, because topic dominates sentence
embeddings, and the lowest language NMI among structured heads, because
topic is near-orthogonal to language. Monosemantic features are there and
the labeled variable we had was the wrong one -- but "a topic head" was
too clean a summary.

### The topic head recurs across seeds (`topichead.py`)

Everything above is one head of one run, and the seed section says no
per-head result survives a seed change. This one does. Partition NMI of
h54 against its best match in each other run, against the median NMI of
the other heads at the same index, on 65,536 held-out rows:

| run | top layer-0 head by eta^2 | eta^2 | next head | h54 matched in it |
|---|---|---|---|---|
| ste_h76 (shipped init, seed 42) | h54 | 0.111 | 0.080 | -- |
| ste_h76_init01 (seed 42) | h54 | 0.117 | 0.062 | h54, NMI 0.355 (others 0.024) |
| ste_h76_i01_s43 (seed 43) | h14 | 0.122 | 0.068 | h14, NMI 0.413 (others 0.010) |

**The topic head is the one head found in every run.** Fixing the
initialization kept it at the same index (same seed, so the same
classifier initialization); a different seed grows it at a different
index, and there too it is the strongest head by nearly a factor of two.
It agrees with the large-atom analysis, where head 54 alone reproduced
at 0.844 on matched contributions. So the head
the language-agnostic embedding should be organized by is both its
strongest partition and the only one the architecture finds reliably --
the first per-head result here that does not inherit the seed caveat.
Its NMI across seeds (0.41) is still well short of the run-to-own-
checkpoint control on layer 0 (0.87), so it is the same variable carved
differently, not the same partition.

```bash
uv run python experiments/ste-arm/topichead.py --model \
    data/out/sonar/multilingual/ste_h76 \
    data/out/sonar/multilingual/ste_h76_init01 \
    data/out/sonar/multilingual/ste_h76_i01_s43
```

**Atoms must be decoded as deviations from the head's mean atom.** An
atom's cosine to that mean is 0.849, so decoding atoms raw returns one
shared sentence 32 times with the details shuffled -- the same "constant
part dwarfs its varying one" as the decoder-sparsity section, at atom
level. `--deviation` subtracts it. Anything resting on `decode_tags.py` is
reading the template rather than the feature, since it decodes raw
atoms.

`autointerp.py`'s `params` mode has the same flaw and worse: it decodes a
code with all `l*h` heads at the measured origin and one forced, so the
origin dominates. `--mode pdev` subtracts the pure-origin decode.
**Legibility is all that buys, though.** Scored by blinded detection over
h54's 32 entries:

| mode | self F1 | null F1 | delta |
|---|---|---|---|
| cacts (LLM reads activating texts) | 0.479 | 0.273 | +0.206 |
| pdev (origin-subtracted decode) | 0.079 | 0.021 | +0.058 |
| params (raw decode) | 0.000 | 0.011 | -0.011 |

The two decode rows were rescored 2026-10-07, decoded at SONAR_NORM after
the encoder fix (first run: pdev 0.052 / 0.014, params 0.031 / 0.000).
The first campaign's harvest was not kept, so they come from a fresh
harvest of the same 32 entries at default settings
(`autointerp.py harvest --feature-ids 1728..1759`, then `texts`,
`describe --mode params` / `pdev` and `score --null`, in
`data/out/sonar/autointerp_h54/`); the cacts row is the first campaign's.

#### Per-latent detection F1 is mostly a density measure

The campaign in `data/out/sonar/autointerp` scores six runs in all three
modes (no null control recorded -- the old `scores.csv` has no `kind`
column -- so these are raw self F1):

| run | density | freq | acts | cacts | n |
|---|---|---|---|---|---|
| m5120_k32 | 0.6% | 0.61% | 0.645 | 0.603 | 250 |
| m5120_k32_bl | 0.6% | 2.62% | 0.578 | 0.470 | 75 |
| m5120_g160top1 | 3.1% | 5.08% | 0.646 | 0.594 | 150 |
| m11264_k5120 | 45.5% | 17.97% | 0.470 | 0.332 | 256 |
| m5120_g160softmax | 100% | 0.00% | 0.449 | 0.364 | 256 |
| onto (5x32, T=0.03) | 100% | 10.80% | 0.473 | 0.392 | 225 |

Read naively this says the Ontologizer is 0.21 below the plain top-k SAE
on cacts, 7.9 sd. **It mostly says the Ontologizer's code is denser.**
Pearson(log nominal density, acts F1) is -0.894 across the six runs, so
density alone carries ~80% of the variance; sparse runs (<5%) average
cacts 0.556 against 0.363 for dense ones (>=40%).

Fitting `cacts ~ a + b log(firing rate)` on the four SAE runs with nonzero
firing rate (slope -0.069 per log unit, r = -0.766) and holding `onto`
out predicts 0.421 against an actual 0.392 -- a residual of **-0.029, or
-1.8 sem**. `onto` sits on the SAE trend. `m5120_g160top1` is the real
outlier at +0.120 above it, which is what makes it look like a damning
control: it is 160 heads of 32 with top-1 per head, the Ontologizer's
own categorical code in single-layer form, and unusually describable for its
density.

Caveats, because the control is thin: the fit has n=4 and a weak r;
`g160softmax` is excluded because its firing rate is exactly 0, and it is
the run most like `onto` in being a softmax code; there is no null
control; and `freq` is thresholded at `2/k`, which is non-monotone in true
density -- it goes to zero for a maximally diffuse code as well as a
maximally sparse one, which is exactly why `g160softmax` reads 0.00%.

The lesson for the benchmark is the durable part: per-latent detection F1
cannot compare architectures at unmatched density, and every Ontologizer
arm is dense by construction.

`params` is at zero, so that mode should be dropped or read only as prose.
It does **not** invalidate a published number: the `0.646 vs 0.645`
per-latent describability figure in `categorical/` is SAE-vs-SAE
(`g160top1` against `k32`) and `replicate.sh` scores it in `--mode acts`,
which never touches the parameter decode. No Ontologizer autointerp score
has been published at all -- `experiments/partition-autointerp` is written
to consume one (`data/out/sonar/autointerp/onto/scores.csv`, `cacts`, on
`resid_nc`) that is not on disk.

`pdev` is also at the floor: a decoded sentence is an INSTANCE of what
an entry writes, and detection needs a criterion to sort snippets by. Read
`pdev` to see what an entry holds; do not read either decode mode as
evidence that an entry is describable. Only `cacts` supports that, and its own null is
high (0.273) because top-activating rows differ from random ones in
surface register, exactly as `autointerp.py`'s header warns.

`info_share_best` in `summary.json` -- the best head's share of the
summed head/language information -- is comparable only between arms of
the same shape. It divides by a sum of uncorrected plug-in mutual
informations over all heads, and the per-head bias tracks how many
entries a head uses, which the null NMI column reports: 0.008 for the
5x76 arms against 0.001 for the flat ones, whose heads are mostly
binary. Across architectures it reverses the probe's verdict, and the
probe is the one without a bias term.

## Steering at the model's own magnitude, against SAEs (`steerembed.py`)

The steering tables above sweep a unit direction over fixed strengths and
rescale the result to |x|. That compares directions at equal budget but
never at the step a real intervention applies. `steerembed.py` now also
scores two things:

- **`native`**: for the Ontologizer, x + (Y_forced - Y_base); for an SAE,
  x + a_j W_dec[j], with a_j latent j's mean nonzero activation. There is
  no rescale, and the reported strength is |delta|/|x|.
- **`native_random`**: a random direction of the same per-row length, as
  the control.

It also scores `sae.py` checkpoints:

- **Runs with heads** (`--groups`, i.e. `g160top1` and `g160softmax`):
  collateral is the share of other heads whose winner changed, the same
  measure as for the Ontologizer. These are not SAE baselines but one-layer
  Ontologizers with linear classifiers, and are reported as simplified
  variants.
- **SAEs proper** (top-k, L1): collateral is the share of the row's other
  active latents that drop out. That is a different measure, and under
  top-k it has a floor of 1/(k-1).

Strength is now a fraction of each row's norm, which changes nothing on
unit-norm SONAR and makes GPT-2 (|x| ~ 122) comparable.

Fixed strength 0.25, the models whose collateral is per head:

| model | decode realized | decode collateral | random collateral |
|---|---|---|---|
| hard-code stack, `ste_h76_init01` | 0.365 | 0.657 | 0.655 |
| hard-code flat, `ste_l1_h380_i01_hm1e4` | **0.996** | 0.328 | 0.368 |
| flat, shipped init (`ste_l1_h380`, earlier run) | 0.966 | 0.086 | 0.093 |
| softmax stack, `sweep_softmax_shm` | 0.769 | 0.566 | 0.559 |
| `g160top1` | 0.582 | **0.026** | 0.232 |
| `g160softmax` | 1.000 | 0.301 | 0.336 |

At native magnitude:

| model | step p50 | realized | collateral | random collateral |
|---|---|---|---|---|
| hard-code stack | 0.129 | 0.077 | 0.536 | 0.531 |
| hard-code flat | 0.028 | 0.169 | 0.043 | 0.051 |
| softmax stack (`_shm`) | 0.064 | 0.189 | 0.250 | 0.248 |
| `g160top1` | 0.186 | **0.410** | **0.014** | 0.176 |
| `g160softmax` | 0.050 | 0.536 | 0.067 | 0.077 |
| `m5120_k32` | 0.072 | 0.456 | 0.062* | 0.070* |
| `m11264_k32` | 0.076 | 0.500 | 0.077* | 0.087* |
| `m11264_k160` | 0.030 | 0.187 | 0.033* | 0.036* |
| `k32_bl` | 0.078 | 0.815 | 0.036* | 0.044* |
| L1 3e-5 | 0.008 | 0.080 | 0.007* | 0.008* |

\* SAE collateral, which has no head axis to count; compare only within
these rows.

**The flat arm's clean addressability was its collapsed heads.** On the
shipped initialization it used 25% of its bits, so its heads had few
entries and few boundaries to cross. With every entry in use, the same
step moves a third of the other heads, not a tenth. The flat arm still
realizes almost everything, and the cascade still doubles collateral (0.33
against the stack's 0.66), so entanglement is part of the story rather
than all of it.

**Collateral is set by step size, not direction, in every Ontologizer.**
Decode and random steps of the same length disturb the same share of other
heads, at fixed strength and at native magnitude. `g160top1` is the only
code whose decode direction is targeted: a tenth of random's collateral.

**At native magnitude the hard-code stack barely steers.** It realizes 7.7%
against 1.6% for random, while moving 54% of other heads, exactly what
random does. A forced entry's own output change hardly moves the classifier
toward that entry. Fixed-strength sweeps flattered SAEs too: at their
activation scale their steps are 3-8% of |x|, and realization falls from
~1.0 to 0.19-0.50 for the top-k SAEs (the bilinear-encoder variant, 0.82,
is the exception).

`k5120` cannot be scored: of its 11,264 latents, 9,660 never fire and 1,090
always do, so none has rows to steer.

The softmax reference in the earlier steering table is `sweep_softmax_shm`
(its old output is under `~/flock`); rerun here it reads 0.769 at 0.25
against the earlier 0.824, a different 60-entry sample.
`sweep_softmax` reads 0.609.

## Supervised steering vectors on the same targets (`steerembed.py`)

The steering comparisons above pit each model's own directions against a
random step. A reviewer asked the right question: the method is pitched
as unsupervised steering, so it should be compared with the supervised
steering vectors it aims to replace. `steerembed.py` now fits, for each
target, the two standard ones -- the difference of means between rows
that select the target and rows that do not (activation addition), and a
class-balanced logistic probe's weight vector -- on the 8,192 reference
rows with the steered rows held out, and scores them with the same
realization and collateral, on the same targets and rows (seed 42
reproduces the earlier runs' targets; the model's own directions agree
with the earlier tables to within sampling).

The labels are the model's own selection, so this is an ORACLE baseline
for reaching the model's categories: how well a supervised method that
already knew the partition would steer into it. It is not a test against
supervised steering of an external concept (language, topic); that needs
labelled targets and is the next version.

Fixed strength 0.25 of |x|, realized / collateral (SAE collateral is the
share of a row's other active latents that drop out; compare within a
row only):

| model | own decode | diff. of means | probe | random |
|---|---|---|---|---|
| softmax stack (`sweep_softmax_shm`) | **0.752** / 0.568 | 0.464 / 0.460 | 0.469 / 0.505 | 0.019 / 0.560 |
| hard-code stack (`ste_h76_init01`) | 0.370 / 0.656 | **0.910** / 0.628 | **0.939** / 0.640 | 0.021 / 0.654 |
| hard-code flat (`ste_l1_h380_i01_hm1e4`) | 0.996 / 0.328 | 0.974 / **0.274** | 0.995 / 0.313 | 0.005 / 0.368 |
| `g160top1` | 0.582 / **0.026** | 0.936 / 0.131 | 0.947 / 0.142 | 0.036 / 0.232 |
| `g160softmax` | 1.000 / 0.301 | 0.985 / 0.270 | 1.000 / 0.309 | 0.012 / 0.336 |
| SAE `m5120_k32` | 0.999 / 0.187 | 0.908 / 0.190 | 0.975 / 0.209 | 0.000 / 0.213 |
| SAE `m11264_k32` | 0.999 / 0.214 | 0.918 / 0.214 | 0.974 / 0.234 | 0.002 / 0.247 |
| SAE `k32_bl` | 1.000 / 0.066 | 0.970 / 0.122 | 0.992 / 0.111 | 0.013 / 0.126 |

At each method's own step (`native`: the model's intervention; `native_dm`:
the difference of means at its own length), realized / collateral, with
the median step as a fraction of |x| and each control's collateral:

| model | own (step) | diff. of means (step) | random collateral, own / dm |
|---|---|---|---|
| softmax stack | 0.194 / 0.251 (0.064) | 0.179 / 0.247 (0.079) | 0.249 / 0.298 |
| hard-code stack | 0.076 / 0.536 (0.129) | **0.266** / 0.465 (0.064) | 0.531 / 0.482 |
| hard-code flat | 0.169 / 0.043 (0.028) | **0.540** / 0.117 (0.087) | 0.051 / 0.170 |
| `g160top1` | 0.410 / **0.014** (0.186) | 0.570 / 0.036 (0.053) | 0.176 / 0.070 |
| `g160softmax` | 0.536 / 0.067 (0.050) | 0.565 / 0.126 (0.107) | 0.077 / 0.162 |
| SAE `m5120_k32` | 0.456 / 0.062 (0.072) | **0.765** / 0.146 (0.180) | 0.070 / 0.165 |
| SAE `m11264_k32` | 0.500 / 0.077 (0.076) | **0.824** / 0.167 (0.172) | 0.087 / 0.194 |
| SAE `k32_bl` | 0.815 / 0.036 (0.078) | 0.881 / 0.078 (0.121) | 0.044 / 0.074 |

- **The softmax stack's own direction beats the supervised vectors** at
  reaching its own categories at equal budget (0.75 against 0.46-0.47),
  the only model where it does by a margin; at native magnitude the two
  tie (0.19 against 0.18 at similar steps).
- **The hard-code stack's own direction is the problem, not its
  categories.** Difference of means and the probe reach the same entries
  on 91-94% of rows where the decode direction reaches 37%, at the same
  collateral. Its categories are linearly reachable; its decode delta does
  not point at them, consistent with `steergeom.py`'s finding that a forced
  entry's output change barely moves the classifier.
- **`g160top1` trades realization for precision.** Its decode direction
  reaches fewer targets than the supervised vectors (0.58 against 0.94)
  but disturbs a fifth as many other heads (0.026 against 0.13); every
  other model's own direction is at random's collateral.
- **Supervised vectors are only modestly more targeted.** Difference of
  means sits about 0.1 below random's collateral in the softmax stack
  (0.46 against 0.56), the flat arm (0.27 against 0.37) and `g160top1`
  (0.13 against 0.23), 0.07 below in `g160softmax`, and at random's level
  in the hard-code stack and the SAEs. Knowing the partition buys some
  precision but not much; the side effects are still mostly a property of
  the step's length.
- **For SAEs the latent's decoder row is already as good as the oracle**
  (0.999 against 0.91-0.97 at equal budget). At native magnitude difference
  of means realizes more (0.77-0.82 against 0.46-0.50), but it takes
  2.3-2.5x longer steps, so that is budget, not direction.

## Why collateral is direction-blind (`steergeom.py`)

`steergeom.py` linearizes each head's logits in the target's layer, with
every upstream code held fixed, so the cascade is excluded by construction.
It reports:

- **off/rand**: how much a unit step moves the other heads' logits, as a
  ratio to a random direction's.
- **flip**: the share of other heads whose winner-vs-runner-up margin a
  step of 0.25 |x| would close, to first order.
- **margin**: the median of that margin.

All layers:

| model | off/rand, decode | flip, decode (random) | median margin |
|---|---|---|---|
| hard-code stack | 0.97 | 0.30 (0.30) | 5.9e-4 |
| hard-code flat | 0.91 | 0.24 (0.26) | 1.5e-4 |
| k=2 binary stack, `ste_k2_h380` (shipped init) | 1.12 | 0.11 (0.14) | 2.0e-2 |
| softmax stack, `_shm` | 0.65 | 0.41 (0.42) | 3.7e-3 |
| softmax stack | 0.87 | 0.38 (0.41) | 5.2e-3 |
| `g160top1` | 0.54 | **0.02** (0.19) | 1.8e-3 |
| `g160softmax` | 0.88 | 0.23 (0.24) | 3.1e-2 |

On `g160top1` the linearized flip share (0.027 on a 16-target check) matches
`steerembed.py`'s measured collateral (0.026), so first order explains it.

**No Ontologizer's decode direction avoids the other heads.** Its
direct-path flip rate equals random's in every arm, so collateral is
direction-blind even without the cascade.

**Avoiding the other heads' logits is not enough.** The `_shm` softmax
stack's decode direction moves other heads' logits at 0.65 of random and
still flips them at random's rate. What matters is the change in each head's
winner-runner-up margin, and only `g160top1` keeps it small. Its decoder
row is at cosine 0.91 to its own encoder column, yet moves other heads'
logits at half random's rate where the encoder column moves them at twice.
`g160softmax` has none of this: decoder rows near copies of encoder columns
(0.98), and off-head change at random's rate. The two differ only in the
selection rule, so winner-take-all training is what aligns them; why is
untested.

**The hard-code Ontologizers' margins are tiny**, 1.5e-4 at layer 0 of the
stack and in the flat arm, so nearly every head sits on a boundary. Binary
heads have margins a hundred times larger and the lowest flip rate of any
Ontologizer; at fixed strength 0.25 the k=2 stack moves 17-19% of other
heads where the k=32 stack moves 66%. That arm is on the shipped
initialization, so the comparison is confounded.

```bash
uv run python experiments/ste-arm/steergeom.py \
    --model data/out/sonar/multilingual/ste_h76_init01
uv run python experiments/ste-arm/steergeom.py \
    --model data/out/sonar/sae_conv/m5120_g160top1/params.npz
```

## The router on GPT-2 (`ste_h20_cat128-sc`, `-sc43`)

`ste_h20_cat128` and `-43` are 5 x 20 heads, k = 256, `ConcatDictBlock` at
d_head = 128 on GPT-2's layer-8 residual. `-sc` and `-sc43` are the same
recipe and seeds with the per-head router (`scaled`, sigmoid-gated). All
four ran 371,750-371,900 steps.

**Analysis scripts were dropping the router.** Every script that replays a
layer under a held or forced classification rebuilt its output as
`combine(hfwd(P))`. That omits the router's per-head gain, and fibers, and
so computes a different model's residual for every layer past the first.
They now call `DictEnc.head_outputs(U, P)`, which applies both as
`withClusts` does, with a test pinning the two equal
(`tests/test_head_outputs.py`). The scripts covered are `pareto.py`,
`steerembed.py`, `steergeom.py`, `headcontrib.py`, `partition.py` and
`headlang.py`. On unrouted models the output is unchanged: the plain
pair's agreement reproduces to four digits. `autointerp.py`'s probe
(`onto_probe`, which `headstruct.py --onto`, `splitting.py` and
`refit.py` consume through `onto_acts_fn`), `textfid.py` and `steerfid.py`
were migrated on 2026-10-06, with `tests/test_autointerp.py` pinning the
probe to the forward, followed by `auxpull.py`, `blendablate.py`,
`codeuse.py`, `gainablate.py`, `initscale.py` and `inputgeom.py` here
(none had been run on a routed model), and `freeze_diag.py` and
`rl_reinforce.py`, whose older probes also skipped the constant
coordinate, the gain-shape split and the gain; they now call `onto_probe`
(see `experiments/layer4-freeze/notes.md`). On `ste_h20_cat128-sc` the old probe's assignments
agreed with the forward's on 2.2%, 1.3%, 0.8%, 0.7% of rows in layers 1-4
(512 tail rows); no recorded analysis had run them on a routed model. `headcontrib.py`'s atom
decomposition assumes contribution = gain x atom, which the router breaks,
so it is skipped for routed models.

| | plain 42 | plain 43 | router 42 | router 43 |
|---|---|---|---|---|
| held-out FVU_w, hard = soft | 0.4208 | 0.4195 | **0.4108** | **0.4114** |
| contribution agreement, mean (null) | 0.041 (0.0017) | | **0.091** (0.0027) | |
| partition NMI, mean (null) | 0.221 (0.179) | | 0.202 (0.179) | |

The router buys 2.4% of FVU and doubles mean contribution agreement. By
layer:

| layer | plain cos | router cos | router size (median share of variance) | router NMI | router control cos |
|---|---|---|---|---|---|
| 0 | 0.171 | **0.244** | 1.66% | 0.259 | 0.986 |
| 1 | 0.017 | **0.124** | 0.64% | 0.192 | 0.918 |
| 2 | 0.007 | **0.081** | 0.31% | 0.187 | 0.843 |
| 3 | 0.005 | 0.003 | **0.000%** | 0.187 | 0.571 |
| 4 | 0.004 | 0.002 | **0.000%** | 0.187 | 0.555 |

Control: the same run against its own checkpoint at step 350,000.

**The router switched off the last two layers.** Their heads' centered
contributions are zero to the precision reported, in both seeds. Whether
the router gates closed or the atoms shrank is not yet checked. The routed
model is effectively three layers
deep, reconstructs slightly better than the five-layer plain one, and its
remaining layers' heads are each larger: 1.66% of target variance at layer
0, against 0.79% for the plain stack.

**Layers 1-2 now reproduce in what they write.** Agreement there rises
seven- to twelvefold, and their heads land in the other seed's same layer
(17/20). What heads carve does not follow: partition NMI is no higher than
the plain pair's past layer 0, and at layer 0 it is lower (0.259 against
0.359). The likeliest reading is that the router's continuous per-head gain
is the reproducible part, and that gain is shared input-dependent magnitude
rather than a reproducible partition; that is untested. Agreement tracks
head size within the routed model (log cosine on size, R^2 0.96), so larger
heads may be all it takes.

Steering, fixed strength 0.25 and native:

| | decode realized / collateral | grad realized | random collateral | native step p50 | native realized / collateral | native random collateral |
|---|---|---|---|---|---|---|
| plain 42 | 0.173 / 0.654 | 0.611 | 0.635 | 0.077 | 0.011 / 0.437 | 0.404 |
| plain 43 | 0.168 / 0.656 | 0.571 | 0.636 | 0.079 | 0.008 / 0.435 | 0.401 |
| router 42 | 0.260 / 0.601 | 0.532 | 0.633 | 0.036 | 0.018 / 0.215 | 0.210 |
| router 43 | 0.240 / 0.608 | 0.470 | 0.634 | 0.034 | 0.018 / 0.232 | 0.228 |

On GPT-2 the classifier's own gradient direction steers three times better
than the decode direction, the reverse of SONAR's aggregate ranking. At
native magnitude almost nothing is realized in either model (1-2%), with
collateral at random's level. The router halves native collateral only
because its native steps are half as long; their 10th percentile is 0.000,
consistent with targets in the switched-off layers. In `steergeom.py` every
GPT-2 model's decode flip share equals random's (0.32-0.34 against
0.31-0.33). Seed twins agree to within 0.07 in every steering condition.

```bash
G=data/out/gpt2_l8; C=data/activations/gpt2_l8.npy
W=data/activations/gpt2_l8.mse_weights_matched.npy
uv run python pareto.py --ckpt $G/ste_h20_cat128-sc --cache $C \
    --mse-weights $W --sae --ms 1 --temperature 1.0 --b 1024
uv run python experiments/ste-arm/steerembed.py --model $G/ste_h20_cat128-sc \
    --cache $C --temperature 1.0 --ref-b 128
uv run python experiments/ste-arm/steergeom.py --model $G/ste_h20_cat128-sc \
    --cache $C --temperature 1.0
uv run python experiments/ste-arm/headcontrib.py \
    --a $G/ste_h20_cat128-sc --b $G/ste_h20_cat128-sc43
uv run python experiments/ste-arm/partition.py --cache $C \
    --a $G/ste_h20_cat128-sc --b $G/ste_h20_cat128-sc43
```

The routed models need smaller batches than the plain ones: `pareto.py` at
its default batch of 4096 and `steerembed.py`'s eigendecomposition at
`--ref-b 512` both ran out of memory.

## k = 128, and atoms without a decoder (`--direct`)

`direct` is ported from the `headline` branch. Dictionary atoms live in
the output space (e_dec = d_out), there is no decoder, and `decode` is the
identity, so every atom is literally a direction in embedding space. It
requires `--signed`, since `abs()`'d atoms in output space could only add
along the positive orthant. `migrate_spec` now maps a `headline` checkpoint
that used `direct` onto the field, rather than refusing it.

Pilots, 3,000 steps each at the fixed initialization's recipe, 5 x 54 heads
at k = 128 (1,890 bits, matching the k = 32 stack's 1,900):

| run | MSE at step 3000 | KL_m | speed | GPU memory | 24 epochs |
|---|---|---|---|---|---|
| k = 128, latent dictionary + decoder | 2.81e-4 | 2.96 | 7.1 it/s | ~6.1 GB | ~14.5 h |
| k = 128, direct | **2.37e-4** | 2.97 | 9.4 it/s | ~4.2 GB | ~11 h |
| k = 32 stack, `ste_h76_init01`, at 3000 | 2.75e-4 | 0.74 | | | |
| k = 32 flat, at 3000 | 2.22e-4 | 0.10 | | | |

Both train cleanly. The initialization fix works for both: a signed
dictionary's layer-0 output starts at 7.3, against 30 for `abs()` rows,
since signed rows partly cancel, and both are rescaled to the 0.1 target.
The direct pilot was ahead of the decoded one and the k = 32 stack at
matched step, which at under 1% of a run ranked nothing.

### Converged: twice the k = 32 stack's error

Both variants at seeds 42 and 43, 24 epochs (369,500 steps), scored on
the held-out tail at T = 0.00015 with the training MSE weights. Per-layer
prefix FVU from `codeuse.py`, against the k = 32 stack `ste_h76_init01`:

| run | layer 0 | 1 | 2 | 3 | 4 (final) | realized bits |
|---|---|---|---|---|---|---|
| `ste_h76_init01`, 5x76, k = 32 | 0.537 | 0.372 | 0.266 | 0.195 | **0.1535** | 1895 (99.7%) |
| `ste_k128_h54` | 0.657 | 0.476 | 0.364 | 0.328 | 0.3207 | 1866 (98.7%) |
| `ste_k128_h54_s43` | 0.658 | 0.477 | 0.365 | 0.331 | 0.3238 | 1865 (98.7%) |
| `ste_k128_h54_direct` | 0.656 | 0.480 | 0.367 | 0.321 | 0.3119 | 1868 (98.8%) |
| `ste_k128_h54_direct_s43` | 0.655 | 0.481 | 0.369 | 0.325 | 0.3152 | 1868 (98.8%) |

**At matched bits, k = 128 is twice as far off as k = 32**, and 6.2-6.5x
the 1890-bit Gaussian reference (0.0499) against 3.1x. Both fill their
codes, so this is not utilization. It is behind at every layer, and the
gap widens with depth: layers 3 and 4 add 0.034-0.046 and 0.007-0.010,
against 0.071 and 0.042 at k = 32. Per layer the k = 128 code sums 54
atoms rather than 76; for an additive code the number of summed vectors
may matter more than the size of each codebook, though nothing here
separates that from the other difference: the temperature, learning
rate and `s_Hm` were tuned at k = 32 and carried over. The direct variant
is ahead of the decoded one by 0.009 at both seeds, which is three times
the seed spread (0.003); the decoder buys nothing.

**Seed reproducibility is lower than at k = 32.** `partition.py`, seed 42
against 43 within each variant:

| pair | mean matched NMI | null | max | same layer (null) |
|---|---|---|---|---|
| decoded, 42 vs 43 | 0.0638 | 0.0590 | 0.186 | 64% (22%) |
| direct, 42 vs 43 | 0.0640 | 0.0591 | 0.220 | 53% (22%) |
| decoded vs direct, both seed 42 | **0.1800** | 0.0590 | 0.432 | **100%** (27%) |

NMI at k = 128 has a larger chance floor than at k = 32 (0.059 against
0.005), so read the excess: +0.005 here, against +0.007 for the k = 32
pair. Layer 0 is again the only layer above the rest (0.076-0.078), and
the depth order reproduces loosely, as at k = 32.

The third row is the striking one. **Two different architectures from the
same seed share their partitions far more than one architecture across
seeds**: every head matches the head at its own index (100% same layer;
the other heads' same-index median NMI at layer 0 is 0.35), falling from
0.375 at layer 0 to 0.080 at layer 4. The seed sets the classifier
initialization, which has the same shape in both variants; whatever
carves the partitions is fixed largely by it, not by the dictionary or
whether a decoder follows it. That is consistent
with the seed section's conclusion -- the function is determined by the
data, the features by the initialization -- and sharpens it: the
partitions are not even a property of the architecture.

**A strongest layer-0 head recurs, more weakly.** `topichead.py`, keyed on
seed 42's top head by eta^2:

| variant | seed 42 top head (eta^2) | seed 43 top head (eta^2) | best match across seeds | other heads' same-index median |
|---|---|---|---|---|
| decoded | h34 (0.083) | h19 (0.103) | h19, NMI 0.132 | 0.038 |
| direct | h22 (0.094) | h19 (0.091) | h6 (seed 43's second, 0.088), NMI 0.198 | 0.039 |
| k = 32 stack | h54 (0.117) | h14 (0.122) | h14, NMI 0.413 | 0.010 |

In each variant the strongest head's best match across seeds is a top-two
head of the other seed, so a dominant partition recurs, but its
agreement is a third to a half of the k = 32 topic head's and its eta^2
0.7-0.9x. Whether it is the same topic-and-genre variable has not been
checked (no decoding or auto-interp). Seed 42's top heads in both variants
(h34, h22) are the decoded run's top two and the direct run's first and
second -- the same-seed sharing again.

```bash
M=data/out/sonar/multilingual; W=data/out/sonar/mse_weights.npy
uv run python experiments/ste-arm/train_ste.py --h 54 --k 128 \
    --temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3 \
    --s-hm 1e-4 --dict-init-scale 0.1 --signed --direct \
    --out $M/ste_k128_h54_direct          # --seed 43 for the _s43 replica
uv run python pareto.py --ckpt $M/ste_k128_h54 --temperature 0.00015 \
    --mse-weights $W --ms 1 --b 1024 --out $M/ste_k128_h54/pareto
uv run python experiments/ste-arm/codeuse.py --model $M/ste_k128_h54 \
    --temperature 0.00015 --mse-weights $W
uv run python experiments/ste-arm/partition.py --a $M/ste_k128_h54 \
    --b $M/ste_k128_h54_s43 --temperature 0.00015
uv run python experiments/ste-arm/topichead.py --temperature 0.00015 \
    --head 34 --model $M/ste_k128_h54 $M/ste_k128_h54_s43
```

Logs: `experiments/logs/eval_k128/`.

## Classifier geometry from the weights (`bilinspec.py`)

2026-10-07. Per (layer, head, entry), the signed cos(w, v) of the
bilinear pair over the semantic block of the classifier input, the
odd share (the constant coordinate's linear term, as a share of the
logit's variance for isotropic unit u), and rho = |a| / |w_s|. Eight
geometrically spaced checkpoints per run. Closed-form spectra match
dense `eigvalsh` to 1e-14. Random-pair |cos| at d = 1024 is about 0.025.

Final checkpoint, median |cos| [IQR], share of features with dominant
eigenvalue negative ("neg-dom"), median odd share, median rho_w:

| layer | `ste_h76_init01` (369.5k) | | | | `resid_nc` (372.6k) | | | |
|---|---|---|---|---|---|---|---|---|
| | \|cos\| | neg-dom | odd | rho_w | \|cos\| | neg-dom | odd | rho_w |
| 0 | 0.535 [0.50–0.56] | 98% | 0.72 | 0.032 | 0.638 [0.48–0.72] | 96% | 0 (no const) | — |
| 1 | 0.096 [0.07–0.14] | 98% | 0.68 | 0.026 | 0.938 [0.92–0.95] | 99% | 0.96 | 0.178 |
| 2 | 0.109 [0.08–0.14] | 99% | 0.67 | 0.025 | 0.799 [0.73–0.83] | 93% | 0.99 | 0.211 |
| 3 | 0.155 [0.12–0.19] | 100% | 0.61 | 0.022 | 0.517 [0.38–0.59] | 87% | 0.99 | 0.242 |
| 4 | 0.262 [0.22–0.30] | 100% | 0.48 | 0.018 | 0.176 [0.10–0.28] | 63% | 1.00 | 0.227 |

- **The softmax stack goes rank-1; the hard stack mostly does not.**
  `resid_nc` layer 1 is the `(u·x)²` regime lineage §6 described for
  `resid_nc_hm` (86% of features above 0.9), and layers 1–3 sit at
  0.5–0.94. In `ste_h76_init01` only layer 0 leaves the random-pair
  floor far (0.54); layers 1–3 stay at 0.10–0.16, near-orthogonal
  factors, which is a saddle `(w·x)(v·x)` that sees each projection's
  sign.
- **Alignment is learned late, after the anneal.** In both runs every
  layer is at the random floor at 10k–20k steps. `resid_nc` departs
  from 30k and climbs to the end; `ste_h76_init01` stays at the floor
  until about 80k (layer 0) and 130k (the rest).
- **The dominant eigenvalue is negative almost everywhere** (c < 0):
  the rank-1 features are `−λ(u·x)²`, high where the input is
  orthogonal to u, not aligned with it.
- **resid_nc's constant coordinate carries weight; ste_h76_init01's
  barely does.** rho_w is about 0.2 in `resid_nc` layers 1–4, so the
  constant dominates a factor for inputs within cosine 0.2 of w_s,
  which is most of a 1024-d sphere; in `ste_h76_init01` it is 0.02–0.03.
  The odd share is near 1 in `resid_nc` but is inflated by
  construction: an isotropic projection is only ~|w|/√d, so a small
  constant already dominates that reference measure. On data the share
  depends on how inputs align with w_s, which this script does not
  measure. Sign-blindness of `resid_nc`'s rank-1 layers is therefore
  broken by the constant at least for weakly aligned inputs.
- One seed per run; weights only, no data.

Outputs: `<run>/bilinspec/` (`bilin_ecdf.png`, `bilin_steps.png`,
`bilin_heads.png`, `bilinspec.npz`, `bilinspec.csv`).

```bash
uv run python bilinspec.py --ckpt data/out/sonar/multilingual/ste_h76_init01
uv run python bilinspec.py --ckpt data/out/sonar/multilingual/resid_nc
```

## What one head's removal does to the decoded text (`textstrip.py`)

2026-10-07, after the SONAR encoder fix, decoded at SONAR_NORM (see the
encoder section below; a first pass at the 3-sentence scale, 0.203,
reached the same conclusions). 12 random eval-tail rows (seed 42), the 2
heads per layer whose uniform ablation moves x_hat furthest, so 120 (row,
head) cells per model; one seed and checkpoint each (`resid_nc` step
372.6k at t=0.03, `ste_h76_init01` step 369.5k at t=0.00015). Every
condition is decoded and scored by dNLL: the decoder's per-token NLL of a
reference decode under the condition's embedding, less under the
reference's own (decode(x_hat) for ablations and the null, decode(x) for
recon, prefixes and a head's write alone). chrF between decodes is not
usable for comparisons: greedy decoding flips on rounding-level
differences. At the last layer, where the live and frozen ablations are
one embedding up to float32 rounding (max |diff| 1.2e-3 and 4.5e-4),
5/24 (`resid_nc`) and 3/24 (`ste_h76_init01`) pairs decode to different
text (9/24 and 3/24 at the 0.203 scale).

Machinery: the probes reproduce the model's forward to 7e-4 / 1e-5 abs;
per-head contributions sum to x_hat to 6e-4 / 3e-4; a live uniform ablation
equals `withArgs` with `h_unif` (unit test). Two runs with identical inputs
agree on 779/780 decodes, max dNLL difference 0.011.

Mean dNLL (nats/token), SE over cells (cells share rows, so the SE is
optimistic):

| condition | `resid_nc` | `ste_h76_init01` |
|---|---|---|
| recon vs decode(x) | 0.022 ± 0.008 | 0.694 ± 0.105 |
| prefixes x̂_1..x̂_4 vs decode(x) | 0.218 ± 0.044 | 1.385 ± 0.120 |
| uniform ablation, live | 0.015 ± 0.003 | 0.151 ± 0.018 |
| uniform ablation, frozen | 0.071 ± 0.011 | 0.042 ± 0.012 |
| zero ablation, live | 0.015 ± 0.003 | 0.150 ± 0.018 |
| null step, frozen's length | 0.076 ± 0.013 | 0.028 ± 0.007 |
| head's write alone | 3.296 ± 0.106 | 3.253 ± 0.107 |
| head's mean write alone | 3.375 ± 0.113 | 3.314 ± 0.113 |

Live minus frozen, by layer of the ablated head:

| layer | `resid_nc` live / frozen / null | `ste_h76_init01` live / frozen / null |
|---|---|---|
| 0 | 0.019 / 0.270 / 0.274 | 0.395 / 0.200 / 0.111 |
| 1 | 0.012 / 0.026 / 0.050 | 0.218 / 0.003 / 0.012 |
| 2 | 0.000 / 0.012 / 0.011 | 0.104 / 0.000 / 0.001 |
| 3 | 0.027 / 0.028 / 0.026 | 0.033 / 0.005 / 0.005 |
| 4 | 0.019 / 0.019 / 0.019 | 0.004 / 0.004 / 0.008 |

- **Removing what a head wrote costs no more text than a random step of
  the same size.** Frozen ablation minus null: −0.005 (`resid_nc`, worse in
  53% of cells) and +0.015 (`ste_h76_init01`, 56%). The null steps along
  the difference of two random tail rows at the frozen delta's whitened
  length (median 6.5% and 3.4% of x_hat). So at this resolution, the text
  damage of an ablation is set by its size, not by what the head carried.
- **Downstream layers repair a soft ablation and amplify a hard one.** In
  `resid_nc` the live ablation does less damage than the frozen one
  (0.015 vs 0.071, live worse in 30% of cells; layer 0: 0.019 vs 0.270),
  as later layers re-read the residual and re-classify toward the
  original. In `ste_h76_init01` it does more (0.151 vs 0.042, live worse
  in 78% of cells; layer 1: 0.218 vs 0.003): later hard classifiers flip,
  and the flips add error. This fits the residual cascade amplifying
  disturbance in the depth results, seen here in text.
- **A head's write alone carries almost none of the sentence.** Its own
  write scores 0.06–0.08 nats better than its average write over 1024 tail
  rows (better in 54% of cells), against about 3.3 nats for either.
- The hard code's reconstruction is far from x in decode likelihood
  (0.694 vs 0.022), consistent with its FVU.

Outputs: `<run>/textstrip/` (`strips.html`, `strips.jsonl`, `summary.json`,
`meta.json`).

```bash
uv run python textstrip.py --ckpt data/out/sonar/multilingual/resid_nc --device cuda
uv run python textstrip.py --ckpt data/out/sonar/multilingual/ste_h76_init01 --device cuda
```

## The SONAR encoder ran with dropout outside `encode_corpus.py`

Found 2026-10-07; fixed in `ontologize/data/pretrained.py`, which now
returns the encoder in eval mode. Its SONAR branch builds `M2M100Encoder`
directly, so the module started in training mode with the config's
dropout 0.1. `encode_corpus.py` and `experiments/fresh-eval/encode_fresh.py`
called `.eval()` themselves, so the embedding caches are unaffected. Every
other caller put only the decoder in eval mode, so its encodes were
stochastic:

- **Decode scale only:** `textfid.py`, `autointerp.py`,
  `experiments/task-naturalness/naturalness.py` and
  `experiments/partition-autointerp/headinterp.py`. Each rescales an
  embedding to `ref_norm` before decoding, and `ref_norm` came from the
  encoder: 0.247–0.258 across processes with dropout, 0.2030 without. So
  their decodes were made at a scale that changed from run to run.
- **Decode scale and re-encoded text:** `steerfid.py`,
  `experiments/steering-overlay/steer_overlay.py` and
  `experiments/rl-classifications/rl_reinforce.py`'s cycle tier. Each also
  re-encodes generated text to measure its effect, so those activations
  carried dropout noise. The recorded REINFORCE result used the
  in-embedding tier, which never decodes, and is unaffected.
- **Encoded text:** `headcoh.py`'s description view,
  `experiments/ste-arm/decodehead.py`, and the `decode.py` /
  `decode_tags.py` REPLs.

### The decode scale

`ref_norm` was the mean-pooled norm of three short English reference
sentences, and that is not the scale the decoder was trained on. The
cache's rows are L2-normalized, so their own norms are gone; re-encoding
2048 mC4 corpus texts with the eval-mode encoder gives a raw mean-pooled
norm of median 0.307 (SD 0.040, 5–95% 0.235–0.365; per-language medians
0.30–0.34). Decode-then-re-encode fidelity follows it: on 128 tail rows,
cos(x, cycle(x)) is 0.240 decoding at 0.15, 0.338 at 0.203, 0.352 at
0.25, 0.376 at 0.30 and 0.374 at 0.40. So the dropout-inflated scale
(about 0.25) happened to sit closer to the right one than the
deterministic 3-sentence value (0.203). Every script now rescales to
`textfid.SONAR_NORM = 0.307`, one shared constant, and `SonarDecoder`
no longer loads the encoder at all; `decode.py` still rescales to its
input text's own norm, which is correct now that the encoder is in eval
mode.

### Reruns (2026-10-07)

Results that sit in their own sections are updated there: the round trip
("Steering, scored in embedding space" above),
`experiments/steering-overlay/notes.md` and
`experiments/task-naturalness/notes.md`. The rest, the earlier value in
parentheses:

**Text fidelity** (`textfid.py`, 512 tail rows, the SAEs in
`data/out/sonar/sae_conv`, byte-identical to the checkpoints the first
runs used). The NLL ceiling is now one value, 1.560 for every model, and
the corpus-mean floor 3.999 (chrF 0.176, earlier about 0.20):

| model | chrF | exact | NLL of decode(x) under recon |
|---|---|---|---|
| m5120_k32 | 0.228 (0.235) | 0.000 | 3.156 |
| m5120_k32_p5 | 0.229 (0.229) | 0.000 | 3.178 |
| m11264_k32 | 0.230 (0.232) | 0.000 | 3.160 |
| m5120_k32_bl | 0.204 (0.214) | 0.000 | 3.365 |
| m11264_k160 | 0.282 (0.285) | 0.000 | 2.434 |
| m5120_g160top1 | 0.275 (0.275) | 0.000 | 2.407 |
| resid_nc, soft | 0.552 (0.538) | 0.070 (0.090) | 1.585 |
| m5120_g160softmax | 0.591 (0.592) | 0.094 (0.119) | 1.568 |
| m11264_k5120 | 0.841 (0.856) | 0.562 (0.592) | 1.560 |

**Description-text coherence** (`headcoh.py --desc-mode cacts`, the same
campaign descriptions, encoded in eval mode): permutation z of
within-head description similarity is +2.03 for `resid_nc` (+1.92),
+1.55 for `g160top1` (+2.42) and +1.09 for `g160softmax` (+0.79). The
hard variant no longer clears 2 and the Ontologizer now has the largest
z; on the plain `acts` descriptions `resid_nc` reads +0.14. The
decode-direction z-scores in the same runs are unchanged, as they never
touch the encoder.

**Parameter-space decodes** (`autointerp.py describe --mode params`,
`eig` for the bilinear SAE, re-decoded at SONAR_NORM from the campaign's
own harvests into `data/out/sonar/autointerp_scalefix/`, then self-scored
by the Haiku judge without a null, as the first campaign was): mean
detection F1

| run | params | eig |
|---|---|---|
| onto (resid_nc) | 0.039 (0.020) | |
| m5120_k32 | 0.139 (0.158) | |
| m5120_k32_bl | 0.136 (0.061) | 0.034 (0.021) |
| m5120_g160top1 | 0.111 (0.039) | |
| m5120_g160softmax | 0.086 (0.025) | |
| m11264_k5120 | 0.010 (0.001) | |

Three SAE-family runs now read 0.09–0.14 where the first campaign had
them at or under 0.06, so "at most 0.06 everywhere except the plain SAE"
no longer holds. The conclusion does: activation-based descriptions of
the same features score 0.45–0.65.

The sweep's text fidelity and `steerfid.py --n-features 64` (the
selection sweep, the straight-through arms and now the k32 SAE) are in
"Downstream: textfid and steerfid" above; the text strips, the
`decodehead.py` reading of L0 h54 and h54's `params`/`pdev` scores in
their own sections; the head-level pass in
`experiments/partition-autointerp/notes.md`.

What changed a conclusion: the straight-through arm's text-cycle
steering is now indistinguishable from random (hit 0.055 against a
random 0.125), forcing a classification on `resid_nc_hm` moves decodes
0.64x as far as supervised steering rather than 0.43x, the hard
variant's description coherence no longer clears its null, and three
SAE-family `params` scores rose past 0.06. Everything else held to within
its noise: text fidelity, the round trip, naturalness, the s_Hm
confound, the text strips and both head-level lenses.

## Run

See [`README.md`](README.md) for training and scoring commands.

## Caveats

- The temperature is hand-matched to a logit scale measured once, and
  that scale drifts as the classifier trains. The principled version
  normalizes the logits per head, or controls T against a measured
  `spread/T` setpoint, and needs a package change.
- The configuration sweep under "The calibration" is one seed per arm
  at 4000 steps, which is early. It ranks configurations; it does not
  predict final quality.
- `p_revive` works under `ste` only since `k_sel = 1` there: the guard
  had read a non-`top<k>` rule as dense and returned immediately. Dead
  entries are recoverable without it anyway, since the softmax surrogate
  gives every entry gradient.
