# ste-arm

A straight-through (hard-code) Ontologizer arm, and the instruments built to
measure it. The results, condensed, are in [`findings.md`](findings.md);
the full lab notebook, with every measurement, superseded reading and
command, is [`notes.md`](notes.md). The writeup's account is
`writeup/sections/results.tex` and its appendix.

## Motivation

The softmax Ontologizer's argmax entries are not its code. `pareto.py`'s argmax row for
`sweep_softmax_shm` is whitened FVU 8.7e6 against 0.0007 for the same model's
soft forward, so the tag assignment that interventions and `decode_tags.py`
act on is not what reconstructs the embedding. Under `select="ste"` the
forward pass *is* the argmax, and the discrete code is the model.

A hard code also transmits exactly `l * h * log2(k)` bits, and reverse
water-filling on the whitened SONAR tail puts the best distortion any 800-bit
code can reach at FVU 0.226. The softmax architecture spends exactly 800, so a
discrete ontology needs more heads: this arm's default is 5 x 76 heads at
k = 32, 1,900 bits. Heads are the cheap axis -- bits and parameters are both
linear in `h`, and heads run in parallel where layers are sequential.

A hard forward needs its own calibration (`notes.md`, "The calibration"):
the argmax is scale-invariant, so nothing pushes the logits up, and noise,
surrogate temperature and the mean-entropy weight all have to be set relative
to that.

## Layout

`train_ste.py` imports a base configuration module and reads every unset
constant from it, so an arm cannot drift from its base; only the flags it is
given are overridden. `--base sonar` (default) is the live SONAR run;
`--base gpt2_l8` (`gpt2_l8.py`) overrides only the widths, cache, output
directory and MSE weighting for GPT-2's layer-8 residual stream.

Several flags replace `Hyperparams` with a subclass defined here, and those
subclasses do not compose except where noted:

| module | flag | what it changes |
|---|---|---|
| `initscale.py` | `--dict-init-scale` | rescales each dictionary at init so a layer writes that fraction of its input (the "fixed initialization") |
| `dictwd.py` | `--dict-wd` | weight decay on the dictionary alone; subclasses `initscale`, so the two compose |
| `layerft.py` | `--train-layer`, `--init-from` | trains one layer with the rest frozen |
| `whiten_enc.py` | `--zca` | installs a fixed ZCA whitener (`make_zca.py`) as the encoder |
| `hsic_ste.py` | `--s-hsic-heads` | pairwise head-independence penalty (unbiased CKA) |

Architecture flags pass straight to `Ontologizer`: `--h/--k/--l`, `--signed`
(no `abs()` on the dictionary), `--concat --d-head D` (`ConcatDictBlock`),
`--direct` (entries in output space, no decoder; needs `--signed`),
`--scaled` (the per-head router), `--forward`, `--encoded`.

The remaining scripts are read-only analyses of trained checkpoints. All load
through `pareto.load_onto`, so legacy and `headline`-branch specs migrate.
Scripts that replay a layer under a held or forced classification do it with
`DictEnc.head_outputs`, which applies the router and fibers as the forward
does.

| question | script |
|---|---|
| **reconstruction and capacity** | |
| where the code's capacity goes, per prefix and per head | `codeuse.py` |
| what the per-layer gain carries outside the code | `gainablate.py` |
| what the residual cascade contributes (blend toward unpaired references) | `blendablate.py` |
| how concentrated each classifier's input is | `inputgeom.py` |
| what the final layer contributes | `lastlayer.py` |
| which auxiliary losses do anything, at what weight | `auxpull.py` |
| the layer-0 logit scale against the temperature and the training noise, at initialization and over training | `logitscale.py` |
| **dictionary geometry** | |
| within-head row geometry from the weights | `dictgeom.py` |
| whether heads occupy disjoint coordinates (support overlap) | `headsupport.py` |
| whether `ConcatDictBlock` changes the non-negative correlation floor | `concat_floor.py` |
| whether entries in a layer contribute equally | `entryshare.py` |
| how a layer's large atoms are spread across its heads | `bigatoms.py` |
| **reproducibility** | |
| do two models learn the same partitions, in the same layers | `partition.py` |
| do two models' heads write the same thing; what head size is made of | `headcontrib.py` |
| do two checkpoints' heads hold the same atoms, whatever order each stores its entries in | `atomgram.py` |
| do two runs compute the same function or only reach the same error; Ontologizers and SAEs | `reconseeds.py` |
| is the strongest layer-0 head the same head across runs | `topichead.py` |
| **what heads encode** | |
| language identity, per head and per probe budget; whether the top heads are complete | `headlang.py` |
| script, per-label cells, conjunctions | `headscript.py` |
| embedding variance explained by each head's partition | `headeta.py` |
| language geometry of the SONAR embedding | `embedgeom.py` |
| one head's entries decoded to text | `decodehead.py` |
| whether auto-interp F1 tracks code density | `aidensity.py` |
| **steering** | |
| realization and collateral of an input-space step, at fixed and native magnitude, against isotropic and data-shaped random steps, and split into the data's high- and low-variance subspaces; Ontologizers and SAEs | `steerembed.py` |
| whether a direction avoids the other heads' boundaries, to first order | `steergeom.py` |

Every script documents its own measurement, nulls and caveats in its
docstring; `-h` prints it.

## Usage

```bash
# the shipped SONAR arm at the fixed initialization (fresh dir; resumable)
uv run python experiments/ste-arm/train_ste.py \
    --temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3 \
    --s-hm 1e-4 --dict-init-scale 0.1 \
    --out data/out/sonar/multilingual/ste_h76_init01

# k = 128 at matched bits (5 x 54 x 7 = 1,890), with and without a decoder
uv run python experiments/ste-arm/train_ste.py --h 54 --k 128 \
    --temperature 0.00015 --lr 1e-5 --noise-k batchnorm --sd-k 0.3 \
    --s-hm 1e-4 --dict-init-scale 0.1 --signed --direct \
    --out data/out/sonar/multilingual/ste_k128_h54_direct

# a short configuration pilot: exposes only the first N steps of the cache
# and holds out the rest, eval tail included
uv run python experiments/ste-arm/train_ste.py ... --steps 3000

# the GPT-2 model organism
uv run python experiments/ste-arm/train_ste.py --base gpt2_l8 --dict-init-scale 1.0

# score: held-out whitened FVU, hard and soft rows
uv run python pareto.py --ckpt <dir> --temperature <the arm's T>
# GPT-2: add --cache data/activations/gpt2_l8.npy
#        --mse-weights data/activations/gpt2_l8.mse_weights_matched.npy --sae

# across seeds: atoms modulo entry order, and the reconstruction itself
# (--a/--b of reconseeds.py may also be two sae.py params.npz)
uv run python experiments/ste-arm/atomgram.py \
    --a data/out/sonar/multilingual/ste_h76_init01 --step-a 369500 \
    --b data/out/sonar/multilingual/ste_h76_i01_s43 --step-b 369500
uv run python experiments/ste-arm/reconseeds.py \
    --a data/out/sonar/multilingual/ste_h76_init01 --step-a 369500 \
    --b data/out/sonar/multilingual/ste_h76_i01_s43 --step-b 369500 \
    --ctrl-step 350000

# language per head, with completeness rows around the top 32
uv run python experiments/ste-arm/headlang.py --temperature 0.00015 \
    --model data/out/sonar/multilingual/ste_h76_init01 \
            data/out/sonar/multilingual/ste_h76_i01_s43

# the layer-0 logit scale and noise flips at every checkpoint on disk
uv run python experiments/ste-arm/logitscale.py \
    --model data/out/sonar/multilingual/ste_h76_init01

# steering in embedding space: own directions, supervised vectors, the
# isotropic and data-shaped controls, the on/off-manifold split
uv run python experiments/ste-arm/steerembed.py \
    --model data/out/sonar/multilingual/ste_h76_init01 --step 369500 \
    --temperature 0.00015 --out data/out/sonar/steerembed_null/ste_h76_init01
```

Analyses default to the SONAR cache; pass `--cache data/activations/gpt2_l8.npy`
(and the matching `--mse-weights` where a script takes one) for GPT-2. An
arm trained with a constant surrogate temperature should be scored at that
temperature; under `ste` it does not change the argmax, only soft readouts.

Resources, measured on one 24 GB GPU: the k = 32 arm and the k = 128 pilots
each take 4--6 GB, at 7--9 steps/s for k = 128 (about 11--15 hours for 24
epochs). Run at most three trainings at once; six exhausted the device.
