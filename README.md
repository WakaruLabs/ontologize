# ontologize

A training and architectural framework for mixture-of-experts-based mechanistic interpretability.

## Introduction

Ideally we would like mechanistic interpretability to provide guarantees about
the computational process used by a model. We would like features to be
1. causal
2. composable
3. closed

However most existing techniques, such as sparse autoencoders (SAEs),
produce features that can perform arbitrarily badly on these desiderata.
Feature splitting breaks compositionality. 
They can't usually be statically analyzed (though bilinear SAE variants can be).
Sparsity trades off against reconstruction fidelity, limiting the extent to which 
features can be used for steering.

Arguably the simplest interpretability technique is a linear probe,
and the simplest control technique is a steering vector.
They are remarkably robust compared to unsupervised techniques such as SAEs,
but require labeled contrastive datasets.
We propose a MoEfication technique based on decomposing a model into a set of 
contrastive classifiers. Each classification corresponds to a representative vector.
Inputs are encoded as a series of classifier outputs, and reconstructed 
from a weighted average of the representative vectors.
This aims at several advantages over SAEs:
1. Steering for free: forcing a head's classification is a steering move.
2. Outputs are bounded.
3. The representative vectors can be statically interpreted.
4. Feature space is structured.

We present the `Ontologizer` architecture, which aims to embed activations in an 
interpretable-by-construction feature space. It consists of a stack of `l` 
dictionary learning layers, each of which has `h` heads and `k` labels per head.
Each layer has two subnets. A classifier network encodes the input as soft label
assignments. A dictionary network reconstructs the input as a weighted average of
representative vectors for each label.
The goal is for each classifier to partition latent space into simplices where the 
vertices represent contrastive concepts.

The result decomposes the activations of a frozen pretrained model into a structured code — `l` layers × `h` heads, each head classifying its input over `k` learned dictionary entries ("ontofeatures" or *tags*). Unlike a sparse autoencoder, whose features must be read off activations after the fact, the Ontologizer's interpretable objects *are* its parameters: every tag is an explicit weight vector you can decode, every classification is a point on a product of simplices you can intervene on, and the classifiers are bilinear, so the encoding computation itself admits closed-form weight analysis.

The main target is the [SONAR](https://github.com/facebookresearch/SONAR) multilingual sentence-embedding space (1024-d, ~86 languages of mC4). Because SONAR embeddings pair with an M2M100 *decoder*, the learned ontology is directly legible: dictionary entries decode to text, and causal interventions on the code (force a tag, ablate a head) decode to *changed* text. GPT-2 small's residual stream (layer 8) serves as a second substrate, where sparse dictionary learning is known to work and a reconstruction can be scored by the loss GPT-2 recovers when it is spliced back in.

**Status: research code in active flux.** Expect rough edges, dead branches, and configuration by editing constants.

## Findings so far

The working write-up is [`writeup/findings.tex`](writeup/findings.tex) (compiled: `writeup/findings.pdf`); the straight-through arm's full lab notebook, including superseded readings, is [`experiments/ste-arm/notes.md`](experiments/ste-arm/notes.md), condensed in [`experiments/ste-arm/findings.md`](experiments/ste-arm/findings.md). In brief, and preliminary:

- **Trained heads are contrast sets.** Within a softmax head, entries' decode directions *repel* (96% of heads below a random-group null) — well-separated alternatives rather than a semantic cluster.
- **A soft code's argmax is not its code.** The softmax model reconstructs well, but its discrete reading is orders of magnitude worse. Training with hard, straight-through selection (`select="ste"`) makes the discrete code the model: on SONAR, 1900 index bits reach whitened FVU 0.154, and 0.158 with the four per-layer gains (its only continuous numbers) pinned to their means. A top-k SAE held to the same 1900 bits cannot get below 0.236 however its coefficients are coded, but with every code's coefficients quantized and entropy-coded (`quantrate.py`) an L1 SAE reaches 0.128: its latents fire unevenly and are cheap to name, while the hard code's entries are used almost uniformly. On GPT-2, an 800-bit hard code with signed entries in activation space recovers 97.6% of GPT-2's loss, against 96.2% for a k=32 SAE at nearly the same FVU.
- **Depth helps reconstruction and costs addressability.** At equal bits, parameters and code utilization, the residual stack beats a single wide layer by 1.79× in error — but any input-space push reclassifies ~60% of downstream heads, where a flat layer's heads are independently addressable.
- **Individual heads are not yet reproducible.** Two seeds reach identical error, yet past the first layer no head's partition or contribution reproduces above a same-shape null (on SONAR and on GPT-2, under two different designs). Per-head interpretations should be read with that in mind.
- **Per-latent evaluation lenses mis-measure classifier codes.** k-sparse probing ranks the Ontologizer worst, and blinded auto-interp detection F1 mostly tracks code density.
- **Heads follow meaning before language.** The strongest SONAR head encodes topic and genre, as a language-agnostic semantic embedding should be organized. Language — probed because it has free labels, not because a head was expected to encode it — is carried as neighbourhoods across heads.

## Background

### Sparse autoencoders

The dominant approach to decomposing neural representations is the sparse autoencoder (SAE): train a wide autoencoder with a sparsity penalty on activations sampled from a model, and read features off as dictionary directions with sparse coefficients (Bricken et al., 2023; Cunningham et al., 2023). SAEs demonstrated that superposed representations can be unfolded into far more monosemantic units than the ambient dimensionality suggests.

### Limitations of activation-based interpretability

An SAE is a *separate model of the activations*, not an account of the computation:

- Features exist only relative to a dataset of sampled activations; discovering what a feature means requires searching data for activating examples.
- The feature dictionary is unstructured — a flat bag of directions with no factorization into heads, levels of abstraction, or an explicit "generic" point to measure deviations from.
- Nothing in the SAE's own weights explains *why* an input activates a feature; the encoder is a trained approximation whose errors (and hence whose causal fidelity under intervention) are unbounded.
- Reconstruction quality and sparsity trade off through a scalar penalty, with the effective code structure left implicit.

### Bilinear MLPs

Pearce et al. (2025), *[Bilinear MLPs enable weight-based mechanistic interpretability](https://arxiv.org/abs/2410.08417)* (ICLR 2025), showed that a GLU with the elementwise nonlinearity removed — output $g(x) = (W_0 x) \odot (W_1 x)$ — remains competitive while being expressible as a third-order tensor. Each output coordinate is a quadratic form $x^\top B\, x$ in the input, so the layer's behavior can be analyzed *from its weights alone* via the eigendecomposition of the symmetric interaction matrices $B$, without sampling activations.

### Parameter-based interpretability

`ontologize` combines the two ideas: dictionary learning where both sides of the dictionary live in parameters.

- **The decoder side** is literal: a `DictBlock` holds an `(h, k, d)` tensor of dictionary vectors (non-negative by default). A tag's meaning is a row of a weight matrix, decodable through the frozen pretrained decoder with no dataset in the loop (see [Feature annotation](#feature-annotation)).
- **The encoder side** is a multihead bilinear classifier, so *why* an input receives a classification is a weight-eigendecomposition question, per head and per tag (see [`Bilinear` feature extraction](#bilinear-feature-extraction)).
- **The code itself** is a first-class interface: classifications are points on `h` simplices per layer, and causal interventions are simplex operations (force, add, ablate) applied in place, then decoded (see [`DictBlock` feature intervention](#dictblock-feature-intervention)).

### Inductive bias towards low complexity

The architecture is deliberately biased toward low-description-length codes rather than relying on a sparsity penalty alone:

- classifications are softmax mixtures over `k` entries per head, annealed from soft (exploration, usage diversity) toward hard; or hard from the start, with an argmax forward and a straight-through softmax gradient (`select="ste"`), so that the discrete code is the model;
- dictionary entries are `abs()`'d by default, so features are additive-only — non-negative parts rather than signed directions (`signed=True` drops this);
- layers refine residually (RVQ-style): each layer classifies what the prefix before it left unexplained, with per-prefix reconstruction losses enforcing monotone refinement;
- each layer separates *which* (the direction of what it is correcting, which it classifies) from *how much* (that residual's norm, which scales its output), so the code carries shape and the measured gain carries magnitude;
- regularizers target code structure directly: a sample-similarity penalty, a support-overlap penalty pushing heads onto disjoint coordinates of the dictionary space, and a batch mean-entropy bonus anchoring average usage to uniform (load balancing).

## Installation

Requires Python ≥ 3.13 and [`uv`](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

JAX/Flax owns the Ontologizer; PyTorch + HuggingFace `transformers` are used only to run frozen pretrained models (the SONAR encoder/decoder, GPT-2), bridged into JAX zero-copy via `dlpack`. A CUDA GPU is used when available (`jax[cuda12]` is a hard dependency); training from a prebuilt activation cache is the intended GPU workload, and analysis/tests run fine on CPU.

## Quickstart

```bash
# 1. Precompute SONAR embeddings for the mC4 corpus into data/sonar_embeddings/
#    (resumable; also writes a .langs.npy language sidecar)
uv run python encode_corpus.py

# 2. Train an Ontologizer from the cache (configuration: edit the constants
#    at the top of sonar.py, or override some with ONTO_* environment
#    variables; writes checkpoints + loss.csv to the `out` dir)
uv run python sonar.py

# 3. Interactive reconstruction + intervention REPL against the checkpoint
#    (pass the checkpoint dir sonar.py wrote; the default is an older path)
uv run python decode.py data/out/sonar/multilingual/<run>

# 4. Decode the learned dictionary itself to text annotations
uv run python decode_tags.py data/out/sonar/multilingual/<run>
```

Other entry points:

```bash
# The straight-through (hard-code) arm: reads sonar.py's configuration and
# overrides it with flags. --base gpt2_l8 trains on GPT-2 activations instead.
uv run python experiments/ste-arm/train_ste.py -h

# Harvest GPT-2 residual-stream activations into the same cache layout
uv run python encode_acts.py -h

# A field-standard top-k / L1 SAE baseline on the same cache and objective
uv run python sae.py -h
```

`experiments/gpt2/` holds the GPT-2 hard-code runs with entries in activation space; they need options from the `headline` branch (`direct`, `resid_first`, `gain_clip`) that this branch does not implement, and its `results/README.md` has the commands.

The test suite is fast and CPU-only (safe to run next to a live training run):

```bash
uv run pytest
```

## Layer types

### `Sparse` supertype

Dense layer supporting sparsity metrics, additive noise, and ghost gradients.
Allows specification of activation function and `dtype` as strings for serialization.

### `Bilinear`

A bilinear MLP providing eigenfeature extraction methods.

### `NLinear`

Generalizes a bilinear MLP to varable `n`.

### `NLinearBlock`

A block of `h` parallel `NLinear`s.

### `DictBlock`

Encodes logits as a linear combination of `h` heads each of width `k`. 
Provides methods for causal intervention on features, and optionally head dropout and fibers (below).

### `ConcatDictBlock`

A `DictBlock` whose heads write disjoint slices of the output (`d // h` each) and concatenate, instead of each writing all of it and summing (`concat=True`). Heads then use disjoint coordinates of the dictionary space by construction, and the dictionary costs `h` times fewer parameters. The disjointness is in the dictionary space only: the shared decoder maps the slices onto overlapping output subspaces.

### `DictEnc`

An `NLinearBlock` classifier followed by a `DictBlock`, with the optional router and fiber branches.

## `Ontologizer` architecture

One layer (`DictEnc`) is classifier → dictionary, with two optional branches:

```
input ỹ (d)
  └─ gain-shape split: u = ỹ / ‖ỹ‖, G = ‖ỹ‖ (stop-gradient)
       ├─ classifier: BilinearBlock(u)          (h heads × k logits)
       │    └─ softmax(K / T) or STE = P        (the code; head dropout acts here)
       ├─ router: |Bilinear(u)| = S             (optional: one scale per head)
       └─ fiber read: b·tanh(A u / b) = c       (optional: r coordinates per head)
  DictBlock: Fₕ = Sₕ Σₖ Pₕₖ (|dₕₖ| + Uₕₖ cₕ)
  └─ Σₕ (or concatenate) ── × G ──► layer output (e)
```

The `Ontologizer` stacks `l` of these on a shared residual, with a shared linear decoder at the top:

- **Residual forwarding** (`forward="resid"`): layer *i+1* classifies the stop-gradiented reconstruction residual `X − decode(R_i)` — each layer explains what its prefix missed. (The stop-gradient prevents lower layers from writing a communication code into the residual instead of reducing it. The alternative `forward="labels"`, where each layer reads the previous layer's flattened classification, starves upper layers once codes harden.)
- **Residual conditioning**: bilinear logits scale with ‖input‖² and are even functions of their input, so raw residuals leave upper classifiers untrainable and sign-blind. `resid_gain` (on by default) classifies the unit direction of each layer's input and scales the layer's output by its norm; `resid_const` appends a constant coordinate, giving the quadratic form linear terms while preserving the eigendecomposition analysis (over d+1 dims). `resid_norm` is the older normalization without the gain.
- **Deep supervision** (`deepsup`): every prefix of layers must reconstruct, so the loss is the mean over `l` prefix reconstructions. `deepsup_sg` optionally stop-gradients each prefix at the residual accumulated by earlier layers, making credit assignment exactly layer-local (pure RVQ/boosting semantics) without changing forward values.
- **Selection** (`select`): `"softmax"` (dense), `"top<k>"` (softmax over each head's top-k logits, the rest exactly zero), or `"ste"`/`"argmax"` (hard forward, straight-through gradient). A hard code needs its own calibration — see `experiments/ste-arm/train_ste.py` and `experiments/ste-arm/notes.md`.
- **Router** (`scaled`): a second bilinear map from the same shaped input to one non-negative scale per head, so the classifier decides *what* a head says and the router *how much*.
- **Fibers** (`fiber_rank`): each entry carries a local basis, and the head reads `r` coordinates from its input to move the output within it — a discrete base point plus a continuous correction. Unconstrained, the correction can take over, so the base is kept meaningful by fiber dropout, a norm cap, a base-only loss term (`base_aux`), or by training the fibers after freezing the base.
- **Head dropout** (`p_head_drop`): in training, a head's classification is replaced by its batch mean, so heads cannot co-adapt and ablating a head to its mean stays in distribution.

`writeup/tikz/` has diagrams of the stack (`ontologizer.tex`), one layer (`dictenc.tex`, and `dictenc_full.tex` with the router and fibers) and a head's geometry (`dict.tex`, `fiber.tex`).

All string-valued config (`activation`, `noise`, `select`, dtypes) resolves through lookup tables in `fns/keys.py`, so model specs are JSON-serializable dataclasses.

## `Bilinear` feature extraction

Every classifier logit is a quadratic form: for head *h*, entry *k*, the logit is $x^\top B_{hk}\, x$ with $B_{hk}$ the symmetrized product of the two weight matrices. `BilinearBlock` (`layers/nlinear.py`) exposes the analysis toolkit:

- `bilinearTensor()` — the symmetric interaction tensor `(h, k, d, d)`;
- `decompose()` — its eigendecomposition: each tag's logit is a weighted sum of squared projections onto orthogonal *eigenfeatures*, readable directly from weights;
- `jacobian(x)` / `interactionMat(Y)` — local linearization and output-weighted interaction matrices;
- `rev(Y)` — an approximate adjoint mapping classification deltas back to input space along each logit's top-|eigenvalue| eigenfeature; this is what `Ontologizer.intervene` uses for input-space steering. Eigenvector signs are fixed by `nlinear.orient` (by the eigenvalue's sign, with a leading-coordinate tie-break), so `rev` is well-defined.

For steering from the input, the direction matters: on the SONAR straight-through arm the classifier's own gradient works where the input reaches the classifier directly (layer 0), while moving the input toward what the model emits with the tag active (`decode`) works best deeper in the stack (`experiments/ste-arm/steerembed.py`).

## `DictBlock` feature intervention

`DictBlock.intervene` operates on classifications in place; `Ontologizer.withArgs` applies one `DictIntervention` per layer during a forward pass:

| field | effect on head(s) |
|---|---|
| `h_set` / `k_set` | force a one-hot: entry `k_set` gets probability 1 |
| `h_unif` | ablate to the uniform mixture over its `k` entries |
| `h_zero` | zero the head's classification entirely |
| `h_add` / `k_add` | add 1 to an entry's coefficient |
| `h_sub` / `k_sub` | subtract 1 from an entry's coefficient |
| `scale` | scalar multiple per head |

The decode is linear in the flattened classification, so interventions at the last layer compose exactly and a no-op intervention is the identity — own-entry forcing serves as a control condition. Under residual forwarding, an intervention at an earlier layer also changes what later layers see, and they reclassify; measured on SONAR, two set-interventions still compose approximately additively (median cosine ≥ 0.95 between the joint effect and the sum of the singles), with the non-additive part shrinking as fewer layers sit below the intervention (`compose.py`). On the SONAR runs, intervened outputs are decoded through the M2M100 decoder to read the causal effect as text.

For neutral ablations, prefer the measured mean classification `E[p]` over the uniform mixture: on the reference softmax model the uniform code lands off the data manifold (cosine 0.067 to the data mean), while `E[p]` decodes to the corpus-mean text.

`ontologize/inference/steerable.py` (`Steerable`) wraps checkpoint loading + `withArgs` for scripted intervention experiments.

## Training

Training runs from `sonar.py`; there is no CLI — configuration is the block of documented constants at the top of the file, some of which can be overridden with `ONTO_*` environment variables. The cache's last 32,768 rows (`holdout`) are excluded from training, so the evaluation scripts score out-of-sample.

The loss is (optionally per-dimension whitened) MSE over the deep-supervision prefixes, plus weighted stat terms computed inside the forward pass. Each stat is `stop_gradient`-gated unless its weight is nonzero, so unused terms cost nothing and everything is always logged.

### Configuration

Key knobs in `sonar.py`:

| group | constants |
|---|---|
| model | `d`, `e_dec`, `k`, `h`, `l`, `n`/`gate` (2/"none" = pure bilinear), `fwd_mode`, `select`, `norm_rows`, `deepsup`, `deepsup_sg`, `resid_norm`, `resid_const`, `scaled` (router) |
| annealing | `temperature` → `temperature_end` over `anneal_steps` (geometric); `p_drop_start` → `p_drop` (linear winner dropout: masks the argmax entry so runners-up receive gradient); `sd_K` → `sd_K_end` (classifier logit noise; enters the softmax as `sd_K / T`); `p_revive` (dead-entry revival under hard selection) |
| regularizers | `s_L1F` (feature sparsity; `L1F_target` holds it at a setpoint instead), `s_bcossim` (sample similarity), `s_support` (support overlap between heads), `s_kcossim` (within-head row collinearity; `KCOS_target` holds its per-layer max at a setpoint), `s_Hm` (batch mean-entropy bonus: KL(E[p] ‖ uniform)), `s_H`, `s_L1K` (disrecommended) |
| data | `cache` (embedding `.npy` from `encode_corpus.py`; keeps the frozen encoder out of the training loop entirely), `holdout`, `b`, `epochs`, `mse_weights` (inverse-variance target whitening) |
| logging | `save_each` (rolling checkpoints + `loss.csv` flush), `checkpoint_each` (kept checkpoints), `out` |

`loss.csv` has one row per step and 19 columns, documented with their
provenance at `ontologize/visualize/loss.py:COLUMNS` — import that list
rather than restating it here, since the layout is append-only and copies
of it have gone stale and mislabelled columns before:

`loss, MSE, MSE_ghost, L1_K, L1_F, entropy, cossim_b, cossim_h, cossim_k,
cossim_k_hmax, KL_m, KL_pwak, L2_pwak, s_L1F, s_kcossim, cossim_k_max,
cossim_flat, L1_S, support`

`cossim_h` and `cossim_flat` are retired and read NaN in new runs (their
slots stay so older rows still align); `support` replaces both, and their
weights `s_hcossim`/`s_flatcos` now raise.
`visualize/loss.py` plots each against step (called automatically after
training). Note `cossim_k` (mean over heads, summed over layers),
`cossim_k_hmax` (per-head max within a layer, summed over layers) and
`cossim_k_max` (max over heads and layers) are three different columns.

### Data Types

#### `Hyperparams`

Contains arguments used for setting up the model/optimizer and calculating loss.

#### `Metadata`

Contains arguments for model serialization, data loading, and checkpoint frequency

#### `Ontostate`

Contains training state.

#### `TrainingEnv`

Encapsulates `Hyperparams`, `Metadata`, `Ontostate`.

### Serialization

Checkpointing uses an `orbax` `CheckpointManager` with two items per step: `state` (the full `TrainState`, including optimizer state and the rolling stats buffer) and `spec` (the `Ontologizer` dataclass as a plain dict, so `Ontologizer(**spec)` rebuilds the architecture without the training script). On startup, `TrainingEnv` resumes from the latest checkpoint in `out` automatically — annealing schedules are indexed by absolute step, `loss.csv`/`log.jsonl` are truncated to the resumed step, and the setpoint controllers pick up their last multipliers from `loss.csv`. Rolling checkpoints are kept every `save_each` steps (last 10), permanent ones every `checkpoint_each`.

Every loader rebuilds a model through `serialize.migrate_spec`, which maps renamed fields forward (including the `headline` branch's `signed_dict` and `private_heads`) and drops `headline`-only fields only when they are switched off, so a checkpoint either rebuilds as the model that was trained or refuses to.

## Evaluation

Each protocol is a standalone script scoring on the cache's held-out tail; most take `-h`:

| script | measures |
|---|---|
| `pareto.py` | reconstruction against code capacity (coefficients and index bits), Ontologizer against SAEs |
| `headstruct.py`, `headcoh.py` | whether groups behave as categorical heads; whether a head's entries cohere or form a contrast set |
| `splitting.py` | feature splitting and seed-to-seed reproducibility |
| `textfid.py` | text fidelity of reconstructions through the M2M100 decoder |
| `langprobe.py` | sparse probing of language identity |
| `refit.py` | shrinkage: refit coefficients on each frozen support |
| `quantrate.py` | operational rate: everything a code sends quantized and entropy-coded, FVU against total bits |
| `steerfid.py` | steering effect against collateral, through the text round trip |
| `compose.py` | whether two interventions compose additively |
| `autointerp.py` | blinded auto-interpretability of tags and SAE latents |

`experiments/ste-arm/` adds diagnostics for the straight-through arm and for head-level analysis — among them `steerembed.py` (steering scored in embedding space), `partition.py` and `headcontrib.py` (seed reproducibility of head partitions and contributions), `entryshare.py` and `bigatoms.py` (how entries share a layer), `headlang.py` (language per head) and `dictgeom.py` (dictionary geometry). The results are in `writeup/`.

## Interventions on `Ontologized` model

### Chat CLI

`decode.py` loads the trained checkpoint plus the SONAR encoder/decoder pair and drops into a REPL (`ontologize/chat.py`): type text, get its reconstruction decoded back to text, with interventions applied per layer.

```
ChatEnv> help
  set <layer> <field> <val1,val2,...>  # e.g., set 0 k_add 5,10
  clear <layer>                        # clear interventions for a layer
  clear_all                            # clear all interventions
  run <text>                           # run the model with current interventions
  q, quit, exit                        # exit
```

`<field>` is any `DictIntervention` field from the table above — e.g. `set 1 h_set 8` + `set 1 k_set 17` forces entry 17 in head 8 of layer 1 before running.

## Feature annotation

`decode_tags.py` decodes the dictionary itself: for each (layer, head, entry) it builds the synthetic code where that entry is one-hot and every other head sits at the uniform mixture (`Ontologizer.decodeUniform`; `decodeEntries` decodes raw entries), pushes it through the shared decoder into embedding space, rescales to the decoder's reference norm, and generates text with the M2M100 decoder — a per-tag text annotation of the learned ontology, produced entirely from weights.

Interpretation caveats, from the findings:

- The informative object is a tag's *deviation* from its head's mean, not the tag vector in isolation. On the straight-through arm an entry sits at cosine 0.85 to its head's mean, so decoding entries raw returns one shared template sentence with the details shuffled; `experiments/ste-arm/decodehead.py --deviation` decodes deviations instead.
- Uniform-elsewhere decoding is not the generic point it was meant to be: the uniform code lands off the data manifold on the reference model. The measured mean classification `E[p]` is the better origin.
- A decoded sentence is an *instance* of what a tag writes, not a description of when it fires: in blinded detection, parameter-space decodes almost never let a judge recognize a tag's activating texts. Naming a tag still needs its activating examples (`autointerp.py --mode cacts`).
