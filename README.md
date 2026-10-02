# ontologize

A training and architectural framework for mixture-of-experts-based mechanistic interpretability.

worked:
pointless vector space
subspaces of cluster
graph theory

didn't:
dimension reduction 
algebraic 
topology

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
This has several advantages over SAEs:
1. It effectively gives steering for free.
2. Outputs are bounded.
3. The representative vectors can be statically interpreted.
4. Feature space is structured.



only provide unstructured features for a given input.
This presents several problems for feature analysis:

We present the `Ontologizer` architecture, which aims to embed activations in an 
interpretable-by-construction feature space. It consists of a stack of `l` 
dictionary learning layers, each of which has `h` heads and `k` labels per head.
Each layer has two subnets. A classifier network encodes the input as soft label
assignments. A dictionary network reconstructs the input as a weighted average of
representative vectors for each label.
The goal is for each classifier to partition latent space into simplices where the 
vertices represent contrastive concepts. 
that decomposes the activations of a frozen pretrained encoder into a structured, discrete-leaning code — `l` layers × `h` heads, each head classifying its input over `k` learned dictionary entries ("ontofeatures" or *tags*). Unlike a sparse autoencoder, whose features must be read off activations after the fact, the Ontologizer's interpretable objects *are* its parameters: every tag is an explicit weight vector you can decode, every classification is a point on a product of simplices you can intervene on, and the classifiers are bilinear, so the encoding computation itself admits closed-form weight analysis.

The current target is the [SONAR](https://github.com/facebookresearch/SONAR) multilingual sentence-embedding space (1024-d, ~86 languages of mC4). Because SONAR embeddings pair with an M2M100 *decoder*, the learned ontology is directly legible: dictionary entries decode to text, and causal interventions on the code (force a tag, ablate a head) decode to *changed* text.

**Status: research code in active flux.** Expect rough edges, dead branches, and configuration by editing constants. `mnist.py` (a raw-pixel toy config) is currently stale and does not run.

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

- **The decoder side** is literal: a `DictBlock` holds an `(h, k, d)` tensor of non-negative dictionary vectors. A tag's meaning is a row of a weight matrix, decodable through the frozen pretrained decoder with no dataset in the loop (see [Feature annotation](#feature-annotation)).
- **The encoder side** is a multihead bilinear classifier, so *why* an input receives a classification is a weight-eigendecomposition question, per head and per tag (see [`Bilinear` feature extraction](#bilinear-feature-extraction)).
- **The code itself** is a first-class interface: classifications are points on `h` simplices per layer, and causal interventions are simplex operations (force, add, ablate) applied in place, then decoded (see [`DictBlock` feature intervention](#dictblock-feature-intervention)).

### Inductive bias towards low complexity

The architecture is deliberately biased toward low-description-length codes rather than relying on a sparsity penalty alone:

- classifications are softmax mixtures over `k` entries per head, annealed from soft (exploration, usage diversity) to hard (content forced into tag *identity* rather than mixture coefficients);
- dictionary entries are `abs()`'d, so features are additive-only — non-negative parts rather than signed directions;
- layers refine residually (RVQ-style): each layer classifies what the prefix before it left unexplained, with per-prefix reconstruction losses enforcing monotone refinement;
- regularizers target code structure directly: sample-similarity and head-similarity penalties push toward support partitioning, and a batch mean-entropy bonus anchors average usage to uniform (load balancing, and a well-defined "generic" origin for the code).

## Installation

Requires Python ≥ 3.13 and [`uv`](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

JAX/Flax owns the Ontologizer; PyTorch + HuggingFace `transformers` are used only to run the frozen SONAR encoder/decoder (M2M100), bridged into JAX zero-copy via `dlpack`. A CUDA GPU is used when available (`jax[cuda12]` is a hard dependency); training from a prebuilt embedding cache is the intended GPU workload, and analysis/tests run fine on CPU.

## Quickstart

```bash
# 1. Precompute SONAR embeddings for the mC4 corpus into data/sonar_embeddings/
#    (resumable; also writes a .langs.npy language sidecar)
uv run python encode_corpus.py

# 2. Train an Ontologizer from the cache (configuration: edit the constants
#    at the top of sonar.py; writes checkpoints + loss.csv to the `out` dir)
uv run python sonar.py          # or ./sonar.sh to run under tmux

# 3. Interactive reconstruction + intervention REPL against the checkpoint
uv run python decode.py

# 4. Decode the learned dictionary itself to text annotations
uv run python decode_tags.py
```

The test suite is fast and CPU-only (safe to run next to a live training run):

```bash
uv run pytest
```

`run_chat.py`, `run_decode.py`, and `run_decode_tags.py` are non-interactive harnesses that drive the REPLs with scripted stdin — useful as templates.
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
Provides methods for causal intervention on features.

### `DictEnc`

An `NLinearBlock` classifier followed by a `DictBlock`.

## `Ontologizer` architecture

One layer (`DictEnc`) is classifier → dictionary:

```
input E (d)
  └─ classifier: BilinearBlock          (h heads × k logits; no nonlinearity)
       └─ softmax(K / T) = P            (h simplex points -- the code)
            └─ DictBlock: Σₖ Pₕₖ·|Wₕₖ|  (weighted sum of dictionary entries)
                 └─ F (h, d) ── Σₕ ──► layer output (d)
```

The `Ontologizer` stacks `l` of these on a shared residual, with a shared linear decoder at the top:

- **Residual forwarding** (`forward="resid"`): layer *i+1* classifies the stop-gradiented reconstruction residual `X − decode(R_i)` — each layer explains what its prefix missed. (The stop-gradient prevents lower layers from writing a communication code into the residual instead of reducing it. The alternative `forward="labels"`, where each layer reads the previous layer's flattened classification, starves upper layers once codes harden.)
- **Residual conditioning**: bilinear logits scale with ‖input‖² and are even functions of their input, so raw residuals leave upper classifiers untrainable and sign-blind. `resid_norm` classifies the unit-normalized residual direction; `resid_const` appends a constant coordinate, giving the quadratic form linear terms while preserving the eigendecomposition analysis (over d+1 dims).
- **Deep supervision** (`deepsup`): every prefix of layers must reconstruct, so the loss is the mean over `l` prefix reconstructions. `deepsup_sg` optionally stop-gradients each prefix at the residual accumulated by earlier layers, making credit assignment exactly layer-local (pure RVQ/boosting semantics) without changing forward values.

All string-valued config (`activation`, `noise`, `select`, dtypes) resolves through lookup tables in `fns/keys.py`, so model specs are JSON-serializable dataclasses.

## `Bilinear` feature extraction

Every classifier logit is a quadratic form: for head *h*, entry *k*, the logit is $x^\top B_{hk}\, x$ with $B_{hk}$ the symmetrized product of the two weight matrices. `BilinearBlock` (`layers/nlinear.py`) exposes the analysis toolkit:

- `bilinearTensor()` — the symmetric interaction tensor `(h, k, d, d)`;
- `decompose()` — its eigendecomposition: each tag's logit is a weighted sum of squared projections onto orthogonal *eigenfeatures*, readable directly from weights;
- `jacobian(x)` / `interactionMat(Y)` — local linearization and output-weighted interaction matrices;
- `rev(Y)` — an approximate adjoint mapping classification deltas back to input space along each logit's top-|eigenvalue| eigenfeature; this is what `Ontologizer.intervene` uses for input-space steering. (Eigenvector sign is arbitrary — harmless for the even form, but verify realized classifications when steering and tune `strength`.)

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

Because the model output is linear in each layer's classification, interventions compose additively and a no-op intervention is exactly the identity — so own-entry forcing serves as a control condition. On the SONAR runs, intervened outputs are decoded through the M2M100 decoder to read the causal effect as text.

`inference/steerable.py` (`Steerable`) wraps checkpoint loading + `withArgs` for scripted intervention experiments.

## Training

Training runs from `sonar.py`; there is no CLI — configuration is the block of documented constants at the top of the file.

The loss is (optionally per-dimension whitened) MSE over the deep-supervision prefixes, plus weighted stat terms computed inside the forward pass. Each stat is `stop_gradient`-gated unless its weight is nonzero, so unused terms cost nothing and everything is always logged.

### Configuration

Key knobs in `sonar.py`:

| group | constants |
|---|---|
| model | `d`, `e_dec`, `k`, `h`, `l`, `n`/`gate` (2/"none" = pure bilinear), `fwd_mode`, `deepsup`, `deepsup_sg`, `resid_norm`, `resid_const` |
| annealing | `temperature` → `temperature_end` over `anneal_steps` (geometric); `p_drop_start` → `p_drop` (linear winner dropout: masks the argmax entry so runners-up receive gradient — the live anti-collapse mechanism); `sd_K` → `sd_K_end` (classifier logit noise; enters the softmax as `sd_K / T`) |
| regularizers | `s_L1F` (feature sparsity), `s_bcossim` (sample similarity), `s_hcossim` (head similarity), `s_Hm` (batch mean-entropy bonus: KL(E[p] ‖ uniform) — load balancing + generic-origin anchoring), `s_H`, `s_L1K` (disrecommended) |
| data | `cache` (embedding `.npy` from `encode_corpus.py`; keeps the frozen encoder out of the training loop entirely), `b`, `epochs`, `mse_weights` (inverse-variance target whitening) |
| logging | `save_each` (rolling checkpoints + `loss.csv` flush), `checkpoint_each` (kept checkpoints), `out` |

`loss.csv` columns: `loss, MSE, MSE_ghost, L1_K, L1_F, entropy, cossim_b, cossim_h, KL_m`; `visualize/loss.py` plots each against step (called automatically after training).

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

Checkpointing uses an `orbax` `CheckpointManager` with two items per step: `state` (the full `TrainState`, including optimizer state and the rolling stats buffer) and `spec` (the `Ontologizer` dataclass as a plain dict, so `Ontologizer(**spec)` rebuilds the architecture without the training script). On startup, `TrainingEnv` resumes from the latest checkpoint in `out` automatically — annealing schedules are indexed by absolute step, and `loss.csv`/`log.jsonl` are truncated to the resumed step. Rolling checkpoints are kept every `save_each` steps (last 10), permanent ones every `checkpoint_each`.

## Interventions on `Ontologized` model

### Chat CLI

`decode.py` loads the trained checkpoint plus the SONAR encoder/decoder pair and drops into a REPL (`chat.py`): type text, get its reconstruction decoded back to text, with interventions applied per layer.

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

Interpretation caveat: the informative object is a tag's *deviation from the generic point*, not the tag vector in isolation — uniform-elsewhere decoding bakes this in by construction, and the `s_Hm` bonus is what entitles "uniform" to mean "generic".
