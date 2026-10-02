# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ontologizer is a JAX/Flax implementation of dictionary learning for neural network interpretability. It trains sparse autoencoder-like models (`Ontologizer`) to decompose activations from pretrained transformer encoders (currently SONAR/M2M100 sentence embeddings, or raw MNIST pixels) into sparse combinations of learned "ontofeatures," and supports causal interventions on those features for mechanistic interpretability experiments.

**The codebase is research-grade and in active flux.** Expect dead code, half-migrated modules, and root-level scratch scripts alongside the real package — see "Known dead/stale code" below before assuming a file is load-bearing.

## Project Setup

- Requires Python >= 3.13, managed with `uv` from the repository root (there is no `pytorch/` subdirectory despite what older docs may say — the package lives at `ontologize/` in the repo root).
- Install: `uv sync`
- Stack: JAX/Flax (linen) for the model, `optax` for optimization, `orbax.checkpoint` for checkpointing, `grain` for data loading, PyTorch + HuggingFace `transformers` only for loading pretrained encoder/decoder models (SONAR/M2M100), bridged into JAX via `dlpack`.

## Common Commands

There is no real CLI. `python -m ontologize.main` is currently just a `print("Hello from ontologize!")` stub — do not treat it as the entry point. The actual entry points are root-level scripts, run with `uv run`:

```bash
# Precompute SONAR embeddings for the mC4 corpus into a .npy cache
# (resumable; sonar.py trains from the cache when its `cache` config is set,
# which keeps the frozen encoder out of the training loop entirely)
uv run python encode_corpus.py

# Train on SONAR multilingual sentence embeddings (mC4 data)
uv run python sonar.py
# or via tmux wrapper:
./sonar.sh

# Train an MNIST toy autoencoder (currently broken/stale, see below)
uv run python mnist.py

# Interactive reconstruction + intervention REPL against a trained checkpoint
uv run python decode.py

# Decode a trained model's learned dictionary tags to text
uv run python decode_tags.py

# Non-interactive smoke-test harnesses that drive decode.py / decode_tags.py
# with scripted stdin (useful as a template for manual testing changes to the REPL)
uv run python run_chat.py
uv run python run_decode.py
uv run python run_decode_tags.py

# SAE-baseline comparison suite (all score on the cache tail sae.py holds out):
uv run python sae.py           # train a field-standard top-k SAE on the same
                               # cache + whitened MSE (CLI flags; built to be swept).
                               # --groups/--group-fn/--prefixes add Ontologizer-style
                               # structure one rung at a time (heads / deepsup layers)
uv run python pareto.py        # reconstruction-vs-code-capacity Pareto table:
                               # Ontologizer top-m deviation codes vs trained SAEs
uv run python autointerp.py -h # 4-stage auto-interp pipeline (harvest/texts/
                               # describe/score) over Ontologizer tags & SAE latents
uv run python headstruct.py -h # post-hoc head-structure discovery: do SAE latents
                               # form exhaustive/exclusive groups? (split-half + nulls)
uv run python headcoh.py -h    # head/group SEMANTIC coherence: within-head decode-
                               # direction similarity vs size-matched nulls (quota-
                               # free), plus description-text view over autointerp
                               # artifacts; --assignment scores discovered groups;
                               # --metric whitened = objective's inverse-variance
                               # inner product (writes <out>_w)
uv run python compose.py -h    # intervention composition: do two-head set-
                               # interventions compose additively in decode space?
                               # (final layer exact by linearity = numerical floor;
                               # deviation above it = downstream reclassification)
uv run python splitting.py -h  # cross-model feature splitting via activation
                               # containment (parents/children/matches, absorption)
uv run python textfid.py -h    # downstream text fidelity ("loss recovered"): decode
                               # x vs recon(x) through M2M100, chrF + NLL vs floor
uv run python langprobe.py -h  # SAEBench-style k-sparse probing of language
                               # identity (langs sidecar); dense-probe ceiling
uv run python refit.py -h      # shrinkage: refit LS coefficients on each frozen
                               # support; selection-vs-magnitude error split
uv run python steerfid.py -h   # steering fidelity: effect (cycle-consistency
                               # activation gain) vs collateral (1-chrF) at matched
                               # magnitude; W_dec / eigenfeature / withArgs steering
```

Seed stability: train a replica with `sae.py --seed 43` (run dirs get an
`_s43` suffix) and feed the pair to `splitting.py` — its "A matched"/"B
matched" columns are the reproducible-feature fractions.

Tests: the curated suite lives in `tests/`, wired via `[tool.pytest.ini_options] testpaths = ["tests"]` in `pyproject.toml`, so a bare `uv run pytest` runs only it (fast, CPU-only — `tests/conftest.py` forces `JAX_PLATFORMS=cpu`, so it is safe to run alongside a live GPU training run). It covers: `deepsup_sg` gradient semantics, `resid_norm`/`resid_const` conditioning (including bilinear sign-blindness), the `s_Hm`/`KL_m` batch mean-entropy bonus, noise-path NaN regressions, checkpoint save/restore including the 8→9 `n_stats` stats-width migration, and a short end-to-end training loop per live run configuration.

```bash
uv run pytest -v
```

(A generation of root-level `test_*.py` ad hoc debugging scripts was deleted when this suite landed; if one resurfaces from git history, treat it as a scratch script, not a spec.)

No linter, formatter, or type checker is configured (no ruff/mypy/black config exists).

## Architecture

### Model stack (`ontologize/`)

Layers build on a common base in `layers/sparse.py`:

- **`Sparse`** (`layers/sparse.py`) — base `nn.Module` for nearly all layers. Resolves string-keyed config (`activation`, `noise`, `dtype_str`) into callables via `fns/keys.py`, and provides shared stats (`l1`, `cossim`, `bcossim`, `entropy`) that are `stop_gradient`-wrapped unless explicitly enabled as loss terms, plus noise injection and a `ghost()` hook for ghost-gradient dead-feature resurrection.
- **`Linear`** (`layers/linear.py`) — standard dense layer with `fwd`/`rev` (raw matmul, no bias/activation, used to compose ghost gradients across layers).
- **`NLinear`/`Bilinear`** and their multihead variants **`NLinearBlock`/`BilinearBlock`** (`layers/nlinear.py`) — generalized bilinear layers (`Bilinear` implements eigendecomposition-based analysis per Pearce et al. 2025-style bilinear MLPs). `NLinearBlock`/`BilinearBlock` are what `DictEnc`'s classifier actually instantiates.
- **`DictBlock`** (`layers/dictblock.py`) — the core dictionary/tag lookup. Holds a `(h, k, d)` weight tensor (heads × dict entries × feature dim; always `abs()`'d so features can't subtract). Given per-head classification probabilities, does a weighted sum over dictionary entries (`fwd`/`hfwd`), and exposes the intervention interface (`intervene`: uniform-ablate, zero-ablate, set, add, subtract, scale — see `DictIntervention` below).
- **`DictEnc`** (`layers/dictenc.py`) — composes an `NLinearBlock` classifier (activations → per-head logits), a `DictBlock` (logits → reconstructed features), and `Linear`/`NLinear` encoder/decoder/scaling. One `DictEnc` = one dictionary-learning layer.
- **`Ontologizer`** (`ontologizer.py`) — stacks `l` `DictEnc` layers residually: layer *i+1* consumes the flattened classification of layer *i*. Holds the shared `encoder`/`decoder` `Linear` (or `Sparse` passthrough) around the stack. Key methods: `__call__` (plain forward), `withStats`/`withGhost`/`withArgs` (forward + validation stats / ghost-gradient / per-layer intervention list), `decodeEntries`/`decodeUniform` (decode the learned dictionary itself, rather than model input, into output space — used by `decode_tags.py`).
- **`DictIntervention`** / **`OntologizerIntervention`** (`ontologizer.py`) — dataclasses describing a causal intervention (scale a head, force/add/subtract specific dict entries, zero/uniform-ablate a head) applied per `Ontologizer` layer via `withArgs`.

`fns/keys.py` holds the string→callable lookup tables (activations, dtypes, losses, noise, data source type) that let all of the above be JSON-serializable dataclasses instead of holding raw callables. `fns/loss.py` has the actual math (mse/l1/l0/entropy/cossim/ghost-gradient). `fns/classify.py` has classification-specific activations (softmax, straight-through estimator).

### Training (`ontologize/training/`)

- **`config.py`** — `Hyperparams` (loss weights, optimizer, temperature/noise settings; builds the `Ontologizer` and drives `init`/`train`), `Metadata` (paths, checkpoint cadence, data source type; builds the orbax `CheckpointManager` and the `SampleLoader`), `TrainingEnv` (glues `Ontologizer` spec + `Hyperparams` + `Metadata`, handles resume-from-checkpoint and truncating `loss.csv`/`log.jsonl` on resume).
- **`ontostate.py`** — `OntoState` (a Flax `TrainState` subclass carrying the model instance itself as a static field, plus a rolling stats buffer). `state_init`, `update` (jitted optax step with optional global-norm grad clipping), `train` (the actual training loop, checkpointing + stats-flushing every `save_each` steps).
- **`serialize.py`** — a second, simpler save/load helper pair using orbax's `StandardSave`/`StandardRestore`. This overlaps with `OntoState.save`/`Metadata.manager` and is **not** used by the main training path (`Hyperparams.load`/`TrainingEnv.init` use the manager/spec-item approach in `config.py`/`ontostate.py` instead) — don't assume both are equally live.

Model serialization is plain `dataclasses.asdict(model)` (there is no `@model_spec()` decorator or dedicated spec-class pattern in this codebase — if you see references to one, it's stale).

### Data (`ontologize/data/`)

- **`loaders.py`** — Grain-based (`grain.python`) pipeline: `SampleLoader` (base), `TextLoader` (adds tokenization with SONAR BCP-47 language routing), `ImageLoader` (flattens/normalizes images, e.g. for MNIST), `EmbeddingLoader` (no-op transform over precomputed embedding vectors; the `srctype="embedding"` loader). `HFDataSource`/`JSONLDataSource`/`NpyDataSource` adapt HuggingFace `datasets`, raw JSONL, or an on-disk `.npy` (memory-mapped; e.g. the embedding cache written by root-level `encode_corpus.py`) into Grain data sources.
- **`pretrained.py`** — loads pretrained PyTorch/HuggingFace encoders (special-cased for SONAR: loads the `M2M100Encoder` directly from a `pytorch_model.bin`, bypassing `AutoModel`). `encode()`/`l2_pooling()` run the torch model, then hand the hidden state to JAX via `jnp.from_dlpack` and mean-pool + L2-normalize it (SONAR embeddings are unit-normed). `decode()` runs the paired M2M100 decoder's `.generate()` with an `Ontologizer` output substituted in as `last_hidden_state`.
- **`multilingual.py`** / **`langs.py`** — loads and interleaves per-language HuggingFace datasets (mC4/C4), with a static ISO→SONAR/NLLB BCP-47 language code map.

This is the load-bearing pattern to know: **JAX/Flax owns the `Ontologizer` model; PyTorch is used only to run the frozen pretrained SONAR encoder/decoder, with `dlpack` as the zero-copy bridge between the two frameworks.**

### Inference / interventions

- **`chat.py`** (`ChatEnv`) — interactive CLI REPL for setting per-layer `DictIntervention`s (`set <layer> <field> <values>`, `clear`, `clear_all`, `run <text>`), used by `decode.py`.
- **`inference/steerable.py`** (`Steerable`, and a *second, different* `ChatEnv` dataclass) — loads a trained `Ontologizer` checkpoint and runs `withArgs` with a list of interventions. Note the name collision: this `ChatEnv` is not the same class as `chat.py`'s — don't assume they're interchangeable.

### Visualization

`visualize/loss.py` reads `<output_dir>/loss.csv` (fixed columns: `loss, MSE, MSE_ghost, L1_K, L1_F, entropy, cossim_b, cossim_h, KL_m`, matching `OntoState`'s stats buffer layout; runs from before the `KL_m` batch mean-entropy stat have 8 columns) and plots each column vs. step with matplotlib. Called from `sonar.py` after training.

## Known dead/stale code

- `ontologize/layers/dictblock_enhanced.py`, `dict_interpreter.py`, `integration_clean.py`, `vae_integration.py` all import from a nonexistent `ontologize.jax.layers.*` path (a leftover from a prior multi-backend `pytorch/`/`jax/`/`haskell/`/`julia/` layout that has since collapsed into the flat `ontologize/` package). They are never imported anywhere else and contain their own "placeholder, not implemented" docstrings. Treat them as dead — the real dictionary-learning code is `layers/dictblock.py`/`dictenc.py`.
- `mnist.py` imports `from ontologize.training.data import ImageLoader`, but that module doesn't exist (`ImageLoader` now lives in `ontologize/data/loaders.py`); the file is also truncated mid-argument-list. It needs fixing before it will run.
