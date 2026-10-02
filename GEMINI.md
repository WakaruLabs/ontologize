# Ontologizer

## Project Overview
Ontologizer is a dictionary learning framework in JAX/Flax for neural network interpretability. It trains sparse autoencoder-like architectures (`Ontologizer`) to decompose continuous activations—specifically pretrained multilingual sentence embeddings from SONAR/M2M100—into sparse, categorical combinations of learned "ontofeatures" across multiple residual layers. The learned representations support causal interventions (clamping/setting, adding, subtracting, scaling, zeroing, or uniform-ablating dictionary heads) for mechanistic interpretability experiments. The repository also includes a comprehensive evaluation and structural-ablation suite comparing Ontologizer against standard Sparse Autoencoders (SAEs).

## Setup & Environment
- **Runtime:** Python >= 3.13 managed via `uv` (`pyproject.toml`, `uv.lock`, `.python-version`).
- **Installation:** `uv sync`
- **Framework Stack:**
  - JAX and Flax (linen) for neural network layers and model execution.
  - Optax for optimization and gradient transformations.
  - Orbax Checkpoint (`orbax-checkpoint`) for model and training state checkpointing.
  - Grain (`grain.python`) for dataset pipelines and batching.
  - PyTorch (`torch`) and HuggingFace `transformers` for running the frozen pretrained SONAR text encoder (`cointegrated/SONAR_200_text_encoder`) and M2M100 decoder (`raxtemur/SONAR_200_text_decoder`), bridged zero-copy into JAX via DLPack (`jnp.from_dlpack` / `torch.from_dlpack`).
  - `nvidia-cudnn-cu12` (pinned via `[tool.uv.override-dependencies]` in `pyproject.toml`) with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_ALLOCATOR=platform` to allow JAX and PyTorch to share GPU VRAM without preallocation collisions.
  - Optional TPU support via `torch-xla>=2.9.0` (activated when `USE_TPU` is set).

## Entry Points & Scripts
Execution is driven by root-level scripts rather than a package CLI (`python -m ontologize.main` is only a minimal hello stub). Run Python scripts with `uv run python <script>.py`.

### Training & Corpus Processing
- `sonar.py`: Primary training script for multi-layer Ontologizer models on multilingual SONAR sentence embeddings (from precomputed `.npy` cache or streaming mC4).
- `sonar.sh`: Shell wrapper running `uv run sonar.py` inside a tmux session named `training`.
- `sonar_hc.py`: Loss ablation variant of `sonar.py` isolating the head-similarity penalty (`s_hcossim=1e-5`, `s_Hm=0.0`) by patching and running `sonar.py`.
- `sae.py`: Standalone top-k and ReLU+L1 sparse autoencoder baseline trainer with structural-ablation flags (`--groups`, `--prefixes`, `--enc bilinear`).
- `sae_ladder.sh`: Shell runner executing the 4-rung structural ablation training ladder (`m5120_k32`, `m5120_g160top1`, `m5120_g160softmax`, `m5120_k32_p5`), followed by `pareto.py` and `headstruct.py`.
- `sae_evals.sh`: Shell runner finalizing remaining SAE training rungs and orchestrating the full evaluation battery (`pareto`, `headstruct`, `splitting`, `textfid`, `langprobe`, `refit`).
- `encode_corpus.py`: Precomputes SONAR embeddings for the interleaved multilingual mC4 corpus into a resumable `.npy` cache file.
- `encode_discord.py`: Precomputes SONAR embeddings for Discord chat exports into paired `.npy` embeddings and `.jsonl` metadata files.
- `mnist.py`: Broken/stale toy autoencoder training script on MNIST images (see Known Dead/Stale Code).

### Decoding & Interventions
- `decode.py`: Interactive CLI REPL for text reconstruction and causal feature interventions against a trained Ontologizer checkpoint via M2M100.
- `decode_tags.py`: Decodes trained dictionary entries (`decodeEntries`) and uniform tag embeddings (`decodeUniform`) into natural language text using M2M100.
- `classify_acts.py`: Computes and caches per-layer classification probabilities over an embedding cache using `Ontologizer.classify` (in float64 for numerical stability).

### Evaluation & Interpretability Battery
- `autointerp.py`: 4-stage automated interpretability pipeline (harvest, texts, describe, score) comparing Ontologizer tags and SAE latents via Claude Haiku.
- `autointerp_run.sh`: Automated shell driver executing the full multi-model auto-interp campaign across Ontologizer and SAE variants.
- `compose.py`: Evaluates intervention additivity in decode space when applying two head set-interventions to measure downstream reclassification.
- `pareto.py`: Computes reconstruction error vs. code capacity (Pareto frontier) comparing Ontologizer deviation codes against top-k SAEs.
- `headcoh.py`: Evaluates head semantic coherence by measuring within-head decode direction and description similarities against null baselines.
- `headstruct.py`: Analyzes whether trained or greedily discovered SAE latent groups exhibit categorical head-like structure (exhaustiveness, exclusivity, sum stability).
- `splitting.py`: Measures cross-model feature splitting and seed stability (e.g. seeds 42 vs 43) via activation containment and decoder cosine similarity.
- `steerfid.py`: Measures steering fidelity (target activation gain cycle-consistency vs. text distortion `1 - chrF`) at matched intervention magnitudes.
- `textfid.py`: Measures downstream text reconstruction fidelity (chrF and NLL loss recovered) decoded through the M2M100 decoder.
- `refit.py`: Decomposes reconstruction error into feature selection versus magnitude shrinkage error via unconstrained least-squares refits on frozen support.
- `langprobe.py`: Trains k-sparse and dense logistic probes to predict language identity from cache embeddings.

### Smoke Tests & Harnesses
- `run_chat.py`: Non-interactive smoke-test harness driving `decode.py` via a pseudo-terminal (PTY) with scripted commands.
- `run_chat2.py`: Non-interactive smoke test executing `decode.py` under CPU mode (`CUDA_VISIBLE_DEVICES=""`) with scripted intervention commands.
- `run_decode.py`: Non-interactive test script feeding sample sentences to `decode.py` under CPU mode for text reconstruction smoke tests.
- `run_decode_tags.py`: Non-interactive test runner executing `decode_tags.py` against a saved checkpoint.

## Package Architecture (`ontologize/`)
`ontologize` is an implicit namespace package (no root `__init__.py`). Key modules and subpackages:

- **Root Modules:**
  - `ontologizer.py`: Defines `Ontologizer` (stacks `DictEnc` layers residually with `forward="resid"` or `"labels"`, supporting `deepsup`, `deepsup_sg`, `resid_norm`, `resid_const`, and wrapping encoder/decoder `Linear` layers), `DictIntervention` (dataclass for causal interventions: scale, set, uniform-ablate, zero-ablate, add, subtract), and `OntologizerIntervention`.
  - `chat.py`: Defines `ChatEnv`, the interactive CLI environment for configuring `DictIntervention`s dynamically in `decode.py`.
  - `main.py`: Minimal stub printing `"Hello from ontologize!"`.
- **`layers/`:**
  - `sparse.py`: Base `Sparse` module resolving string-keyed activations/dtypes/noise and tracking conditionally gated statistics (`l1`, `cossim`, `bcossim`, `entropy`).
  - `linear.py`: Dense `Linear` layer with `fwd` and `rev` methods (raw matmuls) used for multi-layer ghost-gradient propagation.
  - `nlinear.py`: Multi-linear and bilinear modules (`NLinear`, `Bilinear`, `NLinearBlock`, `BilinearBlock`) implementing bilinear MLPs and Pearce et al. (2025) eigendecomposition analysis.
  - `dictblock.py`: `DictBlock` dictionary lookup with non-negative weights (`abs(weights)`), per-head softmax clustering, and intervention hooks (`intervene`).
  - `dictenc.py`: Single dictionary-learning layer combining multi-head classification (`BilinearBlock`/`NLinearBlock`), `DictBlock`, and optional bilinear scaling (encoder/decoder reside in `Ontologizer`).
- **`training/`:**
  - `config.py`: `Hyperparams` (loss scalings, noise/temperature annealing, optimizer configuration), `Metadata` (filesystem paths, checkpoint intervals, data loader dispatch), and `TrainingEnv` (training lifecycle, checkpoint resume, and CSV logging).
  - `ontostate.py`: `OntoState` (Flax `TrainState` carrying the model instance and rolling stats buffer), `state_init`, `update` (jitted training step with gradient clipping), and `train` loop.
  - `serialize.py`: Standalone save/restore helper using Orbax `StandardSave`/`StandardRestore` (not used by the main training pipeline).
- **`data/`:**
  - `loaders.py`: Grain pipelines (`SampleLoader`, `TextLoader`, `ImageLoader`, `EmbeddingLoader`) and data sources (`HFDataSource`, `JSONLDataSource`, `NpyDataSource`).
  - `pretrained.py`: Pretrained PyTorch transformer integration; loads `M2M100Encoder` for SONAR, applies mean-pooling + L2 normalization, and bridges tensors to JAX via DLPack.
  - `multilingual.py`: Helpers for loading and interleaving HuggingFace `mC4`/`C4` language datasets.
  - `langs.py`: Mapping table from HuggingFace dataset language codes (ISO 639) to SONAR/NLLB BCP-47 language codes.
- **`fns/`:**
  - `keys.py`: String-to-callable registry mapping strings to activation functions, dtypes, loss functions, loader types, and noise functions.
  - `loss.py`: Mathematical functions for loss terms (MSE, L1, L0, Shannon entropy, batch cossim), noise injection (`addnoise_batchnorm`, `addnoise_featvar`), and dead-feature ghost gradients (`ghostgrad`, `l2_ghost`).
  - `classify.py`: Classification activation functions (`softmax_cl` and straight-through estimator `ste`).
- **`inference/`:**
  - `steerable.py`: Experimental `Steerable` class for checkpoint restoration and forward passes with interventions; contains a separate `ChatEnv` dataclass (unreferenced; see Known Dead/Stale Code).
- **`visualize/`:**
  - `loss.py`: Reads `<output_dir>/loss.csv` (`loss, MSE, MSE_ghost, L1_K, L1_F, entropy, cossim_b, cossim_h, KL_m`) and plots training metric trajectories using matplotlib.

## Running Tests
Tests are located in `tests/` (24 test suites) and executed using `pytest`:
```bash
uv run pytest
# or with verbose output:
uv run pytest -v
```
- Configured via `pyproject.toml` (`[tool.pytest.ini_options] testpaths = ["tests"]`).
- `tests/conftest.py` forces `JAX_PLATFORMS=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`, allowing tests to run fast on CPU without consuming GPU VRAM or conflicting with live training jobs.
- The test suite covers gradient semantics (`deepsup_sg`), bilinear adjoints, residual conditioning, batch mean-entropy bonus (`KL_m`), noise regression, checkpoint save/restore, SAE baseline modules, interpretability evaluation metrics, data loaders, and training steps.

## Known Dead or Stale Code
- `ontologize/layers/dictblock_enhanced.py`: Imports nonexistent `ontologize.jax.layers.dictblock` (leftover from legacy multi-backend directory layout).
- `ontologize/layers/integration_clean.py`: Imports nonexistent `ontologize.jax.layers.*` modules (`DictEnc`, `DictBlock`, `DictInterpreter`).
- `ontologize/layers/vae_integration.py`: Imports nonexistent `ontologize.jax.layers.dictenc` and contains stub/placeholder implementations.
- `ontologize/layers/dict_interpreter.py`: Stale utility using `FlaxAutoModelForCausalLM` ("gpt2") to generate text descriptions; orphaned and unreferenced by any active training or evaluation code.
- `ontologize/inference/steerable.py`: Unreferenced checkpoint loader/runner defining `Steerable` and a colliding `ChatEnv` dataclass; active decoding and REPL code in `decode.py` uses `ocp.CheckpointManager` and `ontologize.chat.ChatEnv` directly.
- `ontologize/training/serialize.py`: Alternative checkpointing utility using `ocp.args.StandardSave`; unreferenced by the primary training and inference pipeline, which uses `OntoState.save` and `ocp.CheckpointManager` composite PyTree items (`state` and `spec`).
- `ontologize/main.py`: A stub printing `"Hello from ontologize!"`; not a functional CLI entry point.
- `mnist.py`: Stale toy script containing an invalid import `from ontologize.training.data import ImageLoader` (`ImageLoader` resides in `ontologize.data.loaders`), incorrect constructor arguments for `Hyperparams`, `Ontologizer`, and `Metadata`, and invalid `kwargs_loader` argument on `TrainingEnv`.
