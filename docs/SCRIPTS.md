# Root Script Index (`docs/SCRIPTS.md`)

## 1. Overview and Architecture

The root directory of `ontologize` contains **27 executable scripts** comprising **23 Python scripts (`*.py`)** and **4 Bash shell scripts (`*.sh`)**. These scripts drive corpus precomputation, model training, interactive causal decoding, sparse autoencoder baseline comparison, and multi-metric interpretability evaluation.

Execution is managed via `uv`:
```bash
# Python scripts
uv run python <script>.py [args]

# Shell scripts
./<script>.sh
```

### JAX and PyTorch Runtime Architecture
The scripts operate across a dual JAX/PyTorch framework stack:
- **JAX / Flax**: Model layers, autoencoder architectures, Optax optimizers, Orbax checkpoints, and GPU tensor computations.
- **PyTorch / Transformers**: Pretrained frozen SONAR text encoder (`cointegrated/SONAR_200_text_encoder`) and M2M100 conditional text decoder (`raxtemur/SONAR_200_text_decoder`).
- **Memory Management**: To prevent JAX from claiming 90% of GPU VRAM and starving PyTorch, scripts configure `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_ALLOCATOR=platform`, and `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- **Zero-Copy Bridge**: Tensors transfer between PyTorch and JAX using DLPack (`t.from_dlpack` / `jnp.from_dlpack`).

---

## 2. Master Script Index Table (All 27 Scripts)

| Script | Category | Purpose / One-Line Summary | Main Inputs | Main Outputs | Covering Test in `tests/` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `autointerp.py` | SAE baseline + evaluation suite | 4-stage automated interpretability pipeline (harvest activations, stream texts, generate LLM/parameter descriptions, score detection) | CLI subcommands (`harvest`, `texts`, `describe`, `score`), `.npy` cache, checkpoints, `ANTHROPIC_API_KEY` | `features.npz`, `texts.jsonl`, `descriptions.jsonl`, `scores.csv` | `tests/test_autointerp.py` |
| `autointerp_run.sh` | SAE baseline + evaluation suite | Shell driver orchestrating the 4-stage autointerp campaign across Ontologizer and 5 SAE variants | `ANTHROPIC_API_KEY`, optional `AI_RATE`, checkpoints in `data/out/sonar/` | Artifacts in `data/out/sonar/autointerp/`, `.campaign_done` | None |
| `classify_acts.py` | Data preparation / harvesting | Runs `Ontologizer.classify` in float64 over embedding caches to generate row-aligned per-layer post-softmax activations | `--ckpt`, `--emb`, `--temperature`, `--dtype`, `--compute-dtype`, `-b`, `-c`, `-n` | `<out>.npy` (shape `(N, l*h*k)`), `<out>.meta.json`, `<out>.progress.json` | None |
| `compose.py` | SAE baseline + evaluation suite | Evaluates intervention additivity by comparing joint dual-head interventions against summed single-intervention deltas | `--ckpt`, `--cache`, `--eval-rows`, `--rows`, `--singles-per-layer`, `--cross-pairs`, `--mse-weights` | `<out>/pairs.csv`, `<out>/summary.json`, console metrics | `tests/test_compose.py` |
| `decode.py` | Interactive / decoding / chat | Interactive CLI shell to encode text to SONAR space, apply head interventions via `Ontologizer.withArgs`, and decode back to text via M2M100 | Checkpoint directory (`data/out/sonar/linear`), interactive stdin commands via `ChatEnv` | Interactive console output (reconstruction MSE, decoded strings) | None |
| `decode_tags.py` | Interactive / decoding / chat | Batch decodes all Ontologizer dictionary entries and uniform reference tags into natural language text via M2M100 | Checkpoint directory (`data/out/sonar/linear`), pretrained SONAR encoder and M2M100 decoder | `<ckpt>/entries.txt`, `<ckpt>/uniform.txt` | None |
| `encode_corpus.py` | Data preparation / harvesting | Streams multilingual mC4 text from HuggingFace, computes SONAR embeddings, and saves a crash-resilient `.npy` memmap disk cache | `-n`, `-b`, `-m`, `-o`, HuggingFace `allenai/c4` dataset, SONAR text encoder | `<out>/<name>.npy`, `<out>/<name>.langs.npy`, `<out>/<name>.meta.json`, `<out>/<name>.progress.json` | None |
| `encode_discord.py` | Data preparation / harvesting | Indexes Discord JSON exports and encodes dynamic length-bucketed SONAR embeddings into `.npy` | `-s`, `-o`, `-m`, `-b`, `-c`, `-l`, `--maxlen`, Discord JSON export directory, SONAR encoder | `<out>/<name>.npy`, `<out>/<name>.jsonl`, `<out>/<name>.meta.json`, `<out>/<name>.progress.json` | None |
| `headcoh.py` | SAE baseline + evaluation suite | Evaluates semantic coherence of heads/groups via pairwise cosine similarities and SVD energy of decode directions vs nulls | `--model`, `--ckpt`, `--assignment`, `--desc-dir`, `--desc-mode`, `--metric`, `--n-null`, checkpoint | `<out>/groups.csv`, `<out>/summary.json`, console z-scores | `tests/test_headcoh.py` |
| `headstruct.py` | SAE baseline + evaluation suite | Discovers and scores latent head-like categorical groupings in SAEs (exhaustiveness, exclusivity, and sum stability vs nulls) | `--sae`, `--cache`, `--rows`, `--b`, `--size`, `--topn`, `--nulls`, `--trained-groups`, SAE `params.npz` | `<out>/groups.csv`, `<out>/assignment.npy`, summary table | `tests/test_headstruct.py` |
| `langprobe.py` | SAE baseline + evaluation suite | Evaluates language identity representation using k-sparse and dense logistic probes across SAE latents and Ontologizer activations | `--model`, `--cache`, `--langs`, `--train-rows`, `--test-rows`, `--ks`, model checkpoint, `.langs.npy` sidecar | `<out>/langs.csv`, summary Macro-F1 | `tests/test_langprobe.py` |
| `mnist.py` | Scratch / stale / experimental | Broken legacy toy script attempting to train an Ontologizer on MNIST image pixels using obsolete/missing module APIs | Hardcoded parameters, HuggingFace `mnist` dataset | Checkpoint directory `data/out/mnist/L1` (crashes on import) | None |
| `pareto.py` | SAE baseline + evaluation suite | Computes and plots capacity-Pareto frontier (whitened reconstruction FVU vs active coefficients and index bits) comparing Ontologizer vs SAEs | `--ckpt`, `--step`, `--temperature`, `--ms`, `--code`, `--origin`, `--sae`, `--cache`, `--eval-rows`, `--plot` | `<out>/pareto.csv`, optional `<out>/pareto.png`, stdout table | `tests/test_pareto.py` |
| `refit.py` | SAE baseline + evaluation suite | Isolates selection vs magnitude (shrinkage) error by solving closed-form whitened least-squares coefficients on frozen feature supports | `--sae`, `--onto`, `--ms`, `--cache`, `--mse-weights`, `--rows`, `--b`, `--ridge`, checkpoints | `<out>/refit.csv`, stdout comparison table | `tests/test_refit.py` |
| `run_chat.py` | Scratch / stale / experimental | Drives `decode.py` interactively through a pseudo-terminal (`pty`) with scripted commands (`set`, `run`, `q`) | Hardcoded command list, checkpoint `data/out/sonar/linear/multilingual` | Decoded stdout from pseudo-terminal to console | None |
| `run_chat2.py` | Scratch / stale / experimental | Headless CPU-only subprocess wrapper (`CUDA_VISIBLE_DEVICES=""`) running `decode.py` with piped input commands | Hardcoded input string, checkpoint `data/out/sonar/linear/multilingual` | Captured STDOUT and STDERR to console | None |
| `run_decode.py` | Scratch / stale / experimental | Feeds a batch of hardcoded test sentences into `decode.py` via a CPU-only subprocess and prints reconstruction results | Hardcoded multi-line input text, checkpoint `data/out/sonar/linear/multilingual` | Captured STDOUT and STDERR to console | None |
| `run_decode_tags.py` | Scratch / stale / experimental | Runs `decode_tags.py` inside a subprocess with restricted XLA memory allocation (`XLA_PYTHON_CLIENT_MEM_FRACTION=".10"`) | Hardcoded checkpoint `data/out/sonar/linear/multilingual` | Captured STDOUT and STDERR to console | None |
| `sae.py` | SAE baseline + evaluation suite | Standalone trainer and model suite for Sparse Autoencoder (SAE) baselines (Top-K, ReLU+L1, grouped, prefix deepsup, bilinear encoder) on SONAR | `--m`, `--topk`, `--l1`, `--groups`, `--group-fn`, `--prefixes`, `--enc`, `--lr`, `--epochs`, `--cache` | Checkpoints (`params.npz`, `state.npz`), `loss.csv`, `summary.json` | `tests/test_sae.py` *(also imported by `tests/test_refit.py`)* |
| `sae_evals.sh` | SAE baseline + evaluation suite | Shell driver for Phase 2 evaluations, executing remaining SAE training runs and the full evaluation battery | Checkpoints in `data/out/sonar/sae/`, cache `mc4_4M.npy` | Evaluation artifacts across all metrics, `.evals_done` | None |
| `sae_ladder.sh` | SAE baseline + evaluation suite | Shell driver training the 4-rung architectural ablation ladder for SAEs and running initial evaluations (`pareto`, `headstruct`) | Cache `mc4_4M.npy`, GPU monitor | Trained SAE checkpoints in `data/out/sonar/sae/`, `.ladder_done` | None |
| `sonar.py` | Training / fine-tuning | Primary training script for multi-layer Ontologizer autoencoder on SONAR text embeddings with deep supervision and regularization | Module-level globals (no CLI flags), cache `mc4_4M.npy`, `mse_weights.npy`, SONAR text encoder | Checkpoints in `data/out/sonar/multilingual/resid_nc_hm`, `loss.csv`, `loss.png` | None |
| `sonar.sh` | Training / fine-tuning | Launches `sonar.py` inside a persistent detached `tmux` session named `training` | None | Detached tmux session `training` | None |
| `sonar_hc.py` | Training / fine-tuning | Loss ablation training script isolating the head-similarity penalty by monkey-patching `sonar.py` loss weights | Inherits module globals from `sonar.py` with `s_hcossim=1e-5`, `s_Hm=0.0` | Checkpoints and logs in `data/out/sonar/multilingual/resid_nc_hc` | None |
| `splitting.py` | SAE baseline + evaluation suite | Measures cross-model feature splitting, containment, and seed stability between coarse and fine models | `--a`, `--b`, `--cache`, `--rows`, `--tau`, `--taus`, coarse checkpoint A, fine checkpoint B | `<out>/parents.csv`, `<out>/meta.json`, sweep table | `tests/test_splitting.py` |
| `steerfid.py` | SAE baseline + evaluation suite | Evaluates steering fidelity and collateral damage (feature cycle consistency vs chrF text degradation) under matched intervention magnitudes | `--model`, `--mode`, `--cache`, `--mags`, `--sub-rows`, `--device`, checkpoint, SONAR encoder, M2M100 decoder | `<out>/steer.csv`, `<out>/steers.jsonl` | `tests/test_steerfid.py` |
| `textfid.py` | SAE baseline + evaluation suite | Evaluates downstream text reconstruction fidelity by computing character n-gram chrF and M2M100 per-token negative log-likelihood | `--model`, `--cache`, `--rows`, `--b`, `--b-decode`, `--device`, checkpoint, SONAR encoder, M2M100 decoder | `<out>/summary.json`, `<out>/rows.jsonl` | `tests/test_textfid.py` |

---

## 3. Category 1: Data Preparation / Harvesting (3 Scripts)

### `encode_corpus.py`
- **Purpose**: Large-scale offline precomputation of SONAR sentence embeddings for the multilingual mC4 corpus into a memory-mapped NumPy disk cache.
- **Mechanism**:
  - Streams text samples from HuggingFace dataset `allenai/c4` (interleaved across languages defined in `ontologize.data.multilingual.MC4_TO_SONAR`).
  - Tokenizes text with `cointegrated/SONAR_200_text_encoder` using `TokenizeTransform(maxlen=512)`.
  - Computes mean-pooled, L2-normalized 1024-dimensional embeddings via PyTorch.
  - Implements crash resilience: writes to `<name>.npy.part` and `<name>.langs.npy.part` with atomic progress checkpoints saved to `<name>.progress.json`. Flushes every 50 batches. Resuming skips previously encoded samples without re-encoding.
- **Main Inputs**:
  - `-n`, `--n`: Corpus size in samples (default `4_000_000`).
  - `-b`, `--b`: Encoding batch size (default `256`).
  - `-m`, `--name`: Cache name (default `mc4_4M`).
  - `-o`, `--out`: Output directory (default `data/sonar_embeddings`).
  - External resources: HuggingFace dataset `allenai/c4` (train split), pretrained model `cointegrated/SONAR_200_text_encoder`.
- **Main Outputs**:
  - `<out>/<name>.npy`: Shape `(n, 1024)`, `float32` memory-mapped embedding array.
  - `<out>/<name>.langs.npy`: Shape `(n,)`, string dtype `U8` tracking ISO/BCP-47 language codes.
  - `<out>/<name>.meta.json`: Encoding configuration metadata (pooling, sequence length, model ID).
  - `<out>/<name>.progress.json`: Ephemeral progress tracking JSON file during generation.
- **Covering Test in `tests/`**: None *(referenced in comments of `tests/test_loaders.py`)*.

---

### `encode_discord.py`
- **Purpose**: Builds an indexed JSONL and precomputes SONAR embeddings from DiscordChatExporter JSON exports.
- **Mechanism**:
  - Scans an export directory containing channel JSON dumps.
  - Builds an initial row-aligned metadata index (`<name>.jsonl`) storing message text, author, channel ID, and timestamp.
  - Sorts message texts by sequence length in 32,768-row chunk windows for dynamic length-bucketing, minimizing padding waste during transformer encoding.
  - Encodes using SONAR and scatters embeddings back to original index row order.
- **Main Inputs**:
  - `-s`, `--src`: Source directory containing Discord export JSON files (default `../act-i-export`).
  - `-o`, `--out`: Output directory (default `data/sonar_embeddings`).
  - `-m`, `--name`: Cache name (default `act_i`).
  - `-b`, `--b`: Encoding batch size (default `64`).
  - `-c`, `--chunk`: Rows per bucketing window and checkpoint interval (default `32768`).
  - `-l`, `--lang`: SONAR source language token (default `eng_Latn`).
  - `--maxlen`: Maximum token sequence length (default `512`).
- **Main Outputs**:
  - `<out>/<name>.npy`: Shape `(n, 1024)`, `float32` embedding cache.
  - `<out>/<name>.jsonl`: Line-delimited message metadata mapping row indices to Discord messages.
  - `<out>/<name>.meta.json`: Encoding metadata.
  - `<out>/<name>.progress.json`: Progress checkpoint sidecar.
- **Covering Test in `tests/`**: None.

---

### `classify_acts.py`
- **Purpose**: Runs `Ontologizer.classify` over a precomputed embedding cache to extract row-aligned per-layer post-softmax classification activations.
- **Mechanism**:
  - Restores an Ontologizer checkpoint via Orbax (`CheckpointManager`).
  - Evaluates the model forward classification pass in `float64` precision. Double precision is necessary because residual normalization amplifies `float32` rounding errors across successive residual layers, causing batch-shape-dependent argmax flips in layers 3–5.
  - Flattens layer, head, and tag activations into a single 1D index per sample ($f = (l_i \cdot h + h_i) \cdot k + k_i$), matching the feature indexing convention of `autointerp.py`.
- **Main Inputs**:
  - `--ckpt`: Checkpoint directory path (default `data/out/sonar/multilingual/resid_nc_hm`).
  - `--step`: Checkpoint step (default latest).
  - `--emb`: Input embedding cache `.npy` (default `data/sonar_embeddings/act_i.npy`).
  - `--out`: Output `.npy` path (default `<ckpt>/<emb_stem>_classify.npy`).
  - `--temperature`: Softmax temperature (default `0.03`).
  - `--dtype`: Output array storage dtype (`float32` or `float16`, default `float32`).
  - `--compute-dtype`: Forward-pass execution dtype (`float64` or `float32`, default `float64`).
  - `-b`, `--batch`: Evaluation batch size (default `1024`).
  - `-c`, `--chunk`: Flush and checkpoint interval (default `65536`).
  - `-n`, `--limit`: Optional row count limit for smoke testing.
- **Main Outputs**:
  - `<out>.npy`: Post-softmax activation matrix of shape `(N, l * h * k)`.
  - `<out>.meta.json`: Execution metadata recording temperature, checkpoint step, and source files.
  - `<out>.progress.json`: Progress tracking sidecar.
- **Covering Test in `tests/`**: None.

---

## 4. Category 2: Training / Fine-Tuning (3 Scripts)

### `sonar.py`
- **Purpose**: Primary training script for multi-layer Ontologizer dictionary learning models on multilingual SONAR sentence embeddings.
- **Mechanism**:
  - Configures and trains a deep residual `Ontologizer` model with residual recurrence (`forward="resid"`), deep prefix supervision (`deepsup=True`), residual normalization (`resid_norm=True`), and constant coordinate augmentation (`resid_const=True`).
  - Employs a bilinear classifier (`n=2`, `gate="none"`), geometric noise annealing, temperature hardening schedules, and batch mean-entropy penalty ($s_{Hm}$).
  - Uses JIT-compiled training steps with Optax Adam and gradient clipping.
  - Can train from cached `.npy` embeddings (via `cache = "data/sonar_embeddings/mc4_4M.npy"`) or stream live text from HuggingFace mC4.
- **Main Inputs**:
  - Configured via top-level module variables (no CLI flags):
    - `cache`: Path to embedding `.npy` (e.g. `data/sonar_embeddings/mc4_4M.npy`).
    - `d`: 1024 (embedding dimension).
    - `e_enc`: 2048 (encoder width), `e_dec`: 2048 (decoder height).
    - `k`: 32 (tags per head), `h`: 32 (heads per layer), `l`: 5 (DictEnc layers).
    - `epochs`: 24, `b`: 256, `lr`: 5e-5.
    - Loss scalings: `s_MSE`, `s_g` (ghost gradient), `s_L1K`, `s_L1F`, `s_cossim_b`, `s_hcossim`, `s_Hm`.
- **Main Outputs**:
  - Checkpoints under `data/out/sonar/multilingual/resid_nc_hm/` managed by Orbax (`state`, `spec`, and `metadata.json`).
  - Step metrics log in `<out>/loss.csv` (`loss, MSE, MSE_ghost, L1_K, L1_F, entropy, cossim_b, cossim_h, KL_m`).
  - Loss curve plot in `<out>/loss.png`.
- **Covering Test in `tests/`**: None *(referenced in comments of `tests/test_loaders.py` and `tests/test_training_step.py`)*.

---

### `sonar_hc.py`
- **Purpose**: Loss ablation variant of `sonar.py` isolating the head-similarity penalty ($s_{hcossim}$) versus the batch mean-entropy bonus ($s_{Hm}$).
- **Mechanism**:
  - Imports `sonar.py` as a module and directly monkey-patches module-level globals:
    - `sonar.s_hcossim = 1e-5` (matching phase 1 of `resid_nc_hm`).
    - `sonar.s_Hm = 0.0` (disabling the entropy bonus while retaining the logged stat).
    - `sonar.out = sonar.path / "out/sonar/multilingual/resid_nc_hc"`.
  - Invokes `sonar.main()`.
- **Main Inputs**:
  - Inherits all configuration, data paths, and parameters from `sonar.py`.
- **Main Outputs**:
  - Checkpoints and logs under `data/out/sonar/multilingual/resid_nc_hc/`.
- **Covering Test in `tests/`**: None.

---

### `sonar.sh`
- **Purpose**: Shell launcher that executes `sonar.py` inside a detached `tmux` session named `training`.
- **Mechanism**:
  - Executes `tmux new -s training` and `uv run sonar.py`.
- **Main Inputs**: None.
- **Main Outputs**: Detached background `tmux` session.
- **Covering Test in `tests/`**: None.

---

## 5. Category 3: Interactive / Decoding / Chat (2 Scripts)

### `decode.py`
- **Purpose**: Interactive CLI shell for encoding arbitrary text, executing head-level dictionary interventions via `Ontologizer.withArgs`, and decoding reconstructed embeddings back to natural language text.
- **Mechanism**:
  - Loads the PyTorch SONAR text encoder (`cointegrated/SONAR_200_text_encoder`) and M2M100 conditional text decoder (`raxtemur/SONAR_200_text_decoder`).
  - Restores an Orbax Ontologizer checkpoint and initializes `ChatEnv`.
  - In an interactive loop:
    1. Reads user prompt text and intervention commands (`set <layer> <head> <tag>`, `run <text>`, `q`).
    2. Encodes input text to a 1024-d SONAR embedding and records the raw L2 norm.
    3. Bridges embedding to JAX via DLPack and runs `Ontologizer.withArgs` at temperature `0.1`.
    4. Computes reconstruction MSE.
    5. Converts reconstructed embedding back to PyTorch, renormalizes to the reference embedding norm, and decodes to English (`eng_Latn`) text using M2M100 beam search (beam width 4, length 128).
- **Main Inputs**:
  - Positional CLI argument: `checkpoint` directory (default `data/out/sonar/linear`).
  - Interactive stdin commands managed by `ChatEnv`.
- **Main Outputs**:
  - Console prints showing reconstruction MSE and decoded text strings.
- **Covering Test in `tests/`**: None.

---

### `decode_tags.py`
- **Purpose**: Batch decodes every trained dictionary entry and uniform tag representation into natural language text via M2M100.
- **Mechanism**:
  - Restores an Ontologizer checkpoint and queries `Ontologizer.decodeEntries` and `Ontologizer.decodeUniform`.
  - Computes an empirical reference norm across 6 standard English reference sentences.
  - Scales tag embeddings by the reference norm and feeds them into M2M100 in batches of 16.
  - Generates decoded text labels for all $l \times h \times k$ entries and uniform head tags.
- **Main Inputs**:
  - Positional CLI argument: `checkpoint` directory (default `data/out/sonar/linear`).
  - Pretrained encoder `cointegrated/SONAR_200_text_encoder` and decoder `raxtemur/SONAR_200_text_decoder`.
- **Main Outputs**:
  - `<checkpoint>/entries.txt`: Natural language text strings for every dictionary entry.
  - `<checkpoint>/uniform.txt`: Text strings decoded from uniform tag distributions per head.
- **Covering Test in `tests/`**: None *(referenced in docstrings of `tests/test_code_geometry.py`)*.

---

## 6. Category 4: SAE Baseline + Evaluation Suite (14 Scripts)

### `sae.py`
- **Purpose**: Standalone trainer and model suite for Sparse Autoencoder (SAE) baselines (Top-K, ReLU+L1, grouped competition, prefix deep supervision, and bilinear encoders).
- **Mechanism**:
  - Trains standard Top-K SAEs (Gao et al. 2024) or ReLU+L1 SAEs using whitened MSE on the same precomputed SONAR embedding cache as `sonar.py`.
  - Supports 4 structural-ablation rungs:
    1. Standard Top-K: `--topk K`.
    2. Grouped Competition: `--groups G --group-fn top1|softmax` (latents partitioned into competing heads; Top-1 allows silence, Softmax emits dense distributions).
    3. Prefix Deep Supervision: `--prefixes P` (nested Matryoshka prefix losses over contiguous latent blocks).
    4. Bilinear Encoders: `--enc bilinear` (Pearce et al. 2025 quadratic encoders with closed-form eigenfeatures).
  - Implements aux-k dead latent revival and pre-subtracted decoder bias.
- **Main Inputs**:
  - CLI flags: `--m` (latents, default 5120), `--topk` (default 32), `--l1` (default 0.0), `--groups` (default 0), `--group-fn` (`top1` or `softmax`), `--prefixes` (default 1), `--enc` (`linear` or `bilinear`), `--epochs` (default 20), `--lr` (default 1e-4), `--b` (default 4096), `--cache` (default `data/sonar_embeddings/mc4_4M.npy`), `--mse-weights` (default `data/out/sonar/mse_weights.npy`), `--eval-rows` (default 32768), `--aux-k` (default 512), `--dead-steps` (default 1000), `--seed` (default 42).
- **Main Outputs**:
  - Output directory `data/out/sonar/sae/<run_name>/`:
    - `params.npz`: Trained weights (`W_dec`, `b_dec`, `b_enc`, `W_enc` or `W_enc1`/`W_enc2`).
    - `state.npz`: Optimizer state and step counters.
    - `loss.csv`: Step-by-step training progress (`step, loss, mse_w, fvu, l0, n_dead`).
- **Covering Test in `tests/`**: `tests/test_sae.py` *(also imported by `tests/test_refit.py`)*.

---

### `pareto.py`
- **Purpose**: Computes and plots the capacity-Pareto frontier (reconstruction whitened FVU vs transmission capacity) comparing Ontologizer deviation codes against Top-K SAEs.
- **Mechanism**:
  - Compares continuous coefficients transmitted per sample against discrete index side-channel bits on held-out cache tail rows (`--eval-rows`).
  - Evaluates two Ontologizer code types against the corpus-mean origin ($E[p]$):
    - `dev`: Keeps the top $m$ entries per head with largest $|p - E[p]|$, pinning the remainder to the origin.
    - `head`: Keeps the full soft distribution for the $m$ lowest-entropy heads per layer and mean-ablates the rest.
  - Automatically parses and plots SAE baseline runs from `data/out/sonar/sae/`.
- **Main Inputs**:
  - `--ckpt`: Ontologizer checkpoint directory (default `data/out/sonar/multilingual/resid_nc`).
  - `--step`: Checkpoint step (default 0, latest).
  - `--temperature`: Softmax temperature (default 0.03).
  - `--ms`: List of deviation counts per head or heads per layer (default `[1, 2, 3, 4, 6, 8, 16]`).
  - `--code`: Code type (`dev`, `head`, or `both`, default `dev`).
  - `--origin`: Baseline origin (`meanp` or `uniform`, default `meanp`).
  - `--origin-rows`: Training rows used to estimate corpus-mean $E[p]$ (default 65536).
  - `--sae`: Paths to SAE `params.npz` files (default discovers all under `data/out/sonar/sae/*/`).
  - `--cache`: Embedding cache path (default `data/sonar_embeddings/mc4_4M.npy`).
  - `--eval-rows`: Held-out tail rows (default 32768).
  - `--plot`: Flag to generate log-log plot.
- **Main Outputs**:
  - `<out>/pareto.csv`: Columns `label, coeffs, index_bits, fvu_w`.
  - `<out>/pareto.png`: Log-log plot of FVU vs active coefficients (when `--plot` is passed).
  - Formatted stdout table.
- **Covering Test in `tests/`**: `tests/test_pareto.py`.

---

### `refit.py`
- **Purpose**: Isolates feature selection error from magnitude shrinkage error by solving unconstrained whitened least-squares coefficients on frozen feature supports.
- **Mechanism**:
  - For each sample, freezes the active feature support (active latents for SAEs; top-$m$ deviation entries for Ontologizer).
  - Computes the closed-form ridge least-squares reconstruction on that support:
    $$\min_p \| (X - b_{dec}) - p A \|_W^2 + \lambda \|p\|^2$$
  - Measures the gap between raw model FVU and refit FVU to quantify magnitude shrinkage distortion.
- **Main Inputs**:
  - `--sae`: Paths to SAE `params.npz` files (default all under `data/out/sonar/sae/*/`).
  - `--onto`: Optional Ontologizer checkpoint directory.
  - `--ms`: List of deviation counts for Ontologizer (default `[1, 2, 4, 8, 16]`).
  - `--cache`: Embedding cache path (default `data/sonar_embeddings/mc4_4M.npy`).
  - `--rows`: Evaluation tail rows (default 32768).
  - `--ridge`: Ridge regularization constant (default 1e-6).
  - `--max-support`: Maximum active features threshold to avoid underdetermined solves (default 2600).
- **Main Outputs**:
  - `<out>/refit.csv`: Columns `model, k, raw_fvu, refit_fvu, delta, rel_gain`.
  - Comparison table printed to stdout.
- **Covering Test in `tests/`**: `tests/test_refit.py`.

---

### `headcoh.py`
- **Purpose**: Evaluates semantic coherence of trained heads or discovered groups by measuring within-head decode direction and description similarities against null baselines.
- **Mechanism**:
  - Geometry view: Computes mean off-diagonal pairwise cosine similarity and top-singular-value energy fraction over decode directions ($G$ rows for Ontologizer via `refit.onto_linear_model`, or $W_{dec}$ rows for SAEs). Z-scores observed metrics against size-matched random subsets.
  - Description view (`--desc-dir`): Encodes autointerp description texts via SONAR and evaluates within-head description cosine similarity against label permutations.
- **Main Inputs**:
  - `--model`: `onto` or `sae` (required).
  - `--ckpt`: Checkpoint path (required).
  - `--assignment`: Discovered group assignment array `.npy` (for post-hoc SAE groups).
  - `--metric`: `raw` SONAR cosine or `whitened` inner product (default `raw`).
  - `--desc-dir`: Directory containing `descriptions.jsonl` from `autointerp.py`.
  - `--desc-mode`: Description mode to analyze (default `cacts`).
  - `--n-null`: Number of random permutations for geometry null (default 500).
  - `--n-perm`: Number of permutations for description null (default 2000).
- **Main Outputs**:
  - `<out>/groups.csv`: Columns `group, pool, size, mean_cos, z_cos, sv1, z_sv`.
  - `<out>/summary.json`: Global mean z-scores and description coherence statistics.
- **Covering Test in `tests/`**: `tests/test_headcoh.py`.

---

### `headstruct.py`
- **Purpose**: Discovers and scores latent categorical head structure in SAEs based on paradigmatic substitution.
- **Mechanism**:
  - Clusters SAE latents by co-firing profile similarity with other latents while penalizing mutual co-firing ($PMI < 1$).
  - Evaluates discovered groups (or trained groups with `--trained-groups`) on three criteria:
    1. Exhaustiveness: $P(\text{group fires})$.
    2. Exclusivity: $E[\# \text{fired} \mid \ge 1]$.
    3. Sum Stability: Coefficient of variation $CV(\sum z)$.
  - Scores metrics against size-matched random partitions (`--nulls`) using a split-half cross-validation design (clusters discovered on first half of `--rows`, evaluated on second half).
- **Main Inputs**:
  - `--sae`: Path to SAE `params.npz` (required).
  - `--cache`: Embedding cache path (default `data/sonar_embeddings/mc4_4M.npy`).
  - `--rows`: Training rows for co-firing statistics (default 65536).
  - `--size`: Group size cap matching Ontologizer head size $k$ (default 32).
  - `--topn`: Candidate neighbours per latent (default 64).
  - `--min-fire`: Minimum latent firing frequency (default 1e-3).
  - `--fire-thr`: Threshold defining an activation (default 0.0).
  - `--nulls`: Random partition iterations (default 20).
  - `--trained-groups`: Flag to score `meta.json` partition rather than discovering clusters.
- **Main Outputs**:
  - `<out>/groups.csv`: Per-group size, exhaustiveness, exclusivity, and sum stability z-scores.
  - `<out>/assignment.npy`: Integer cluster assignment array of length $m$.
  - Formatted stdout summary table.
- **Covering Test in `tests/`**: `tests/test_headstruct.py`.

---

### `langprobe.py`
- **Purpose**: Evaluates language identity representation using sparse and dense linear probes on SAE latents and Ontologizer activations.
- **Mechanism**:
  - Labels samples using the `.langs.npy` sidecar from mC4 (~86 languages).
  - Selects top latents per language using two-sample t-statistics on training rows.
  - Fits and evaluates one-vs-rest logistic probes on held-out test rows:
    - $k=1$: Single best latent with threshold tuning.
    - $k>1$: $k$-sparse logistic regression probes.
    - Dense: Full 1024-dimensional raw embedding baseline ceiling.
- **Main Inputs**:
  - `--model`: Path to SAE `params.npz` or Ontologizer checkpoint directory (required).
  - `--cache`: Embedding cache path (default `data/sonar_embeddings/mc4_4M.npy`).
  - `--langs`: Language sidecar file (default `<cache>.langs.npy`).
  - `--train-rows`: Training rows (default 65536).
  - `--test-rows`: Held-out evaluation rows (default 16384).
  - `--ks`: Active latent counts for sparse probes (default `[1, 4, 16]`).
  - `--min-count`: Minimum language frequency in training split (default 200).
  - `--no-dense`: Flag to skip raw dense baseline.
- **Main Outputs**:
  - `<out>/langs.csv`: Detailed per-language probe F1 scores across $k$ values.
  - Macro-F1 summary table printed to stdout.
- **Covering Test in `tests/`**: `tests/test_langprobe.py`.

---

### `splitting.py`
- **Purpose**: Measures cross-model feature splitting, containment, and seed stability between coarse and fine models.
- **Mechanism**:
  - Compares model A (coarse) against model B (fine) on shared embedding cache rows.
  - Evaluates activation containment:
    - Child: $P(A_i \mid B_j) \ge \tau$ and $P(B_j \mid A_i) < \tau$.
    - Match: Bidirectional containment $\ge \tau$.
    - Split: Parent $A_i$ with $\ge 2$ children in model B.
  - Evaluates child union coverage, child pairwise overlap, and geometric decoder cosine alignment.
  - Used for seed stability benchmarking by comparing two identical SAE models trained with different random seeds (`seed 42` vs `seed 43`).
- **Main Inputs**:
  - `--a`: Coarse model (`params.npz` or Ontologizer directory, required).
  - `--b`: Fine model (`params.npz` or Ontologizer directory, required).
  - `--cache`: Embedding cache path (default `data/sonar_embeddings/mc4_4M.npy`).
  - `--rows`: Rows to evaluate (default 524288).
  - `--tau`: Primary containment threshold for CSV detail (default 0.7).
  - `--taus`: Containment threshold sweep for summary (default `[0.3, 0.5, 0.7, 0.9]`).
  - `--min-fires`: Minimum firing count for feature inclusion (default 50).
- **Main Outputs**:
  - `<out>/parents.csv`: Per-parent multiplicity, coverage, and overlap statistics at threshold $\tau$.
  - `<out>/meta.json`: Configuration and sweep summary.
  - Formatted stdout table.
- **Covering Test in `tests/`**: `tests/test_splitting.py`.

---

### `steerfid.py`
- **Purpose**: Evaluates causal steering fidelity by measuring intended effect against text degradation at matched intervention magnitudes.
- **Mechanism**:
  - Steers held-out embeddings along native feature directions:
    - SAE: Decoder row $W_{dec}[j]$ (`mode dec`) or closed-form bilinear eigenfeature (`mode eig`).
    - Ontologizer: Forced tag assignment via `withArgs` minus base reconstruction.
    - Control: Random unit directions (`--n-random`).
  - Decodes steered embeddings through M2M100.
  - Re-encodes generated text with SONAR to measure cycle consistency (effect) vs $1 - \text{chrF}(\text{base}, \text{steered})$ (collateral text damage).
- **Main Inputs**:
  - `--model`: SAE `params.npz` or Ontologizer checkpoint directory (required).
  - `--mode`: Steering direction mode (`dec` or `eig`, default `dec`).
  - `--cache`: Embedding cache path (default `data/sonar_embeddings/mc4_4M.npy`).
  - `--n-features`: Number of live features to steer (default 16).
  - `--n-samples`: Samples steered per feature (default 8).
  - `--mags`: Normalized intervention magnitudes (default `[0.25, 0.5, 1.0]`).
  - `--n-random`: Random control direction count (default 8).
  - `--sub-rows`: Reference rows for firing quantiles (default 16384).
  - `--device`: PyTorch device for M2M100 generation (default `cpu`).
- **Main Outputs**:
  - `<out>/steer.csv`: Aggregate effect and collateral damage per magnitude.
  - `<out>/steers.jsonl`: Qualitative log of base text, steered text, and cycle-consistency metrics.
- **Covering Test in `tests/`**: `tests/test_steerfid.py`.

---

### `textfid.py`
- **Purpose**: Evaluates downstream generation fidelity by measuring text reconstruction chrF and negative log-likelihood (NLL).
- **Mechanism**:
  - Decodes original embeddings and reconstructed embeddings through M2M100 (forced `eng_Latn`).
  - Evaluates two text fidelity metrics:
    - `chrF`: Character n-gram F2 score between original and reconstructed text decodes.
    - `NLL`: Mean per-token negative log-likelihood under M2M100, comparing the reconstruction against the original embedding ceiling and corpus-mean floor.
- **Main Inputs**:
  - `--model`: SAE `params.npz` or Ontologizer checkpoint directory (required).
  - `--cache`: Embedding cache path (default `data/sonar_embeddings/mc4_4M.npy`).
  - `--rows`: Held-out cache rows to decode (default 512).
  - `--b`: Reconstruction batch size in JAX (default 4096).
  - `--b-decode`: Generation batch size in PyTorch (default 16).
  - `--device`: PyTorch device for M2M100 (default `cpu`).
- **Main Outputs**:
  - `<out>/summary.json`: Aggregate chrF, exact-match percentage, and NLL recovered.
  - `<out>/rows.jsonl`: Sample-by-sample text decodes and individual scores.
- **Covering Test in `tests/`**: `tests/test_textfid.py`.

---

### `compose.py`
- **Purpose**: Tests causal intervention composition and additivity in decode space.
- **Mechanism**:
  - Evaluates pairs of set-interventions on two distinct heads: $(l_1, h_1) \to k_1$ and $(l_2, h_2) \to k_2$.
  - Compares the joint intervention decode delta $D_{12}$ against the linear sum of single-intervention deltas $D_1 + D_2$.
  - Computes cosine similarity $\cos(D_{12}, D_1 + D_2)$ and relative error $\|D_{12} - (D_1 + D_2)\| / \|D_1 + D_2\|$ under both raw and whitened metrics.
  - Isolates downstream layer reclassification effects (final layer pairs compose additively by construction).
- **Main Inputs**:
  - `--ckpt`: Ontologizer checkpoint directory (required).
  - `--cache`: Embedding cache path (default `data/sonar_embeddings/mc4_4M.npy`).
  - `--eval-rows`: Held-out tail slice (default 32768).
  - `--rows`: Sampled evaluation rows (default 512).
  - `--singles-per-layer`: Single-head interventions sampled per layer (default 8).
  - `--cross-pairs`: Random cross-layer pairs evaluated (default 60).
  - `--mse-weights`: Inverse-variance metric weights (default `data/out/sonar/mse_weights.npy`).
- **Main Outputs**:
  - `<out>/pairs.csv`: Per-pair metrics across intra-layer and cross-layer strata.
  - `<out>/summary.json`: Median cosine and relative error per stratum.
- **Covering Test in `tests/`**: `tests/test_compose.py`.

---

### `autointerp.py`
- **Purpose**: 4-stage automated interpretability pipeline (harvest, texts, describe, score) comparing Ontologizer tags and SAE latents on an equal footing.
- **Mechanism**:
  - **Stage 1 (`harvest`)**: Scans cache slice to record top-activating sample row indices, activation quantiles, and firing rates. Computes parameter decode embeddings ($W_{dec}$ row for SAEs, one-hot tag decode for Ontologizer).
  - **Stage 2 (`texts`)**: Re-streams mC4 corpus deterministically in `encode_corpus.py` order to extract text snippets for harvested row indices.
  - **Stage 3 (`describe`)**: Generates natural language feature descriptions using three modes:
    - `params`: Decodes parameter embeddings directly via M2M100 without an LLM.
    - `acts`: Prompts Anthropic Claude LLM judge to summarize top-activating snippets.
    - `cacts`: Contrastive prompt showing top-activating snippets alongside random corpus draws to eliminate generic web-scrape noise descriptions.
  - **Stage 4 (`score`)**: Blinded detection benchmark. The LLM judge evaluates held-out positive snippets and random low-activation negatives, outputting JSON matching decisions to compute precision, recall, and F1. `--null` scores items against mismatched feature descriptions to measure the chance floor.
- **Main Inputs**:
  - Positional subcommands: `harvest`, `texts`, `describe`, `score`.
  - Common flags: `--model`, `--ckpt`, `--topk`, `--groups`, `--n-features`, `--dir`, `--mode`, `--judge`, `--rate`, `--null`, `--dry-run`.
  - Environment / API: `ANTHROPIC_API_KEY` (required for LLM describe and score stages).
- **Main Outputs**:
  - `<dir>/features.npz`: Harvested activations, quantiles, and parameter embeddings.
  - `<dir>/texts.jsonl`: Extracted corpus text snippets.
  - `<dir>/descriptions.jsonl`: Generated feature descriptions.
  - `<dir>/scores.csv`: Detailed scoring metrics (precision, recall, F1, null F1).
- **Covering Test in `tests/`**: `tests/test_autointerp.py`.

---

### `autointerp_run.sh`
- **Purpose**: Automated shell driver orchestrating the multi-model autointerp campaign across Ontologizer (`resid_nc`) and 5 SAE variants.
- **Mechanism**:
  - Verifies `ANTHROPIC_API_KEY` and checks GPU availability ($\ge 6000$ MB free).
  - Executes `autointerp.py` through stages 1 to 4 across 6 benchmark models:
    `onto resid_nc`, `m5120_k32`, `m5120_k32_bl`, `m5120_g160top1`, `m5120_g160softmax`, `m11264_k5120`.
  - Employs artifact guarding: resumes interrupted runs without duplicating API queries.
- **Main Inputs**:
  - `ANTHROPIC_API_KEY`, optional `AI_RATE` (default 1000).
  - Trained model checkpoints in `data/out/sonar/`.
- **Main Outputs**:
  - Campaign directories under `data/out/sonar/autointerp/`.
  - Sentinel marker file `data/out/sonar/autointerp/.campaign_done`.
- **Covering Test in `tests/`**: None.

---

### `sae_ladder.sh`
- **Purpose**: Shell driver automating training and initial evaluation for the 4-rung SAE structural-ablation ladder.
- **Mechanism**:
  - Monitors GPU memory availability before launching runs.
  - Trains 4 architectural variants of `sae.py` at $m=5120$, $b=4096$, $lr=4\times 10^{-4}$ for 150 epochs:
    1. Rung 1: Plain Top-K (`--topk 32`).
    2. Rung 2a: Hard grouped Top-1 (`--groups 160 --group-fn top1`).
    3. Rung 2b: Soft grouped Softmax (`--groups 160 --group-fn softmax`).
    4. Rung 3: Top-K with 5 nested prefix losses (`--topk 32 --prefixes 5`).
  - Executes `pareto.py` and `headstruct.py` on all completed rungs.
- **Main Inputs**:
  - Embedding cache `data/sonar_embeddings/mc4_4M.npy` and `mse_weights.npy`.
- **Main Outputs**:
  - Checkpoints under `data/out/sonar/sae/`.
  - Sentinel marker file `data/out/sonar/sae/.ladder_done`.
- **Covering Test in `tests/`**: None.

---

### `sae_evals.sh`
- **Purpose**: Shell driver automating Phase 2 of the SAE evaluation battery across all rungs and the Ontologizer.
- **Mechanism**:
  - Completes remaining training runs: `m11264_k5120`, `m5120_k32_bl` (bilinear encoder), and seed replica `m5120_k32_s43`.
  - Executes the full evaluation suite sequentially with artifact guards:
    - `pareto.py`
    - `headstruct.py` (discovered and trained groups)
    - `splitting.py` (width scaling and seed stability)
    - `textfid.py`
    - `langprobe.py`
    - `refit.py`
- **Main Inputs**:
  - Checkpoints in `data/out/sonar/sae/` and `data/out/sonar/multilingual/resid_nc`.
- **Main Outputs**:
  - Evaluation CSV files and JSON summaries across all evaluation directories.
  - Sentinel marker file `data/out/sonar/sae/.evals_done`.
- **Covering Test in `tests/`**: None.

---

## 7. Category 5: Scratch / Stale / Experimental (5 Scripts)

### `mnist.py`
- **Purpose**: Abandoned legacy script attempting to train an Ontologizer autoencoder on MNIST image pixels.
- **Status**: **Completely Broken and Inoperable.**
  - Line 7 contains an invalid import: `from ontologize.training.data import ImageLoader`. The `ontologize.training` package does not contain a `data` module (`ImageLoader` resides in `ontologize.data.loaders`).
  - Lines 39–43 instantiate `Ontologizer`, `Hyperparams`, and `Metadata` with outdated and malformed positional arguments that do not match current constructor signatures.
- **Main Inputs**: None (hardcoded configuration).
- **Main Outputs**: None (fails immediately on import).
- **Covering Test in `tests/`**: None.

---

### `run_chat.py`
- **Purpose**: Ad-hoc interactive test harness driving `decode.py` through a pseudo-terminal (`pty`).
- **Mechanism**:
  - Allocates a pseudo-terminal pair via `pty.openpty()` and runs `decode.py` in a subprocess with `XLA_PYTHON_CLIENT_MEM_FRACTION=".10"`.
  - Simulates interactive user commands via stdin: `set 0 k_add 5,10`, `run The quick brown fox jumps over the lazy dog.`, and `q`.
  - Captures terminal escape codes and stdout to console.
- **Main Inputs**: Hardcoded strings and checkpoint path `data/out/sonar/linear/multilingual`.
- **Main Outputs**: Console output.
- **Covering Test in `tests/`**: None.

---

### `run_chat2.py`
- **Purpose**: Non-interactive headless CPU smoke test for `decode.py`.
- **Mechanism**:
  - Sets `CUDA_VISIBLE_DEVICES=""` to enforce CPU-only execution.
  - Pipes hardcoded commands into `decode.py` via `subprocess.run` with `capture_output=True`.
- **Main Inputs**: Hardcoded command string and checkpoint `data/out/sonar/linear/multilingual`.
- **Main Outputs**: Captured STDOUT and STDERR printed to console.
- **Covering Test in `tests/`**: None.

---

### `run_decode.py`
- **Purpose**: Non-interactive test script feeding a batch of paragraphs into `decode.py` for CPU text reconstruction smoke testing.
- **Mechanism**:
  - Runs `decode.py` in a CPU-only subprocess (`CUDA_VISIBLE_DEVICES=""`).
  - Passes a hardcoded multi-line input text discussing signal anti-aliasing and research funding.
- **Main Inputs**: Hardcoded paragraph text and checkpoint `data/out/sonar/linear/multilingual`.
- **Main Outputs**: Captured STDOUT and STDERR printed to console.
- **Covering Test in `tests/`**: None.

---

### `run_decode_tags.py`
- **Purpose**: Subprocess execution wrapper for `decode_tags.py`.
- **Mechanism**:
  - Runs `decode_tags.py data/out/sonar/linear/multilingual` inside a subprocess with restricted XLA client memory fraction (`XLA_PYTHON_CLIENT_MEM_FRACTION=".10"`).
- **Main Inputs**: Hardcoded checkpoint path.
- **Main Outputs**: Captured STDOUT and STDERR printed to console.
- **Covering Test in `tests/`**: None.

---

## 8. Cross-Script Dependencies and Coupling

The root scripts exhibit significant **horizontal coupling**, importing functions and classes directly from one another at the repository root level:

```
sae.py <──────────────── pareto.py <─────────────── compose.py (loads pareto.load_onto)
  ▲                         ▲
  │                         ├── refit.py (loads pareto.parse_sae_name, autointerp.onto_acts_fn)
  │                         │     ▲
  │                         │     └── headcoh.py (loads refit.onto_linear_model, autointerp.ENCODER_ID)
  │                         │
  │                         ├── splitting.py (loads pareto.parse_sae_name, autointerp.onto_acts_fn)
  │                         │     ▲
  │                         │     └── langprobe.py (loads splitting.load_model)
  │                         │
  │                         ├── steerfid.py (loads pareto.load_onto, textfid.chrf)
  │                         │
  │                         └── textfid.py (loads pareto.parse_sae_name, pareto.load_onto)
  │
  ├── headstruct.py (loads sae, pareto.parse_sae_name)
  └── autointerp.py (loads sae)

sonar.py <─────────────── sonar_hc.py (imports and mutates sonar globals)

decode.py <────────────── run_chat.py, run_chat2.py, run_decode.py (subprocess invocations)
decode_tags.py <───────── run_decode_tags.py (subprocess invocation)
```

### Unit Test Invocations (`tests/`)
11 test files in `tests/` import root-level scripts as top-level modules:
1. `tests/test_autointerp.py` imports `autointerp` prompt templates and helpers.
2. `tests/test_compose.py` imports `compose.additivity_stats`, `compose.sample_singles`.
3. `tests/test_headcoh.py` imports `headcoh` pure helper functions.
4. `tests/test_headstruct.py` imports `headstruct`.
5. `tests/test_langprobe.py` imports `langprobe`.
6. `tests/test_pareto.py` imports `pareto`.
7. `tests/test_refit.py` imports `refit` and `sae`.
8. `tests/test_sae.py` imports `sae`.
9. `tests/test_splitting.py` imports `splitting`.
10. `tests/test_steerfid.py` imports `steerfid`.
11. `tests/test_textfid.py` imports `textfid.chrf`.

---

## 9. Catalog of Observed Bugs, Dead Code, and Code Oddities

*(Observed during script survey; recorded here without altering existing code)*

1. **Dead Code and Broken Imports in `mnist.py`**:
   - `from ontologize.training.data import ImageLoader` fails with `ModuleNotFoundError`.
   - Constructor arguments for `Hyperparams` and `Metadata` do not align with library dataclasses.
2. **Global Variable Monkey-Patching in `sonar_hc.py`**:
   - `sonar_hc.py` mutates `sonar.s_hcossim`, `sonar.s_Hm`, and `sonar.out` at the module level before calling `sonar.main()`.
3. **Hardcoded Module Globals in Training Scripts**:
   - `sonar.py` and `sonar_hc.py` do not support CLI arguments via `argparse`; configurations require directly editing source files.
4. **Self-Restarting CuDNN Process Replacement**:
   - `decode.py` and `decode_tags.py` inspect `CUDNN_INJECTED` and call `os.execv(sys.executable, ...)` to inject CuDNN into `LD_LIBRARY_PATH`.
5. **Hardcoded Paths in Scratch Scripts**:
   - `run_chat.py`, `run_chat2.py`, `run_decode.py`, and `run_decode_tags.py` hardcode a legacy checkpoint path (`data/out/sonar/linear/multilingual`).
6. **External Path Dependency in `encode_discord.py`**:
   - Defaults to `src = "../act-i-export"`, pointing outside the repository workspace.
7. **Root-Level Script Import Coupling**:
   - Evaluation scripts treat sibling root scripts as top-level Python modules (e.g. `import sae`), preventing easy relocation into subdirectories without `sys.path` or packaging adjustments.
