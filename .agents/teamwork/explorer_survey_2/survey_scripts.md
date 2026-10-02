# Comprehensive Census and Survey of Root-Level Scripts

**Date:** 2026-09-29  
**Agent:** explorer_survey_2 (Workstream 2: Root Script Index)  
**Target Repository:** `/home/jade/disk2/ontologizercleanup/ontologize`  
**Scope:** Exhaustive audit of all 27 root-level scripts (23 `.py` and 4 `.sh` files).

---

## 1. Executive Summary

A complete census of the repository root confirmed exactly 27 executable scripts: **23 Python scripts (`*.py`)** and **4 Bash shell scripts (`*.sh`)**.

- **11 scripts (all evaluation/SAE suite modules)** are directly imported and tested by unit tests in `tests/`.
- **16 scripts** have no direct unit tests (`Covering Test: None`), including all 4 shell scripts, all data harvesting scripts, all interactive decoding scripts, both training scripts, and all scratch/one-off wrappers.
- The evaluation suite exhibits extensive **horizontal coupling**: evaluation scripts import helper functions, model loaders, and baseline classes directly from one another at the repository root level (e.g. `langprobe` -> `splitting`, `splitting` -> `pareto` + `autointerp` + `sae`, `steerfid` -> `pareto` + `textfid` + `sae`, `headcoh` -> `refit` + `autointerp`).
- One legacy script (`mnist.py`) is completely broken and obsolete, referencing non-existent submodules (`ontologize.training.data`) and calling outdated constructor signatures.
- Four wrapper scripts (`run_chat.py`, `run_chat2.py`, `run_decode.py`, `run_decode_tags.py`) are ad-hoc scratch drivers invoking `decode.py` and `decode_tags.py` with hardcoded inputs and paths.

---

## 2. Master Script Index Table (27 Scripts)

| Script | Category | Purpose / One-Line Summary | Main Inputs | Main Outputs | Covering Test in `tests/` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `autointerp.py` | SAE baseline + evaluation suite | Multi-stage automated interpretability pipeline (harvest activations, stream texts, generate LLM/param descriptions, score detection) | CLI args (`stage`, `--model`, `--ckpt`, `--topk`, `--judge`, etc.), cache `.npy`, Anthropic API key | `features.npz`, `texts.jsonl`, `descriptions.jsonl`, `scores.csv` | `tests/test_autointerp.py` |
| `autointerp_run.sh` | SAE baseline + evaluation suite | Automated shell driver orchestrating the 4-stage autointerp campaign across Ontologizer and 5 SAE variants | `ANTHROPIC_API_KEY`, checkpoints in `data/out/sonar/` | Campaign artifacts in `data/out/sonar/autointerp/`, `.campaign_done` | None |
| `classify_acts.py` | Data preparation / harvesting | Runs `Ontologizer.classify` in float64 over precomputed embedding caches to generate row-aligned per-layer post-softmax activations | CLI args (`--ckpt`, `--emb`, `--temperature`, `--dtype`, `--compute-dtype`), checkpoint, cache `.npy` | `<out>.npy` (activations `(N, l*h*k)`), `<out>.meta.json`, `<out>.progress.json` | None |
| `compose.py` | SAE baseline + evaluation suite | Evaluates intervention composition and additivity by comparing joint dual-head interventions against summed single-intervention deltas | CLI args (`--ckpt`, `--cache`, `--eval-rows`, `--rows`, `--mse-weights`), checkpoint, cache `.npy` | `<out>/pairs.csv`, `<out>/summary.json`, console metrics | `tests/test_compose.py` |
| `decode.py` | Interactive / decoding / chat | Interactive CLI shell to encode text into SONAR space, apply head interventions via `Ontologizer.withArgs`, and decode back to text via M2M100 | Checkpoint directory (`data/out/sonar/linear`), interactive stdin commands via `ChatEnv` | Interactive console output (MSE, reconstructed strings) | None |
| `decode_tags.py` | Interactive / decoding / chat | Decodes all Ontologizer dictionary entries and uniform reference tags into natural language text via M2M100 using calibrated reference norm scaling | Checkpoint directory (`data/out/sonar/linear`), pretrained SONAR encoder and M2M100 decoder | `<ckpt>/entries.txt`, `<ckpt>/uniform.txt` | None |
| `encode_corpus.py` | Data preparation / harvesting | Streams multilingual mC4 text from HuggingFace, computes SONAR embeddings, and saves a crash-resilient `.npy` memmap disk cache | CLI args (`-n`, `-b`, `-m`, `-o`), HuggingFace `allenai/c4` dataset, SONAR text encoder | `<out>/<name>.npy`, `<out>/<name>.langs.npy`, `<out>/<name>.meta.json`, `<out>/<name>.progress.json` | None |
| `encode_discord.py` | Data preparation / harvesting | Parses DiscordChatExporter JSON files, indexes non-empty messages to `.jsonl`, and encodes dynamic length-bucketed SONAR embeddings into `.npy` | CLI args (`-s`, `-o`, `-m`, `-b`, `-c`, `-l`), Discord export JSON directory (`../act-i-export`), SONAR encoder | `<out>/<name>.npy`, `<out>/<name>.jsonl`, `<out>/<name>.meta.json`, `<out>/<name>.progress.json` | None |
| `headcoh.py` | SAE baseline + evaluation suite | Evaluates semantic coherence of heads/groups via pairwise cosine similarities and SVD energy of decode directions and descriptions vs nulls | CLI args (`--model`, `--ckpt`, `--assignment`, `--desc-dir`, `--metric`), checkpoint, descriptions | `<out>/groups.csv`, `<out>/summary.json`, console z-scores | `tests/test_headcoh.py` |
| `headstruct.py` | SAE baseline + evaluation suite | Discovers and scores latent head-like categorical groupings in SAEs (evaluating exhaustiveness, exclusivity, and sum stability vs nulls) | CLI args (`--sae`, `--cache`, `--rows`, `--size`, `--nulls`, `--trained-groups`), SAE `params.npz`, cache `.npy` | `<out>/groups.csv`, `<out>/assignment.npy`, summary table | `tests/test_headstruct.py` |
| `langprobe.py` | SAE baseline + evaluation suite | Evaluates language identity representation using k-sparse and dense logistic probes across SAE latents and Ontologizer activations | CLI args (`--model`, `--cache`, `--langs`, `--train-rows`, `--ks`), model checkpoint, `.langs.npy` sidecar | `<out>/langs.csv`, summary Macro-F1 | `tests/test_langprobe.py` |
| `mnist.py` | Scratch / stale / experimental | Broken legacy toy script attempting to train an Ontologizer on MNIST image pixels using obsolete/missing module APIs | Hardcoded parameters, HuggingFace `mnist` dataset | Checkpoint directory `data/out/mnist/L1` (currently crashes on import) | None |
| `pareto.py` | SAE baseline + evaluation suite | Computes and plots the capacity-Pareto frontier (whitened reconstruction FVU vs active coefficients and index bits) comparing Ontologizer vs SAEs | CLI args (`--ckpt`, `--ms`, `--code`, `--origin`, `--sae`, `--cache`, `--plot`), checkpoints, cache `.npy` | `<out>/pareto.csv`, optional `<out>/pareto.png`, stdout table | `tests/test_pareto.py` |
| `refit.py` | SAE baseline + evaluation suite | Isolates selection vs magnitude (shrinkage) error by solving closed-form whitened least-squares coefficients on frozen feature supports | CLI args (`--sae`, `--onto`, `--ms`, `--cache`, `--rows`, `--ridge`), checkpoints, cache `.npy` | `<out>/refit.csv`, stdout comparison table | `tests/test_refit.py` |
| `run_chat.py` | Scratch / stale / experimental | Drives `decode.py` interactively through a pseudo-terminal (`pty`) with scripted commands (`set`, `run`, `q`) and captures terminal output | Hardcoded command list, checkpoint `data/out/sonar/linear/multilingual` | Decoded stdout from pseudo-terminal to console | None |
| `run_chat2.py` | Scratch / stale / experimental | Headless CPU-only subprocess wrapper (`CUDA_VISIBLE_DEVICES=""`) running `decode.py` with piped input commands | Hardcoded input string, checkpoint `data/out/sonar/linear/multilingual` | Captured STDOUT and STDERR to console | None |
| `run_decode.py` | Scratch / stale / experimental | Feeds a batch of hardcoded test sentences into `decode.py` via a CPU-only subprocess and prints reconstruction results | Hardcoded multi-line input text, checkpoint `data/out/sonar/linear/multilingual` | Captured STDOUT and STDERR to console | None |
| `run_decode_tags.py` | Scratch / stale / experimental | Runs `decode_tags.py` inside a subprocess with restricted XLA memory allocation (`XLA_PYTHON_CLIENT_MEM_FRACTION=".10"`) | Hardcoded checkpoint `data/out/sonar/linear/multilingual` | Captured STDOUT and STDERR to console | None |
| `sae.py` | SAE baseline + evaluation suite | Standalone trainer and model suite for Sparse Autoencoder (SAE) baselines (Top-K, ReLU+L1, grouped, prefix deepsup, bilinear encoder) on SONAR | CLI args (`--m`, `--topk`, `--l1`, `--groups`, `--prefixes`, `--enc`, `--lr`, `--epochs`), cache `.npy`, `mse_weights.npy` | Checkpoints (`params.npz`, `state.npz`), `loss.csv`, `summary.json` | `tests/test_sae.py` |
| `sae_evals.sh` | SAE baseline + evaluation suite | Automated shell driver for Phase 2 evaluations, executing remaining SAE training runs and the full evaluation battery | Hardcoded run configurations, checkpoint files | Evaluation artifacts across all metrics, `.evals_done` | None |
| `sae_ladder.sh` | SAE baseline + evaluation suite | Automated shell driver training the 4-rung architectural ablation ladder for SAEs and running initial evaluations (`pareto`, `headstruct`) | Hardcoded ladder run configurations, GPU monitor | Trained SAE checkpoints in `data/out/sonar/sae/`, `.ladder_done` | None |
| `sonar.py` | Training / fine-tuning | Primary training script for training the multilayer Ontologizer autoencoder on SONAR text embeddings with deep supervision and regularization | Module-level globals (no CLI args), cache `mc4_4M.npy`, `mse_weights.npy`, SONAR text encoder | Checkpoints in `data/out/sonar/multilingual/resid_nc_hm`, `loss.csv`, `loss.png` | None |
| `sonar.sh` | Training / fine-tuning | Launches `sonar.py` inside a persistent detached `tmux` session named `training` | None | Detached tmux session `training` | None |
| `sonar_hc.py` | Training / fine-tuning | Loss ablation training script isolating the head-similarity penalty by monkey-patching `sonar.py` loss weights before execution | Inherits module globals from `sonar.py` with `s_hcossim=1e-5`, `s_Hm=0.0` | Checkpoints and logs in `data/out/sonar/multilingual/resid_nc_hc` | None |
| `splitting.py` | SAE baseline + evaluation suite | Measures cross-model feature splitting, containment, and seed stability between coarse and fine SAE or Ontologizer models | CLI args (`--a`, `--b`, `--cache`, `--rows`, `--tau`, `--taus`), coarse checkpoint A, fine checkpoint B | `<out>/parents.csv`, `<out>/meta.json`, sweep table | `tests/test_splitting.py` |
| `steerfid.py` | SAE baseline + evaluation suite | Evaluates steering fidelity and collateral damage (feature cycle consistency vs chrF text degradation) under matched intervention magnitudes | CLI args (`--model`, `--mode`, `--cache`, `--mags`, `--sub-rows`, `--device`), checkpoint, SONAR encoder, M2M100 decoder | `<out>/steer.csv`, `<out>/steers.jsonl` | `tests/test_steerfid.py` |
| `textfid.py` | SAE baseline + evaluation suite | Evaluates downstream text reconstruction fidelity by computing character n-gram chrF and M2M100 per-token negative log-likelihood | CLI args (`--model`, `--cache`, `--rows`, `--b`, `--b-decode`, `--device`), checkpoint, SONAR encoder, M2M100 decoder | `<out>/summary.json`, `<out>/rows.jsonl` | `tests/test_textfid.py` |

---

## 3. Detailed Breakdown by Functional Category

### Category 1: Data Preparation / Harvesting (3 scripts)

1. **`encode_corpus.py`**
   - **Role:** Large-scale offline precomputation of SONAR sentence embeddings for the multilingual mC4 corpus.
   - **Mechanism:** Streams from HuggingFace dataset `allenai/c4` (interleaved across languages), tokenizes using NLLB tokenizer (`cointegrated/SONAR_200_text_encoder`), computes mean-pooled L2-normalized embeddings via PyTorch, and stores them directly into a memory-mapped NumPy array.
   - **Resilience:** Writes to `<name>.npy.part` and `<name>.langs.npy.part` with atomic progress sidecar `<name>.progress.json`. Flushes every 50 batches. Resuming skips previously encoded rows. Safely truncates array if stream exhausts early.
   - **Inputs:** CLI args `-n` (corpus size, default 4M), `-b` (batch size, default 256), `-m` (name, default `mc4_4M`), `-o` (output dir).
   - **Outputs:** `data/sonar_embeddings/mc4_4M.npy`, `mc4_4M.langs.npy`, `mc4_4M.meta.json`, `mc4_4M.progress.json`.
   - **Test Coverage:** None.

2. **`encode_discord.py`**
   - **Role:** Offline embedding cache generation from DiscordChatExporter JSON exports (e.g. `../act-i-export`).
   - **Mechanism:** Builds row-aligned line-based JSONL index (`<name>.jsonl`) storing message metadata (channel, author, timestamp, content). Reads texts, sorts by sequence length within 32,768-row chunk windows for dynamic length-bucketing efficiency, encodes using SONAR, and scatters back to original order.
   - **Inputs:** CLI args `-s` (source directory), `-o` (output dir), `-m` (name, default `act_i`), `-b` (batch size, default 64), `-c` (chunk size), `-l` (language, default `eng_Latn`), `--maxlen` (512).
   - **Outputs:** `data/sonar_embeddings/act_i.npy`, `act_i.jsonl`, `act_i.meta.json`, `act_i.progress.json`.
   - **Test Coverage:** None.

3. **`classify_acts.py`**
   - **Role:** Precomputes full-corpus Ontologizer classification activations across all layers for downstream interpretability analysis.
   - **Mechanism:** Runs `Ontologizer.classify` over an embedding cache in float64 precision. The float64 precision is mandatory because residual normalization amplifies float32 rounding errors with depth, causing batch-shape dependent argmax flips in upper layers. Flattens activations to layout $f = (l_i \cdot h + h_i) \cdot k + k_i$ to align with `autointerp.py` tag indexing.
   - **Inputs:** CLI args `--ckpt`, `--step`, `--emb`, `--out`, `--temperature` (default 0.03), `--dtype` (default float32), `--compute-dtype` (default float64), `-b`, `-c`, `-n`.
   - **Outputs:** `<out>.npy` of shape `(N, l*h*k)`, `<out>.meta.json`, `<out>.progress.json`.
   - **Test Coverage:** None.

---

### Category 2: Training / Fine-Tuning (3 scripts)

1. **`sonar.py`**
   - **Role:** Primary training driver for the multilingual Ontologizer architecture on SONAR sentence embeddings.
   - **Mechanism:** Instantiates `Ontologizer` model with residual recurrence (`forward="resid"`), deep supervision (`deepsup=True`), residual normalization (`resid_norm=True`), and constant coordinate (`resid_const=True`). Employs winner dropout, geometric noise annealing, temperature hardening schedules, and batch mean-entropy penalty ($s_{Hm}$). Trains against cached `mc4_4M.npy` or on-the-fly streaming text.
   - **Inputs:** Configured via top-level script variables (no CLI flags): `cache`, `epochs` (24), `b` (256), `lr` (5e-5), `d` (1024), `e_dec` (2048), `h` (32), `k` (32), `l` (5), `mse_weights`.
   - **Outputs:** Checkpoints in `data/out/sonar/multilingual/resid_nc_hm`, step logs in `loss.csv`, training curve plot `loss.png`.
   - **Test Coverage:** None directly (referenced in comments of `test_loaders.py` and `test_training_step.py`).

2. **`sonar_hc.py`**
   - **Role:** Specialized loss ablation script isolating the head-similarity penalty ($s_{hcossim}$) versus the batch mean-entropy bonus ($s_{Hm}$).
   - **Mechanism:** Imports `sonar.py`, directly mutates module-level globals (`sonar.s_hcossim = 1e-5`, `sonar.s_Hm = 0.0`, `sonar.out = ...`), and calls `sonar.main()`.
   - **Inputs:** Inherits all configuration from `sonar.py`.
   - **Outputs:** Checkpoints and `loss.csv` in `data/out/sonar/multilingual/resid_nc_hc`.
   - **Test Coverage:** None.

3. **`sonar.sh`**
   - **Role:** Lightweight execution wrapper launching `sonar.py` inside a detached `tmux` session named `training`.
   - **Mechanism:** Runs `tmux new -s training` and `uv run sonar.py`.
   - **Inputs:** None.
   - **Outputs:** Detached interactive shell session.
   - **Test Coverage:** None.

---

### Category 3: Interactive / Decoding / Chat (2 core scripts)

1. **`decode.py`**
   - **Role:** Interactive CLI tool for testing text encoding, Ontologizer dictionary intervention steering, and generation via M2M100.
   - **Mechanism:** Loads SONAR encoder, M2M100 text decoder, and restored JAX Ontologizer checkpoint. Spawns `ChatEnv` interactive prompt. For each user prompt, encodes text to SONAR space, executes `Ontologizer.withArgs` with specified tag adjustments at temperature 0.1, rescales reconstructed embedding to reference norm, and decodes back to text via beam search with M2M100.
   - **Inputs:** Positional `checkpoint` directory argument (default `data/out/sonar/linear`), stdin interactive commands (`set`, `run`, `q`).
   - **Outputs:** Console output displaying reconstruction MSE and decoded strings.
   - **Test Coverage:** None.

2. **`decode_tags.py`**
   - **Role:** Non-interactive batch decoder generating text labels for every individual dictionary entry and uniform tag representation.
   - **Mechanism:** Uses `Ontologizer.decodeEntries` and `Ontologizer.decodeUniform` to produce embedding representations. Calibrates embedding scale using empirical norm of a set of 6 reference sentences. Generates text via M2M100 in batches of 16.
   - **Inputs:** Positional `checkpoint` directory argument (default `data/out/sonar/linear`).
   - **Outputs:** Text files `<checkpoint_dir>/entries.txt` and `<checkpoint_dir>/uniform.txt`.
   - **Test Coverage:** None (mentioned in docstrings of `test_code_geometry.py`).

---

### Category 4: SAE Baseline + Evaluation Suite (14 scripts)

1. **`sae.py`**
   - **Role:** Complete standalone trainer and model suite for Sparse Autoencoder (SAE) baselines, used as the benchmark against Ontologizer.
   - **Features:** Supports standard Top-K SAE, vanilla ReLU+L1 SAE, grouped Top-1 SAE, grouped Softmax SAE, prefix Matryoshka deep supervision, and bilinear encoders with closed-form eigenfeatures.
   - **Inputs:** Full CLI flags (`--m`, `--topk`, `--l1`, `--groups`, `--group-fn`, `--prefixes`, `--enc`, `--lr`, `--epochs`, `--cache`, `--mse-weights`, `--eval-rows`, etc.).
   - **Outputs:** `params.npz`, `state.npz`, `loss.csv`, `summary.json` under `data/out/sonar/sae/<run_name>/`.
   - **Test Coverage:** `tests/test_sae.py` (imports `sae`).

2. **`pareto.py`**
   - **Role:** Benchmarks Ontologizer deviation and head-ablation codes against top-k SAE baselines on a capacity-Pareto frontier.
   - **Mechanism:** Evaluates reconstruction whitened FVU ($FVU_w$) against transmission capacity (active continuous coefficients and discrete index bits). Scored on identical held-out cache tail rows.
   - **Inputs:** CLI args (`--ckpt`, `--step`, `--temperature`, `--ms`, `--code`, `--origin`, `--sae`, `--cache`, `--eval-rows`, `--plot`).
   - **Outputs:** `<out>/pareto.csv`, optional `<out>/pareto.png`, stdout table.
   - **Test Coverage:** `tests/test_pareto.py` (imports `pareto`).

3. **`refit.py`**
   - **Role:** Shrinkage / refit-FVU analysis isolating selection error from magnitude error (Gao et al. 2024 refinement).
   - **Mechanism:** For each sample, freezes the active feature support and solves the closed-form ridge least-squares coefficients on that support. Measures the gap between raw FVU and refit FVU.
   - **Inputs:** CLI args (`--sae`, `--onto`, `--ms`, `--cache`, `--mse-weights`, `--rows`, `--b`, `--ridge`, `--out`).
   - **Outputs:** `<out>/refit.csv`, stdout table.
   - **Test Coverage:** `tests/test_refit.py` (imports `refit`).

4. **`headcoh.py`**
   - **Role:** Evaluates whether trained heads or discovered groups represent semantically coherent dimensions.
   - **Mechanism:** Measures mean off-diagonal pairwise cosine similarity and top-singular-value energy fraction over decode directions ($G$ rows or $W_{dec}$ rows), z-scored against size-matched random null subsets. Optionally encodes autointerp descriptions to compute description-space coherence.
   - **Inputs:** CLI args (`--model`, `--ckpt`, `--assignment`, `--desc-dir`, `--desc-mode`, `--metric`, `--n-null`, `--out`).
   - **Outputs:** `<out>/groups.csv`, `<out>/summary.json`.
   - **Test Coverage:** `tests/test_headcoh.py` (imports `group_zscores`, `mean_pairwise_cos`, `perm_z`, `sv_energy`).

5. **`headstruct.py`**
   - **Role:** Discovers and scores latent categorical head structure in SAE latents.
   - **Mechanism:** Clusters SAE latents by paradigmatic substitution (similar co-firing contexts with other latents, but negative mutual co-firing). Evaluates discovered groups on exhaustiveness ($P(\text{fires})$), exclusivity ($E[\#\text{fired} \mid \ge 1]$), and sum stability ($CV(\sum z)$) vs size-matched random partitions.
   - **Inputs:** CLI args (`--sae`, `--cache`, `--rows`, `--b`, `--size`, `--topn`, `--nulls`, `--trained-groups`, `--out`).
   - **Outputs:** `<out>/groups.csv`, `<out>/assignment.npy`, summary table.
   - **Test Coverage:** `tests/test_headstruct.py` (imports `headstruct`).

6. **`langprobe.py`**
   - **Role:** Sparse and dense linear probing of language identity on learned representations.
   - **Mechanism:** Uses `.langs.npy` labels from mC4 (~86 languages). Selects top latents by two-sample t-statistic, then trains and evaluates 1-sparse, k-sparse, and full dense logistic regression probes.
   - **Inputs:** CLI args (`--model`, `--cache`, `--langs`, `--train-rows`, `--test-rows`, `--ks`, `--out`).
   - **Outputs:** `<out>/langs.csv`, summary Macro-F1 table.
   - **Test Coverage:** `tests/test_langprobe.py` (imports `langprobe`).

7. **`splitting.py`**
   - **Role:** Cross-model feature splitting and seed stability benchmark.
   - **Mechanism:** Measures activation containment between coarse model A and fine model B ($P(A \mid B)$ vs $P(B \mid A)$) across threshold sweeps ($\tau$), quantifying parent-child multiplicity, union coverage, and pairwise overlap.
   - **Inputs:** CLI args (`--a`, `--b`, `--cache`, `--rows`, `--tau`, `--taus`, `--min-fires`, `--out`).
   - **Outputs:** `<out>/parents.csv`, `<out>/meta.json`, sweep table to stdout.
   - **Test Coverage:** `tests/test_splitting.py` (imports `splitting`).

8. **`steerfid.py`**
   - **Role:** Steering fidelity evaluation measuring effect vs collateral damage at matched intervention magnitudes.
   - **Mechanism:** Steers embeddings along native feature directions ($W_{dec}$ rows, eigenfeatures, or Ontologizer forced tags), decodes via M2M100, re-encodes with SONAR to check cycle consistency (effect), and computes $1 - \text{chrF}(\text{base}, \text{steered})$ (collateral).
   - **Inputs:** CLI args (`--model`, `--mode`, `--cache`, `--n-features`, `--n-samples`, `--mags`, `--n-random`, `--sub-rows`, `--device`, `--out`).
   - **Outputs:** `<out>/steer.csv`, `<out>/steers.jsonl`.
   - **Test Coverage:** `tests/test_steerfid.py` (imports `steerfid`).

9. **`textfid.py`**
   - **Role:** Evaluates downstream text reconstruction fidelity in generation space.
   - **Mechanism:** Decodes original and reconstructed embeddings through M2M100; evaluates generation-space chrF n-gram overlap and M2M100 per-token negative log-likelihood (NLL).
   - **Inputs:** CLI args (`--model`, `--cache`, `--rows`, `--b`, `--b-decode`, `--device`, `--out`).
   - **Outputs:** `<out>/summary.json`, `<out>/rows.jsonl`.
   - **Test Coverage:** `tests/test_textfid.py` (imports `chrf`).

10. **`compose.py`**
    - **Role:** Tests causal intervention composition and additivity in decode space.
    - **Mechanism:** Compares joint dual-head forced assignments $D_{12}$ against the linear sum of single-head interventions $D_1 + D_2$ in decode space, isolating downstream layer reclassification effects.
    - **Inputs:** CLI args (`--ckpt`, `--step`, `--cache`, `--eval-rows`, `--rows`, `--singles-per-layer`, `--cross-pairs`, `--mse-weights`, `--out`).
    - **Outputs:** `<out>/pairs.csv`, `<out>/summary.json`.
    - **Test Coverage:** `tests/test_compose.py` (imports `additivity_stats`, `sample_singles`).

11. **`autointerp.py`**
    - **Role:** Bills et al.-style automated interpretability pipeline for Ontologizer tags and SAE latents.
    - **Mechanism:** 4-stage pipeline: (1) `harvest` activations and compute parameter decodes; (2) `texts` recover corpus text from deterministic mC4 stream; (3) `describe` generate descriptions via M2M100 ("params"), Anthropic LLM ("acts"), or contrastive LLM ("cacts"); (4) `score` blinded detection benchmark measuring precision/recall/F1 and null-control chance floor.
    - **Inputs:** Positional `stage` (`harvest`, `texts`, `describe`, `score`), `--model`, `--ckpt`, `--topk`, `--features`, `--dir`, `--mode`, `--judge`, `--rate`, `--null`, `--dry-run`, `ANTHROPIC_API_KEY`.
    - **Outputs:** `features.npz`, `texts.jsonl`, `descriptions.jsonl`, `scores.csv`, optional `prompts/`.
    - **Test Coverage:** `tests/test_autointerp.py` (imports prompt templates and scoring helpers).

12. **`autointerp_run.sh`**
    - **Role:** Automated shell pipeline driver for the autointerp campaign across Ontologizer and 5 SAE variants.
    - **Mechanism:** Checks GPU free memory ($\ge 6000$ MB), calls `autointerp.py` for harvest, texts, describe (plain and contrastive), and score with null controls.
    - **Inputs:** `ANTHROPIC_API_KEY`, `AI_RATE`, checkpoints in `data/out/sonar/`.
    - **Outputs:** Artifacts in `data/out/sonar/autointerp/`, `.campaign_done`.
    - **Test Coverage:** None.

13. **`sae_evals.sh`**
    - **Role:** Automated shell pipeline driver for Phase 2 evaluations across all SAE rungs and Ontologizer.
    - **Mechanism:** Finalizes remaining training runs, runs `pareto.py`, `headstruct.py`, `splitting.py`, `textfid.py`, `langprobe.py`, and `refit.py`.
    - **Inputs:** Checkpoints and cache.
    - **Outputs:** Evaluation CSVs and JSON summaries across all metrics, `.evals_done`.
    - **Test Coverage:** None.

14. **`sae_ladder.sh`**
    - **Role:** Automated shell driver training the 4-rung SAE structural-ablation ladder.
    - **Mechanism:** Runs `sae.py` across 4 configurations (plain topk, grouped top1, grouped softmax, prefix deepsup), followed by `pareto.py` and `headstruct.py`.
    - **Inputs:** GPU availability, embedding cache.
    - **Outputs:** Trained SAE checkpoints, `.ladder_done`.
    - **Test Coverage:** None.

---

### Category 5: Scratch / Stale / Experimental (5 scripts)

1. **`mnist.py`**
   - **Role:** Abandoned experimental toy script attempting to train an Ontologizer on MNIST image pixels.
   - **Status:** **Completely broken and obsolete.** Contains invalid import `from ontologize.training.data import ImageLoader` (no such module exists) and invokes `Hyperparams` and `Metadata` with outdated and malformed positional arguments.
   - **Inputs:** None (hardcoded).
   - **Outputs:** None (fails at runtime on import).
   - **Test Coverage:** None.

2. **`run_chat.py`**
   - **Role:** Ad-hoc test driver executing `decode.py` interactively through a pseudo-terminal.
   - **Mechanism:** Uses Python `pty.openpty()` and `select.select()` to simulate interactive user input (`set 0 k_add 5,10`, `run ...`, `q`) against `data/out/sonar/linear/multilingual`.
   - **Inputs:** Hardcoded script strings and path.
   - **Outputs:** Prints raw terminal output.
   - **Test Coverage:** None.

3. **`run_chat2.py`**
   - **Role:** Headless CPU-only subprocess test runner for `decode.py`.
   - **Mechanism:** Invokes `decode.py` via `subprocess.run` with `CUDA_VISIBLE_DEVICES=""` and captured stdout/stderr.
   - **Inputs:** Hardcoded strings and path.
   - **Outputs:** Prints stdout and stderr.
   - **Test Coverage:** None.

4. **`run_decode.py`**
   - **Role:** Subprocess test runner feeding multi-line text input into `decode.py`.
   - **Mechanism:** Invokes `subprocess.run` on `decode.py` with multi-line paragraphs (discussing signal anti-aliasing and PIBBSS funding) to test batch text reconstruction.
   - **Inputs:** Hardcoded multi-line string.
   - **Outputs:** Prints stdout and stderr.
   - **Test Coverage:** None.

5. **`run_decode_tags.py`**
   - **Role:** Subprocess test runner for `decode_tags.py`.
   - **Mechanism:** Invokes `decode_tags.py` with environment variable `XLA_PYTHON_CLIENT_MEM_FRACTION=".10"`.
   - **Inputs:** Hardcoded checkpoint path.
   - **Outputs:** Prints stdout and stderr.
   - **Test Coverage:** None.

*(Note on Classification: While `run_chat.py`, `run_chat2.py`, `run_decode.py`, and `run_decode_tags.py` interact with the decoding scripts, they are hardcoded scratch runners without CLI arguments or generic utility, making them best classified as Scratch/Stale/Experimental).*

---

## 4. Coupling and Dependency Analysis

### 4.1 Inter-Script Import Graph (Root Level)

The root scripts are tightly coupled through direct module imports at the root level:

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

### 4.2 Shell Script Invocations

All four shell scripts invoke root scripts via relative Python calls assuming the root directory is current working directory:
- `sonar.sh` -> `uv run sonar.py`
- `sae_ladder.sh` -> `uv run python sae.py`, `uv run python pareto.py`, `uv run python headstruct.py`
- `sae_evals.sh` -> `uv run python sae.py`, `pareto.py`, `headstruct.py`, `splitting.py`, `textfid.py`, `langprobe.py`, `refit.py`
- `autointerp_run.sh` -> `uv run python autointerp.py harvest|texts|describe|score`

### 4.3 Test Suite Invocations (`tests/`)

11 test files in `tests/` directly import root scripts as top-level modules:
1. `tests/test_autointerp.py`: `from autointerp import (CAL_PROMPT_CHARS, CONTRAST_PROMPT, SUMMARIZE_PROMPT, ...)`
2. `tests/test_compose.py`: `from compose import additivity_stats, sample_singles`
3. `tests/test_headcoh.py`: `from headcoh import (group_zscores, mean_pairwise_cos, perm_z, ...)`
4. `tests/test_headstruct.py`: `import headstruct`
5. `tests/test_langprobe.py`: `import langprobe`
6. `tests/test_pareto.py`: `import pareto`
7. `tests/test_refit.py`: `import refit`, `import sae`
8. `tests/test_sae.py`: `import sae`
9. `tests/test_splitting.py`: `import splitting`
10. `tests/test_steerfid.py`: `import steerfid`
11. `tests/test_textfid.py`: `from textfid import chrf`

---

## 5. Catalog of Bugs, Dead Code, and Code Oddities

The survey revealed several notable bugs, architectural oddities, and dead code patterns across the root scripts (documented here without modifying any code):

1. **Broken Imports and Stale Code in `mnist.py`:**
   - Line 7: `from ontologize.training.data import ImageLoader`. The `ontologize.training` package does not contain a `data` module.
   - Lines 40–43: `Hyperparams` and `Metadata` constructor calls pass incorrect positional parameters (missing `epochs`, passing `stddev` where `temperature` is expected).
   - This script cannot be imported or run; it is completely stale.

2. **Global Variable Mutation in `sonar_hc.py`:**
   - Lines 33–37: `sonar_hc.py` imports `sonar` and overwrites its module-level global variables (`sonar.s_hcossim = 1e-5`, `sonar.s_Hm = 0.0`, `sonar.out = ...`) prior to calling `sonar.main()`. This tightly binds `sonar_hc.py` to the exact internal implementation of `sonar.py`.

3. **Missing CLI Support in Core Training Scripts:**
   - Both `sonar.py` and `sonar_hc.py` lack command-line argument parsing (`argparse`). All hyperparameters, paths, and flags are hardcoded module globals. In contrast, the SAE baseline scripts (`sae.py`, `pareto.py`, etc.) feature full CLI argument parsers.

4. **Self-Restarting Process Re-Execution in `decode.py` and `decode_tags.py`:**
   - Lines 6–15 in both scripts inspect `os.environ` for `CUDNN_INJECTED`. If missing, they attempt to locate the CuDNN library path and call `os.execv(sys.executable, [sys.executable] + sys.argv)`. This dynamic shell-bypass workaround can cause unexpected behavior in non-standard execution environments or subshells.

5. **Hardcoded Ad-Hoc Scripts (`run_chat*.py`, `run_decode*.py`):**
   - Four separate scripts exist solely to pass fixed byte strings or text samples into `decode.py` and `decode_tags.py`. They contain hardcoded references to a legacy checkpoint path (`data/out/sonar/linear/multilingual`) and represent scratch debugging artifacts rather than reusable tools.

6. **External Path Dependency in `encode_discord.py`:**
   - Line 55: `src = "../act-i-export"` relies on an uncommitted directory located outside the repository workspace root.

7. **Extensive Cross-Script Module Dependencies:**
   - Root scripts treat each other as top-level modules (e.g. `import sae`, `from pareto import load_onto`, `from splitting import load_model`). None of these utilities are packaged under `ontologize/`, meaning root scripts cannot be moved into subdirectories without breaking cross-imports unless `PYTHONPATH` is configured or code is refactored into a proper package structure.
