# Comprehensive Coupling Survey & Repository Reorganization Proposal

**Document Version:** 1.0  
**Date:** 2026-09-29  
**Author:** explorer_survey_3 (Workstream 3: Reorganization Proposal & Coupling Analysis)  
**Status:** Analytical Proposal (Zero filesystem mutations executed)

---

## 1. Executive Summary

The `ontologize` repository currently houses 27 executable scripts at the repository root: **23 Python scripts (`*.py`)** and **4 Bash scripts (`*.sh`)**. While functional, this flat root layout creates significant cognitive overhead, namespace collisions, and hidden coupling constraints.

This survey provides a comprehensive architectural and dependency analysis across the codebase to prepare for a clean, non-disruptive reorganization (`docs/REORG_PROPOSAL.md`).

### Key Discoveries:
1. **Test Suite Coupling (11 Scripts):** Exactly 11 test files in `tests/` directly import root scripts as top-level modules (`autointerp`, `compose`, `headcoh`, `headstruct`, `langprobe`, `pareto`, `refit`, `sae`, `splitting`, `steerfid`, `textfid`), enabled by `sys.path.insert(0, str(Path(__file__).parents[1]))` in `tests/conftest.py`. Importantly, tests verify **pure mathematical, statistical, and algorithmic helper functions** rather than CLI execution loops.
2. **Dense Sibling Dependency Graph (20 Directed Edges):** Root scripts exhibit dense inter-dependencies. Two scripts act as central utility hubs: `sae.py` (imported by 7 sibling scripts) and `pareto.py` (imported by 6 sibling scripts). Secondary hubs include `autointerp.py` (imported by 3), `refit.py` (imported by 1), `splitting.py` (imported by 1), `textfid.py` (imported by 1), and `sonar.py` (imported by 1).
3. **Execution Script Coupling (8 Scripts):**
   - 4 shell scripts (`autointerp_run.sh`, `sae_ladder.sh`, `sae_evals.sh`, `sonar.sh`) invoke root Python scripts via relative paths (e.g., `uv run python sae.py`).
   - 4 Python wrapper scripts (`run_chat.py`, `run_chat2.py`, `run_decode.py`, `run_decode_tags.py`) invoke `decode.py` and `decode_tags.py` via `subprocess.Popen` / `subprocess.run` with hardcoded relative paths.
4. **Structural Duplication:** Multiple evaluation scripts independently re-implement boilerplate for Orbax checkpoint loading, parameter extraction, and baseline loading.
5. **Code Oddities & Bugs:** A fatal import bug exists in `mnist.py` (`from ontologize.training.data import ImageLoader`, where `ImageLoader` resides in `ontologize.data.loaders`), `sonar_hc.py` monkeypatches global variables in `sonar.py`, and divergent CuDNN loading mechanisms (`os.execv` vs `ctypes.CDLL`) are used across scripts.

---

## 2. Exhaustive Coupling Constraints Analysis

### 2.1 Test Suite Coupling (`tests/`)

In `tests/conftest.py` (lines 9-11):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
```
This line prepends the repository root to `sys.path`. As a result, Python allows test modules in `tests/` to import root scripts directly as top-level modules.

Out of 25 files in `tests/`, **11 test files directly import root scripts**:

| Test File | Imported Script | Import Statement (Line #) | Tested Symbols & Functional Scope |
|-----------|-----------------|---------------------------|-----------------------------------|
| `tests/test_autointerp.py` | `autointerp` | `from autointerp import (CAL_PROMPT_CHARS, CONTRAST_PROMPT, SUMMARIZE_PROMPT, call_weight, contrast_rows, detection_items, cap_pairing, next_slot, null_pairing, read_scores, snippet, stratified_features)` (lines 6-10) | Tests pure helper functions for prompt rate limiting (`next_slot`), token billing (`call_weight`), feature selection (`stratified_features`), and contrastive pairing (`contrast_rows`, `detection_items`). |
| `tests/test_compose.py` | `compose` | `from compose import additivity_stats, sample_singles` (line 6) | Tests vector math for linear intervention additivity (`additivity_stats`), relative errors under whitened metric, and head sampling (`sample_singles`). |
| `tests/test_headcoh.py` | `headcoh` | `from headcoh import (group_zscores, mean_pairwise_cos, perm_z, pooled_within_cos, sv_energy, unit)` (lines 5-6) | Tests semantic coherence metrics, singular value energy ratio (`sv_energy`), permutation z-scores (`perm_z`, `group_zscores`), and cosine similarity math. |
| `tests/test_headstruct.py` | `headstruct` | `import headstruct` (line 8) | Tests categorical head discovery algorithms: graph affinity calculation (`affinity_edges`), greedy clustering with size caps (`greedy_groups`), moments (`group_moments`), and exclusivity/exhaustiveness metrics (`metrics`). |
| `tests/test_langprobe.py` | `langprobe` | `import langprobe` (line 7) | Tests sparse language classification probing: two-sample t-statistic selection (`t_stats`), threshold search (`best_f1_threshold`), F1 scoring (`f1_at`), and logistic regression fitting (`fit_logistic`). |
| `tests/test_pareto.py` | `pareto` | `import pareto` (line 8) | Tests deviation code capacity truncation (`topdev`), head entropy filtering (`tophead`), and SAE naming parser (`parse_sae_name`). |
| `tests/test_refit.py` | `refit`, `sae` | `import refit` (line 8)<br>`import sae` (line 50) | Tests masked weighted least-squares reconstruction (`refit_recon`) and SAE support mask extraction (`sae_supports`). |
| `tests/test_sae.py` | `sae` | `import sae` (line 10) | Tests standalone SAE baseline trainer: parameter initialization (`init_params`), Top-K sparse autoencoder training (`train`), snapshot checkpoint resumption, eigenfeature extraction (`eigenfeatures`), and evaluation metric builders (`make_eval`, `evaluate`). |
| `tests/test_splitting.py` | `splitting` | `import splitting` (line 8) | Tests cross-model feature splitting analysis: activation containment matching (`split_children`), union coverage calculation (`union_cover_counts`), and child overlap verification (`child_overlap`). |
| `tests/test_steerfid.py` | `steerfid` | `import steerfid` (line 7) | Tests vector normalization (`unit`), active feature selection (`pick_features`), steering effect vs collateral scoring (`effect_scores`), and conditional firing quantiles (`firing_quantiles`). |
| `tests/test_textfid.py` | `textfid` | `from textfid import chrf` (line 6) | Tests character n-gram F-score implementation (`chrf`) with beta recall weighting. |

#### Critical Finding on Test Structure
None of the tests execute CLI commands (`main()` or `argparse`), none perform live disk writes to production paths, and none require network calls. Every single test file imports **pure helper functions** to verify mathematical algorithms, statistical metrics, and tensor manipulations.

---

### 2.2 Sibling Script Cross-Imports (Inter-Script Couplings)

A systematic AST scan of all 23 root Python scripts revealed **20 directed dependency edges** connecting 11 root scripts.

```
                    +-------------------+
                    |      sae.py       |<------------------------+
                    +-------------------+                         |
                      ^   ^   ^   ^   ^                           |
         +------------+   |   |   |   +-------------+             |
         |                |   |   |                 |             |
+-----------------+       |   |   |         +-----------------+   |
|  autointerp.py  |       |   |   +---------|  headstruct.py  |   |
+-----------------+       |   |             +-----------------+   |
  ^             ^         |   |               |                   |
  |             |         |   |               v                   |
  |      +------+---------+   |             +-----------------+   |
  |      |      |             |             |    pareto.py    |---+
  |      |      |             |             +-----------------+   |
  |      |      |             |               ^   ^   ^   ^   ^   |
  |      |      |             +---------+     |   |   |   |   |   |
  |      |      |                       |     |   |   |   |   |   |
  |   +-----------------+               |     |   |   |   |   |   |
  |   |    refit.py     |---------------+-----+   |   |   |   |   |
  |   +-----------------+                         |   |   |   |   |
  |     ^                                         |   |   |   |   |
  |     |                                         |   |   |   |   |
  |   +-----------------+                         |   |   |   |   |
  +---|   headcoh.py    |                         |   |   |   |   |
      +-----------------+                         |   |   |   |   |
                                                  |   |   |   |   |
  +-----------------+                             |   |   |   |   |
  |   splitting.py  |-----------------------------+   |   |   |   |
  +-----------------+ (also imports autointerp & sae) |   |   |   |
    ^                                                 |   |   |   |
    |                                                 |   |   |   |
  +-----------------+                                 |   |   |   |
  |   langprobe.py  |                                 |   |   |   |
  +-----------------+                                 |   |   |   |
                                                      |   |   |   |
  +-----------------+                                 |   |   |   |
  |   compose.py    |---------------------------------+   |   |   |
  +-----------------+                                     |   |   |
                                                          |   |   |
  +-----------------+                                     |   |   |
  |   textfid.py    |-------------------------------------+   |   |
  +-----------------+ (also imports sae)                      |   |
    ^                                                         |   |
    |                                                         |   |
  +-----------------+                                         |   |
  |   steerfid.py   |-----------------------------------------+   |
  +-----------------+ (also imports sae, textfid)                 |
                                                                  |
  +-----------------+                                             |
  |   sonar_hc.py   |---> [sonar.py]                              |
  +-----------------+                                             |
```

#### Detailed Cross-Import Inventory:

| Consumer Script | Imported Sibling | Scope | Imported Line Content | Purpose / Symbols Used |
|-----------------|------------------|-------|-----------------------|------------------------|
| `autointerp.py` | `sae` | Inner (`sae_acts_fn`:452, `harvest`:550) | `import sae as sae_mod` | Calls `sae_mod.encode()` to compute SAE activations; calls `sae_mod.eigenfeatures()` for bilinear SAE decode directions. |
| `compose.py` | `pareto` | Inner (`main`:89) | `from pareto import load_onto` | Loads Ontologizer checkpoint and restores spec/params (`model, mparams, step = load_onto(...)`). |
| `headcoh.py` | `refit` | Inner (`onto_directions`:145) | `from refit import onto_linear_model` | Extracts linear decode direction matrix $G$ for Ontologizer heads. |
| `headcoh.py` | `autointerp` | Inner (`encode_texts`:186) | `from autointerp import ENCODER_ID` | Obtains SONAR text encoder HuggingFace model ID string. |
| `headstruct.py` | `sae` | Top-level (line 55) | `import sae` | Calls `sae.encode()` on evaluation batches. |
| `headstruct.py` | `pareto` | Inner (`main`:184) | `from pareto import parse_sae_name` | Parses width $m$ and $k$ from SAE artifact directory names. |
| `langprobe.py` | `splitting` | Top-level (line 39) | `from splitting import load_model` | Unified loader for both SAE `.npz` parameter files and Ontologizer checkpoints. |
| `pareto.py` | `sae` | Top-level (line 62) | `import sae` | Calls `sae.evaluate()` and `sae.make_eval()` to compute FVU and L0 across SAE runs. |
| `refit.py` | `sae` | Top-level (line 50) | `import sae` | Calls `sae.encode()` to determine active latent supports and `sae.decode()` for baseline reconstruction. |
| `refit.py` | `autointerp` | Inner (`onto_linear_model`:125) | `from autointerp import onto_acts_fn` | Obtains Ontologizer activation function and classification matrices. |
| `refit.py` | `pareto` | Inner (`main`:170) | `from pareto import parse_sae_name` | Parses $k$ from SAE checkpoint path. |
| `sonar_hc.py` | `sonar` | Top-level (line 33) | `import sonar` | **Direct monkeypatching:** modifies `sonar.s_hcossim = 1e-5`, `sonar.s_Hm = 0.0`, `sonar.out = ...`, then invokes `sonar.main()`. |
| `splitting.py` | `sae` | Top-level (line 54) | `import sae` | Calls `sae.encode()` inside JIT-compiled activation function. |
| `splitting.py` | `pareto` | Inner (`load_model`:91) | `from pareto import parse_sae_name` | Extracts model dimensions from directory name. |
| `splitting.py` | `autointerp` | Inner (`load_model`:97) | `from autointerp import onto_acts_fn` | Obtains Ontologizer classification activations. |
| `steerfid.py` | `sae` | Top-level (line 49) | `import sae` | Calls `sae.encode()` and `sae.eigenfeatures()` for steering directions. |
| `steerfid.py` | `pareto` | Inner (`load_steerable`:139) | `from pareto import load_onto` | Restores Ontologizer checkpoint for steering interventions. |
| `steerfid.py` | `textfid` | Inner (`main`:183) | `from textfid import chrf` | Computes character n-gram F-score between original and steered decodes. |
| `textfid.py` | `sae` | Top-level (line 44) | `import sae` | Calls `sae.encode()` and `sae.decode()` for embedding reconstruction. |
| `textfid.py` | `pareto` | Inner (`recon_fn`:105, 115) | `from pareto import parse_sae_name`<br>`from pareto import load_onto` | Parses SAE parameters and loads Ontologizer model/state. |

---

### 2.3 Shell Scripts Invocations & Execution Environment

The 4 root-level shell scripts automate training pipelines and evaluation suites. They invoke Python scripts using relative paths from the repository root:

```
autointerp_run.sh  --->  autointerp.py (harvest, texts, describe, score)
sae_ladder.sh      --->  sae.py, pareto.py, headstruct.py
sae_evals.sh       --->  sae.py, pareto.py, headstruct.py, splitting.py, textfid.py, langprobe.py, refit.py
sonar.sh           --->  sonar.py
```

#### Detailed Shell Script Analysis:

#### 1. `autointerp_run.sh` (102 lines)
- **Working Directory Pin:** Line 22: `cd "$(dirname "$0")"` (assumes the shell script is in repo root).
- **Environment Requirements:**
  - `export PATH="$HOME/.local/bin:$PATH"`
  - `ANTHROPIC_API_KEY`: Mandatory. Reads from environment variable or file pointed to by `$ANTHROPIC_API_KEY`. Fails fast if unset.
  - `AI_RATE`: Judge pacing rate in queries per hour (default 1000).
- **GPU Management:** `wait_gpu()` checks `nvidia-smi` until at least 6,000 MiB free VRAM is available.
- **Root Script Invocations:**
  - `uv run python autointerp.py harvest --model onto --ckpt $ONTO --out $AI/onto` (line 65)
  - `uv run python autointerp.py harvest --model sae --ckpt $SAE/$m/params.npz --out $AI/$m` (line 68)
  - `uv run python autointerp.py texts --out $AI --features ...` (line 73)
  - `uv run python autointerp.py describe --dir ... --out $AI --mode ... --rate $RATE` (lines 77, 81, 86)
  - `uv run python autointerp.py score --dir ... --out $AI --judge haiku --jobs 4 --rate $RATE --null --null-n 100` (line 91)
- **Idempotency:** Guarded by `data/out/sonar/autointerp/.campaign_done` and per-model artifact presence checks.

#### 2. `sae_ladder.sh` (67 lines)
- **Working Directory Pin:** Line 20: `cd "$(dirname "$0")"`.
- **GPU Management:** `wait_gpu()` polling loop (6,000 MiB minimum free).
- **Root Script Invocations:**
  - `uv run python sae.py --lr 4e-4 --epochs 150 --m 5120 --topk 32` (line 49)
  - `uv run python sae.py --lr 4e-4 --epochs 150 --m 5120 --topk 0 --groups 160` (line 50)
  - `uv run python sae.py --lr 4e-4 --epochs 150 --m 5120 --topk 0 --groups 160 --group-fn softmax` (line 51)
  - `uv run python sae.py --lr 4e-4 --epochs 150 --m 5120 --topk 32 --prefixes 5` (line 52)
  - `uv run python pareto.py` (line 55)
  - `uv run python headstruct.py --rows 1048576 --sae ...` (lines 56-63)
- **Idempotency:** Guarded by `data/out/sonar/sae/.ladder_done`.

#### 3. `sae_evals.sh` (88 lines)
- **Working Directory Pin:** Line 13: `cd "$(dirname "$0")"`.
- **GPU Management:** `wait_gpu()` polling loop.
- **Root Script Invocations:**
  - `uv run python sae.py --m 11264 --topk 5120 --epochs 23` (line 42)
  - `uv run python sae.py --m 5120 --topk 32 --enc bilinear --lr 4e-4 --epochs 150` (line 45)
  - `uv run python sae.py --m 5120 --topk 32 --lr 4e-4 --epochs 150 --seed 43` (line 47)
  - `uv run python pareto.py` (line 52)
  - `uv run python headstruct.py --sae $SAE/m11264_k160/params.npz --rows 1048576` (line 55)
  - `uv run python splitting.py --a ... --b ...` (lines 59, 62)
  - `uv run python textfid.py --model $SAE/$run/params.npz --device cuda --b-decode 64` (line 68)
  - `uv run python langprobe.py --model $SAE/$run/params.npz` (line 71)
  - `uv run python textfid.py --model $ONTO --device cuda --b-decode 64` (line 74)
  - `uv run python langprobe.py --model $ONTO` (line 76)
  - `uv run python refit.py --onto $ONTO` (line 79)
- **Idempotency:** Guarded by `data/out/sonar/sae/.evals_done` and `step()` artifact guards.

#### 4. `sonar.sh` (9 lines)
- **Invocation:**
  ```bash
  #!/bin/bash
  tmux new -s training
  uv run sonar.py
  ```
- **Flaws:** Unlike the other 3 scripts, `sonar.sh` lacks `cd "$(dirname "$0")"` and invokes `uv run sonar.py` rather than `uv run python sonar.py`.

---

### 2.4 Subprocess Invocation Wrappers

Four Python scripts in root exist solely to wrap other root scripts via `subprocess`:

```
run_chat.py         ---+
run_chat2.py        ---+---> [subprocess] ---> decode.py
run_decode.py       ---+
run_decode_tags.py  -------> [subprocess] ---> decode_tags.py
```

| Wrapper Script | Target Script | Subprocess Method | Relative Path Used | Execution Arguments & Environment |
|----------------|---------------|-------------------|--------------------|-----------------------------------|
| `run_chat.py` | `decode.py` | `subprocess.Popen` via `pty.openpty()` | `"decode.py"` | `["uv", "run", "python", "decode.py", "data/out/sonar/linear/multilingual"]`<br>Sets `XLA_PYTHON_CLIENT_MEM_FRACTION=".10"`, sends interactive pty commands (`set 0 k_add 5,10`, `run ...`). |
| `run_chat2.py` | `decode.py` | `subprocess.run` | `"decode.py"` | `["uv", "run", "python", "decode.py", "data/out/sonar/linear/multilingual"]`<br>Sets `CUDA_VISIBLE_DEVICES=""`, feeds piped stdin commands. |
| `run_decode.py` | `decode.py` | `subprocess.run` | `"decode.py"` | `["uv", "run", "python", "decode.py", "data/out/sonar/linear/multilingual"]`<br>Sets `CUDA_VISIBLE_DEVICES=""`, feeds multi-line text input string. |
| `run_decode_tags.py` | `decode_tags.py` | `subprocess.run` | `"decode_tags.py"` | `["uv", "run", "python", "decode_tags.py", "data/out/sonar/linear/multilingual"]`<br>Sets `XLA_PYTHON_CLIENT_MEM_FRACTION=".10"`. |

If `decode.py` or `decode_tags.py` are moved without updating these wrappers, all 4 wrappers will immediately fail with `FileNotFoundError`.

---

### 2.5 Redundant Implementation Boilerplate

A major finding is that multiple evaluation scripts independently implement identical checkpoint loading and parameter extraction logic:

1. **Orbax Ontologizer Checkpoint Loading:**
   - `classify_acts.py` defines `load_model(ckpt: Path, step)` (lines 45-60)
   - `pareto.py` defines `load_onto(ckpt, step)` (lines 95-115)
   - `splitting.py` defines `load_model(path, cfg)` (lines 80-111)
   - `steerfid.py` defines `load_steerable(path, cfg)` (lines 104-142)
   - `refit.py` defines `onto_linear_model(ckpt, step, temperature)` (lines 123-143)
   - `autointerp.py` defines `onto_acts_fn(ckpt, step, temperature)` (lines 396-446)
   Each function initializes `ocp.CheckpointManager`, fetches `latest_step()`, calls `manager.restore()`, recursively unpacks `params` dictionaries (`while 'params' in params: params = params['params']`), and instantiates `Ontologizer(**spec)`.
2. **SAE Parameter Parsing:**
   `pareto.py:parse_sae_name` (extracting $m$ and $k$ from directory regex `m{m}_k{k}`) is imported by 4 other scripts (`headstruct.py`, `refit.py`, `splitting.py`, `textfid.py`).
3. **Hardcoded Model Constants:**
   The HuggingFace model string `ENCODER_ID = "cointegrated/SONAR_200_text_encoder"` is defined redundantly across `autointerp.py`, `encode_corpus.py`, `encode_discord.py`, `steerfid.py`, and `textfid.py`.

---

## 3. Proposed Target Directory Layout

To eliminate root clutter, preserve modularity, and cleanly separate responsibilities, we propose organizing scripts under a structured `scripts/` hierarchy.

```
ontologize/                     # Main repository
├── ontologize/                 # Python package (core layers, training, data, inference)
│   ├── layers/
│   ├── training/
│   ├── data/
│   ├── fns/
│   ├── inference/
│   └── visualize/
├── tests/                      # Pytest suite
│   ├── conftest.py
│   └── test_*.py
├── docs/                       # Project documentation & indexes
│   ├── SCRIPTS.md              # Exhaustive script inventory (Workstream 2)
│   └── REORG_PROPOSAL.md       # Target reorganization proposal (Workstream 3)
└── scripts/                    # Proposed target location for all 27 scripts
    ├── __init__.py
    ├── data/                   # Data harvesting, preprocessing, corpus encoding
    │   ├── __init__.py
    │   ├── encode_corpus.py
    │   ├── encode_discord.py
    │   └── classify_acts.py
    ├── training/               # Ontologizer model training & ablations
    │   ├── __init__.py
    │   ├── sonar.py
    │   ├── sonar_hc.py
    │   ├── mnist.py
    │   └── sonar.sh
    ├── interactive/            # Interactive inspection, chat, and tag decoding
    │   ├── __init__.py
    │   ├── decode.py
    │   ├── decode_tags.py
    │   └── runners/            # Subprocess test drivers & batch runners
    │       ├── __init__.py
    │       ├── run_chat.py
    │       ├── run_chat2.py
    │       ├── run_decode.py
    │       └── run_decode_tags.py
    ├── sae/                    # Standalone SAE baseline trainer & ladder
    │   ├── __init__.py
    │   ├── sae.py
    │   └── sae_ladder.sh
    └── eval/                   # Interpretability evaluation battery & comparisons
        ├── __init__.py
        ├── autointerp.py
        ├── autointerp_run.sh
        ├── compose.py
        ├── headcoh.py
        ├── headstruct.py
        ├── langprobe.py
        ├── pareto.py
        ├── refit.py
        ├── sae_evals.sh
        ├── splitting.py
        ├── steerfid.py
        └── textfid.py
```

### Architectural Rationale for Grouping:
- **`scripts/data/` (3 scripts):** Groups offline data-generation workflows that prepare `.npy` embedding caches before any training runs.
- **`scripts/training/` (4 scripts):** Groups Ontologizer model training loops, loss ablation scripts, and the launcher shell script.
- **`scripts/interactive/` (6 scripts):** Separates user-facing inspection tools (`decode.py`, `decode_tags.py`) and places their ad-hoc subprocess drivers in an explicit subpackage (`runners/`).
- **`scripts/sae/` (2 scripts):** Isolates the independent baseline SAE implementation and its multi-stage training ladder from Ontologizer code.
- **`scripts/eval/` (12 scripts):** Co-locates the entire scientific evaluation suite (representation geometry, feature splitting, language probes, auto-interp, Pareto frontier, text fidelity, and steering fidelity) alongside their orchestration shell scripts.

---

## 4. Concrete Move Map for All 27 Scripts

The following table provides the exhaustive 1-to-1 move map for every root script, documenting its functional role, inter-script couplings, and covering tests.

| # | Current Root Path | Proposed Target Path | Functional Category | Inbound Couplings (Imported By) | Outbound Couplings (Imports Sibling) | Covering Test File |
|---|-------------------|----------------------|---------------------|---------------------------------|--------------------------------------|--------------------|
| 1 | `encode_corpus.py` | `scripts/data/encode_corpus.py` | Data Prep | None | None | None |
| 2 | `encode_discord.py` | `scripts/data/encode_discord.py` | Data Prep | None | None | None |
| 3 | `classify_acts.py` | `scripts/data/classify_acts.py` | Data Prep | None | None | None |
| 4 | `sonar.py` | `scripts/training/sonar.py` | Training | `sonar_hc.py`, `sonar.sh` | None | None (mirrored in `test_training_step.py`) |
| 5 | `sonar_hc.py` | `scripts/training/sonar_hc.py` | Training | None | `sonar` (direct monkeypatch) | None |
| 6 | `mnist.py` | `scripts/training/mnist.py` | Training / Stale | None | None | None |
| 7 | `sonar.sh` | `scripts/training/sonar.sh` | Training | None | `sonar.py` (shell invocation) | None |
| 8 | `decode.py` | `scripts/interactive/decode.py` | Interactive | `run_chat.py`, `run_chat2.py`, `run_decode.py` | None | None |
| 9 | `decode_tags.py` | `scripts/interactive/decode_tags.py` | Interactive | `run_decode_tags.py` | None | None (referenced in `test_code_geometry.py`) |
| 10 | `run_chat.py` | `scripts/interactive/runners/run_chat.py` | Interactive | None | `decode.py` (subprocess) | None |
| 11 | `run_chat2.py` | `scripts/interactive/runners/run_chat2.py` | Interactive | None | `decode.py` (subprocess) | None |
| 12 | `run_decode.py` | `scripts/interactive/runners/run_decode.py` | Interactive | None | `decode.py` (subprocess) | None |
| 13 | `run_decode_tags.py` | `scripts/interactive/runners/run_decode_tags.py` | Interactive | None | `decode_tags.py` (subprocess) | None |
| 14 | `sae.py` | `scripts/sae/sae.py` | SAE Baseline | `autointerp.py`, `headstruct.py`, `pareto.py`, `refit.py`, `splitting.py`, `steerfid.py`, `textfid.py`, `sae_ladder.sh`, `sae_evals.sh` | None | `tests/test_sae.py`, `tests/test_refit.py` |
| 15 | `sae_ladder.sh` | `scripts/sae/sae_ladder.sh` | SAE Baseline | None | `sae.py`, `pareto.py`, `headstruct.py` | None |
| 16 | `autointerp.py` | `scripts/eval/autointerp.py` | Evaluation | `headcoh.py`, `refit.py`, `splitting.py`, `autointerp_run.sh` | `sae` | `tests/test_autointerp.py` |
| 17 | `autointerp_run.sh` | `scripts/eval/autointerp_run.sh` | Evaluation | None | `autointerp.py` (shell invocation) | None |
| 18 | `compose.py` | `scripts/eval/compose.py` | Evaluation | None | `pareto` | `tests/test_compose.py` |
| 19 | `headcoh.py` | `scripts/eval/headcoh.py` | Evaluation | None | `refit`, `autointerp` | `tests/test_headcoh.py` |
| 20 | `headstruct.py` | `scripts/eval/headstruct.py` | Evaluation | `sae_ladder.sh`, `sae_evals.sh` | `sae`, `pareto` | `tests/test_headstruct.py` |
| 21 | `langprobe.py` | `scripts/eval/langprobe.py` | Evaluation | `sae_evals.sh` | `splitting` | `tests/test_langprobe.py` |
| 22 | `pareto.py` | `scripts/eval/pareto.py` | Evaluation | `compose.py`, `headstruct.py`, `refit.py`, `splitting.py`, `steerfid.py`, `textfid.py`, `sae_ladder.sh`, `sae_evals.sh` | `sae` | `tests/test_pareto.py` |
| 23 | `refit.py` | `scripts/eval/refit.py` | Evaluation | `headcoh.py`, `sae_evals.sh` | `sae`, `autointerp`, `pareto` | `tests/test_refit.py` |
| 24 | `sae_evals.sh` | `scripts/eval/sae_evals.sh` | Evaluation | None | `sae.py`, `pareto.py`, `headstruct.py`, `splitting.py`, `textfid.py`, `langprobe.py`, `refit.py` | None |
| 25 | `splitting.py` | `scripts/eval/splitting.py` | Evaluation | `langprobe.py`, `sae_evals.sh` | `sae`, `autointerp`, `pareto` | `tests/test_splitting.py` |
| 26 | `steerfid.py` | `scripts/eval/steerfid.py` | Evaluation | None | `sae`, `pareto`, `textfid` | `tests/test_steerfid.py` |
| 27 | `textfid.py` | `scripts/eval/textfid.py` | Evaluation | `steerfid.py`, `sae_evals.sh` | `sae`, `pareto` | `tests/test_textfid.py` |

---

## 5. Transition & Compatibility Strategies

To move these 27 files without breaking the test suite, shell scripts, or internal import paths, several strategies are available. We evaluate four distinct approaches and present a recommended hybrid implementation plan.

### 5.1 Comparative Analysis of Strategies

| Strategy | Description | Pros | Cons | Feasibility / Risk |
|----------|-------------|------|------|--------------------|
| **Strategy A: Test Runner PYTHONPATH Adjustment** | Modify `tests/conftest.py` (or `pytest.ini`) to add `scripts/eval`, `scripts/sae`, etc., to `sys.path`. | Zero changes required to the 11 test files; tests keep their current `import sae` statements. | Pollutes `sys.path` with multiple flat directories; hides module namespace structure; does not solve shell script invocations. | High feasibility, low immediate risk. |
| **Strategy B: Test Import Path Modernization** | Treat `scripts` as a package and update `tests/` imports to `from scripts.eval.autointerp import ...` or `from scripts.sae.sae import ...`. | Clean, explicit, standard Python package architecture; zero root pollution; enforces true dependency hierarchy. | Requires editing 11 test files and updating internal sibling imports across scripts. | Medium work, zero runtime ambiguity. |
| **Strategy C: Backward-Compatibility Root Shims** | Leave thin forwarding stubs at the root (e.g. `sae.py` containing `from scripts.sae.sae import *`). | 100% backward compatible with existing systemd units, external cron jobs, shell scripts, and tests without changing any existing caller. | Leaves files in root directory; defeats the visual goal of a clean root until shims are deprecated. | Trivial implementation, zero breaking changes. |
| **Strategy D: Core Library Elevation (Long-Term Ideal)** | Extract pure mathematical/statistical functions from scripts into `ontologize.analysis` or `ontologize.eval` and `ontologize.sae`. Leave only thin CLI wrappers in `scripts/`. | Eliminates script-to-script coupling entirely; tests only test library code; scripts only handle CLI/I/O; eliminates repeated boilerplate. | Requires code refactoring beyond simple file movement. | Highest architectural quality, medium-high effort. |

---

### 5.2 Recommended Implementation Strategy: 3-Phase Migration

To achieve complete cleanliness while ensuring zero downtime or test breakage, we recommend a phased approach:

#### Phase 1: Structural Move with Compatibility Shims (Immediate Safe Move)
1. Move the 27 scripts to their designated target paths under `scripts/`.
2. Add `__init__.py` files to `scripts/`, `scripts/data/`, `scripts/training/`, `scripts/interactive/`, `scripts/sae/`, and `scripts/eval/`.
3. In the repository root, install temporary thin forwarding stubs for the 11 scripts imported by tests and shell scripts (e.g. `sae.py`, `pareto.py`, `autointerp.py`). Each stub re-exports symbols:
   ```python
   # sae.py (temporary backward-compatibility stub)
   import sys
   from pathlib import Path
   _root = Path(__file__).resolve().parent
   if str(_root) not in sys.path:
       sys.path.insert(0, str(_root))
   from scripts.sae.sae import *
   from scripts.sae.sae import __all__  # if defined
   ```
4. Update `tests/conftest.py` so that `sys.path` includes both `scripts/eval` and `scripts/sae`:
   ```python
   root = Path(__file__).parents[1]
   sys.path.insert(0, str(root))
   sys.path.insert(0, str(root / "scripts" / "sae"))
   sys.path.insert(0, str(root / "scripts" / "eval"))
   ```
   *Result:* All 25 test files continue to pass without a single change to test code!

#### Phase 2: Updating Shell Scripts & Subprocess Callers
1. Update `autointerp_run.sh`, `sae_ladder.sh`, `sae_evals.sh`, and `sonar.sh`:
   - Point `cd "$(dirname "$0")"` to repo root (`cd "$(dirname "$0")/../.."` if shell scripts move to subdirectories, or keep shell scripts in `scripts/` with `cd "$(dirname "$0")/.."`).
   - Change `uv run python sae.py` to `uv run python scripts/sae/sae.py` (or `uv run python -m scripts.sae.sae`).
2. Update the 4 interactive runners (`run_chat.py`, `run_chat2.py`, `run_decode.py`, `run_decode_tags.py`) to resolve `decode.py` and `decode_tags.py` relative to `Path(__file__).parent / "decode.py"`.

#### Phase 3: Sibling Import Normalization & Shim Removal
1. Update inter-script imports within `scripts/` to use absolute package imports:
   - In `scripts/eval/refit.py`: `from scripts.sae import sae`
   - In `scripts/eval/steerfid.py`: `from scripts.eval.textfid import chrf`
2. Remove root stubs.
3. (Optional / Future): Consolidate repeated Orbax loading boilerplate into `ontologize.training.ontostate` (`load_checkpoint_model(ckpt, step)`).

---

## 6. Catalog of Observed Bugs, Dead Code, and Oddities

During the survey, several bugs, anti-patterns, and architectural anomalies were identified across the 27 scripts. As per integrity instructions, **none of these were modified**. They are cataloged here for future remediation:

### 6.1 Fatal Bugs & Broken Imports
1. **Broken Import in `mnist.py` (Line 7):**
   ```python
   from ontologize.training.data import ImageLoader
   ```
   *Anomaly:* There is no module or package named `ontologize.training.data`. The class `ImageLoader` is located in `ontologize.data.loaders`. Running `python mnist.py` immediately crashes with `ModuleNotFoundError: No module named 'ontologize.training.data'`.
2. **Truncated Argument List in `mnist.py` (Lines 44-45):**
   `TrainingEnv` instantiation passes parameters that do not match the current signature in `ontologize.training.config.TrainingEnv`, and the file is an abandoned experimental script.

### 6.2 Architectural Anti-Patterns & Fragile Couplings
3. **Module-Level Monkeypatching in `sonar_hc.py` (Lines 33-42):**
   ```python
   import sonar
   sonar.s_hcossim = 1e-5
   sonar.s_Hm = 0.0
   sonar.out = sonar.path / "out/sonar/multilingual/resid_nc_hc"
   print(f"resid_nc_hc: s_hcossim={sonar.s_hcossim} s_Hm={sonar.s_Hm} -> {sonar.out}", flush=True)
   sonar.main()
   ```
   *Anomaly:* `sonar_hc.py` mutates module-level globals in `sonar.py` before calling `sonar.main()`. If `sonar.py` were refactored to use standard CLI arguments or an encapsulation class, `sonar_hc.py` would silently fail or misconfigure the ablation.
4. **Divergent CuDNN Dynamic Injections:**
   - `decode.py` (lines 6-15) and `decode_tags.py` (lines 6-15) perform **process replacement** via `os.execv`:
     ```python
     if "CUDNN_INJECTED" not in os.environ:
         # mutate LD_LIBRARY_PATH
         os.environ["CUDNN_INJECTED"] = "1"
         os.execv(sys.executable, [sys.executable] + sys.argv)
     ```
     This restarts the Python process from scratch with a new PID to force glibc to reload CuDNN libraries.
   - In contrast, `encode_corpus.py` (lines 23-32), `encode_discord.py` (lines 34-41), and `sonar.py` (lines 19-32) dynamically load `libcudnn.so` using `ctypes.CDLL(str(cudnn_path), mode=os.RTLD_GLOBAL)`.
   *Anomaly:* Two completely different strategies are used to bypass the same CUDA/CuDNN library resolution issue. `os.execv` breaks wrappers expecting a continuous child process.

### 6.3 Shell Script Inconsistencies
5. **Inconsistent Shell Invocations in `sonar.sh`:**
   - Lines 3-4:
     ```bash
     tmux new -s training
     uv run sonar.py
     ```
   - *Anomaly:* Lacks `cd "$(dirname "$0")"` (fails if invoked outside repo root). Invokes `uv run sonar.py` rather than `uv run python sonar.py` as used in all other 3 shell scripts.

### 6.4 Redundancies and Dead Code
6. **Hexa-duplicate Model Loading Logic:**
   Six distinct evaluation scripts (`classify_acts.py`, `pareto.py`, `splitting.py`, `steerfid.py`, `refit.py`, `autointerp.py`) contain copy-pasted Orbax checkpoint managers and parameter unpacking loops (`while 'params' in params: params = params['params']`).
7. **Dead Legacy Modules in Core Package:**
   As confirmed by `ORIGINAL_REQUEST.md`, four modules in `ontologize/layers/` are completely dead and unreferenced by any script or test:
   - `ontologize/layers/dictblock_enhanced.py`
   - `ontologize/layers/dict_interpreter.py`
   - `ontologize/layers/integration_clean.py`
   - `ontologize/layers/vae_integration.py`
8. **Comment Anachronism in `pyproject.toml` (Lines 32-33):**
   ```toml
   # only the curated suite -- root-level test_*.py are ad hoc debug scripts,
   # not tests, and several execute GPU/checkpoint code at import time
   testpaths = ["tests"]
   ```
   *Anomaly:* There are currently no `test_*.py` files in the repository root. This refers to an older state of the codebase.

---

## 7. Next Steps for Implementation

When authorization is granted to execute the reorganization:
1. Create target directories under `scripts/`.
2. Move files according to the Move Map (Section 4).
3. Apply Phase 1 compatibility shims and `conftest.py` path adjustments.
4. Execute `pytest` to confirm 100% test pass rate.
5. Update shell scripts and subprocess runners (Phase 2).
6. Verify syntax with `py_compile` across all relocated scripts.
