# Repository Root Script Reorganization Proposal

**Document Version:** 1.0.0  
**Status:** Analytical Proposal & Migration Specification (No files moved)  
**Target Repository:** `ontologize`  
**Author:** worker_m5 (Milestone 5)  

---

## 1. Executive Summary & Problem Statement

The `ontologize` repository root currently contains **27 standalone scripts** (23 Python scripts `*.py` and 4 Bash shell scripts `*.sh`). While this flat structure facilitated rapid experimentation during the initial research phase, it introduces severe architectural and operational liabilities:

1. **Namespace & Cognitive Clutter:** The repository root conflates core package code, offline data preparation, multi-node GPU training, interactive causal intervention REPLs, baseline Sparse Autoencoder (SAE) training, downstream interpretability evaluation batteries, and automated test-driving subprocess wrappers.
2. **Hidden Test Suite Couplings:** 11 of the 25 test suites in `tests/` bypass the installable `ontologize` library package and import root scripts directly as top-level modules (`import sae`, `import pareto`, `from autointerp import ...`). These imports rely on an implicit `sys.path.insert(0, str(Path(__file__).parents[1]))` configured in `tests/conftest.py`.
3. **Dense Sibling Dependencies:** Root evaluation scripts do not execute in isolation; they form an interconnected dependency web with **20 directed import edges** across 11 scripts, centered on two major hubs: `sae.py` (7 inbound dependents) and `pareto.py` (6 inbound dependents).
4. **Shell Script & Subprocess Fragility:** 4 orchestration shell scripts (`autointerp_run.sh`, `sae_ladder.sh`, `sae_evals.sh`, `sonar.sh`) and 4 Python test drivers (`run_chat.py`, `run_chat2.py`, `run_decode.py`, `run_decode_tags.py`) rely on hardcoded relative paths from the repository root. Any uncoordinated file rename or movement immediately breaks training workflows and evaluation pipelines.
5. **Code Duplication & Architectural Drift:** Reusable scientific logic (such as character n-gram F-score calculations, greedy group clustering, and Orbax checkpoint restoration) is implemented redundantly across scripts rather than exposed through the `ontologize/` package.

This proposal provides an exhaustive coupling analysis, introduces a clean **5-domain modular structure** under `scripts/`, details a complete **27-script move map**, and specifies a **phased, zero-downtime transition strategy** that preserves 100% test compatibility and shell workflow integrity.

---

## 2. Exhaustive Coupling Constraints Analysis

Any restructuring of the root scripts must account for three primary coupling vectors: direct imports from `tests/`, inter-script sibling imports, and filesystem path dependencies in shell scripts and subprocess wrappers.

### 2.1 Test Suite Coupling (`tests/`)

In `tests/conftest.py` (lines 9–11), the repository root is prepended to Python's module search path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
```

Because of this runtime configuration, test modules in `tests/` resolve root scripts as top-level Python modules. Exactly **11 test files** (out of 25 in the test suite) depend directly on root scripts:

| Test File | Direct Imported Script(s) | Import Statement & Line | Tested Symbols & Functional Scope |
|---|---|---|---|
| `tests/test_autointerp.py` | `autointerp` | `from autointerp import (CAL_PROMPT_CHARS, CONTRAST_PROMPT, SUMMARIZE_PROMPT, call_weight, contrast_rows, detection_items, cap_pairing, next_slot, null_pairing, read_scores, snippet, stratified_features)` (lines 6–10) | Rate-limiting pacing (`next_slot`), token quota weighting (`call_weight`), feature stratification (`stratified_features`), contrastive row generation (`contrast_rows`), detection items (`detection_items`), and calibration prompts. |
| `tests/test_compose.py` | `compose` | `from compose import additivity_stats, sample_singles` (line 6) | Causal intervention additivity statistics (`additivity_stats`), relative errors under whitened metric, and single head sampling (`sample_singles`). |
| `tests/test_headcoh.py` | `headcoh` | `from headcoh import (group_zscores, mean_pairwise_cos, perm_z, pooled_within_cos, sv_energy, unit)` (lines 5–6) | Head semantic coherence metrics, singular value energy ratio (`sv_energy`), permutation z-score calculations (`perm_z`, `group_zscores`), and pooled within-head cosine similarity. |
| `tests/test_headstruct.py` | `headstruct` | `import headstruct` (line 8) | Categorical head discovery algorithms: graph affinity calculation (`affinity_edges`), greedy group clustering with size caps (`greedy_groups`), moments (`group_moments`), and exclusivity/exhaustiveness metrics. |
| `tests/test_langprobe.py` | `langprobe` | `import langprobe` (line 7) | Sparse language probing: two-sample t-statistic feature selection (`t_stats`), optimal F1 threshold search (`best_f1_threshold`), F1 evaluation (`f1_at`), and logistic regression fitting (`fit_logistic`). |
| `tests/test_pareto.py` | `pareto` | `import pareto` (line 8) | Deviation code capacity truncation (`topdev`), head entropy filtering (`tophead`), and SAE checkpoint directory name parsing (`parse_sae_name`). |
| `tests/test_refit.py` | `refit`<br>`sae` | `import refit` (line 8)<br>`import sae` (line 50) | Unconstrained masked least-squares reconstruction (`refit_recon`) and SAE active latent support extraction (`sae_supports`). |
| `tests/test_sae.py` | `sae` | `import sae` (line 10) | Standalone SAE baseline trainer: parameter initialization (`init_params`), Top-K sparse autoencoder training loop (`train`), checkpoint snapshot resumption, eigenfeature extraction (`eigenfeatures`), and evaluation metric builders (`make_eval`, `evaluate`). |
| `tests/test_splitting.py` | `splitting` | `import splitting` (line 8) | Cross-model feature splitting analysis: activation containment matching (`split_children`), union coverage calculation (`union_cover_counts`), and child overlap verification (`child_overlap`). |
| `tests/test_steerfid.py` | `steerfid` | `import steerfid` (line 7) | Vector normalization (`unit`), active feature selection (`pick_features`), steering target gain vs collateral distortion scoring (`effect_scores`), and firing quantiles (`firing_quantiles`). |
| `tests/test_textfid.py` | `textfid` | `from textfid import chrf` (line 6) | Standalone character n-gram F-score implementation (`chrf`) with beta recall weighting. |

#### Architectural Insight on Test Nature
None of the 11 tests execute CLI entry points (`main()`), invoke `argparse`, or perform disk I/O on production models. Instead, **every test file targets pure mathematical, algorithmic, or statistical helper functions** that happen to be co-located with CLI scripts. Moving scripts naively into subfolders without adjusting `sys.path` or providing module re-exports would instantly fail 11 test suites with `ModuleNotFoundError`.

---

### 2.2 Sibling Script Cross-Imports (Inter-Script Couplings)

A systematic AST scan reveals **20 unique directed dependency edges** connecting 11 root Python scripts. The graph exhibits two major structural hubs: `sae.py` and `pareto.py`.

```
                        +-------------------+
                        |      sae.py       |<--------------------------+
                        +-------------------+                           |
                          ^   ^   ^   ^   ^                             |
             +------------+   |   |   |   +---------------+             |
             |                |   |   |                   |             |
    +-----------------+       |   |   |           +-----------------+   |
    |  autointerp.py  |       |   |   +-----------|  headstruct.py  |   |
    +-----------------+       |   |               +-----------------+   |
      ^             ^         |   |                 |                   |
      |             |         |   |                 v                   |
      |      +------+---------+   |               +-----------------+   |
      |      |      |             |               |    pareto.py    |---+
      |      |      |             |               +-----------------+   |
      |      |      |             |                 ^   ^   ^   ^   ^   |
      |      |      |             +-----------+     |   |   |   |   |   |
      |      |      |                         |     |   |   |   |   |   |
      |   +-----------------+                 |     |   |   |   |   |   |
      |   |    refit.py     |-----------------+-----+   |   |   |   |   |
      |   +-----------------+                           |   |   |   |   |
      |     ^                                           |   |   |   |   |
      |     |                                           |   |   |   |   |
      |   +-----------------+                           |   |   |   |   |
      +---|   headcoh.py    |                           |   |   |   |   |
          +-----------------+                           |   |   |   |   |
                                                        |   |   |   |   |
      +-----------------+                               |   |   |   |   |
      |   splitting.py  |-------------------------------+   |   |   |   |
      +-----------------+ (also imports autointerp & sae)   |   |   |   |
        ^                                                   |   |   |   |
        |                                                   |   |   |   |
      +-----------------+                                   |   |   |   |
      |   langprobe.py  |                                   |   |   |   |
      +-----------------+                                   |   |   |   |
                                                            |   |   |   |
      +-----------------+                                   |   |   |   |
      |   compose.py    |-----------------------------------+   |   |   |
      +-----------------+                                       |   |   |
                                                                |   |   |
      +-----------------+                                       |   |   |
      |   textfid.py    |---------------------------------------+   |   |
      +-----------------+ (also imports sae)                        |   |
        ^                                                           |   |
        |                                                           |   |
      +-----------------+                                           |   |
      |   steerfid.py   |-------------------------------------------+   |
      +-----------------+ (also imports sae, textfid)                   |
                                                                        |
      +-----------------+                                               |
      |   sonar_hc.py   |---> [sonar.py] (direct module monkeypatch)     |
      +-----------------+                                               |
```

#### Detailed Inventory of Sibling Dependencies:

| # | Consumer Script | Target Script | Line & Statement | Import Scope | Symbols / Purpose |
|---|---|---|---|---|---|
| 1 | `autointerp.py` | `sae.py` | Line 452: `import sae as sae_mod`<br>Line 550: `import sae as sae_mod` | Function-level (`sae_acts_fn`, `harvest`) | Encodes embeddings into SAE latents; extracts bilinear SAE decode directions via `eigenfeatures()`. |
| 2 | `compose.py` | `pareto.py` | Line 89: `from pareto import load_onto` | Function-level (`main`) | Restores Ontologizer model instance and parameters from Orbax checkpoint. |
| 3 | `headcoh.py` | `refit.py` | Line 145: `from refit import onto_linear_model` | Function-level (`onto_directions`) | Extracts linear decode weight matrix $G$ for Ontologizer heads. |
| 4 | `headcoh.py` | `autointerp.py` | Line 186: `from autointerp import ENCODER_ID` | Function-level (`encode_texts`) | Reads HuggingFace pretrained SONAR encoder ID string. |
| 5 | `headstruct.py` | `sae.py` | Line 55: `import sae` | Module-level | Encodes batch embeddings into Top-K / Group SAE representations. |
| 6 | `headstruct.py` | `pareto.py` | Line 184: `from pareto import parse_sae_name` | Function-level (`main`) | Parses SAE directory name to extract parameters $m$ and $k$. |
| 7 | `langprobe.py` | `splitting.py` | Line 39: `from splitting import load_model` | Module-level | Unified model loader supporting both SAE `.npz` files and Ontologizer checkpoints. |
| 8 | `pareto.py` | `sae.py` | Line 62: `import sae` | Module-level | Evaluates SAE checkpoints via `sae.make_eval()` and `sae.evaluate()`. |
| 9 | `refit.py` | `sae.py` | Line 50: `import sae` | Module-level | Encodes latents to establish support masks and decodes reconstructions. |
| 10 | `refit.py` | `autointerp.py` | Line 125: `from autointerp import onto_acts_fn` | Function-level (`onto_linear_model`) | Loads Ontologizer activation function and classification weights. |
| 11 | `refit.py` | `pareto.py` | Line 170: `from pareto import parse_sae_name` | Function-level (`main`) | Extracts latent capacity $k$ from SAE artifact directory name. |
| 12 | `sonar_hc.py` | `sonar.py` | Line 33: `import sonar` | Module-level | **Direct monkeypatching:** Mutates `sonar.s_hcossim = 1e-5`, `sonar.s_Hm = 0.0`, `sonar.out = ...`, then calls `sonar.main()`. |
| 13 | `splitting.py` | `sae.py` | Line 54: `import sae` | Module-level | Evaluates SAE forward encoder inside JIT-compiled activation function. |
| 14 | `splitting.py` | `pareto.py` | Line 91: `from pareto import parse_sae_name` | Function-level (`load_model`) | Parses model dimensions from SAE directory string. |
| 15 | `splitting.py` | `autointerp.py` | Line 97: `from autointerp import onto_acts_fn` | Function-level (`load_model`) | Obtains Ontologizer classification activation matrices. |
| 16 | `steerfid.py` | `sae.py` | Line 49: `import sae` | Module-level | Encodes representations and extracts decode directions via `eigenfeatures()`. |
| 17 | `steerfid.py` | `pareto.py` | Line 139: `from pareto import load_onto` | Function-level (`load_steerable`) | Loads Ontologizer checkpoint and restores model PyTree for causal steering. |
| 18 | `steerfid.py` | `textfid.py` | Line 183: `from textfid import chrf` | Function-level (`main`) | Computes character n-gram F-score between original and intervened decodes. |
| 19 | `textfid.py` | `sae.py` | Line 44: `import sae` | Module-level | Encodes embeddings and reconstructs text through SAE decoders. |
| 20 | `textfid.py` | `pareto.py` | Line 105: `from pareto import parse_sae_name`<br>Line 115: `from pareto import load_onto` | Function-level (`recon_fn`) | Parses SAE parameters and restores Ontologizer checkpoint state. |

---

### 2.3 Shell Scripts Invocations & Execution Environment

The 4 shell scripts in the repository root automate multi-stage training ladders and evaluation campaigns. They invoke Python scripts by relative filenames assuming execution from the repository root:

#### 1. `autointerp_run.sh` (102 lines)
- **Directory Pinning:** Line 22 executes `cd "$(dirname "$0")"`.
- **Environment Prerequisites:**
  - `export PATH="$HOME/.local/bin:$PATH"`
  - Requires `ANTHROPIC_API_KEY` (either in environment or pointing to a secret file). Fails immediately if missing.
  - Pacing rate `AI_RATE` defaults to 1000 queries/hour.
- **Hardware Polling:** Custom `wait_gpu()` function queries `nvidia-smi` until at least 6,000 MiB free VRAM is available.
- **Python Script Invocations:**
  - `uv run python autointerp.py harvest --model onto --ckpt $ONTO --out $AI/onto` (line 65)
  - `uv run python autointerp.py harvest --model sae --ckpt $SAE/$m/params.npz --out $AI/$m` (line 68)
  - `uv run python autointerp.py texts --out $AI --features ...` (line 73)
  - `uv run python autointerp.py describe --dir ... --out $AI --mode ... --rate $RATE` (lines 77, 81, 86)
  - `uv run python autointerp.py score --dir ... --out $AI --judge haiku --jobs 4 --rate $RATE --null --null-n 100` (line 91)
- **Idempotency:** State guarded by `data/out/sonar/autointerp/.campaign_done` and per-run artifact checks.

#### 2. `sae_ladder.sh` (67 lines)
- **Directory Pinning:** Line 20 executes `cd "$(dirname "$0")"`.
- **Hardware Polling:** `wait_gpu()` loop (minimum 6,000 MiB free VRAM).
- **Python Script Invocations:**
  - `uv run python sae.py --lr 4e-4 --epochs 150 --m 5120 --topk 32` (line 49)
  - `uv run python sae.py --lr 4e-4 --epochs 150 --m 5120 --topk 0 --groups 160` (line 50)
  - `uv run python sae.py --lr 4e-4 --epochs 150 --m 5120 --topk 0 --groups 160 --group-fn softmax` (line 51)
  - `uv run python sae.py --lr 4e-4 --epochs 150 --m 5120 --topk 32 --prefixes 5` (line 52)
  - `uv run python pareto.py` (line 55)
  - `uv run python headstruct.py --rows 1048576 --sae ...` (lines 56–63)
- **Idempotency:** Guarded by `data/out/sonar/sae/.ladder_done`.

#### 3. `sae_evals.sh` (88 lines)
- **Directory Pinning:** Line 13 executes `cd "$(dirname "$0")"`.
- **Hardware Polling:** `wait_gpu()` loop.
- **Python Script Invocations:**
  - `uv run python sae.py --m 11264 --topk 5120 --epochs 23` (line 42)
  - `uv run python sae.py --m 5120 --topk 32 --enc bilinear --lr 4e-4 --epochs 150` (line 45)
  - `uv run python sae.py --m 5120 --topk 32 --lr 4e-4 --epochs 150 --seed 43` (line 47)
  - `uv run python pareto.py || FAIL=1` (line 52)
  - `uv run python headstruct.py --sae $SAE/m11264_k160/params.npz --rows 1048576` (line 55)
  - `uv run python splitting.py --a $SAE/m5120_k32/params.npz --b $SAE/m11264_k32/params.npz` (line 59)
  - `uv run python splitting.py --a $SAE/m5120_k32/params.npz --b $SAE/m5120_k32_s43/params.npz` (line 62)
  - `uv run python textfid.py --model $SAE/$run/params.npz --device cuda --b-decode 64` (line 68)
  - `uv run python langprobe.py --model $SAE/$run/params.npz` (line 71)
  - `uv run python textfid.py --model $ONTO --device cuda --b-decode 64` (line 74)
  - `uv run python langprobe.py --model $ONTO` (line 76)
  - `uv run python refit.py --onto $ONTO` (line 79)
- **Idempotency:** Guarded by `data/out/sonar/sae/.evals_done` and step-level artifact presence checks.

#### 4. `sonar.sh` (9 lines)
- **Directory Pinning:** **Missing.** Does not execute `cd "$(dirname "$0")"`.
- **Invocation:**
  ```bash
  #!/bin/bash
  tmux new -s training
  uv run sonar.py
  ```
- **Flaw:** Invokes `uv run sonar.py` rather than `uv run python sonar.py`, and fails if invoked from any directory other than root.

---

### 2.4 Subprocess Invocation Wrappers

Four Python scripts in the root directory serve exclusively as test-driving wrappers that invoke `decode.py` and `decode_tags.py` via `subprocess`:

| Wrapper Script | Target Script | Invocations & Arguments | Mechanism & Execution Flags |
|---|---|---|---|
| `run_chat.py` | `decode.py` | `["uv", "run", "python", "decode.py", "data/out/sonar/linear/multilingual"]` | `subprocess.Popen` via `pty.openpty()` to simulate interactive terminal inputs (`set 0 k_add 5,10`, `run ...`). Sets `XLA_PYTHON_CLIENT_MEM_FRACTION=".10"`. |
| `run_chat2.py` | `decode.py` | `["uv", "run", "python", "decode.py", "data/out/sonar/linear/multilingual"]` | `subprocess.run` with piped stdin strings. Sets `CUDA_VISIBLE_DEVICES=""` for CPU execution. |
| `run_decode.py` | `decode.py` | `["uv", "run", "python", "decode.py", "data/out/sonar/linear/multilingual"]` | `subprocess.run` feeding multi-line text input string. Sets `CUDA_VISIBLE_DEVICES=""`. |
| `run_decode_tags.py` | `decode_tags.py` | `["uv", "run", "python", "decode_tags.py", "data/out/sonar/linear/multilingual"]` | `subprocess.run` capturing stdout. Sets `XLA_PYTHON_CLIENT_MEM_FRACTION=".10"`. |

If `decode.py` or `decode_tags.py` are moved without updating these path strings, all 4 runners immediately crash with `FileNotFoundError`.

---

## 3. Proposed Target Directory Layout

To eliminate root clutter, cleanly isolate operational concerns, and maintain strict modular boundaries, we propose migrating all 27 root scripts into a dedicated `scripts/` top-level directory partitioned into **5 functional domains**:

```
ontologize/
├── ontologize/                     # Core Python package (layers, training, data, fns, inference)
│   ├── layers/
│   ├── training/
│   ├── data/
│   ├── fns/
│   ├── inference/
│   └── visualize/
├── tests/                          # Automated Pytest suite
│   ├── conftest.py
│   └── test_*.py
├── docs/                           # Documentation and architectural specifications
│   ├── SCRIPTS.md                  # Comprehensive root script inventory & catalog
│   └── REORG_PROPOSAL.md           # Reorganization proposal & migration plan (this document)
└── scripts/                        # Target directory for all standalone scripts
    ├── __init__.py
    ├── data/                       # Domain 1: Data preparation, harvest, corpus embedding caching
    │   ├── __init__.py
    │   ├── encode_corpus.py
    │   ├── encode_discord.py
    │   └── classify_acts.py
    ├── training/                   # Domain 2: Ontologizer model training & loss ablations
    │   ├── __init__.py
    │   ├── sonar.py
    │   ├── sonar_hc.py
    │   ├── mnist.py
    │   └── sonar.sh
    ├── interactive/                # Domain 3: Interactive decoding, inspection CLI, tag decoders
    │   ├── __init__.py
    │   ├── decode.py
    │   ├── decode_tags.py
    │   └── runners/                # Subprocess drivers & smoke harnesses
    │       ├── __init__.py
    │       ├── run_chat.py
    │       ├── run_chat2.py
    │       ├── run_decode.py
    │       └── run_decode_tags.py
    ├── sae/                        # Domain 4: Standalone Sparse Autoencoder baseline trainer & ladder
    │   ├── __init__.py
    │   ├── sae.py
    │   └── sae_ladder.sh
    └── eval/                       # Domain 5: Evaluation battery, representation geometry, auto-interp
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

### Architectural Rationale by Domain

1. **`scripts/data/` (3 scripts):** Houses offline data pipelines. These scripts run before training to tokenize, encode, and serialize multilingual embeddings into memory-mapped `.npy` files. They have zero inbound runtime dependencies from other scripts or tests.
2. **`scripts/training/` (4 scripts):** Groups Ontologizer training entry points (`sonar.py`), hyperparameter loss ablations (`sonar_hc.py`), toy experiments (`mnist.py`), and the tmux runner (`sonar.sh`).
3. **`scripts/interactive/` (6 scripts):** Separates human-facing causal steering REPLs (`decode.py`, `decode_tags.py`) and isolates their 4 ad-hoc subprocess test runners into an explicit `runners/` subpackage.
4. **`scripts/sae/` (2 scripts):** Completely encapsulates the independent Sparse Autoencoder baseline implementation and its multi-stage training ladder (`sae_ladder.sh`). This isolates SAE logic from Ontologizer model definitions while making it straightforward to import across evaluation tools.
5. **`scripts/eval/` (12 scripts):** Co-locates the entire scientific evaluation suite (Pareto frontiers, semantic head coherence, categorical head structure, cross-model feature splitting, steering fidelity, downstream text reconstruction fidelity, and automated Claude-based interpretability) alongside their orchestration shell scripts.

---

## 4. Concrete Move Map for All 27 Scripts

The table below defines the exact 1-to-1 mapping for all 27 root scripts from their current location to the proposed target layout, summarizing their functional roles, inbound/outbound dependencies, and test coverage:

| # | Current Root Path | Proposed Target Path | Domain | Inbound Dependencies (Imported/Called By) | Outbound Dependencies (Imports Sibling) | Covering Test File in `tests/` |
|---|---|---|---|---|---|---|
| 1 | `encode_corpus.py` | `scripts/data/encode_corpus.py` | Data | None | None | None |
| 2 | `encode_discord.py` | `scripts/data/encode_discord.py` | Data | None | None | None |
| 3 | `classify_acts.py` | `scripts/data/classify_acts.py` | Data | None | None | None |
| 4 | `sonar.py` | `scripts/training/sonar.py` | Training | `sonar_hc.py`, `sonar.sh` | None | None (training step unit-tested in `test_training_step.py`) |
| 5 | `sonar_hc.py` | `scripts/training/sonar_hc.py` | Training | None | `sonar` (direct monkeypatch) | None |
| 6 | `mnist.py` | `scripts/training/mnist.py` | Training | None | None (contains broken import) | None |
| 7 | `sonar.sh` | `scripts/training/sonar.sh` | Training | None | `sonar.py` (shell invocation) | None |
| 8 | `decode.py` | `scripts/interactive/decode.py` | Interactive | `run_chat.py`, `run_chat2.py`, `run_decode.py` | None | None |
| 9 | `decode_tags.py` | `scripts/interactive/decode_tags.py` | Interactive | `run_decode_tags.py` | None | None (decoder geometry covered in `test_code_geometry.py`) |
| 10 | `run_chat.py` | `scripts/interactive/runners/run_chat.py` | Interactive | None | `decode.py` (subprocess PTY) | None |
| 11 | `run_chat2.py` | `scripts/interactive/runners/run_chat2.py` | Interactive | None | `decode.py` (subprocess stdin) | None |
| 12 | `run_decode.py` | `scripts/interactive/runners/run_decode.py` | Interactive | None | `decode.py` (subprocess stdin) | None |
| 13 | `run_decode_tags.py` | `scripts/interactive/runners/run_decode_tags.py` | Interactive | None | `decode_tags.py` (subprocess) | None |
| 14 | `sae.py` | `scripts/sae/sae.py` | SAE Baseline | `autointerp.py`, `headstruct.py`, `pareto.py`, `refit.py`, `splitting.py`, `steerfid.py`, `textfid.py`, `sae_ladder.sh`, `sae_evals.sh` | None | `tests/test_sae.py`<br>`tests/test_refit.py` |
| 15 | `sae_ladder.sh` | `scripts/sae/sae_ladder.sh` | SAE Baseline | None | `sae.py`, `pareto.py`, `headstruct.py` | None |
| 16 | `autointerp.py` | `scripts/eval/autointerp.py` | Eval | `headcoh.py`, `refit.py`, `splitting.py`, `autointerp_run.sh` | `sae` | `tests/test_autointerp.py` |
| 17 | `autointerp_run.sh` | `scripts/eval/autointerp_run.sh` | Eval | None | `autointerp.py` (shell invocation) | None |
| 18 | `compose.py` | `scripts/eval/compose.py` | Eval | None | `pareto` | `tests/test_compose.py` |
| 19 | `headcoh.py` | `scripts/eval/headcoh.py` | Eval | None | `refit`, `autointerp` | `tests/test_headcoh.py` |
| 20 | `headstruct.py` | `scripts/eval/headstruct.py` | Eval | `sae_ladder.sh`, `sae_evals.sh` | `sae`, `pareto` | `tests/test_headstruct.py` |
| 21 | `langprobe.py` | `scripts/eval/langprobe.py` | Eval | `sae_evals.sh` | `splitting` | `tests/test_langprobe.py` |
| 22 | `pareto.py` | `scripts/eval/pareto.py` | Eval | `compose.py`, `headstruct.py`, `refit.py`, `splitting.py`, `steerfid.py`, `textfid.py`, `sae_ladder.sh`, `sae_evals.sh` | `sae` | `tests/test_pareto.py` |
| 23 | `refit.py` | `scripts/eval/refit.py` | Eval | `headcoh.py`, `sae_evals.sh` | `sae`, `autointerp`, `pareto` | `tests/test_refit.py` |
| 24 | `sae_evals.sh` | `scripts/eval/sae_evals.sh` | Eval | None | `sae.py`, `pareto.py`, `headstruct.py`, `splitting.py`, `textfid.py`, `langprobe.py`, `refit.py` | None |
| 25 | `splitting.py` | `scripts/eval/splitting.py` | Eval | `langprobe.py`, `sae_evals.sh` | `sae`, `autointerp`, `pareto` | `tests/test_splitting.py` |
| 26 | `steerfid.py` | `scripts/eval/steerfid.py` | Eval | None | `sae`, `pareto`, `textfid` | `tests/test_steerfid.py` |
| 27 | `textfid.py` | `scripts/eval/textfid.py` | Eval | `steerfid.py`, `sae_evals.sh` | `sae`, `pareto` | `tests/test_textfid.py` |

---

## 5. Transition Strategies & Non-Breaking Migration Plan

To move these 27 files without breaking test execution, shell orchestration, or sibling imports, we evaluate four technical transition options and formulate a phased, risk-free implementation plan.

### 5.1 Technical Strategy Comparison

| Strategy | Mechanism | Pros | Cons | Recommendation |
|---|---|---|---|---|
| **Strategy 1: PYTHONPATH Extension (`conftest.py` / `pyproject.toml`)** | Append `scripts/eval/` and `scripts/sae/` to `sys.path` in `tests/conftest.py` or configure `pythonpath = ["scripts/eval", "scripts/sae"]` in `pyproject.toml`. | Zero test file edits required; 100% of existing `import sae` and `from autointerp import ...` statements resolve unchanged. | Flattens module namespaces; conceals true file origins; does not solve shell script invocations. | **Adopt for Phase 1.** Provides an immediate safety net for tests. |
| **Strategy 2: Root Forwarding Stubs (Re-Export Shims)** | Retain thin shim files at the root (e.g. `sae.py` containing `from scripts.sae.sae import *`). | 100% backward-compatible with external scripts, existing systemd user services, and legacy shell commands. | Keeps files in the root directory temporarily until consumers are migrated. | **Adopt for Phase 1–2.** Ensures zero external breakage. |
| **Strategy 3: Shell Script & Subprocess Path Updates** | Update script invocation paths in shell scripts and Python runners (e.g. `uv run python scripts/sae/sae.py`). | Fixes root path assumptions; ensures execution consistency across directories. | Requires updating 4 shell scripts and 4 runner scripts. | **Adopt for Phase 2.** Essential for pipeline functionality. |
| **Strategy 4: Core Library Elevation (Long-Term Ideal)** | Extract pure mathematical/algorithmic logic from scripts into `ontologize.analysis` / `ontologize.eval` and `ontologize.sae`. Leave only CLI wrappers in `scripts/`. | Eliminates inter-script couplings completely; adheres to standard Python packaging best practices; tests only test library code. | Higher refactoring effort; modifies existing test imports to point to `ontologize.*`. | **Adopt for Phase 4 (Future Work).** |

---

### 5.2 Recommended Phased Migration Plan

We recommend a **4-Phase Migration** that delivers complete reorganization without breaking any existing workflows at any intermediate step:

#### Phase 1: Physical Relocation with Backward-Compatibility Shims (Immediate Safe Move)
1. **Directory Creation:** Create `scripts/`, `scripts/data/`, `scripts/training/`, `scripts/interactive/`, `scripts/interactive/runners/`, `scripts/sae/`, and `scripts/eval/`. Add empty `__init__.py` files to each directory.
2. **Move Files:** Relocate all 27 scripts according to the Move Map (Section 4).
3. **Configure Test Search Path:** In `tests/conftest.py`, extend `sys.path` to include the script domain directories:
   ```python
   root = Path(__file__).parents[1]
   sys.path.insert(0, str(root))
   sys.path.insert(0, str(root / "scripts" / "sae"))
   sys.path.insert(0, str(root / "scripts" / "eval"))
   ```
4. **Deploy Compatibility Shims at Root:** For the 11 scripts imported by tests and sibling scripts, place thin forwarding stubs at the repository root:
   ```python
   # Example: root-level sae.py shim
   import sys
   from pathlib import Path
   _root = Path(__file__).resolve().parent
   if str(_root) not in sys.path:
       sys.path.insert(0, str(_root))
   from scripts.sae.sae import *  # noqa: F401, F403
   ```
   *Verification:* Run `uv run pytest`. All 25 test suites pass with zero regressions.

#### Phase 2: Updating Shell Scripts & Subprocess Runners
1. **Shell Script Directory Pinning & Paths:**
   - In `scripts/eval/autointerp_run.sh`:
     - Update repo root resolution: `REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$REPO_ROOT"`.
     - Update invocations: `uv run python scripts/eval/autointerp.py harvest ...`, `describe ...`, `score ...`.
   - In `scripts/sae/sae_ladder.sh`:
     - Update repo root resolution: `REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$REPO_ROOT"`.
     - Update invocations: `uv run python scripts/sae/sae.py ...`, `uv run python scripts/eval/pareto.py`, `uv run python scripts/eval/headstruct.py`.
   - In `scripts/eval/sae_evals.sh`:
     - Update repo root resolution: `REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" && cd "$REPO_ROOT"`.
     - Update invocations to target `scripts/sae/sae.py`, `scripts/eval/pareto.py`, `scripts/eval/splitting.py`, etc.
   - In `scripts/training/sonar.sh`:
     - Add directory pinning: `cd "$(cd "$(dirname "$0")/../.." && pwd)"`.
     - Update command: `uv run python scripts/training/sonar.py`.
2. **Subprocess Test Drivers:**
   - Update `run_chat.py`, `run_chat2.py`, `run_decode.py`, and `run_decode_tags.py` to resolve target scripts relative to `Path(__file__).resolve().parents[1] / "decode.py"` (or via absolute path from repo root).

#### Phase 3: Sibling Import Normalization & Shim Deprecation
1. **Modernize Inter-Script Imports:** Update all sibling imports in `scripts/` to use standard package-level imports:
   - In `scripts/eval/refit.py`: Change `import sae` to `from scripts.sae import sae`.
   - In `scripts/eval/steerfid.py`: Change `from textfid import chrf` to `from scripts.eval.textfid import chrf`.
   - In `scripts/training/sonar_hc.py`: Change `import sonar` to `from scripts.training import sonar`.
2. **Update Test Imports:** Update the 11 test files to import directly from `scripts.*`:
   - `from scripts.sae import sae`
   - `from scripts.eval.autointerp import ...`
   - `from scripts.eval.textfid import chrf`
3. **Deprecate Root Shims:** Once all tests, shell scripts, and systemd service units point to `scripts/`, remove the temporary forwarding stubs from the repository root.

#### Phase 4: Library Elevation & Boilerplate Consolidation (Future Polish)
1. **Consolidate Orbax Loading Boilerplate:** Extract the hexa-duplicate model restoration logic into `ontologize.training.ontostate.load_checkpoint_model(ckpt_path, step=None)`.
2. **Elevate Pure Math Functions:** Move reusable metrics (`chrf`, `greedy_groups`, `sv_energy`, `refit_recon`) into an `ontologize.eval` or `ontologize.analysis` package namespace.

---

## 6. Catalog of Observed Bugs, Dead Code, and Oddities

During the exhaustive survey of the 27 scripts and existing package files, several bugs, anti-patterns, and architectural anomalies were identified. In accordance with the non-destructive mandate, **none of these were altered**. They are cataloged below for remediation during future refactoring passes:

### 6.1 Fatal Bugs & Broken Imports
1. **Broken Import in `mnist.py` (Line 7):**
   ```python
   from ontologize.training.data import ImageLoader
   ```
   *Anomaly:* No module named `ontologize.training.data` exists. The `ImageLoader` class is located in `ontologize.data.loaders`. Executing `python mnist.py` immediately crashes with `ModuleNotFoundError: No module named 'ontologize.training.data'`.
2. **Truncated Constructor Call in `mnist.py` (Lines 44–45):**
   `TrainingEnv` instantiation passes arguments that do not match the current signature in `ontologize.training.config.TrainingEnv`. `mnist.py` is an unmaintained experimental stub.

### 6.2 Architectural Anti-Patterns & Fragile Couplings
3. **Module-Level Monkeypatching in `sonar_hc.py` (Lines 33–42):**
   ```python
   import sonar
   sonar.s_hcossim = 1e-5
   sonar.s_Hm = 0.0
   sonar.out = sonar.path / "out/sonar/multilingual/resid_nc_hc"
   print(f"resid_nc_hc: s_hcossim={sonar.s_hcossim} s_Hm={sonar.s_Hm} -> {sonar.out}", flush=True)
   sonar.main()
   ```
   *Anomaly:* Rather than passing CLI flags or using an inheritance pattern, `sonar_hc.py` mutates global variables in the imported `sonar` module before calling `sonar.main()`. Any refactoring of `sonar.py` that encapsulates state inside a class or function will silently break `sonar_hc.py`.
4. **Divergent CuDNN Initialization Mechanisms:**
   - `decode.py` (lines 6–15) and `decode_tags.py` (lines 6–15) perform **process replacement** via `os.execv`:
     ```python
     if "CUDNN_INJECTED" not in os.environ:
         # mutate LD_LIBRARY_PATH
         os.environ["CUDNN_INJECTED"] = "1"
         os.execv(sys.executable, [sys.executable] + sys.argv)
     ```
     This restarts the Python process with a new PID to force glibc to reload CuDNN libraries.
   - In contrast, `encode_corpus.py` (lines 23–32), `encode_discord.py` (lines 34–41), and `sonar.py` (lines 19–32) load `libcudnn.so` using `ctypes.CDLL(str(cudnn_path), mode=os.RTLD_GLOBAL)`.
   *Anomaly:* Two conflicting methods solve the exact same dynamic library linking issue. The `os.execv` approach breaks subprocess drivers that expect a persistent PID.
5. **Hexa-Duplicate Orbax Model Loading Logic:**
   Six distinct evaluation and inference scripts (`classify_acts.py`, `pareto.py`, `splitting.py`, `steerfid.py`, `refit.py`, `autointerp.py`) contain duplicate implementations of checkpoint restoration and dictionary unwrapping (`while 'params' in params: params = params['params']`).
6. **Repeated Model Constant Strings:**
   The HuggingFace identifier `ENCODER_ID = "cointegrated/SONAR_200_text_encoder"` is redundantly defined across `autointerp.py`, `encode_corpus.py`, `encode_discord.py`, `steerfid.py`, and `textfid.py`.

### 6.3 Shell Script Inconsistencies
7. **Shell Invocation Inconsistency in `sonar.sh`:**
   Unlike the other 3 shell scripts, `sonar.sh` lacks `cd "$(dirname "$0")"` and invokes `uv run sonar.py` instead of `uv run python sonar.py`. It fails if triggered from outside the repository root.

### 6.4 Dead Modules in Core Package
8. **Four Dead Modules in `ontologize/layers/`:**
   As specified in project requirements, four modules in `ontologize/layers/` contain broken imports or references to obsolete package layouts (`ontologize.jax.layers.*`):
   - `ontologize/layers/dictblock_enhanced.py`
   - `ontologize/layers/dict_interpreter.py`
   - `ontologize/layers/integration_clean.py`
   - `ontologize/layers/vae_integration.py`
   None of these are imported by any active script, test, or package module.
9. **Outdated Comment in `pyproject.toml` (Lines 32–33):**
   ```toml
   # only the curated suite -- root-level test_*.py are ad hoc debug scripts,
   # not tests, and several execute GPU/checkpoint code at import time
   testpaths = ["tests"]
   ```
   *Anomaly:* No `test_*.py` files exist in the repository root. The comment references a historical repository state.

---

## 7. Verification Procedures & Checklist

When authorization is granted to execute the reorganization, the following verification commands must be executed to ensure zero regressions:

```bash
# 1. Verify all 27 scripts are present in their target directories:
ls -1 scripts/data/*.py | wc -l          # Expected: 3
ls -1 scripts/training/*.py scripts/training/*.sh | wc -l # Expected: 4
ls -1 scripts/interactive/*.py scripts/interactive/runners/*.py | wc -l # Expected: 6
ls -1 scripts/sae/*.py scripts/sae/*.sh | wc -l # Expected: 2
ls -1 scripts/eval/*.py scripts/eval/*.sh | wc -l # Expected: 12
# Total: 27

# 2. Verify all relocated Python scripts compile cleanly without syntax errors:
python3 -m py_compile scripts/**/*.py

# 3. Verify that the complete Pytest suite passes:
uv run pytest

# 4. Verify AST invariance of core library files:
git diff ontologize/
```

---
*End of Reorganization Proposal.*
