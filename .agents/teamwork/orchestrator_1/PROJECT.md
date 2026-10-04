# Project: Ontologize Documentation-and-Organization Pass

## Architecture
The `ontologize` repository contains:
1. `ontologize/`: Core Python library for dictionary learning and interpretability on top of JAX/Flax and PyTorch (SONAR/M2M100).
   - `layers/`: Neural network layers (`DictBlock`, `DictEnc`, `Linear`, `NLinear`, `Bilinear`, `Sparse`).
   - `training/`: Training environment (`Hyperparams`, `Metadata`, `TrainingEnv`), state management (`OntoState`), and serialization.
   - `data/`: Grain dataset pipelines, multi-lingual dataset loading, and pretrained SONAR text encoders.
   - `fns/`: Activations, classification utilities, and loss functions (MSE, ghost grads, entropy, batch cossim).
   - `inference/`: Causal steering and model inference (`Steerable`).
   - `visualize/`: Training metric plotting utilities.
   - Root modules: `ontologizer.py` (main multi-layer model), `chat.py` (CLI interactive environment), `main.py` (entry stub).
2. Root scripts: 27 scripts (23 `.py`, 4 `.sh`) spanning data prep, training, decoding, SAE baseline and evaluation suite, and scratch wrappers.
3. `tests/`: 25 test files executed via pytest under CPU mode.
4. `docs/`: Documentation target directory for `docs/SCRIPTS.md` and `docs/REORG_PROPOSAL.md`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | M1 Layers Docstrings | Add/improve docstrings for `dictblock.py`, `dictenc.py`, `linear.py`, `nlinear.py`, `sparse.py`, `__init__.py`. Document tensor shapes (e.g. `(h, k, d)`). | M1 | Survey |
| 2 | M1 Dead Modules Skip | Strictly skip dead modules: `dictblock_enhanced.py`, `dict_interpreter.py`, `integration_clean.py`, `vae_integration.py`. | M1 | Survey |
| 3 | M2 Training Docstrings | Add/improve docstrings for `config.py`, `ontostate.py`, `serialize.py`. | M2 | Survey |
| 4 | M2 Data Docstrings | Add/improve docstrings for `__init__.py`, `langs.py`, `loaders.py`, `multilingual.py`, `pretrained.py`. | M2 | Survey |
| 5 | M3 Fns Docstrings | Add/improve docstrings for `classify.py`, `keys.py`, `loss.py`. | M3 | Survey |
| 6 | M3 Inference/Vis Docstrings | Add/improve docstrings for `steerable.py`, `visualize/loss.py`. | M3 | Survey |
| 7 | M3 Package Root Docstrings | Add/improve docstrings for `chat.py`, `ontologizer.py`, `main.py`. | M3 | Survey |
| 8 | M4 Root Script Census | Exhaustive catalog of all 27 root scripts (23 .py, 4 .sh) across 5 categories in `docs/SCRIPTS.md`. | M4 | Survey |
| 9 | M4 Inputs/Outputs & Tests | Specify summary, inputs, outputs, and covering `tests/` file for every script in `docs/SCRIPTS.md`. | M4 | Survey |
| 10 | M5 Coupling Analysis | Analyze `tests/` direct imports, shell script calls, inter-script imports in `docs/REORG_PROPOSAL.md`. | M5 | Survey |
| 11 | M5 Reorg Target & Move Map | Formulate 5-domain target layout under `scripts/`, move map, and transition strategies in `docs/REORG_PROPOSAL.md`. | M5 | Survey |
| 12 | M6 Invariance Verification | AST syntactic invariance check (AST nodes identical before vs after docstrings). | M6 | Survey |
| 13 | M6 Git Diff Scope Verification | Verify only allowed `ontologize/` files modified and new files in `docs/`. | M6 | Survey |
| 14 | M6 py_compile Verification | Run `python3 -m py_compile` across all modified `ontologize/*.py` files. | M6 | Survey |
| 15 | M6 Forensic Integrity Audit | Independent integrity verification by Forensic Auditor. | M6 | Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1_Docstrings_Layers | Docstrings in `ontologize/layers/` (`dictblock.py`, `dictenc.py`, `linear.py`, `nlinear.py`, `sparse.py`, `__init__.py`). Skip 4 dead modules. | none | DONE |
| 2 | M2_Docstrings_Training_Data | Docstrings in `ontologize/training/` (`config.py`, `ontostate.py`, `serialize.py`) and `ontologize/data/` (`langs.py`, `loaders.py`, `multilingual.py`, `pretrained.py`, `__init__.py`). | none | DONE |
| 3 | M3_Docstrings_Fns_Root | Docstrings in `ontologize/fns/` (`classify.py`, `keys.py`, `loss.py`), `ontologize/inference/steerable.py`, `ontologize/visualize/loss.py`, `chat.py`, `ontologizer.py`, `main.py`. | none | DONE |
| 4 | M4_Script_Index | Create `docs/SCRIPTS.md` indexing all 27 root scripts with categories, inputs, outputs, and tests/ coverage. | none | DONE |
| 5 | M5_Reorg_Proposal | Create `docs/REORG_PROPOSAL.md` with coupling analysis, target layout, move map, and non-breaking migration plans. | none | DONE |
| 6 | M6_Verification_Audit | AST invariance check, `py_compile`, git status & diff check, pytest run, and Forensic Integrity Audit. | M1, M2, M3, M4, M5 | DONE |

Gate Result: **PASS** (All 6 Milestones Completed & Independently Verified).

## Code Layout & Write Boundaries
- Worker 1 (M1):
  - EXCLUSIVE WRITE: `ontologize/layers/dictblock.py`, `ontologize/layers/dictenc.py`, `ontologize/layers/linear.py`, `ontologize/layers/nlinear.py`, `ontologize/layers/sparse.py`, `ontologize/layers/__init__.py`.
  - FORBIDDEN: Any other file.
- Worker 2 (M2):
  - EXCLUSIVE WRITE: `ontologize/training/config.py`, `ontologize/training/ontostate.py`, `ontologize/training/serialize.py`, `ontologize/data/langs.py`, `ontologize/data/loaders.py`, `ontologize/data/multilingual.py`, `ontologize/data/pretrained.py`, `ontologize/data/__init__.py`.
  - FORBIDDEN: Any other file.
- Worker 3 (M3):
  - EXCLUSIVE WRITE: `ontologize/fns/classify.py`, `ontologize/fns/keys.py`, `ontologize/fns/loss.py`, `ontologize/inference/steerable.py`, `ontologize/visualize/loss.py`, `chat.py`, `ontologizer.py`, `main.py`.
  - FORBIDDEN: Any other file.
- Worker 4 (M4):
  - EXCLUSIVE WRITE: `docs/SCRIPTS.md`.
  - FORBIDDEN: Any other file.
- Worker 5 (M5):
  - EXCLUSIVE WRITE: `docs/REORG_PROPOSAL.md`.
  - FORBIDDEN: Any other file.

## Interface Contracts & Hard Constraints
- **CRITICAL HARD RULE:** Change ONLY docstrings. No code edits, no renames, no reformatting, no type-hint changes, and no comment removal.
- **Dead modules to skip entirely:**
  - `ontologize/layers/dictblock_enhanced.py`
  - `ontologize/layers/dict_interpreter.py`
  - `ontologize/layers/integration_clean.py`
  - `ontologize/layers/vae_integration.py`
- Do NOT modify `GEMINI.md`, `README.md`, `CLAUDE.md`, `pyproject.toml`, `tests/`, or any root-level script.
- Do NOT move or rename any existing files.
- Do NOT install any packages, run model training, or download any datasets or checkpoints.
- Verify AST invariance: ASTs stripped of docstrings must match original ASTs exactly.
