# Original User Request

## 2026-09-29T22:16:57Z

Documentation-and-organization pass on the Python research repo at `/home/jade/disk2/ontologizercleanup/ontologize`. Split the work across the team into three distinct workstreams: Docstrings, Script Index, and Reorganization Proposal.

Working directory: `/home/jade/disk2/ontologizercleanup/ontologize`
Integrity mode: development

## Requirements

### R1. Ontologize Package Docstrings
Add or improve docstrings throughout the `ontologize/` package: module-level docstrings, class docstrings, and docstrings for public functions and methods. State purpose, arguments, return values, and tensor/array shapes wherever knowable from the code (e.g. `DictBlock` weight is `(h, k, d)`).
Divide the work by subpackage:
- `ontologize/layers/` (`dictblock.py`, `dictenc.py`, `linear.py`, `nlinear.py`, `sparse.py`, `__init__.py`)
- `ontologize/training/` (`config.py`, `ontostate.py`, `serialize.py`, `__init__.py`)
- `ontologize/data/` (`__init__.py`, `langs.py`, `loaders.py`, `multilingual.py`, `pretrained.py`)
- `ontologize/fns/` (`classify.py`, `keys.py`, `loss.py`) + `ontologize/inference/` (`steerable.py`) + `ontologize/visualize/` (`loss.py`) + `ontologize/chat.py` + `ontologize/ontologizer.py` + `ontologize/main.py`

**HARD RULE:** Change ONLY docstrings. No code edits, no renames, no reformatting, no type-hint changes, and no comment removal.
**Dead modules to skip entirely:**
- `ontologize/layers/dictblock_enhanced.py`
- `ontologize/layers/dict_interpreter.py`
- `ontologize/layers/integration_clean.py`
- `ontologize/layers/vae_integration.py`

### R2. Root Script Index (`docs/SCRIPTS.md`)
Write `docs/SCRIPTS.md`: an exhaustive table indexing every root-level `*.py` and `*.sh` script (27 in total: 23 `.py` files and 4 `.sh` files). Group scripts by purpose:
- Data preparation / harvesting
- Training / fine-tuning
- Interactive / decoding / chat
- SAE baseline + evaluation suite
- Scratch / stale / experimental

For each script, provide:
- One-line summary of what it does
- Main inputs and outputs
- Covering test file in `tests/` (if any, determined by inspecting `tests/` imports and script logic)

### R3. Reorganization Proposal (`docs/REORG_PROPOSAL.md`)
Write `docs/REORG_PROPOSAL.md` proposing (NOT executing) a cleaner layout for the root-level scripts.
Analyze and detail existing coupling constraints:
- `tests/` import several root scripts directly as top-level modules (`pareto`, `sae`, `refit`, `autointerp`, `splitting`, `compose`, `steerfid`, `headstruct`, `headcoh`, `langprobe`, `textfid`).
- Shell scripts call root scripts by relative path (e.g., `sonar.sh`, `sae_evals.sh`, `sae_ladder.sh`, `autointerp_run.sh`).
Include:
- Proposed target directory layout
- A concrete move map for each file
- The exact changes required to keep tests passing and shell scripts working (e.g., test runner pythonpath, entry-point scripts, or module import paths)

### R4. Worker Summaries & Code Oddity Logging
Each worker must write a summary of changes made and catalog any bugs, dead code, or oddities noticed in the codebase (do NOT fix them).

### Constraints & Guardrails
- Do NOT modify `GEMINI.md`, `README.md`, `CLAUDE.md`, `pyproject.toml`, `tests/`, or any root-level script.
- Do NOT move or rename any existing files.
- Do NOT install any packages, run model training, or download any datasets or checkpoints.

## Verification

### Verification Mechanisms
- **AST / Syntactic Invariance Check:** Verify that comparing ASTs (or code stripped of docstrings) before and after changes yields identical AST nodes for all modified files in `ontologize/`.
- **Git Status & Diff Scope:** `git status -s` must show edits strictly inside allowed `ontologize/` files and untracked files only in `docs/`. `git diff --stat` must show 0 modifications to `tests/`, root scripts, `pyproject.toml`, or markdown root files.
- **Python Syntax Compilation:** Run `python3 -m py_compile` across all modified `ontologize/*.py` files to ensure zero syntax errors.
- **Script Table Completeness:** Validate that all 27 root `*.py` and `*.sh` files are accounted for in `docs/SCRIPTS.md`.

## Acceptance Criteria

### Workstream 1: Docstrings
- [ ] Module, class, and public function/method docstrings added or updated across active files in `ontologize/layers/`, `ontologize/training/`, `ontologize/data/`, `ontologize/fns/`, `ontologize/inference/`, `ontologize/visualize/`, `chat.py`, `ontologizer.py`, and `main.py`.
- [ ] Purpose, arguments, return values, and tensor/array dimensions are documented where inferable.
- [ ] Strictly zero modifications to executable code, type annotations, variable names, or existing comments.
- [ ] The four dead modules (`dictblock_enhanced.py`, `dict_interpreter.py`, `integration_clean.py`, `vae_integration.py`) remain completely untouched.
- [ ] All modified Python files compile cleanly with `py_compile`.

### Workstream 2: Script Index
- [ ] `docs/SCRIPTS.md` is created with a structured Markdown table or grouped sections.
- [ ] All 27 root scripts (23 `.py` and 4 `.sh`) are listed with purpose, inputs/outputs, and corresponding `tests/` coverage identified.

### Workstream 3: Reorganization Proposal
- [ ] `docs/REORG_PROPOSAL.md` is created without any root files being physically moved.
- [ ] Concrete move mapping is provided for all root scripts.
- [ ] Explains how to handle `tests/` direct module imports and shell script invocations.

### Guardrails & Summaries
- [ ] `GEMINI.md`, `README.md`, `CLAUDE.md`, `pyproject.toml`, `tests/`, and root scripts are untouched.
- [ ] Zero package installations, training runs, or network downloads occurred.
- [ ] Summary of changes and catalog of observed bugs/oddities recorded.
