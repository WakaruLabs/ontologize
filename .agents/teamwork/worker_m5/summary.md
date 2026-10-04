# Milestone 5 Summary: Root Script Reorganization Proposal

**Worker:** worker_m5  
**Milestone:** Milestone 5 — Reorganization Proposal (`docs/REORG_PROPOSAL.md`)  
**Date:** 2026-09-29  
**Status:** Completed  

---

## 1. Summary of Changes

In accordance with the task instructions and strict non-destructive constraints, **zero files were renamed, moved, or deleted**, and no executable code was modified. 

The primary deliverable created is:
- `docs/REORG_PROPOSAL.md`: A comprehensive, production-ready architectural specification proposing a clean, modular reorganization of all 27 root-level scripts into a structured `scripts/` hierarchy.

### Key Contents of `docs/REORG_PROPOSAL.md`:
1. **Coupling Constraints Analysis:**
   - **`tests/` direct module imports:** Detailed analysis of all 11 test files that import root scripts (`autointerp`, `compose`, `headcoh`, `headstruct`, `langprobe`, `pareto`, `refit`, `sae`, `splitting`, `steerfid`, `textfid`) enabled by `tests/conftest.py`. Noted that all 11 tests exclusively target pure mathematical/statistical helper functions rather than CLI execution loops.
   - **Shell scripts invocations:** Analysis of the 4 shell scripts (`autointerp_run.sh`, `sae_ladder.sh`, `sae_evals.sh`, `sonar.sh`), detailing working directory pinning, hardware/GPU polling loops, environment variable dependencies (`ANTHROPIC_API_KEY`, `AI_RATE`), idempotency guards, and script invocation syntax.
   - **Sibling script dependencies:** Documented the full directed graph of 20 dependency edges across 11 root scripts, identifying `sae.py` (7 inbound dependents) and `pareto.py` (6 inbound dependents) as the two primary structural hubs.
   - **Subprocess invocation wrappers:** Analyzed `run_chat.py`, `run_chat2.py`, `run_decode.py`, and `run_decode_tags.py` which invoke `decode.py` and `decode_tags.py` via hardcoded subprocess calls.
   - **Boilerplate & Architectural Redundancies:** Identified repeated Orbax checkpoint loading routines across 6 scripts, shared SAE directory regex parsing, and duplicate model constants.

2. **Proposed Target Layout:**
   - Designed a 5-domain modular structure under `scripts/`:
     - `scripts/data/` (3 scripts: `encode_corpus.py`, `encode_discord.py`, `classify_acts.py`)
     - `scripts/training/` (4 scripts: `sonar.py`, `sonar_hc.py`, `mnist.py`, `sonar.sh`)
     - `scripts/interactive/` (6 scripts: `decode.py`, `decode_tags.py`, and `runners/` subpackage for `run_chat.py`, `run_chat2.py`, `run_decode.py`, `run_decode_tags.py`)
     - `scripts/sae/` (2 scripts: `sae.py`, `sae_ladder.sh`)
     - `scripts/eval/` (12 scripts: `autointerp.py`, `autointerp_run.sh`, `compose.py`, `headcoh.py`, `headstruct.py`, `langprobe.py`, `pareto.py`, `refit.py`, `sae_evals.sh`, `splitting.py`, `steerfid.py`, `textfid.py`)

3. **Concrete Move Map:**
   - 100% complete mapping table covering every one of the 27 root scripts (23 `.py`, 4 `.sh`) with current path, target path, domain category, inbound dependencies, outbound dependencies, and covering test files.

4. **Exact Transition Strategies & Phased Migration Plan:**
   - Evaluated 4 strategies: PYTHONPATH extension, root compatibility forwarding stubs, shell script invocation updates, and core library elevation.
   - Formulated a recommended 4-phase zero-downtime migration plan:
     - *Phase 1:* Physical move + root compatibility forwarding stubs + `tests/conftest.py` path additions (guarantees 100% test pass rate with zero test code edits).
     - *Phase 2:* Update shell script path resolutions and subprocess test drivers.
     - *Phase 3:* Modernize inter-script and test imports to standard package imports (`from scripts.*`) and deprecate root stubs.
     - *Phase 4:* Consolidate repeated checkpoint restoration logic into `ontologize.training.ontostate` and elevate pure mathematical helper functions into library modules.

---

## 2. Catalog of Observed Bugs, Dead Code, and Code Oddities

The following anomalies, bugs, and anti-patterns were observed across the codebase during this pass. Per instructions, **none were fixed**:

| # | File & Location | Category | Description & Impact |
|---|---|---|---|
| 1 | `mnist.py:7` | Fatal Import Bug | `from ontologize.training.data import ImageLoader` causes immediate `ModuleNotFoundError: No module named 'ontologize.training.data'`. `ImageLoader` actually resides in `ontologize.data.loaders`. |
| 2 | `mnist.py:44-45` | API Mismatch | `TrainingEnv` constructor arguments are out of sync with `ontologize.training.config.TrainingEnv`. The script appears to be an abandoned experimental prototype. |
| 3 | `sonar_hc.py:33-42` | Anti-Pattern | Mutates module-level globals in `sonar.py` (`sonar.s_hcossim = 1e-5`, `sonar.s_Hm = 0.0`, `sonar.out = ...`) prior to calling `sonar.main()`. Any refactoring of `sonar.py` that encapsulates state inside a class or function will break `sonar_hc.py`. |
| 4 | `decode.py:6-15`<br>`decode_tags.py:6-15` vs `sonar.py:19-32` | Divergent Runtime Injections | `decode.py` and `decode_tags.py` use `os.execv` to re-execute Python with modified `LD_LIBRARY_PATH` to resolve CuDNN, changing the process PID. In contrast, `sonar.py`, `encode_corpus.py`, and `encode_discord.py` load `libcudnn.so` using `ctypes.CDLL(..., mode=os.RTLD_GLOBAL)`. |
| 5 | `sonar.sh:3-4` | Shell Inconsistency | Lacks `cd "$(dirname "$0")"` and invokes `uv run sonar.py` rather than `uv run python sonar.py`, making it fragile when executed outside repo root. |
| 6 | Hexa-Duplicate Checkpoint Loading | Code Duplication | `classify_acts.py`, `pareto.py`, `splitting.py`, `steerfid.py`, `refit.py`, and `autointerp.py` each duplicate identical Orbax restoration loops and nested dictionary parameter unpacking (`while 'params' in params: params = params['params']`). |
| 7 | Duplicate Constants | Code Duplication | `ENCODER_ID = "cointegrated/SONAR_200_text_encoder"` is independently hardcoded across `autointerp.py`, `encode_corpus.py`, `encode_discord.py`, `steerfid.py`, and `textfid.py`. |
| 8 | `ontologize/layers/` | Dead Code | Four modules in `ontologize/layers/` (`dictblock_enhanced.py`, `dict_interpreter.py`, `integration_clean.py`, `vae_integration.py`) are completely unreferenced, contain broken legacy imports (`ontologize.jax.layers.*`), or are non-functional stubs. |
| 9 | `pyproject.toml:32-33` | Stale Comments | Comments state `# only the curated suite -- root-level test_*.py are ad hoc debug scripts, not tests...`, but no `test_*.py` files exist in the repository root. |
