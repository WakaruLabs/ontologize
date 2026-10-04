# Handoff Report — Reorganization Proposal & Coupling Analysis

**Agent:** explorer_survey_3 (Workstream 3)  
**Date:** 2026-09-29  
**Type:** Hard Handoff (Task Complete)

---

## 1. Observation

1. **Root Script Inventory:** Exactly 27 scripts reside in the repository root:
   - 23 Python files: `autointerp.py`, `classify_acts.py`, `compose.py`, `decode.py`, `decode_tags.py`, `encode_corpus.py`, `encode_discord.py`, `headcoh.py`, `headstruct.py`, `langprobe.py`, `mnist.py`, `pareto.py`, `refit.py`, `run_chat.py`, `run_chat2.py`, `run_decode.py`, `run_decode_tags.py`, `sae.py`, `sonar.py`, `sonar_hc.py`, `splitting.py`, `steerfid.py`, `textfid.py`.
   - 4 Bash files: `autointerp_run.sh`, `sae_evals.sh`, `sae_ladder.sh`, `sonar.sh`.
2. **Test Suite Coupling (`tests/`):**
   - In `tests/conftest.py` line 11: `sys.path.insert(0, str(Path(__file__).parents[1]))` prepends the root directory to `sys.path`.
   - 11 of 25 test files directly import root scripts as top-level modules:
     - `tests/test_autointerp.py:6`: `from autointerp import (CAL_PROMPT_CHARS, CONTRAST_PROMPT, SUMMARIZE_PROMPT, call_weight, contrast_rows, detection_items, cap_pairing, next_slot, null_pairing, read_scores, snippet, stratified_features)`
     - `tests/test_compose.py:6`: `from compose import additivity_stats, sample_singles`
     - `tests/test_headcoh.py:5`: `from headcoh import (group_zscores, mean_pairwise_cos, perm_z, pooled_within_cos, sv_energy, unit)`
     - `tests/test_headstruct.py:8`: `import headstruct`
     - `tests/test_langprobe.py:7`: `import langprobe`
     - `tests/test_pareto.py:8`: `import pareto`
     - `tests/test_refit.py:8, 50`: `import refit`, `import sae`
     - `tests/test_sae.py:10`: `import sae`
     - `tests/test_splitting.py:8`: `import splitting`
     - `tests/test_steerfid.py:7`: `import steerfid`
     - `tests/test_textfid.py:6`: `from textfid import chrf`
   - Every one of these test files imports pure algorithmic, mathematical, or statistical helper functions (e.g. `chrf`, `additivity_stats`, `affinity_edges`, `t_stats`, `refit_recon`, `split_children`), never CLI handlers or `main()`.
3. **Sibling Script Dependencies:**
   - 20 directed sibling imports connect 11 root scripts.
   - `sae` is imported by: `autointerp.py:452,550`, `headstruct.py:55`, `pareto.py:62`, `refit.py:50`, `splitting.py:54`, `steerfid.py:49`, `textfid.py:44`.
   - `pareto` is imported by: `compose.py:89`, `headstruct.py:184`, `refit.py:170`, `splitting.py:91`, `steerfid.py:139`, `textfid.py:105,115`.
   - `autointerp` is imported by: `headcoh.py:186`, `refit.py:125`, `splitting.py:97`.
   - `refit` is imported by: `headcoh.py:145`.
   - `splitting` is imported by: `langprobe.py:39`.
   - `textfid` is imported by: `steerfid.py:183`.
   - `sonar` is imported and monkeypatched by `sonar_hc.py:33-42`.
4. **Shell Script Invocations:**
   - `autointerp_run.sh`: pins `cd "$(dirname "$0")"`, calls `uv run python autointerp.py <subcommand>`. Requires `ANTHROPIC_API_KEY`.
   - `sae_ladder.sh`: pins `cd "$(dirname "$0")"`, calls `uv run python sae.py ...`, `uv run python pareto.py`, `uv run python headstruct.py ...`.
   - `sae_evals.sh`: pins `cd "$(dirname "$0")"`, calls `uv run python sae.py ...`, `pareto.py`, `headstruct.py`, `splitting.py`, `textfid.py`, `langprobe.py`, `refit.py`.
   - `sonar.sh`: calls `uv run sonar.py` without `cd "$(dirname "$0")"` or `python` keyword.
5. **Subprocess Wrappers:**
   - `run_chat.py:12`, `run_chat2.py:11`, `run_decode.py:15` invoke `["uv", "run", "python", "decode.py", "data/out/sonar/linear/multilingual"]`.
   - `run_decode_tags.py:9` invokes `["uv", "run", "python", "decode_tags.py", "data/out/sonar/linear/multilingual"]`.
6. **Code Oddities / Bugs:**
   - `mnist.py:7`: `from ontologize.training.data import ImageLoader` fails with `ModuleNotFoundError` (`ImageLoader` is in `ontologize.data.loaders`).
   - `sonar_hc.py:35-37`: Mutates `sonar.s_hcossim`, `sonar.s_Hm`, `sonar.out` at module level before calling `sonar.main()`.
   - `decode.py` and `decode_tags.py` use `os.execv` to re-execute Python with mutated `LD_LIBRARY_PATH`, whereas `encode_corpus.py`, `encode_discord.py`, and `sonar.py` use `ctypes.CDLL`.
   - Hexa-duplicate Orbax checkpoint loading routines in `classify_acts.py`, `pareto.py`, `splitting.py`, `steerfid.py`, `refit.py`, `autointerp.py`.

---

## 2. Logic Chain

1. **From Observation 1 & 2:** Root scripts cannot be simply relocated into subdirectories without breaking 11 test files, because `tests/conftest.py` adds only repository root to `sys.path`, and test files rely on top-level module names (`import sae`, `from autointerp import ...`).
2. **From Observation 2:** Because tests exclusively target pure mathematical/statistical helper functions, the true design problem is the co-location of reusable library logic and CLI entry points in the same root scripts.
3. **From Observation 3:** The inter-script import graph has two primary hubs (`sae.py` and `pareto.py`). Any move must preserve cross-module access between `scripts/eval/` and `scripts/sae/` (or package-level imports `from scripts.sae import sae`).
4. **From Observation 4 & 5:** Moving `decode.py`, `decode_tags.py`, `autointerp.py`, `sae.py`, `pareto.py`, etc., without updating the 4 shell scripts and 4 subprocess wrappers will result in runtime `FileNotFoundError` or script crashes.
5. **From Observations 1–6 to Reorganization Plan:**
   - A subpackage structure under `scripts/` (`scripts/data/`, `scripts/training/`, `scripts/interactive/`, `scripts/sae/`, `scripts/eval/`) cleanly partitions the 27 scripts by function while grouping interdependent evaluation scripts together.
   - A phased migration using root forwarding stubs and `tests/conftest.py` path extensions guarantees that tests continue to pass 100% and existing shell invocations remain completely uninterrupted during transition.

---

## 3. Caveats

1. **Zero Filesystem Mutation:** In strict compliance with instructions, no files were renamed, moved, or deleted. All findings are purely analytical.
2. **External Systemd Units:** Shell scripts contain comments mentioning systemd user units (`sae-evals`, `sae-ladder`, `autointerp-run`). If those systemd units invoke root shell scripts or root python scripts directly on the host machine, root forwarding stubs or updating the unit configuration files will be required when moves are implemented.
3. **Hardware Availability:** Scripts using `wait_gpu()` or CUDA (e.g. M2M100 decode in `textfid.py`, `autointerp.py`, `decode.py`) require GPU memory when executed in full production mode, though their unit tests run purely on CPU via JAX mocking.

---

## 4. Conclusion

The 27 root scripts should be reorganized into `scripts/` using a 5-domain layout (`data/`, `training/`, `interactive/`, `sae/`, `eval/`).

The recommended migration path is:
- **Phase 1:** Move scripts to `scripts/`, add `__init__.py` files, update `tests/conftest.py` to add `scripts/eval` and `scripts/sae` to `sys.path`, and place thin backward-compatibility stubs in root.
- **Phase 2:** Update path invocations in the 4 shell scripts and 4 subprocess wrappers.
- **Phase 3:** Transition sibling imports to standard package imports (`from scripts.sae import sae`) and retire root stubs.
- **Phase 4:** Fix `mnist.py` import and extract repeated checkpoint loading boilerplate into `ontologize.training.ontostate`.

All findings and move mappings are fully documented in `survey_reorg.md`.

---

## 5. Verification Method

To independently verify the observations and analysis:
1. **Verify Root Scripts Count:**
   ```bash
   ls -1 *.py *.sh | wc -l
   # Expected output: 27 (23 .py, 4 .sh)
   ```
2. **Verify Test Couplings:**
   ```bash
   python3 -c "import ast, glob; [(print(f'{f}: {n.module or n.names[0].name}')) for f in sorted(glob.glob('tests/*.py')) for n in ast.walk(ast.parse(open(f).read())) if isinstance(n, (ast.Import, ast.ImportFrom)) and (n.module or n.names[0].name) in ['autointerp', 'compose', 'headcoh', 'headstruct', 'langprobe', 'pareto', 'refit', 'sae', 'splitting', 'steerfid', 'textfid']]"
   ```
3. **Verify Sibling Couplings:**
   Inspect Section 2.2 of `survey_reorg.md` against AST parse of root scripts.
4. **Verify Broken Import in `mnist.py`:**
   ```bash
   python3 -c "import mnist"
   # Verifies ModuleNotFoundError: No module named 'ontologize.training.data'
   ```
