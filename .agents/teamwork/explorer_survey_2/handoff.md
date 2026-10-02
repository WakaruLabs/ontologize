# Handoff Report — Explorer Survey 2 (Root Script Index)

## 1. Observation

- **Script Inventory:** Direct inspection of `/home/jade/disk2/ontologizercleanup/ontologize` confirmed exactly 27 root-level executable scripts: 23 Python files (`*.py`) and 4 Bash shell scripts (`*.sh`).
  - Python files: `autointerp.py`, `classify_acts.py`, `compose.py`, `decode.py`, `decode_tags.py`, `encode_corpus.py`, `encode_discord.py`, `headcoh.py`, `headstruct.py`, `langprobe.py`, `mnist.py`, `pareto.py`, `refit.py`, `run_chat.py`, `run_chat2.py`, `run_decode.py`, `run_decode_tags.py`, `sae.py`, `sonar.py`, `sonar_hc.py`, `splitting.py`, `steerfid.py`, `textfid.py`.
  - Shell scripts: `autointerp_run.sh`, `sae_evals.sh`, `sae_ladder.sh`, `sonar.sh`.

- **Test Suite Coverage:** Searching `/home/jade/disk2/ontologizercleanup/ontologize/tests/` revealed direct imports of root scripts in 11 test files:
  - `tests/test_autointerp.py:6`: `from autointerp import (CAL_PROMPT_CHARS, CONTRAST_PROMPT, SUMMARIZE_PROMPT, ...)`
  - `tests/test_compose.py:6`: `from compose import additivity_stats, sample_singles`
  - `tests/test_headcoh.py:5`: `from headcoh import (group_zscores, mean_pairwise_cos, perm_z, ...)`
  - `tests/test_headstruct.py:8`: `import headstruct`
  - `tests/test_langprobe.py:7`: `import langprobe`
  - `tests/test_pareto.py:8`: `import pareto`
  - `tests/test_refit.py:8,50`: `import refit`, `import sae`
  - `tests/test_sae.py:10`: `import sae`
  - `tests/test_splitting.py:8`: `import splitting`
  - `tests/test_steerfid.py:7`: `import steerfid`
  - `tests/test_textfid.py:6`: `from textfid import chrf`
  - The remaining 16 root scripts have no importing test files in `tests/` (`Covering Test: None`).

- **Cross-Script Coupling:** Root scripts import functions and classes from each other directly:
  - `splitting.py:54,91,97`: imports `sae`, `from pareto import parse_sae_name`, `from autointerp import onto_acts_fn`.
  - `langprobe.py:39`: `from splitting import load_model`.
  - `compose.py:89`: `from pareto import load_onto`.
  - `steerfid.py:49,139,183`: imports `sae`, `from pareto import load_onto`, `from textfid import chrf`.
  - `textfid.py:44,105,115`: imports `sae`, `from pareto import parse_sae_name`, `from pareto import load_onto`.
  - `headstruct.py:55,184`: imports `sae`, `from pareto import parse_sae_name`.
  - `refit.py:50,125,170`: imports `sae`, `from autointerp import onto_acts_fn`, `from pareto import parse_sae_name`.
  - `headcoh.py:145,186`: `from refit import onto_linear_model`, `from autointerp import ENCODER_ID`.
  - `autointerp.py:452,550`: imports `sae as sae_mod`.
  - `sonar_hc.py:33-37`: `import sonar`, sets `sonar.s_hcossim = 1e-5`, `sonar.s_Hm = 0.0`.

- **Observed Code Defects:**
  - `mnist.py:7`: `from ontologize.training.data import ImageLoader` fails with `ModuleNotFoundError` because `ontologize/training/data.py` does not exist.
  - `mnist.py:40-43`: Calls `Hyperparams` and `Metadata` with invalid and out-of-order arguments.
  - `encode_discord.py:55`: Default path `src = "../act-i-export"` references external sibling directory outside workspace.
  - `decode.py:6-15` and `decode_tags.py:6-15`: Self-restart via `os.execv(sys.executable, [sys.executable] + sys.argv)` for CuDNN `LD_LIBRARY_PATH` injection.

## 2. Logic Chain

1. Starting from the file listing of `/home/jade/disk2/ontologizercleanup/ontologize`, filtering for root-level files matching `*.py` and `*.sh` yielded exactly 23 Python files and 4 shell files (total 27).
2. Inspection of code docstrings, CLI definitions (`argparse`), and execution structures established each script's primary role, leading to unambiguous grouping into the 5 target categories:
   - Data preparation / harvesting (3 scripts: `encode_corpus.py`, `encode_discord.py`, `classify_acts.py`).
   - Training / fine-tuning (3 scripts: `sonar.py`, `sonar_hc.py`, `sonar.sh`).
   - Interactive / decoding / chat (2 scripts: `decode.py`, `decode_tags.py`).
   - SAE baseline + evaluation suite (14 scripts: `sae.py`, `pareto.py`, `headcoh.py`, `headstruct.py`, `langprobe.py`, `splitting.py`, `steerfid.py`, `textfid.py`, `compose.py`, `refit.py`, `autointerp.py`, `autointerp_run.sh`, `sae_evals.sh`, `sae_ladder.sh`).
   - Scratch / stale / experimental (5 scripts: `mnist.py`, `run_chat.py`, `run_chat2.py`, `run_decode.py`, `run_decode_tags.py`).
3. Systematic regex searching across all 25 files in `tests/` identified the exact 11 test modules that import root scripts, proving that all remaining 16 scripts lack direct unit test coverage.
4. Parsing import statements across root scripts mapped an intricate horizontal dependency web among evaluation modules, demonstrating that any reorganization must account for root `sys.path` assumptions or package boundaries.

## 3. Caveats

- Execution testing was not run with live GPUs or external APIs (e.g. Anthropic API calls in `autointerp.py` or large mC4 dataset downloads in `encode_corpus.py`), conforming to the strict read-only/no-training guardrail.
- Categorization of `run_chat.py`, `run_chat2.py`, `run_decode.py`, and `run_decode_tags.py` could alternatively be viewed under "Interactive / decoding / chat" as automated test drivers; however, their hardcoded paths, fixed input strings, and ad-hoc nature place them most accurately in "Scratch / stale / experimental". Both classifications are documented in `survey_scripts.md`.

## 4. Conclusion

The script survey is complete. All 27 root-level scripts are fully cataloged with their purpose, inputs, outputs, test coverage, category assignments, and observed bugs/oddities. The deliverables are written to:
- `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_2/survey_scripts.md`
- `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_2/handoff.md`

These documents provide the complete foundation for Workstream 2 (`docs/SCRIPTS.md`) and Workstream 3 (`docs/REORG_PROPOSAL.md`).

## 5. Verification Method

To independently verify the observations:
1. Count root scripts:
   ```bash
   ls -1 /home/jade/disk2/ontologizercleanup/ontologize/*.py | wc -l  # returns 23
   ls -1 /home/jade/disk2/ontologizercleanup/ontologize/*.sh | wc -l  # returns 4
   ```
2. Verify test imports:
   ```bash
   grep -rnE "^(import|from) (autointerp|classify_acts|compose|decode|decode_tags|encode_corpus|encode_discord|headcoh|headstruct|langprobe|mnist|pareto|refit|run_chat|run_chat2|run_decode|run_decode_tags|sae|sonar|sonar_hc|splitting|steerfid|textfid)" /home/jade/disk2/ontologizercleanup/ontologize/tests/
   ```
   Confirm it returns exactly 11 matching test files.
3. Verify `mnist.py` broken import:
   Inspect lines 4-8 of `/home/jade/disk2/ontologizercleanup/ontologize/mnist.py` and verify `ontologize/training/data.py` does not exist.
