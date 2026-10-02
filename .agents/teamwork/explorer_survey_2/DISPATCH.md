# DISPATCH — Explorer Survey 2 (Root Scripts)
Target: Comprehensive census and analysis of all 27 root-level scripts (23 .py, 4 .sh) and tests/ coverage.
Working directory: /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_2
Original Request: /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/ORIGINAL_REQUEST.md

## 2026-09-29T22:18:30Z
You are explorer_survey_2, an exploration agent for Workstream 2 (Root Script Index).
Your working directory is:
/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_2

MANDATORY: Read /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/ORIGINAL_REQUEST.md before starting work.

YOUR MISSION:
Perform a comprehensive survey of all root-level `*.py` and `*.sh` scripts (27 scripts in total: 23 `.py` and 4 `.sh`).
1. Identify all 27 root scripts:
   - Verify the exact list of 23 `.py` files and 4 `.sh` files in the repository root.
2. For each script:
   - Inspect its code / docstrings / CLI arguments (`argparse`, etc.).
   - Determine its purpose and produce a clear one-line summary.
   - Determine its main inputs (CLI flags, input data formats, config, checkpoints, datasets) and outputs (saved models, metrics, JSON/H5 files, plots, console output).
   - Inspect the `tests/` directory: grep/search for imports of the script or direct testing of the script's functions/classes. Identify the exact covering test file in `tests/` (or "None" if untested).
3. Group the 27 scripts into the 5 required categories:
   - Data preparation / harvesting
   - Training / fine-tuning
   - Interactive / decoding / chat
   - SAE baseline + evaluation suite
   - Scratch / stale / experimental
4. Catalog any observed bugs, dead code, or oddities across these scripts (do NOT fix them).

Deliverables:
- Write your detailed findings and the structured script table to `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_2/survey_scripts.md`.
- Write `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_2/handoff.md` with standard handoff format.
- Send a message to orchestrator parent when done.
