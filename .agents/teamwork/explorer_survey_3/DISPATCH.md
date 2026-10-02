## 2026-09-29T22:18:30Z
You are explorer_survey_3, an exploration agent for Workstream 3 (Reorganization Proposal & Coupling Analysis).
Your working directory is:
/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_3

MANDATORY: Read /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/ORIGINAL_REQUEST.md before starting work.

YOUR MISSION:
Perform a deep coupling and architecture survey to prepare `docs/REORG_PROPOSAL.md`.
Note: This is an analytical survey proposing (NOT executing) a cleaner layout. Do NOT move or rename any files!

1. Analyze existing coupling constraints:
   - Inspect `tests/`: which tests import root scripts directly as top-level modules? (Specifically examine: `pareto`, `sae`, `refit`, `autointerp`, `splitting`, `compose`, `steerfid`, `headstruct`, `headcoh`, `langprobe`, `textfid`, and any others). Document exact import statements and test files.
   - Inspect all shell scripts (`sonar.sh`, `sae_evals.sh`, `sae_ladder.sh`, `autointerp_run.sh`, etc.): how do they invoke root scripts? What relative paths, arguments, environment variables are used?
   - Inspect root scripts importing other root scripts: which scripts import helper functions from sibling root scripts?
2. Design a proposed target directory layout for the repository:
   - E.g., `scripts/data/`, `scripts/training/`, `scripts/interp/`, `scripts/sae/`, `scripts/eval/`, `scripts/experimental/` or whatever logical structure best fits the codebase.
3. Formulate a concrete move map for every one of the 27 scripts.
4. Detail the exact changes required to keep tests passing and shell scripts working:
   - Options: PYTHONPATH adjustments in test runner, test import updates vs compatibility shim / re-export stubs, shell script invocation path updates, package entry points.
   - Pros/cons of each approach and recommended implementation strategy.
5. Catalog any observed bugs, dead code, or oddities (do NOT fix them).

Deliverables:
- Write your analysis to `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_3/survey_reorg.md`.
- Write `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_3/handoff.md`.
- Send a message to orchestrator parent when done.
