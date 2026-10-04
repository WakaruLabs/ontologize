# BRIEFING — 2026-09-29T22:25:00Z

## Mission
Analyze coupling constraints and design a comprehensive reorganization proposal for the 27 root scripts (23 .py, 4 .sh) to produce `survey_reorg.md` and `handoff.md`.

## 🔒 My Identity
- Archetype: Teamwork explorer
- Roles: Exploration agent (Workstream 3: Reorganization Proposal & Coupling Analysis)
- Working directory: /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_3
- Original parent: 5c469005-ec60-4d24-ba33-17f4fec5aef5
- Milestone: Survey & Proposal Generation

## 🔒 Key Constraints
- Read-only investigation — do NOT implement, move, or rename any files!
- Do NOT modify GEMINI.md, README.md, CLAUDE.md, pyproject.toml, tests/, or any root-level script.
- Do NOT install any packages, run training, or download data/checkpoints.
- Write analysis strictly to .agents/teamwork/explorer_survey_3/

## Current Parent
- Conversation ID: 5c469005-ec60-4d24-ba33-17f4fec5aef5
- Updated: not yet

## Investigation State
- **Explored paths**:
  - All 27 root scripts (23 .py, 4 .sh)
  - All 25 test files in `tests/`
  - All shell scripts (`autointerp_run.sh`, `sae_ladder.sh`, `sae_evals.sh`, `sonar.sh`)
  - Subprocess callers (`run_chat.py`, `run_chat2.py`, `run_decode.py`, `run_decode_tags.py`)
- **Key findings**:
  - 11 test files import root scripts directly; all tests target pure helper functions rather than CLI executables.
  - Sibling scripts have 20 directed import edges; `sae.py` (7 importers) and `pareto.py` (6 importers) are central hubs.
  - 4 shell scripts call root scripts by relative paths.
  - 4 Python scripts wrap `decode.py` and `decode_tags.py` via subprocess.
  - Fatal import bug in `mnist.py` (`from ontologize.training.data import ImageLoader`).
  - Dangerous module-level monkeypatching in `sonar_hc.py`.
- **Unexplored areas**: None remaining for this scope.

## Key Decisions Made
- Proposed 5-domain layout under `scripts/`: `data/`, `training/`, `interactive/`, `sae/`, `eval/`.
- Designed 3-phase transition strategy combining temporary root shims, `tests/conftest.py` sys.path injection, and eventual package-level import modernization.
- Authored comprehensive `survey_reorg.md` and complete hard `handoff.md`.

## Artifact Index
- DISPATCH.md — record of incoming dispatch
- BRIEFING.md — situational awareness
- progress.md — liveness heartbeat
- survey_reorg.md — detailed coupling and architecture survey
- handoff.md — 5-component hard handoff report
