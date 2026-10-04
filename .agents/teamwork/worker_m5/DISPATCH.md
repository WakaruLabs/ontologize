## 2026-09-29T22:25:17Z
You are worker_m5, assigned to Milestone 5: Reorganization Proposal (`docs/REORG_PROPOSAL.md`).
Your working directory is:
/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/worker_m5

MANDATORY: Read /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/ORIGINAL_REQUEST.md before starting work.
Read project specification: /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/orchestrator_1/PROJECT.md
Read survey report: /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_3/survey_reorg.md
Read survey handoff: /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_3/handoff.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

EXCLUSIVE WRITE BOUNDARIES:
You own and may ONLY edit:
- `docs/REORG_PROPOSAL.md`
DO NOT MOVE OR RENAME ANY EXISTING FILES. This is a proposal document only!

YOUR MISSION:
Write `docs/REORG_PROPOSAL.md` proposing (NOT executing) a cleaner layout for root scripts.
Must include:
1. Analysis of existing coupling constraints:
   - `tests/` direct module imports (the 11 test files importing root scripts directly).
   - Shell scripts calling root scripts by relative paths (`sonar.sh`, `sae_evals.sh`, `sae_ladder.sh`, `autointerp_run.sh`).
   - Sibling script imports (the 20 directed dependencies across 11 root scripts).
2. Proposed target layout:
   - Clean 5-domain modular structure under `scripts/` (e.g. `data/`, `training/`, `interactive/`, `sae/`, `eval/`).
3. Concrete move map:
   - Mapping table for each of all 27 scripts from current root path to new path.
4. Exact transition strategies to keep tests passing and shell scripts working:
   - Options: PYTHONPATH configuration in `pyproject.toml` / `tests/conftest.py`, root compatibility forwarding stubs / re-export shims, shell script invocation updates, and clean library extraction.
   - Recommended phased migration plan.

VERIFICATION REQUIREMENTS:
- Verify coverage of all 27 scripts in the move map.
- Verify coverage of all 11 importing test files and 4 shell scripts in coupling analysis.

DELIVERABLES:
- Create `docs/REORG_PROPOSAL.md`.
- Write summary of changes and catalog of observed bugs/oddities to `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/worker_m5/summary.md`.
- Write handoff report to `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/worker_m5/handoff.md`.
- Send message to orchestrator parent when complete.
