# BRIEFING — 2026-09-29T22:28:30Z

## Mission
Write `docs/REORG_PROPOSAL.md` proposing a cleaner layout for root scripts without moving or renaming any existing files.

## 🔒 My Identity
- Archetype: worker_m5
- Roles: implementer, qa, specialist
- Working directory: /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/worker_m5
- Original parent: 5c469005-ec60-4d24-ba33-17f4fec5aef5
- Milestone: Milestone 5: Reorganization Proposal (docs/REORG_PROPOSAL.md)

## 🔒 Key Constraints
- Exclusive write boundary: `docs/REORG_PROPOSAL.md` and `.agents/teamwork/worker_m5/*`
- DO NOT MOVE OR RENAME ANY EXISTING FILES. This is a proposal document only!
- Must cover all 27 root scripts in move map
- Must cover all 11 importing test files and 4 shell scripts in coupling analysis
- DO NOT CHEAT or fabricate verification outputs

## Current Parent
- Conversation ID: 5c469005-ec60-4d24-ba33-17f4fec5aef5
- Updated: 2026-09-29T22:28:30Z

## Task Summary
- **What to build**: `docs/REORG_PROPOSAL.md` proposing a cleaner layout for root scripts with coupling analysis, 5-domain modular structure, 27-script move map, and transition strategies.
- **Success criteria**: Comprehensive, accurate analysis of coupling constraints (tests, shell scripts, sibling imports), 5-domain target layout under scripts/, complete 27-script move map, concrete transition strategies with phased migration plan.
- **Interface contracts**: PROJECT.md, survey_reorg.md, handoff.md
- **Code layout**: Proposal doc in `docs/REORG_PROPOSAL.md`

## Key Decisions Made
- Organized scripts into 5 domains under `scripts/`: `data/`, `training/`, `interactive/` (with `runners/` subpackage), `sae/`, and `eval/`.
- Designed a 4-phase transition plan starting with root compatibility forwarding stubs and `tests/conftest.py` path extensions, guaranteeing 100% test compatibility with zero downtime.

## Artifact Index
- `docs/REORG_PROPOSAL.md` — Root script reorganization proposal document
- `.agents/teamwork/worker_m5/summary.md` — Summary of changes and catalog of observed bugs/oddities
- `.agents/teamwork/worker_m5/handoff.md` — Handoff report

## Change Tracker
- **Files modified**: `docs/REORG_PROPOSAL.md` (created)
- **Build status**: Verified document completeness via automated AST/string assertions (100% match)
- **Pending issues**: None

## Quality Status
- **Build/test result**: All 27 scripts, 11 tests, and 4 shell scripts verified in proposal
- **Lint status**: Clean markdown
- **Tests added/modified**: None (proposal only)

## Loaded Skills
- None
