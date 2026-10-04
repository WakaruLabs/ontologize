## 2026-09-29T22:17:29Z

You are the Project Orchestrator for the ontologize repository documentation-and-organization pass.
Your assigned working directory is:
`/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/orchestrator_1`
The project root is:
`/home/jade/disk2/ontologizercleanup/ontologize`

The original user request is stored at:
`/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/ORIGINAL_REQUEST.md`

Please read ORIGINAL_REQUEST.md carefully and orchestrate the execution across the three required workstreams:
1. Workstream 1: Ontologize Package Docstrings
   Add or improve docstrings throughout the `ontologize/` package: module-level, class, public functions and methods. State purpose, arguments, return values, and tensor/array shapes wherever knowable from code.
   Subpackages:
   - `ontologize/layers/` (`dictblock.py`, `dictenc.py`, `linear.py`, `nlinear.py`, `sparse.py`, `__init__.py`)
   - `ontologize/training/` (`config.py`, `ontostate.py`, `serialize.py`, `__init__.py`)
   - `ontologize/data/` (`__init__.py`, `langs.py`, `loaders.py`, `multilingual.py`, `pretrained.py`)
   - `ontologize/fns/` (`classify.py`, `keys.py`, `loss.py`) + `ontologize/inference/` (`steerable.py`) + `ontologize/visualize/` (`loss.py`) + `ontologize/chat.py` + `ontologize/ontologizer.py` + `ontologize/main.py`
   CRITICAL HARD RULE: Change ONLY docstrings. No code edits, no renames, no reformatting, no type-hint changes, and no comment removal.
   DEAD MODULES TO SKIP ENTIRELY:
   - `ontologize/layers/dictblock_enhanced.py`
   - `ontologize/layers/dict_interpreter.py`
   - `ontologize/layers/integration_clean.py`
   - `ontologize/layers/vae_integration.py`

2. Workstream 2: Root Script Index (`docs/SCRIPTS.md`)
   Write `docs/SCRIPTS.md` indexing all 27 root-level `*.py` and `*.sh` scripts (23 .py, 4 .sh).
   Group by purpose (Data preparation / harvesting; Training / fine-tuning; Interactive / decoding / chat; SAE baseline + evaluation suite; Scratch / stale / experimental).
   Provide: one-line summary, main inputs and outputs, covering test file in `tests/` (if any).

3. Workstream 3: Reorganization Proposal (`docs/REORG_PROPOSAL.md`)
   Write `docs/REORG_PROPOSAL.md` proposing (NOT executing) a cleaner layout for root scripts.
   Analyze existing coupling constraints (tests/ direct module imports, shell script calls).
   Include proposed target layout, concrete move map, and exact changes required to keep tests passing and shell scripts working.

4. Worker summaries and code oddities / bugs logging:
   Catalog observed bugs, dead code, or oddities (do NOT fix them).

CONSTRAINTS & GUARDRAILS:
- Do NOT modify `GEMINI.md`, `README.md`, `CLAUDE.md`, `pyproject.toml`, `tests/`, or any root-level script.
- Do NOT move or rename any existing files.
- Do NOT install any packages, run model training, or download any datasets or checkpoints.

VERIFICATION REQUIRED:
- AST / syntactic invariance check: AST nodes of modified files (or code stripped of docstrings) must match before vs after.
- git status & diff scope: only modified files in `ontologize/` (docstring changes only) and new files in `docs/`.
- python3 -m py_compile on all modified files.
- All 27 root scripts accounted for in `docs/SCRIPTS.md`.

Maintain your `BRIEFING.md` and `progress.md` in `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/orchestrator_1/`.
When finished and verified, send a message claiming victory and report your findings to Sentinel.
