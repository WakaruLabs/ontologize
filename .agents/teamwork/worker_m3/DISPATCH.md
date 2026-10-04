## 2026-09-29T22:25:17Z
You are worker_m3, assigned to Milestone 3: Ontologize Package Docstrings for Fns, Inference, Visualize & Root.
Your working directory is:
/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/worker_m3

MANDATORY: Read /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/ORIGINAL_REQUEST.md before starting work.
Read project specification: /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/orchestrator_1/PROJECT.md
Read survey blueprint: /home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/explorer_survey_1/survey_docstrings.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

EXCLUSIVE WRITE BOUNDARIES:
You own and may ONLY edit:
- `ontologize/fns/classify.py`
- `ontologize/fns/keys.py`
- `ontologize/fns/loss.py`
- `ontologize/inference/steerable.py`
- `ontologize/visualize/loss.py`
- `ontologize/chat.py`
- `ontologize/ontologizer.py`
- `ontologize/main.py`
DO NOT TOUCH ANY OTHER FILE.

CRITICAL HARD RULES:
- Change ONLY docstrings.
- ZERO code edits, ZERO renames, ZERO reformatting, ZERO type-hint changes, and ZERO comment removal.
- Add module-level docstring, class docstrings, and docstrings for all public functions/methods.
- State purpose, arguments, return values, and tensor/array shapes wherever knowable from code.

VERIFICATION REQUIREMENTS:
- Run `python3 -m py_compile` on all modified files.
- Run an AST syntactic invariance check (compare AST of file before vs after, with docstrings stripped/ignored) to prove zero code changes.
- Verify `git diff --stat` shows changes only in your assigned files.

DELIVERABLES:
- Write summary of changes and catalog of observed bugs/oddities (do not fix them) to `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/worker_m3/summary.md`.
- Write handoff report to `/home/jade/disk2/ontologizercleanup/ontologize/.agents/teamwork/worker_m3/handoff.md`.
- Send message to orchestrator parent when complete.
