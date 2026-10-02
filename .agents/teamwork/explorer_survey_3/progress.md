# Progress — Workstream 3 (Coupling Analysis & Reorg Proposal)

Last visited: 2026-09-29T22:25:00Z
Status: Complete

- [x] Initialized workspace, DISPATCH.md, BRIEFING.md
- [x] Survey 27 root scripts (23 .py, 4 .sh)
- [x] Survey test suite imports of root scripts (11 test files identified and analyzed)
- [x] Survey shell script invocations and dependencies (4 .sh scripts analyzed)
- [x] Survey inter-script imports (20 directed edges across 11 scripts mapped)
- [x] Survey subprocess wrappers (4 run_* scripts analyzed)
- [x] Analyze bugs, dead code, and code oddities (mnist.py, sonar_hc.py, CuDNN injections, etc.)
- [x] Design proposed target directory layout (5 domain subpackages under `scripts/`)
- [x] Formulate concrete move map for all 27 scripts
- [x] Formulate transition strategies, compatibility shims, PYTHONPATH adjustments
- [x] Write `survey_reorg.md`
- [x] Write `handoff.md`
- [x] Send completion message to parent
