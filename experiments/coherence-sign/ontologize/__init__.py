"""The training path of the project's earlier codebase at commit 4d32e33
(2026-09-11, the "PWAK integration" merge), copied for `train_old.py`.

These are the 17 files that training `resid_nc`'s configuration imports (15
modules and two empty subpackage `__init__.py`s), extracted with `git
archive`. Their code is the commit's; their comments and docstrings have
been corrected where they contradicted it, and `check_copy.py` beside this
package verifies that the code still equals the commit's. Change only
comments and docstrings here; the code is what reproduces those runs.

This file is the only addition: it makes the copy a regular package, so it
shadows rather than merges with this repository's `ontologize`, a namespace
package, when both are importable."""
