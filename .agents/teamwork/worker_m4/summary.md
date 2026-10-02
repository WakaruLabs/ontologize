# Summary of Changes and Code Oddities — Milestone 4 (Worker M4)

## 1. Summary of Changes

Milestone 4 required creating `docs/SCRIPTS.md`: an exhaustive, structured index and guide covering all 27 root-level scripts in the repository (23 Python files and 4 Bash shell scripts).

### Deliverable: `docs/SCRIPTS.md`
- **Scope**: Exactly 27 root scripts accounted for (23 `*.py`, 4 `*.sh`).
- **Structure**:
  1. **Overview & Architecture**: Explains dual JAX/Flax and PyTorch/Transformers runtime stack, memory allocation coordination (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `expandable_segments:True`), and DLPack zero-copy tensor sharing.
  2. **Master Script Index Table**: Comprehensive markdown table indexing all 27 scripts with Category, One-Line Summary, Main Inputs, Main Outputs, and Covering Test in `tests/`.
  3. **Category 1: Data Preparation / Harvesting (3 scripts)**: `encode_corpus.py`, `encode_discord.py`, `classify_acts.py`. Detailed operational mechanisms, CLI flags, memory map layouts, and sidecar files.
  4. **Category 2: Training / Fine-Tuning (3 scripts)**: `sonar.py`, `sonar_hc.py`, `sonar.sh`. Model architectures, deep supervision configs, residual conditioning, and tmux runner.
  5. **Category 3: Interactive / Decoding / Chat (2 scripts)**: `decode.py`, `decode_tags.py`. Interactive causal steering, M2M100 beam search, tag label generation, and reference norm scaling.
  6. **Category 4: SAE Baseline + Evaluation Suite (14 scripts)**: `sae.py`, `pareto.py`, `refit.py`, `headcoh.py`, `headstruct.py`, `langprobe.py`, `splitting.py`, `steerfid.py`, `textfid.py`, `compose.py`, `autointerp.py`, `autointerp_run.sh`, `sae_ladder.sh`, `sae_evals.sh`. Structural ablation ladder rungs, metrics, mathematical formulations, CLI parameters, and artifacts.
  7. **Category 5: Scratch / Stale / Experimental (5 scripts)**: `mnist.py`, `run_chat.py`, `run_chat2.py`, `run_decode.py`, `run_decode_tags.py`. Diagnostic failure analysis and execution wrappers.
  8. **Cross-Script Coupling and Dependency Graph**: Visual ASCII import graph, shell invocation map, and unit test invocation mapping.
  9. **Catalog of Observed Bugs, Dead Code, and Oddities**: Full documentation of observed issues.

---

## 2. Catalog of Observed Bugs, Dead Code, and Oddities

During the exhaustive survey and documentation of all root-level scripts, the following bugs, dead code patterns, and architectural oddities were cataloged:

### 1. Broken Module Imports and Obsolete Constructors in `mnist.py`
- **Location**: `mnist.py:7`, `mnist.py:39-43`.
- **Observation**:
  - `mnist.py:7` attempts `from ontologize.training.data import ImageLoader`. The `ontologize.training` package contains only `config.py`, `ontostate.py`, `serialize.py`, and `__init__.py`. `ImageLoader` actually resides in `ontologize.data.loaders`.
  - `mnist.py:39` calls `Ontologizer(d_in, d_in, e_enc, e_dec, k, h, n_l)` with 7 positional arguments, but `Ontologizer` expects `(d, e_enc, e_dec, k, h, l, ...)` with only one input/output dimension `d`.
  - `mnist.py:40-43` calls `Hyperparams` and `Metadata` passing arguments that do not align with current dataclass field signatures (e.g. passing `b` where `lr` is expected, `stddev` where `temperature` is expected).
- **Impact**: Running `python mnist.py` immediately crashes with `ModuleNotFoundError`. The script is completely stale and abandoned.

### 2. Module-Level Global Variable Monkey-Patching in `sonar_hc.py`
- **Location**: `sonar_hc.py:33-42`.
- **Observation**:
  `sonar_hc.py` imports `sonar` and imperatively mutates its top-level module globals before calling `sonar.main()`:
  ```python
  import sonar
  sonar.s_hcossim = 1e-5
  sonar.s_Hm = 0.0
  sonar.out = sonar.path / "out/sonar/multilingual/resid_nc_hc"
  sonar.main()
  ```
- **Impact**: Binds `sonar_hc.py` tightly to the internal implementation details and variable names of `sonar.py`. If any global variable in `sonar.py` is renamed, `sonar_hc.py` will silently fail or behave unexpectedly.

### 3. Lack of Command-Line Argument Parsing in Core Training Scripts
- **Location**: `sonar.py`, `sonar_hc.py`.
- **Observation**:
  Unlike all SAE baseline and evaluation scripts (which use `argparse` with structured CLI flags), `sonar.py` and `sonar_hc.py` configure all hyperparameters, data paths, and run options via hardcoded top-level module variables.
- **Impact**: Sweeps, hyperparameter tuning, or rerunning on different datasets requires directly editing source code files, increasing git churn and risk of unintended commits.

### 4. Self-Restarting CuDNN Process Replacement in Interactive Decoders
- **Location**: `decode.py:6-15`, `decode_tags.py:6-15`.
- **Observation**:
  Both scripts inspect `os.environ` for `CUDNN_INJECTED`. If absent, they dynamically resolve the pip-installed CuDNN directory, inject it into `LD_LIBRARY_PATH`, set `CUDNN_INJECTED=1`, and call:
  ```python
  os.execv(sys.executable, [sys.executable] + sys.argv)
  ```
- **Impact**: This self-restart bypasses the parent shell and re-executes the process. While intended to resolve library search path issues on specific GPU setups, it causes unusual side effects when invoked inside subprocess wrappers or virtual environments.

### 5. Hardcoded Paths in Ad-Hoc Scratch Runners
- **Location**: `run_chat.py:12`, `run_chat2.py:11`, `run_decode.py:15`, `run_decode_tags.py:9`.
- **Observation**:
  Four separate scripts exist solely to wrap `decode.py` and `decode_tags.py` with hardcoded inputs. All four hardcode the legacy checkpoint path `data/out/sonar/linear/multilingual`.
- **Impact**: These are one-off scratch scripts rather than reusable test tools or utilities. If the legacy checkpoint does not exist, all four fail.

### 6. External Path Dependency in `encode_discord.py`
- **Location**: `encode_discord.py:55`.
- **Observation**:
  `src = "../act-i-export"` points to an uncommitted external directory outside the repository root.
- **Impact**: Running `encode_discord.py` with default arguments in a clean checkout fails because `../act-i-export` does not exist.

### 7. Extensive Cross-Script Module Coupling at Root Level
- **Location**: `compose.py`, `headcoh.py`, `headstruct.py`, `langprobe.py`, `refit.py`, `splitting.py`, `steerfid.py`, `textfid.py`.
- **Observation**:
  The evaluation scripts treat other root-level scripts as top-level Python modules (e.g. `import sae`, `from pareto import load_onto`, `from splitting import load_model`, `from autointerp import onto_acts_fn`). None of these utilities are packaged inside `ontologize/`.
- **Impact**: These scripts cannot be relocated into subfolders (e.g. `scripts/eval/`) without either setting `PYTHONPATH=.`, using relative package imports, or creating CLI entry points.
