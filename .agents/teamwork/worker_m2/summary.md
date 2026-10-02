# Milestone 2 Summary: Ontologize Package Docstrings for Training & Data

## 1. Overview of Changes
In Milestone 2, comprehensive module-level docstrings, class docstrings, and function/method docstrings were added to all assigned modules in `ontologize/training/` and `ontologize/data/`. All additions strictly followed the Zero-Code-Change and AST-Invariance mandates: no executable logic, type hints, variable names, or comments were altered.

### Files Modified & Summary of Docstrings Added

1. **`ontologize/training/__init__.py`** (Created):
   - Added package-level docstring describing the `ontologize.training` subpackage and summarizing its submodules (`config`, `ontostate`, `serialize`).

2. **`ontologize/training/serialize.py`**:
   - Added module-level docstring describing Orbax parameter and configuration serialization and restoration.
   - Enhanced `save_model`: Documented parameters (`checkpoint_dir`, `step`, `params_pytree`, `model`), configuration serialization via `dataclasses.asdict`, and checkpoint directory structure.
   - Enhanced `load_model`: Documented parameters, structured vs. unstructured parameter restore, return values `(restored_params, instantiated_model)`, and `ValueError` exception conditions.

3. **`ontologize/training/ontostate.py`**:
   - Added module-level docstring detailing training state management, optimization, schedules, and training loop orchestration.
   - Enhanced `OntoState`: Documented attributes (`model`, `b`, `save_each`, `n_stats`, `noise_in`, `noise_K`, `noise_F`, `stats`), ring buffer mechanics, and methods (`spec`, `save`, `newstats`, `writestats`).
   - Enhanced `state_init`: Documented initialization of dummy array `(b, model.d_in)`, binding of `apply_fn` (`withGhost` vs. `withStats`), and returned `OntoState`.
   - Enhanced `stats_insert`: Documented ring buffer insertion at `state.step % state.save_each` for 1D stats of shape `(n_stats,)`.
   - Enhanced `load_params`: Documented checkpoint restoration, backwards compatibility handling for legacy param dictionaries, and isolation of the active stats buffer.
   - Enhanced `update`: Documented JIT-compiled single optimization step, loss evaluation, global gradient norm clipping, parameter gradients, and return tuple `(state, L, rng_next)`.
   - Enhanced `schedules`: Documented geometric and linear hyperparameter annealing schedules (`temperature`, `p_drop`, `sd_K`) over the `anneal_steps` horizon.
   - Enhanced `train`: Documented outer dataset training loop, optional pretrained encoder integration, tqdm progress updates, periodic checkpointing, and `loss.csv` flushing.

4. **`ontologize/training/config.py`**:
   - Added module-level docstring outlining hyperparameters, metadata, and execution environments.
   - Enhanced `_mse_weights`: Documented per-dimension MSE weight caching across JIT traces.
   - Enhanced `Hyperparams`: Documented all fields and architectural constraints, loss weightings (`s_g`, `s_L1K`, `s_L1F`, `s_H`, `s_bcossim`, `s_hcossim`, `s_Hm`), winner dropout (`p_drop`), and schedules.
   - Enhanced `Hyperparams` methods: `s_loss` (scaling vector and active loss booleans), `rng`, `loss` (multi-component loss computation and `stats` array assembly), `opt` (Adam vs. AdamW selection), `ontologizer` (Flax model instantiation), `init`, `update`, `train`, and `load`.
   - Enhanced `Metadata`: Documented serialization targets, retention policies, and data loading paths.
   - Enhanced `Metadata` methods: `manager` (`CheckpointManagerOptions` and PyTree checkpointers), `src`, `loader`, and `model`.
   - Enhanced `log_env`: Documented environment logging to `log.jsonl` with custom serialization for `Path` and NumPy/JAX types.
   - Enhanced `TrainingEnv`: Documented container class and methods `init` (constructing state, loader, and manager, with checkpoint resumption and `loss.csv` truncation) and `train` (running full training lifecycle).

5. **`ontologize/data/__init__.py`**:
   - Added package-level docstring summarizing dataset loading, transforms, and pretrained encoder bridges across `langs`, `loaders`, `multilingual`, and `pretrained`.

6. **`ontologize/data/langs.py`**:
   - Added module-level docstring describing the 90-language mapping table `MC4_TO_SONAR` translating HuggingFace ISO 639 dataset identifiers to SONAR / NLLB BCP-47 codes. Preserved existing comments.

7. **`ontologize/data/multilingual.py`**:
   - Added module-level docstring describing dataset downloading, cleaning, and multilingual interleaving.
   - Enhanced `load_lang`: Documented single-language dataset loading, metadata filtering, and language tagging.
   - Enhanced `load_langs`: Documented dataset combination via `datasets.interleave_datasets`.
   - Enhanced `mc4_data`: Documented convenience wrapper over all supported mC4 language splits.

8. **`ontologize/data/pretrained.py`**:
   - Added module-level docstring describing PyTorch transformer integration, tokenization, pooling, and DLPack zero-copy tensor sharing. Preserved top comment.
   - Enhanced `get_torch_dtype`: Documented string to `torch.dtype` resolution.
   - Enhanced `pretrained_transformer`: Documented M2M100/SONAR and AutoModel loading in half precision (`bfloat16`).
   - Enhanced `tokenize`: Documented batch tokenization with padding.
   - Enhanced `l2_pooling`: Documented attention-masked mean pooling and unit-L2 normalization (`||v||_2 = 1`).
   - Enhanced `encode`: Documented evaluation under `torch.no_grad()`, zero-copy bridge via `jnp.from_dlpack`, and pooling.
   - Enhanced `decode`: Documented zero-copy bridge via `t.from_dlpack` and autoregressive sequence generation.

9. **`ontologize/data/loaders.py`**:
   - Added module-level docstring describing Grain data sources, transformation maps, and batch loaders. Preserved all existing comments.
   - Enhanced `TokenizeTransform`: Documented class, `__init__`, and `map` with dynamic language selection from `LANG_MAP`.
   - Enhanced `FlattenTransform`: Documented class, `__init__`, and `map` (image flattening to `(..., height * width)` and scaling to `[0, 1]`).
   - Enhanced `DecodeTransform`: Documented class and `map` (UTF-8 decoding).
   - Enhanced `HFDataSource`: Documented class and `__new__` (dispatching to random-access or streaming iterators).
   - Enhanced `NpyDataSource`: Documented class, `__init__`, `__len__`, `__getitem__`, and `__getstate__` (lazy memory-mapping and clean pickling for Grain multiprocessing).
   - Enhanced `JSONLDataSource`: Documented class, `__init__` (byte offset scanning for O(1) seeking), `__len__`, and `__getitem__`.
   - Enhanced `SampleLoader`: Documented class, `__init__` (Grain pipeline construction with `IndexSampler` and `DataLoader`), and `__iter__` (generator fallbacks).
   - Enhanced `EmbeddingLoader`, `ImageLoader`, `TextLoader`: Documented specialized loader subclasses.

---

## 2. Catalog of Observed Bugs, Dead Code, and Oddities
*(Per task requirements, these items were identified and documented during analysis, but code was NOT modified to fix them.)*

1. **Obsolete Module Reference in `Metadata.loader` Docstring**:
   - In `ontologize/training/config.py` line 256, the docstring states: `"Creates the appropriate data loader type from ontologize.training.data as specified by self.srctype"`.
   - *Oddity*: There is no `ontologize.training.data` package; `SampleLoader` and its subclasses reside in `ontologize.data.loaders`. (This is also the root cause of the broken import in `mnist.py:7`).

2. **Unused `Hyperparams.update` Method**:
   - `Hyperparams.update` defines a wrapper calling `ontologize.ontostate.update`, but it is never called anywhere in the codebase. `Hyperparams.train` bypasses it and directly invokes `ontologize.training.ontostate.train` (which directly invokes `ontostate.update`).

3. **Loss Stats Length Inconsistency**:
   - In `Hyperparams.loss`, the type annotation for `stats` is given as `Float[Array, "4"]`, but in actual execution (and in `DictEnc.withStats` / `Ontologizer.withStats`), `stats` contains 5 or 6 elements (`[L1_K, L1_F, entropy, cossim_b, cossim_h, KL_m]`).

4. **Incomplete Language Table in `loaders.py` vs `langs.py`**:
   - `ontologize/data/loaders.py` defines a local dictionary `LANG_MAP` containing only 5 languages (`en`, `fr`, `es`, `de`, `zh`), whereas `ontologize/data/langs.py` defines `MC4_TO_SONAR` with 90 languages. `TokenizeTransform` uses the 5-language `LANG_MAP` rather than the comprehensive `MC4_TO_SONAR` mapping.

5. **`OntoState.writestats` Missing CSV Header**:
   - `OntoState.writestats` opens `manager.directory / file` in append mode (`'a'`) and writes rows via `csv.writer.writerows(dat)`. It never writes a header row (`loss, MSE, MSE_ghost, L1_K, L1_F, entropy, cossim_b, cossim_h, KL_m`), meaning tools like `ontologize/visualize/loss.py` must hardcode the column names when reading the CSV without headers.

6. **`TrainingEnv.init` Resumption Line Truncation**:
   - `TrainingEnv.init` attempts to truncate `loss.csv` upon resuming with `lines = lines[:self.meta.resume_from]`. If `loss.csv` had a header line or if `save_each > 1`, simple step index slicing does not correspond 1-to-1 with line indices.

7. **Standalone Orbax Serialization in `serialize.py` Unconnected to Training Pipeline**:
   - `ontologize/training/serialize.py` uses `ocp.args.StandardSave` and `ocp.args.StandardRestore`, whereas the active training pipeline in `ontostate.py` uses PyTree checkpointers (`items={'state': self, 'spec': self.spec()}`). `serialize.py` is entirely unreferenced by `sonar.py`, `decode.py`, and `steerable.py`.

8. **`NpyDataSource.__getitem__` Memmap Re-creation**:
   - In `NpyDataSource.__getitem__`, if `self._arr is None` (after unpickling in a Grain worker process), it re-executes `self._arr = np.load(self.file_path, mmap_mode="r")`. This reloads the file handle once per worker, but each index access performs a full row copy `np.array(self._arr[int(idx)])` which avoids pinning but incurs memory allocation overhead for large batches.

9. **`JSONLDataSource` Full-File Offset Scan at Startup**:
   - `JSONLDataSource.__init__` scans the entire JSONL file line by line in binary mode at initialization time to compute `_offsets`. For multi-gigabyte files, this initialization can take significant time and RAM before any training batch is generated.

10. **`load_lang` Schema Filtering Workaround**:
    - In `ontologize/data/multilingual.py`, `load_lang` explicitly strips columns and keeps only `text` because different language shards in HuggingFace C4 have mismatched schemas (e.g. `timestamp` formatted as `string` in some languages and `timestamp[us]` in others).

11. **Local Imports in `pretrained_transformer`**:
    - `ontologize/data/pretrained.py` dynamically imports `M2M100Config`, `M2M100Encoder`, and `hf_hub_download` inside the body of `pretrained_transformer` only when `"SONAR"` is in `model_id`.

---

## 3. Verification Summary
- **AST Syntactic Invariance**: Verified across all 8 modified files + 1 new file using `ast_verify.py`. Comparing ASTs before and after with docstrings stripped confirms exact node-level equivalence. Zero code changes.
- **Python Syntax Compilation**: Verified clean compilation across all modified files using `python3 -m py_compile`.
- **Write Scope Compliance**: Verified via `git status` and `git diff --stat` that only the assigned files in `ontologize/training/` and `ontologize/data/` were modified.
