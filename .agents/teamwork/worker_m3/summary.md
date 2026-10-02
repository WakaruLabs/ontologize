# Milestone 3: Summary of Changes & Catalog of Observed Bugs / Oddities

## Summary of Changes

Milestone 3 covers adding comprehensive module, class, and public function/method docstrings to the following 8 files:
- `ontologize/fns/classify.py`
- `ontologize/fns/keys.py`
- `ontologize/fns/loss.py`
- `ontologize/inference/steerable.py`
- `ontologize/visualize/loss.py`
- `ontologize/chat.py`
- `ontologize/ontologizer.py`
- `ontologize/main.py`

### Hard Constraint Compliance
- **Docstring-only changes**: 100% of modifications were adding or enhancing docstrings.
- **Zero code edits**: Zero statements, expressions, signatures, parameter orders, or logic branches were altered.
- **Zero reformatting & zero renames**: All formatting and naming remained identical.
- **Zero type-hint changes**: No type annotations were modified.
- **Zero comment removal**: All inline and block comments in source files were strictly preserved.
- **Syntactic invariance**: Every file was verified with an AST stripper/comparator tool (`ast_verify.py`), confirming that AST nodes before and after the docstring updates match identically.
- **Compilation check**: Every file compiled cleanly under `python3 -m py_compile`.
- **Write boundary adherence**: Only the 8 assigned files were modified; no other files outside the assigned list were touched.

### Detailed Inventory of Docstring Enhancements

1. `ontologize/fns/classify.py`:
   - Added module-level docstring describing categorical probability functions and straight-through estimators.
   - Added docstring for `softmax_cl` documenting input/output shapes `(..., k)` and `axis=-1` reduction.
   - Enhanced docstring for `ste` explaining forward hard one-hot argmax and backward softmax gradient bypass.

2. `ontologize/fns/keys.py`:
   - Added module-level docstring explaining key-to-callable/key-to-type resolvers for JSON-serializable dataclasses.
   - Preserved existing header comments intact.
   - Added/expanded docstrings for `get_activation`, `get_dtype`, `get_loss`, `get_srctype`, and `get_noise`, listing all valid keys and default fallback callables.

3. `ontologize/fns/loss.py`:
   - Added module-level docstring summarizing similarity metrics, information criteria, noise generators, and ghost gradient implementations.
   - Preserved all inline engineering comments (batchnorm scaling rationale, featvar variance underflow notes, JAX cond tracing notes).
   - Added/expanded docstrings for all 16 functions: `identity`, `cossim`, `bcossim`, `entropy`, `l0`, `l1`, `l2`, `errorct`, `aic`, `bic`, `addnoise`, `addnoise_batchnorm`, `addnoise_featvar`, `l2_ghost`, `ghost`, and `ghostgrad`. Documented all mathematical equations and tensor shapes (`(d,)`, `(..., b, d)`, `(..., b, b)`, etc.).

4. `ontologize/inference/steerable.py`:
   - Added module-level docstring explaining checkpoint restoration and inference steering via `Ontologizer.withArgs`.
   - Added class docstring for `Steerable` documenting attributes (`hyper`, `meta`, `model`, `layer_args`, `state`).
   - Added docstrings for `Steerable.__init__` and `Steerable.__call__` detailing restoration parameters and forward return shapes.
   - Added class docstring for `ChatEnv` noting its role as an inference configuration container.

5. `ontologize/visualize/loss.py`:
   - Added module-level docstring explaining training metric extraction and plotting.
   - Added docstrings for `read_loss`, `plot_stat`, and `plot_loss`, detailing the 9 column names (`loss`, `MSE`, `MSE_ghost`, `L1_K`, `L1_F`, `entropy`, `cossim_b`, `cossim_h`, `KL_m`), logarithmic scaling behavior, and figure cleanup.

6. `ontologize/chat.py`:
   - Added module-level docstring detailing the interactive CLI REPL for ontofeature interventions.
   - Added class docstring for `ChatEnv` detailing attributes (`model`, `num_layers`, `interventions`).
   - Added docstrings for `__init__`, `_print_state`, `_parse_array`, and `interact` explaining commands (`set`, `clear`, `clear_all`, `run`, `help`, `quit`).

7. `ontologize/main.py`:
   - Added module-level docstring describing the entry-point stub.
   - Added docstring for `main()`.

8. `ontologize/ontologizer.py`:
   - Added module-level docstring detailing the multi-layer Ontologizer architecture, discrete dictionary lookup, residual vs label forwarding, deep supervision, and steering interventions.
   - Added class docstrings for `DictIntervention` and `OntologizerIntervention` specifying tensor shapes for all fields.
   - Added docstrings for `DictIntervention.show` and `OntologizerIntervention.show`.
   - Added extensive class docstring for `Ontologizer(nn.Module)` documenting all architectural hyperparameters.
   - Added or updated docstrings for all 18 methods: `dictenc`, `setup`, `fwd_dec`, `prefix`, `resid`, `encode`, `nextinput`, `classify`, `decode`, `__call__`, `withStats`, `withGhost`, `withArgs`, `decodeLayerEntries`, `decodeEntries`, `decodeLayerUnif`, `decodeUniform`, and `intervene`. Documented tensor dimensions, deep supervision prefixes, and adjoint reversal mechanics.

---

## Catalog of Observed Bugs and Oddities

Per instructions, the following issues were cataloged during inspection without modifying any underlying code:

1. **Duplicate Class Names (`ChatEnv`)**:
   - `ontologize/chat.py` defines `class ChatEnv`, an interactive terminal REPL.
   - `ontologize/inference/steerable.py` defines `@dataclass class ChatEnv`, a container holding `(model, hyper, meta)`.
   - *Consequence*: Potential naming collisions when importing from both modules.

2. **`Ontologizer.setup` Intervention Field Type Mismatch**:
   - Line 205 of `ontologize/ontologizer.py`: `self.interventions = [OntologizerIntervention(i, []) for i in range(self.l+1)]`.
   - The dataclass definition for `OntologizerIntervention` specifies `method: DictIntervention`. Initializing with `[]` (empty list) violates the dataclass field type contract.

3. **`Ontologizer.__call__` Unused `arglist` Argument**:
   - `Ontologizer.__call__` accepts `arglist: Optional[List[DictIntervention]] = None`, but does not pass `arglist` into `classify()` or `dictenc.withStats()`. Interventions passed to `__call__` are ignored; users must call `withArgs` to apply interventions.

4. **Return Shape Annotation Mismatch in `Steerable.__call__`**:
   - `Steerable.__call__` is annotated with return type `Tuple[..., Float[Array, "l 5"], ...]`.
   - However, `Ontologizer.withStats` returns a statistics tensor of shape `(l, 6)` (`[L1_K, L1_F, entropy, cossim_b, cossim_h, KL_m]`).

5. **Redundant In-Function Imports**:
   - In `ontologize/ontologizer.py`, line 384 (`withArgs`) has `import dataclasses` inside the layer loop.
   - Line 433 (`decodeLayerUnif`) has `import einops` inside the method, even though `import einops` is already imported at top level (line 7).

6. **Loss History Column Shift on Legacy Checkpoints**:
   - In `ontologize/visualize/loss.py`, `read_loss` assumes 9 columns.
   - As noted in file comments, checkpoints from older runs before `KL_m` was introduced have 8 columns, which results in `KL_m` reading as NaN.
