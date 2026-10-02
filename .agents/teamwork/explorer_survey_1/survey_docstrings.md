# Ontologize Package Comprehensive Docstring Survey

This document provides an exhaustive inventory and specification of all files, classes, public functions, and public methods across the `ontologize/` package. It serves as the authoritative blueprint for Workstream 1 (Ontologize Package Docstrings).

---

## 1. Executive Summary & Global Observations

### Package Architecture
- **Namespace Package**: `ontologize` functions as an implicit namespace package (PEP 420). There is **no root `ontologize/__init__.py`**, nor are there `__init__.py` files in `training/`, `fns/`, `inference/`, or `visualize/`. Only `data/__init__.py` and `layers/__init__.py` exist (both currently 0-byte empty files).
- **Technology Stack**:
  - Core deep learning: JAX / Flax Linen (`flax.linen as nn`, `flax.struct`).
  - Typing: `jaxtyping` (`Array`, `Float`, `Int`, `UInt`, `Bool`, `PRNGKeyArray`) and standard Python `typing`.
  - Data pipelines: Google `grain.python` (`gp.MapTransform`, `gp.DataLoader`, `gp.RandomAccessDataSource`).
  - Pretrained models & tokenization: HuggingFace `transformers` and PyTorch (`torch as t`) via DLPack (`jnp.from_dlpack`, `t.from_dlpack`).
  - Checkpointing: Orbax (`orbax.checkpoint as ocp`).
  - Reshaping: `einops` (`einops.rearrange`).
- **Global Docstring Baseline**:
  - **Module-level docstrings**: **0 / 22** active files have a module docstring (100% missing).
  - **Class docstrings**: Most classes have a 1-line or short summary docstring, but lack parameter/attribute specifications.
  - **Method / Function docstrings**: Highly uneven. Several key methods have concise summary docstrings; many utility and private/dunder methods have none. Crucially, **almost none document explicit tensor shapes, argument types, or return types in standard docstring format**.

---

## 2. Dead Modules (Strictly SKIP)

The following four modules located in `ontologize/layers/` are **dead code** and must be **completely skipped** during docstring passes. No changes should be made to them.

1. **`ontologize/layers/dictblock_enhanced.py`**
   - **Reason**: Imports from `ontologize.jax.layers.dictblock`, a nonexistent path from an old multi-backend repository layout. Contains a stub `InterpretableDictBlock(DictBlock)`.
2. **`ontologize/layers/dict_interpreter.py`**
   - **Reason**: Orphaned JAX/Flax exemplar generator importing `FlaxAutoModelForCausalLM`. Not imported anywhere in active execution paths.
3. **`ontologize/layers/integration_clean.py`**
   - **Reason**: Imports from nonexistent `ontologize.jax.layers.dictenc`, `ontologize.jax.layers.dictblock`, and `ontologize.jax.layers.dict_interpreter`.
4. **`ontologize/layers/vae_integration.py`**
   - **Reason**: Imports from nonexistent `ontologize.jax.layers.dictenc`. Contains `VAEEnhancedDictBlockPlaceholder` and `ExemplarAnalyzer`.

---

## 3. Subpackage Surveys

---

### Subpackage 1: `ontologize/layers/`

#### 1. `ontologize/layers/__init__.py`
- **Current Status**: Empty file (0 bytes).
- **Docstring Status**: Missing module docstring.
- **Contents**: No classes or functions.
- **Target Docstring**: Needs a module-level docstring describing the `ontologize.layers` package (custom neural network layers for dictionary learning and multihead ontofeature decomposition).

---

#### 2. `ontologize/layers/dictblock.py`
Multihead ontofeature submodule. Performs classification-weighted reconstruction from learned concept dictionaries.
- **Module Docstring**: Missing.
- **Class `DictBlock(Sparse)`**:
  - **Current Docstring**: Partial (short paragraph describing high-level concept).
  - **Attributes / Shapes**:
    - `k: int`: Number of dictionary entries (tags/concepts) per head.
    - `d: int`: Embedding / activation dimension.
    - `h: int`: Number of heads.
    - `select: str`: Selection activation key (default `"softmax"`).
    - `activation: str`: Post-combination activation (default `"none"`).
    - `sparse: bool`, `entropy_loss: bool`, `cossim_loss: bool`, `bcossim_loss: bool`, `hmean_loss: bool`: Loss gates.
    - `noise: str`, `sd: float`: Noise hyperparameters.
    - `dtype_str: str` (default `"bfloat16"`), `dtype_p_str: str` (default `"float32"`): Computation and parameter dtypes.
    - Parameters:
      - `weights`: Parameter tensor of shape `(h, k, d)`.
      - `n_tags = k * h`, `n_feat = d * h`.
  - **Methods**:
    1. `setup(self)`:
       - **Status**: Missing docstring.
       - **Purpose**: Initializes selection function and dictionary weight tensor of shape `(h, k, d)`.
    2. `dicts(self) -> Float[Array, "h k d"]`:
       - **Status**: Complete / Partial.
       - **Purpose**: Returns `abs(self.weights)` to enforce non-negative ontofeatures.
       - **Returns**: Dictionary tensor `(h, k, d)`.
    3. `cluster(self, K: Float[Array, "... h k"], temperature: float = 1.0) -> Float[Array, "... h k"]`:
       - **Status**: Partial.
       - **Purpose**: Applies `select` function (e.g. softmax) to scaled logits `K / temperature`.
       - **Arguments**: `K` shape `(..., h, k)`, `temperature` scalar.
       - **Returns**: Probability / weight array `(..., h, k)`.
    4. `fwd(self, P_0: Float[Array, "... h k"], S: Optional[Float[Array, "... h"]] = None, *args, **kwargs) -> Float[Array, "... d"]`:
       - **Status**: Partial.
       - **Purpose**: Forward pass applying interventions and contracting classifications `P` with dictionaries `(h, k, d)`, summing over head axis `h`.
       - **Arguments**: `P_0` shape `(..., h, k)`, `S` optional head scale `(..., h)`.
       - **Returns**: Reconstructed vector `(..., d)`.
    5. `hfwd(self, P_0: Float[Array, "... h k"], S: Optional[Float[Array, "... h"]] = None, *args, **kwargs) -> Float[Array, "... h d"]`:
       - **Status**: Partial.
       - **Purpose**: Forward pass without summing over head axis `h`.
       - **Arguments**: `P_0` shape `(..., h, k)`, `S` optional head scale `(..., h)`.
       - **Returns**: Per-head reconstruction tensor `(..., h, d)`.
    6. `combine(self, Y: Float[Array, "... h d"]) -> Float[Array, "... d"]`:
       - **Status**: Partial.
       - **Purpose**: Sums tensor along the head dimension `h`.
       - **Arguments**: `Y` shape `(..., h, d)`.
       - **Returns**: Combined tensor `(..., d)`.
    7. `__call__(self, K: Float[Array, "... h k"], S: Optional[Float[Array, "... h"]] = None, temperature: float = 1.0, *args, **kwargs) -> Float[Array, "... d"]`:
       - **Status**: Missing docstring.
       - **Purpose**: Full forward call: clusters logits `K`, computes `fwd`, and applies layer activation `fn`.
       - **Arguments**: `K` shape `(..., h, k)`, `S` optional scale `(..., h)`, `temperature` float.
       - **Returns**: Layer output `(..., d)`.
    8. `hrev(self, F: Float[Array, "... h d"]) -> Float[Array, "... h k"]`:
       - **Status**: Partial.
       - **Purpose**: Reverse pass mapping head activations `(..., h, d)` back to logit space `(..., h, k)` using transposed dictionaries.
    9. `rev(self, F: Float[Array, "... d"], S: Optional[Float[Array, "... h"]] = None, *args, **kwargs) -> Float[Array, "... h k"]`:
       - **Status**: Partial.
       - **Purpose**: Reverse pass mapping combined activation `(..., d)` to per-head logits `(..., h, k)`.
    10. `ghost(self, K: Float[Array, "... h k"], F: Float[Array, "... d"], S: Optional[Float[Array, "... h"]] = None, *args, **kwargs) -> Float[Array, "... d"]`:
        - **Status**: Partial.
        - **Purpose**: Computes ghost gradient for dead features mapped over heads.
        - **Returns**: Ghost output `(..., d)`.
    11. `tagGram(self) -> Float[Array, "h k k"]`:
        - **Status**: Partial.
        - **Purpose**: Computes per-head Gram matrices of the dictionary vectors `W_h @ W_h.T`.
        - **Returns**: Tensor of shape `(h, k, k)`.
    12. `bcossim_tags(self, P_0: Float[Array, "... h k"], S: Optional[Float[Array, "... h"]] = None) -> Float[Array, ""]`:
        - **Status**: Complete / Detailed.
        - **Purpose**: Computes batch cosine similarity in tag space via `tagGram` in O(h k^2 d + b^2 h k) time.
        - **Returns**: Scalar float array `()`.
    13. `hmean_kl(self, P_0: Float[Array, "... h k"]) -> Float[Array, ""]`:
        - **Status**: Complete / Detailed.
        - **Purpose**: Computes KL divergence `KL(batch-mean classification || uniform)` in bits, averaged over heads.
        - **Returns**: Scalar float array `()`.
    14. `withClusts(self, K: Float[Array, "... h k"], S: Optional[Float[Array, "... h"]] = None, temperature: float = 1.0, *args, **kwargs) -> Tuple[Float[Array, "... h d"], Float[Array, "... h k"]]`:
        - **Status**: Partial.
        - **Returns**: Tuple of `(Fs, P)` where `Fs` is `(..., h, d)` and `P` is `(..., h, k)`.
    15. `withEntropy(self, K: Float[Array, "... h k"], S: Optional[Float[Array, "... h"]] = None, *args, **kwargs) -> Tuple[Float[Array, "... h d"], Float[Array, "... h k"], Float[Array, ""]]`:
        - **Status**: Partial.
        - **Returns**: Tuple of `(Fs, P, H)` where `H` is scalar entropy.
    16. `withL1(self, Fs: Float[Array, "... h d"], isloss: bool = False) -> Tuple[Float[Array, "... d"], Float[Array, ""]]`:
        - **Status**: Partial.
        - **Returns**: Tuple of `(combined_F, l1_norm)`.
    17. `drop_winners(self, K: Float[Array, "... h k"], p_drop: float = 0.0, rng: Optional[PRNGKeyArray] = None) -> Tuple[Float[Array, "... h k"], Optional[PRNGKeyArray]]`:
        - **Status**: Complete.
        - **Purpose**: Winner dropout: masks argmax logit to `-inf` with probability `p_drop` per head per sample.
    18. `withStats(self, K: Float[Array, "... b h k"], S: Optional[Float[Array, "... b h"]] = None, sd: float = 0.0, rng: Optional[PRNGKeyArray] = None, *args, p_drop: float = 0.0, **kwargs) -> Tuple[Float[Array, "... b d"], Float[Array, "... b h k"], Float[Array, "5"], PRNGKeyArray]`:
        - **Status**: Partial.
        - **Returns**: `(F, P, stats, rng_next)` where `stats` is length-5 float array `[L1, H, cossim_b, cossim_h, KL_m]`.
    19. `tags(self) -> Float[Array, "n_tags d"]`:
        - **Status**: Partial.
        - **Purpose**: Flattens `dicts()` to shape `(h * k, d)`.
    20. `uniform(self, b: int = 1) -> Float[Array, "b h k"]`:
        - **Status**: Partial (Note typo in return shape annotation in code: `"... b"` instead of `"b h k"`).
        - **Purpose**: Returns synthetic uniform classifications with values `1 / k`.
    21. `hmask(self, P: Float[Array, "... h k"], heads: UInt[Array, "n"]) -> Float[Array, "... h k"]`:
        - **Status**: Partial. Creates mask tensor active only at specified `heads`.
    22. `kmask(self, P: Float[Array, "... h k"], tags: UInt[Array, "n"]) -> Float[Array, "... h k"]`:
        - **Status**: Partial. Creates mask tensor active only at specified `tags`.
    23. `mask(self, P: Float[Array, "... h k"], heads: UInt[Array, "n_head"], tags: UInt[Array, "n_tag"]) -> Float[Array, "... h k"]`:
        - **Status**: Partial. Creates mask tensor active at `[..., heads, tags]`.
    24. `uniformAblate(self, P_0: Float[Array, "... h k"], heads: Optional[UInt[Array, "n_head"]] = None) -> Float[Array, "... h k"]`:
        - **Status**: Partial. Sets specified heads in `P_0` to `1 / k`.
    25. `zeroAblate(self, P_0: Float[Array, "... h k"], heads: Optional[UInt[Array, "n_head"]] = None) -> Float[Array, "... h k"]`:
        - **Status**: Partial. Sets specified heads in `P_0` to 0.
    26. `set(self, P_0: Float[Array, "... h k"], heads: UInt[Array, "n_head"], tags: UInt[Array, "n_head"]) -> Float[Array, "... h k"]`:
        - **Status**: Partial. Forces `tags` to 1 and non-tags to 0 on specified `heads`.
    27. `uniformTags(self) -> Float[Array, "n_tags h k"]`:
        - **Status**: Partial (Note: actual return shape is `(1 + h * k, h, k)` = `(1 + n_tags, h, k)`).
        - **Purpose**: Generates base uniform plus one-hot variation for every head-tag pair.
    28. `intervene(self, P_0: Float[Array, "... h k"], scale = None, k_set = None, h_set = None, h_unif = None, h_zero = None, k_add = None, h_add = None, k_sub = None, h_sub = None) -> Float[Array, "... h k"]`:
        - **Status**: Partial. Dispatches causal intervention sequence on `P_0`.

---

#### 3. `ontologize/layers/dictenc.py`
Dictionary Encoder composed of a classifier (`BilinearBlock`/`NLinearBlock`), `DictBlock`, and optional scaling module.
- **Module Docstring**: Missing.
- **Class `DictEnc(nn.Module)`**:
  - **Current Docstring**: Partial.
  - **Attributes / Shapes**:
    - `d_in: int`, `d_out: int`, `k: int`, `h: int`.
    - Submodules: `classifier` mapping `(..., d_in) -> (..., h, k)`, `dict` mapping `(..., h, k) -> (..., d_out)`, optional `scaling` mapping `(..., d_in) -> (..., h)`.
  - **Methods**:
    1. `setup(self)`: Missing docstring.
    2. `fwd_dict(self, P: Float[Array, "... h k"], *args, **kwargs) -> Float[Array, "... d_out"]`: Partial. (NOTE: Contains bug referencing nonexistent `self.fwd_dec`).
    3. `fwd(self, E: Float[Array, "... d_in"], *args, **kwargs) -> Float[Array, "... d_out"]`: Partial.
    4. `rev_dict(self, Y: Float[Array, "... d_out"], *args, **kwargs) -> Float[Array, "... h k"]`: Partial.
    5. `rev(self, Y: Float[Array, "... d_out"], *args, **kwargs) -> Float[Array, "... d_in"]`: Partial.
    6. `classify(self, E: Float[Array, "... d_in"], *args, **kwargs) -> Float[Array, "... h k"]`: Partial.
    7. `scale(self, E: Float[Array, "... d_in"]) -> Float[Array, "... h"]`: Partial.
    8. `__call__(self, E: Float[Array, "... d_in"], *args, **kwargs) -> Float[Array, "... d_out"]`: Partial.
    9. `tags(self) -> Float[Array, "n_tags d_out"]`: Partial. (NOTE: typo in docstring: "sel.dict.dicts()").
    10. `decodeUniform(self, *args, **kwargs) -> Float[Array, "n_tags d_out"]`: Partial.
    11. `withClusts(self, R: Float[Array, "... d_out"], E: Float[Array, "... d_in"], *args, **kwargs) -> Tuple[Float[Array, "... d_out"], Float[Array, "... n_tags"]]`: Partial.
    12. `withStats(self, R: Float[Array, "... d_out"], E: Float[Array, "... d_in"], sd_K: float = 0.0, sd_F: float = 0.0, rng: Optional[PRNGKeyArray] = None, *args, p_drop: float = 0.0, **kwargs) -> Tuple[Float[Array, "... d_out"], Float[Array, "... (h k)"], Float[Array, "6"], PRNGKeyArray]`: Partial.
        - Stats vector shape `(6,)`: `[L1_K, L1_F, entropy, cossim_batch, cossim_heads, KL_mean]`.
    13. `withGhost(self, R: Float[Array, "... d_out"], R_g: Float[Array, "... d_out"], E: Float[Array, "... d_in"], E_g: Optional[Float[Array, "... d_in"]], temperature: float = 1.0, sd_K: float = 0.0, sd_F: float = 0.0, rng: Optional[PRNGKeyArray] = None, *args, p_drop: float = 0.0, **kwargs) -> Tuple[...]`: Partial.
    14. `intervene(self, E: Float[Array, "... d_in"], *args, temperature: float = 1.0, **kwargs) -> Tuple[Float[Array, "... d_out"], Float[Array, "... (h k)"], Float[Array, "... d_in"]]`: Complete.

---

#### 4. `ontologize/layers/linear.py`
Linear layer with optional bias and activation function, supporting `fwd` and `rev` passes and multi-layer ghost gradients.
- **Module Docstring**: Missing.
- **Class `Linear(Sparse)`**:
  - **Current Docstring**: Partial.
  - **Attributes / Shapes**:
    - `d_in: int`, `d_out: int`, `biased: bool`.
    - Weights: `(d_out, d_in)`, Bias: `(d_out,)`.
  - **Methods**:
    1. `setup(self)`: Complete.
    2. `fwd(self, X: Float[Array, "... d_in"]) -> Float[Array, "... d_out"]`: Partial. Unbiased matrix multiplication `X @ W.T`.
    3. `rev(self, Y: Float[Array, "... d_out"]) -> Float[Array, "... d_in"]`: Partial. Reverse projection `Y @ W`.
    4. `__call__(self, X: Float[Array, "... d_in"]) -> Float[Array, "... d_out"]`: Missing docstring.
    5. `ghost(self, X: Float[Array, "... d_in"], Y: Float[Array, "... d_out"], lbound: int = -10.0, ubound: int = 10.0) -> Float[Array, "... d_out"]`: Missing docstring (NOTE: lbound/ubound typed as `int` in code but defaulted to floats).

---

#### 5. `ontologize/layers/nlinear.py`
Multi-linear and bilinear layers and blocks for higher-order interactions and multihead classification.
- **Module Docstring**: Missing.
- **Class `NLinear(Sparse)`**:
  - **Docstring**: Partial.
  - **Attributes / Shapes**: `d_in: int`, `d_out: int`, `n: int` (order, default 2), `weight: (n, d_out, d_in)`, `bias: (d_out,)`.
  - **Methods**:
    1. `setup(self)`: Missing docstring.
    2. `nfwd(self, X: Float[Array, "... d_in"]) -> Float[Array, "n ... d_out"]`: Missing docstring.
    3. `fwd(self, X: Float[Array, "... d_in"]) -> Float[Array, "... d_out"]`: Missing docstring.
    4. `__call__(self, X: Float[Array, "... d_in"]) -> Float[Array, "... d_out"]`: Missing docstring.
    5. `weight_ubind(self) -> Tuple[Float[Array, "d_out d_in"], ...]`: Partial.
    6. `ghost(self, X: Float[Array, "... d_in"], Y: Float[Array, "... d_out"], *args, **kwargs) -> Float[Array, "... d_out"]`: Partial.
- **Class `Bilinear(NLinear)`**:
  - **Docstring**: Partial (Pearce et al. 2025). Forces `n = 2`.
  - **Methods**:
    1. `setup(self)`: Missing docstring.
    2. `bilinearTensor(self) -> Float[Array, "d_out d_in d_in"]`: Partial. Symmetrized bilinear tensor `0.5 * (B + B.T)`.
    3. `interactionMat(self, Y_0: Float[Array, "... d_out"]) -> Float[Array, "... d_in d_in"]`: Partial.
    4. `jacobian(self, X: Float[Array, "... d_in"]) -> Float[Array, "... d_out d_in"]`: Missing docstring.
    5. `decompose(self) -> Tuple[Float[Array, "d_out d_in"], Float[Array, "d_out d_in d_in"]]`: Partial. Eigendecomposition `(vals, vecs)`.
    6. `rev(self, Y: Float[Array, "... d_out"]) -> Float[Array, "... d_in"]`: Partial. Top-1 eigenvector adjoint reconstruction.
    7. `project(self, Y_0: Float[Array, "... d_out"]) -> Tuple[...]`: Partial.
- **Class `NLinearBlock(Sparse)`**:
  - **Docstring**: Partial. Multihead version of `NLinear`.
  - **Attributes / Shapes**: `weight: (n, h, d_out, d_in)`, `bias: (h, d_out)`.
  - **Methods**:
    1. `setup(self)`: Missing docstring.
    2. `nfwd(self, X: Float[Array, "... d_in"]) -> Float[Array, "n ... h d_out"]`: Partial.
    3. `fwd(self, X: Float[Array, "... d_in"]) -> Float[Array, "... h d_out"]`: Partial.
    4. `__call__(self, X: Float[Array, "... d_in"]) -> Float[Array, "... h d_out"]`: Missing docstring.
    5. `weight_ubind(self) -> Tuple[Float[Array, "h d_out d_in"], ...]`: Missing docstring.
    6. `ghost(self, X: Float[Array, "... d_in"], Y: Float[Array, "... h d_out"], *args, **kwargs) -> Float[Array, "... d_out"]`: Partial.
    7. `rev(self, Y: Float[Array, "... h d_out"]) -> Float[Array, "... d_in"]`: Partial. (NOTE: only valid for `n=1`, raises `NotImplementedError` if `n != 1`).
- **Class `BilinearBlock(NLinearBlock)`**:
  - **Docstring**: Partial. Multihead version of `Bilinear` (`n = 2`).
  - **Methods**:
    1. `setup(self)`: Missing docstring.
    2. `bilinearTensor(self) -> Float[Array, "h d_out d_in d_in"]`: Partial.
    3. `jacobian(self, x: Float[Array, "... d_in"]) -> Float[Array, "... h d_out d_in"]`: Partial.
    4. `interactionMat(self, Y_0: Float[Array, "... h d_out"]) -> Float[Array, "... h d_in d_in"]`: Partial.
    5. `decompose(self) -> Tuple[Float[Array, "h d_out d_in"], Float[Array, "h d_out d_in d_in"]]`: Partial.
    6. `project(self, Y_0: Float[Array, "... h d_out"]) -> Tuple[...]`: Partial.
    7. `rev(self, Y: Float[Array, "... h d_out"]) -> Float[Array, "... d_in"]`: Partial. Reversal aggregated across heads.

---

#### 6. `ontologize/layers/sparse.py`
Superclass providing methods to conditionally stop gradients when computing sparsity statistics, activation gating, and noise addition.
- **Module Docstring**: Missing.
- **Class `Sparse(nn.Module)`**:
  - **Docstring**: Partial.
  - **Attributes**: `activation: str`, `sparse: bool`, `entropy_loss: bool`, `cossim_loss: bool`, `bcossim_loss: bool`, `noise: str`, `sd: float`, `dtype_str: str`, `dtype_p_str: str`.
  - **Methods**:
    1. `setup(self)`: Missing docstring.
    2. `l1(self, F: Float[Array, "..."]) -> Float[Array, ""]`: Partial. Conditionally stop-gradients L1 norm.
    3. `cossim(self, F: Float[Array, "... d"], S = None, isloss = None) -> Float[Array, ""]`: Partial.
    4. `bcossim(self, Fs: Float[Array, "... h d"], *args, **kwargs) -> Float[Array, ""]`: Missing docstring.
    5. `entropy(self, K: Float[Array, "... d"], S = None) -> Float[Array, ""]`: Missing docstring.
    6. `addnoise(self, X: Float[Array, "..."], sd: float = 0.0, rng = None) -> Tuple[Float[Array, "..."], Optional[PRNGKeyArray]]`: Missing docstring.
    7. `addbias(self, Y: Float[Array, "... d"], bias = None) -> Float[Array, "... d"]`: Missing docstring.
    8. `haddbias(self, Y: Float[Array, "... h d"], bias = None) -> Float[Array, "... h d"]`: Missing docstring.
    9. `ghost(self, X: Float[Array, "..."], Y: Float[Array, "..."], *args, **kwargs) -> Optional[Float[Array, "..."]]`: Missing docstring (returns `None`).

---

### Subpackage 2: `ontologize/training/`

#### 1. `ontologize/training/config.py`
Dataclasses and configuration helpers for hyperparameters, dataset metadata, checkpointing, and execution environments.
- **Module Docstring**: Missing.
- **Function `_mse_weights(path: str) -> jnp.ndarray`**:
  - **Status**: Partial. Caches per-dimension MSE weights `(d_out,)`.
- **Class `Hyperparams`**:
  - **Status**: Partial class docstring.
  - **Key Fields**: `d_in`, `d_out`, `b`, `epochs`, `lr`, `wd`, `temperature`, `p_drop`, `n_stats`, `lossfn`, `mse_weights`, loss scale factors (`s_g`, `s_L1K`, `s_L1F`, `s_H`, `s_bcossim`, `s_hcossim`, `s_Hm`), annealing schedules (`temperature_end`, `anneal_steps`, `p_drop_start`, `sd_K_end`), `ghost`.
  - **Methods**:
    1. `s_loss(self, *args, **kwargs) -> Tuple[Float[Array, "7"], Tuple[bool, ...]]`: Partial.
    2. `rng(self) -> PRNGKeyArray`: Partial.
    3. `loss(self, X: Float[Array, "... b d_out"], Y: Float[Array, "... b d_out"], X_g: Float[Array, "... b d_out"], stats: Float[Array, "4"]) -> Tuple[Float[Array, ""], Float[Array, "n_stats"]]`: Complete.
    4. `opt(self, *args, **kwargs) -> optax.GradientTransformation`: Partial. Adam or AdamW.
    5. `ontologizer(self, *args, **kwargs) -> Ontologizer`: Missing docstring.
    6. `init(self, model: Ontologizer, save_each: int = 1000) -> OntoState`: Partial.
    7. `update(self, state: OntoState, rng, X, Y, *args, **kwargs) -> OntoState`: Partial. (NOTE: unused method).
    8. `train(self, state: OntoState, dat: SampleLoader, *args, **kwargs) -> OntoState`: Partial.
    9. `load(self, manager: ocp.CheckpointManager, step: int) -> OntoState`: Partial.
- **Class `Metadata`**:
  - **Status**: Partial class docstring.
  - **Key Fields**: `srctype`, `model_id`, `sae_id`, `layer`, `out_path`, `data_path`, `threads`, `checkpoint_each`, `save_each`, `max_to_keep`, `resume_from`, `overwrite`, `test_ratio`, `shuffle`.
  - **Methods**:
    1. `manager(self, *args, **kwargs) -> ocp.CheckpointManager`: Partial.
    2. `src(self, *args, **kwargs) -> gp.ArrayRecordDataSource`: Partial.
    3. `loader(self, *args, **kwargs) -> SampleLoader`: Partial.
    4. `model(self, *args, **kwargs) -> Tuple[Any, Any]`: Partial.
- **Function `log_env(log_path, hyper: Hyperparams, meta: Metadata)`**:
  - **Status**: Partial. Logs configuration as JSONL.
- **Class `TrainingEnv`**:
  - **Status**: Partial class docstring.
  - **Key Fields**: `spec: Ontologizer`, `hyper: Hyperparams`, `meta: Metadata`, and argument lists/dicts for loss, loader, manager.
  - **Methods**:
    1. `init(self, src, keep_spec: bool = False) -> Tuple[OntoState, SampleLoader, ocp.CheckpointManager]`: Complete.
    2. `train(self, src, encoder = None, decoder = None) -> OntoState`: Complete.

---

#### 2. `ontologize/training/ontostate.py`
Flax `TrainState` extension managing training state, stats accumulators, scheduled annealing, and optimization loops.
- **Module Docstring**: Missing.
- **Class `OntoState(TrainState)`**:
  - **Status**: Partial.
  - **Fields**: `model: Ontologizer`, `b: int`, `save_each: int`, `n_stats: int`, `noise_in`, `noise_K`, `noise_F`, `stats: (save_each, n_stats)`.
  - **Methods**:
    1. `spec(self) -> dict`: Partial. Serializes model configuration.
    2. `save(self, manager: ocp.CheckpointManager) -> bool`: Partial.
    3. `newstats(self) -> Float[Array, "save_each n_stats"]`: Partial. Allocates zeroed stats buffer.
    4. `writestats(self, manager: ocp.CheckpointManager, file: str = "loss.csv") -> bool`: Partial. Appends stats to CSV.
- **Functions**:
  1. `state_init(model: Ontologizer, b: int, tx, rng: PRNGKeyArray, save_each: int = 1000, n_stats: int = 9, step: int = 0, ghost: bool = True) -> OntoState`: Complete.
  2. `stats_insert(state: OntoState, s: Float[Array, "n_stats"]) -> OntoState`: Partial. Inserts row into stats buffer.
  3. `load_params(state: OntoState, manager: ocp.CheckpointManager, step) -> OntoState`: Complete.
  4. `update(state: OntoState, lossfn: Callable, rng: PRNGKeyArray, X: Float[Array, "... d_in"], Y_0: Float[Array, "... d_out"], grad_clip: Optional[float] = None, *args, **kwargs) -> Tuple[OntoState, Float[Array, ""], Float[Array, "n_stats"], PRNGKeyArray]`: Complete. JIT-compiled single step.
  5. `schedules(step: int, anneal_steps: int, temperature: float, temperature_end: Optional[float], p_drop: float, p_drop_start: Optional[float], sd_K: float, sd_K_end: Optional[float]) -> Tuple[float, float, float]`: Complete. Computes annealed `(temperature, p_drop, sd_K)`.
  6. `train(state: OntoState, dat, lossfn: Callable, rng: PRNGKeyArray, epochs: int, manager: ocp.CheckpointManager, save_each: int = 0, encoder = None, decoder: Callable = None, dev = ..., ...) -> OntoState`: Complete. Outer training loop with tqdm bar and periodic checkpointing.

---

#### 3. `ontologize/training/serialize.py`
Flax model serialization and restoration utilities using Orbax.
- **Module Docstring**: Missing.
- **Functions**:
  1. `save_model(checkpoint_dir: str, step: int, params_pytree: Any, model: nn.Module) -> None`:
     - **Status**: Complete docstring with Args.
  2. `load_model(checkpoint_dir: str, step: int, model_class: type[nn.Module], dummy_input: Any = None) -> Tuple[Any, nn.Module]`:
     - **Status**: Complete docstring with Args and Returns.

---

### Subpackage 3: `ontologize/data/`

#### 1. `ontologize/data/__init__.py`
- **Current Status**: Empty file (0 bytes).
- **Docstring Status**: Missing module docstring.
- **Contents**: No classes or functions.

---

#### 2. `ontologize/data/langs.py`
Language mappings for multilingual datasets.
- **Module Docstring**: Missing (has a comment on lines 1-2).
- **Contents**: Constant dictionary `MC4_TO_SONAR`: 90-entry mapping between HuggingFace mC4 / C4 dataset split names (ISO 639) and SONAR (NLLB) BCP-47 language codes.
- **Docstring Target**: Needs a module-level docstring explaining the ISO 639 to BCP-47 language dictionary.

---

#### 3. `ontologize/data/loaders.py`
Grain-based dataset sources, transforms, and pipeline loaders.
- **Module Docstring**: Missing.
- **Classes**:
  1. `TokenizeTransform(gp.MapTransform)`:
     - Class docstring: Partial.
     - `__init__(self, tokenizer: Callable, maxlen: int = 512)`: Missing docstring.
     - `map(self, item: Dict[str, str]) -> Dict[str, Int[np.ndarray, "maxlen"]]`: Missing docstring.
  2. `FlattenTransform(gp.MapTransform)`:
     - Class docstring: Partial.
     - `__init__(self, height: int, width: int)`: Missing docstring.
     - `map(self, x: Any) -> np.ndarray`: Missing docstring. Flattens `(..., h, w)` to `(..., h*w)` and scales to `[0, 1]`.
  3. `DecodeTransform(gp.MapTransform)`:
     - Class docstring: Partial.
     - `map(self, x: bytes | str) -> str`: Missing docstring. Decodes bytes to utf-8.
  4. `HFDataSource`:
     - Class docstring: Partial.
     - `__new__(cls, dataset, text_key: str = "text", lang_key: str = "lang")`: Missing docstring. Dispatches to `_HFRandomAccess` or `_HFIterable`.
  5. `NpyDataSource(gp.RandomAccessDataSource)`:
     - Class docstring: Complete.
     - `__init__(self, file_path: str)`: Missing docstring.
     - `__len__(self) -> int`: Missing docstring.
     - `__getitem__(self, idx: int) -> np.ndarray`: Missing docstring.
     - `__getstate__(self) -> dict`: Missing docstring.
  6. `JSONLDataSource(gp.RandomAccessDataSource)`:
     - Class docstring: Partial.
     - `__init__(self, file_path: str, text_key: str = "text")`: Missing docstring. Indexes byte line offsets for O(1) random access.
     - `__len__(self) -> int`: Missing docstring.
     - `__getitem__(self, index: int) -> str`: Missing docstring.
  7. `SampleLoader`:
     - Class docstring: Partial. Generic loader wrapping Grain `DataLoader` or manual generator.
     - `__init__(self, b: int, epochs: int, d: int, src, operations, threads: int = 0, shuffle: bool = False, seed: int = 42, drop_remainder: bool = True)`: Missing docstring.
     - `__iter__(self) -> Iterator[Dict[str, Int[np.ndarray, "b d"]]]`: Missing docstring.
  8. `EmbeddingLoader(SampleLoader)`:
     - Class docstring: Partial. Loader for `.npy` embeddings.
     - `__init__(self, b: int, epochs: int, src, d: int = 0, threads: int = 0, shuffle: bool = False, seed: int = 42, drop_remainder: bool = True)`: Missing docstring.
  9. `ImageLoader(SampleLoader)`:
     - Class docstring: Partial. Applies `FlattenTransform`.
     - `__init__(self, b: int, epochs: int, src, height: int, width: int, threads: int = 0, shuffle: bool = False, seed: int = 42, drop_remainder: bool = True)`: Missing docstring.
  10. `TextLoader(SampleLoader)`:
      - Class docstring: Partial. Applies `DecodeTransform` and `TokenizeTransform`.
      - `__init__(self, b: int, epochs: int, src, tokenizer: Callable, threads: int = 0, shuffle: bool = False, maxlen: int = 512, seed: int = 42, drop_remainder: bool = True)`: Missing docstring.

---

#### 4. `ontologize/data/multilingual.py`
HuggingFace multilingual dataset loader utilities.
- **Module Docstring**: Missing.
- **Functions**:
  1. `load_lang(src: str, lang: str, *args, **kwargs)`:
     - **Status**: Missing docstring.
     - **Purpose**: Loads single language split of HuggingFace dataset, keeps only `"text"`, adds `"lang"` tag.
     - **Returns**: HuggingFace `Dataset`.
  2. `load_langs(src: str, langs: dict, *args, **kwargs)`:
     - **Status**: Missing docstring.
     - **Purpose**: Loads and interleaves datasets for multiple languages.
     - **Returns**: Interleaved dataset.
  3. `mc4_data(src: str, *args, **kwargs)`:
     - **Status**: Missing docstring.
     - **Purpose**: Calls `load_langs` with default `MC4_TO_SONAR` mapping.

---

#### 5. `ontologize/data/pretrained.py`
Pretrained transformer model integration (PyTorch HuggingFace models, tokenization, pooling, and DLPack bridge).
- **Module Docstring**: Missing.
- **Functions**:
  1. `get_torch_dtype(name: str) -> torch.dtype`:
     - **Status**: Partial. Maps string (`"bfloat16"`, `"float32"`) to `torch.dtype`.
  2. `pretrained_transformer(model_id: str, dtype_str: str = "bfloat16", dev = torch.device("cuda")) -> Tuple[Any, Any]`:
     - **Status**: Partial. Loads model and tokenizer, supporting SONAR (M2M100Encoder) or general AutoModel.
     - **Returns**: `(encoder, tokenizer)`.
  3. `tokenize(tokenizer, batch, *args, **kwargs)`:
     - **Status**: Missing docstring. Returns PyTorch tokenized batch.
  4. `l2_pooling(E: Float[Array, "batch seq_len d_model"], attention_mask: Int[Array, "batch seq_len"]) -> Float[Array, "batch d_model"]`:
     - **Status**: Complete docstring. Masked mean pooling followed by L2 normalization (unit-normed sentence embeddings).
  5. `encode(model, inputs, dev = torch.device("cuda")) -> Float[Array, "batch d_model"]`:
     - **Status**: Complete docstring. Runs PyTorch model and bridges to JAX via DLPack.
  6. `decode(model, outputs, E: Float[Array, "... b d"], *args, **kwargs) -> torch.Tensor`:
     - **Status**: Partial. Reconstructs text sequences via model's `generate()` method.

---

### Subpackage 4: `ontologize/fns/`

#### 1. `ontologize/fns/classify.py`
Classification functions and straight-through estimators for discrete activations.
- **Module Docstring**: Missing.
- **Functions**:
  1. `softmax_cl(x: Float[Array, "... k"], *args, **kwargs) -> Float[Array, "... k"]`:
     - **Status**: Missing docstring.
     - **Purpose**: Standard softmax over the trailing axis `-1`.
  2. `ste(x: Float[Array, "... k"], *args, **kwargs) -> Float[Array, "... k"]`:
     - **Status**: Complete docstring. Straight-through estimator: `argmax` on forward pass, `softmax` on backward pass.

---

#### 2. `ontologize/fns/keys.py`
Key-to-callable/type resolvers used for JSON-serializable dataclass configs.
- **Module Docstring**: Missing (has file header comment).
- **Functions**:
  1. `get_activation(name: str) -> Callable`:
     - **Status**: Partial. Maps `"none"`, `"relu"`, `"gelu"`, `"silu"`, `"tanh"`, `"sigmoid"`, `"softmax"`, `"argmax"`, `"ste"`.
  2. `get_dtype(name: str) -> Any`:
     - **Status**: Partial. Maps `"float32"`, `"float16"`, `"bfloat16"`, `"float64"` to `jax.numpy` dtypes.
  3. `get_loss(fn: str) -> Callable`:
     - **Status**: Partial. Maps `"mse"`, `"crossentropy"`, `"crossentropy_int"`, `"binary_crossentropy"`.
  4. `get_srctype(name: str) -> Callable`:
     - **Status**: Partial. Maps `"embedding"`, `"image"`, `"text"` to `SampleLoader` subclasses.
  5. `get_noise(name: str) -> Callable`:
     - **Status**: Partial. Maps `"none"`, `"standard"`, `"normal"`, `"gaussian"`, `"batchnorm"`, `"featvar"`.

---

#### 3. `ontologize/fns/loss.py`
Loss metrics, similarity functions, information criteria, noise generators, and ghost gradient implementations.
- **Module Docstring**: Missing.
- **Functions**:
  1. `identity(x: T, *args, **kwargs) -> T`: Missing docstring. Passthrough identity function.
  2. `cossim(x: Float[Array, "d"], y: Float[Array, "d"]) -> Float[Array, ""]`: Partial. 1D cosine similarity.
  3. `bcossim(X: Float[Array, "... b d"]) -> Float[Array, "... b b"]`: Partial. Batch cosine similarity matrix over sample dimension `-2`.
  4. `entropy(X: Float[Array, "... d"]) -> Float[Array, "..."]`: Partial. Shannon entropy in bits (base 2).
  5. `l0(x: Float[Array, "..."], reduction: Callable = jnp.sum) -> Float[Array, ""]`: Complete. Count of non-zero entries.
  6. `l1(x: Float[Array, "..."], reduction: Callable = jnp.sum) -> Float[Array, ""]`: Complete. Sum of absolute values.
  7. `l2(x: Float[Array, "..."], y: Float[Array, "..."], reduction: Callable = jnp.mean) -> Float[Array, ""]`: Complete. Mean squared error.
  8. `errorct(pred: Float[Array, "... d"], target: UInt[Array, "..."]) -> UInt[Array, ""]`: Missing docstring. Argmax misclassification count.
  9. `aic(k, L) -> Float`: Complete. Akaike information criterion `2k - 2ln(L)`.
  10. `bic(k, n, L) -> Float`: Complete. Bayesian information criterion `k*ln(n) - 2ln(L)`.
  11. `addnoise(x: Float[Array, "..."], rng: PRNGKeyArray, stddev: float = 0.0) -> Float[Array, "..."]`: Complete. Gaussian noise.
  12. `addnoise_batchnorm(x: Float[Array, "... d"], rng: PRNGKeyArray, stddev: float = 0.0) -> Float[Array, "... d"]`: Complete. Noise scaled by per-sample norm.
  13. `addnoise_featvar(x: Float[Array, "b ..."], rng: PRNGKeyArray, stddev: float = 0.0) -> Float[Array, "b ..."]`: Complete. Noise scaled by batch feature standard deviation.
  14. `l2_ghost(W: Float[Array, "d_in d_out"], X: Float[Array, "... b d_in"], Y: Float[Array, "... b d_out"], Yhat: Float[Array, "... b d_out"]) -> Float[Array, ""]`: Complete. Ghost gradient MSE loss for dead features.
  15. `ghost(X: Float[Array, "... b d"], isdead: Bool[Array, "d"], lbound: float = -10.0, ubound: float = 10.0) -> Float[Array, "... b d"]`: Complete. Clips dead features and applies exponential activation.
  16. `ghostgrad(W: Float[Array, "d_out d_in"], X: Float[Array, "... d_in"], Y: Float[Array, "... d_out"], lbound: float = -10.0, ubound: float = 10.0) -> Float[Array, "... d_out"]`: Complete. Computes `exp(W @ X)` for features that are all-zero in `Y`.

---

### Subpackage 5: `ontologize/inference/`

#### 1. `ontologize/inference/steerable.py`
Inference-time steering wrapper for pretrained `Ontologizer` models.
- **Module Docstring**: Missing.
- **Classes**:
  1. `Steerable`:
     - Class docstring: Complete.
     - `__init__(self, hyper: Hyperparams, meta: Metadata, *args, **kwargs)`: Missing docstring. Restores checkpoint, instantiates model, initializes per-layer `DictIntervention` list.
     - `__call__(self, X: Float[Array, "... d_in"], *args, **kwargs) -> Tuple[Float[Array, "... d_out"], Float[Array, "... (h k)"], Float[Array, "l 5"], Optional[PRNGKeyArray]]`: Complete. Runs forward pass applying layer interventions via `Ontologizer.withArgs`.
  2. `ChatEnv`:
     - Class docstring: Missing.
     - Dataclass fields: `model: Steerable`, `hyper: Hyperparams`, `meta: Metadata`. (NOTE: Naming collision with interactive CLI `ChatEnv` in `ontologize/chat.py`).

---

### Subpackage 6: `ontologize/visualize/`

#### 1. `ontologize/visualize/loss.py`
Training loss parsing and visualization via matplotlib.
- **Module Docstring**: Missing.
- **Functions**:
  1. `read_loss(path: str | Path) -> pd.DataFrame`:
     - **Status**: Missing docstring.
     - **Purpose**: Reads `loss.csv` from checkpoint directory, mapping 9 column names (`loss`, `MSE`, `MSE_ghost`, `L1_K`, `L1_F`, `entropy`, `cossim_b`, `cossim_h`, `KL_m`).
  2. `plot_stat(y: Sequence[float], path: Path, stat: str, fmt: str = 'pdf', base: Optional[float] = None, *args, **kwargs) -> None`:
     - **Status**: Missing docstring.
     - **Purpose**: Plots a single training statistic curve vs batch index and saves to file.
  3. `plot_loss(path: str | Path, fmt: str = "pdf", base: Optional[float] = None, *args, **kwargs) -> None`:
     - **Status**: Missing docstring.
     - **Purpose**: Iterates over all columns in `loss.csv` and outputs individual plots.

---

### Package Root Files

#### 1. `ontologize/chat.py`
Interactive terminal command environment for dynamically inspecting and editing ontofeature interventions on an `Ontologizer`.
- **Module Docstring**: Missing.
- **Class `ChatEnv`**:
  - Class docstring: Complete.
  - Methods:
    1. `__init__(self, model: Ontologizer)`: Missing docstring. Initializes `DictIntervention` for each of `model.l` layers.
    2. `_print_state(self) -> None`: Missing docstring. Prints active interventions across all layers.
    3. `_parse_array(self, val_str: str, dtype: type) -> Optional[jnp.ndarray]`: Missing docstring. Parses comma-separated number strings into JAX arrays.
    4. `interact(self) -> Tuple[Optional[str], Optional[List[DictIntervention]]]`: Complete. Main REPL loop accepting `set`, `clear`, `clear_all`, `run`, `help`, `quit`.

---

#### 2. `ontologize/ontologizer.py`
Core Multilayer Ontologizer model architecture, intervention dataclasses, and whole-network intervention/decoding methods.
- **Module Docstring**: Missing.
- **Class `DictIntervention`**:
  - Class docstring: Complete.
  - Fields: `scale: (h,)`, `k_set: (n,)`, `h_set: (n,)`, `h_unif: (n,)`, `h_zero: (n,)`, `k_add: (n,)`, `h_add: (n,)`, `k_sub: (n,)`, `h_sub: (n,)`.
  - `show(self) -> None`: Missing docstring. Pretty-prints intervention fields.
- **Class `OntologizerIntervention`**:
  - Class docstring: Complete.
  - Fields: `layer: int`, `method: DictIntervention` (NOTE: `method` holds a `DictIntervention` instance).
  - `show(self) -> None`: Missing docstring.
- **Class `Ontologizer(nn.Module)`**:
  - Class docstring: Complete.
  - **Key Attributes / Shapes**:
    - `d_in: int`: Input dimension.
    - `d_out: int`: Output reconstruction dimension.
    - `e_dec: int`: Intermediate decoder dimension (output of each `DictBlock`).
    - `k: int`: Tags per head.
    - `h: int`: Number of heads per layer.
    - `l: int`: Number of sequential `DictEnc` layers.
    - `forward: str`: Layer chaining mode (`"labels"` or `"resid"`).
    - `deepsup: bool`: Deep supervision flag (outputs stacked per-prefix decodes).
    - `deepsup_sg: bool`: Stop-gradient on residual prefixes.
    - `resid_norm: bool`: Unit-normalizes residual before next layer.
    - `resid_const: bool`: Appends constant 1 coordinate to residual.
    - Submodules:
      - `encoder`: `Linear` (if `encoded=True`) mapping `d_in -> e_enc`, else `Sparse` passthrough.
      - `dictencs`: List of `l` `DictEnc` instances.
        - Layer 0 input: `e_enc` (or `d_in`).
        - Layers 1..l-1 input: `h * k` (if `"labels"`) or `d_out + int(resid_const)` (if `"resid"`).
      - `decoder`: `Linear` mapping `e_dec -> d_out`.
  - **Methods**:
    1. `dictenc(self, d_enc: int) -> DictEnc`: Partial. Factory method instantiating a `DictEnc` layer with inherited configs.
    2. `setup(self)`: Partial. Instantiates encoder, decoder, and layer list `dictencs`.
    3. `fwd_dec(self, F: Float[Array, "... e_dec"]) -> Float[Array, "... d_out"]`: Partial.
    4. `prefix(self, R: Float[Array, "... e_dec"], R_0: Float[Array, "... e_dec"]) -> Float[Array, "... e_dec"]`: Complete.
    5. `resid(self, X: Float[Array, "... d_in"]) -> Float[Array, "... e_dec"]`: Missing docstring. Allocates zero residual tensor.
    6. `encode(self, X: Float[Array, "... d_in"], sd: float = 0.0, rng = None) -> Tuple[Float[Array, "... e_enc"], Optional[PRNGKeyArray]]`: Missing docstring.
    7. `nextinput(self, X: Float[Array, "... d_out"], R: Float[Array, "... e_dec"], K: Float[Array, "... (h k)"]) -> Array`: Complete. Computes next layer's input based on `self.forward`.
    8. `classify(self, E: Float[Array, "... e_enc"], *args, X_ref = None, rng = None, **kwargs) -> Tuple[Float[Array, "... e_dec"], Float[Array, "l ... (h k)"]]`: Complete.
    9. `decode(self, E: Float[Array, "... e_dec"]) -> Float[Array, "... d_out"]`: Partial.
    10. `__call__(self, X: Float[Array, "... d_in"], *args, sd_in: float = 0.0, rng = None, arglist = None, **kwargs) -> Float[Array, "... d_out"]`: Missing docstring. Full forward inference pass.
    11. `withStats(self, X: Float[Array, "... d_in"], sd_in: float = 0.0, rng = None, *args, **kwargs) -> Tuple[...]`: Complete. Returns `(Y, stats, rng_K)` where `Y` is `(..., d_out)` or `(l, ..., d_out)` under `deepsup`, and `stats` is `(l, 6)`.
    12. `withGhost(self, X: Float[Array, "... d_in"], sd_in: float = 0.0, rng = None, *args, **kwargs) -> Tuple[...]`: Complete. Returns `(Y, Y_g, stats, rng_K)` where `Y_g` is ghost reconstruction.
    13. `withArgs(self, X: Float[Array, "... d_in"], arglist: List[DictIntervention], *args, sd_in: float = 0.0, rng = None, **kwargs) -> Tuple[...]`: Complete. Applies per-layer intervention list.
    14. `decodeLayerEntries(self, layer: int, *args, **kwargs) -> Tuple[Float[Array, "(h k) d_out"], Float[Array, "(h k) (h k)"]]`: Complete. Decodes one-hot dictionary tags for a layer through downstream layers.
    15. `decodeEntries(self, *args, **kwargs) -> Tuple[Float[Array, "(l h k) d_out"], Float[Array, "l (h k) (h k)"]]`: Complete.
    16. `decodeLayerUnif(self, layer: int, *args, **kwargs) -> Float[Array, "(h k) d_out"]`: Complete.
    17. `decodeUniform(self, *args, **kwargs) -> Float[Array, "(l h k) d_out"]`: Complete.
    18. `intervene(self, X: Float[Array, "... d_in"], layer: int, *args, temperature: float = 1.0, strength: float = 1.0, sd_in: float = 0.0, rng = None, **kwargs) -> Tuple[Float[Array, "... d_out"], Float[Array, "... d_in"], Float[Array, "l ... (h k)"]]`: Complete. Backward adjoint steering intervention on input `X`.

---

#### 3. `ontologize/main.py`
CLI entry point stub.
- **Module Docstring**: Missing.
- **Function `main() -> None`**: Missing docstring. Prints `"Hello from ontologize!"`.

---

## 4. Comprehensive Bug, Dead Code, and Oddity Catalog

*(Note: Per project requirements, do NOT modify code to fix these; cataloged for architectural reference and docstring precision.)*

1. **Dead JAX Modules with Broken Imports**:
   - `layers/dictblock_enhanced.py`, `layers/dict_interpreter.py`, `layers/integration_clean.py`, and `layers/vae_integration.py` import from `ontologize.jax.layers.*`, which does not exist. They are obsolete artifacts from a previous multi-backend structure.
2. **Missing `__init__.py` Files Across Subpackages**:
   - Neither the root `ontologize/` package nor `training/`, `fns/`, `inference/`, or `visualize/` contain an `__init__.py`. Python relies on implicit namespace packaging. Meanwhile, `layers/__init__.py` and `data/__init__.py` exist but are 0 bytes.
3. **`DictEnc.fwd_dict` Missing Attribute Bug**:
   - `DictEnc.fwd_dict` (lines 96-99) attempts to call `self.fwd_dec(self.dict.fwd(P, ...))`. However, `fwd_dec` is defined on `Ontologizer`, not on `DictEnc`. Calling `DictEnc.fwd_dict` directly will raise an `AttributeError`.
4. **Duplicate / Overlapping `ChatEnv` Class Names**:
   - `ontologize/chat.py` defines `ChatEnv`, an interactive terminal REPL for managing interventions.
   - `ontologize/inference/steerable.py` also defines a `@dataclass class ChatEnv` containing `model`, `hyper`, `meta`.
5. **Typo in `DictEnc.tags()` Docstring**:
   - Line 146 of `dictenc.py` has `"sel.dict.dicts()"` instead of `"self.dict.dicts()"`.
6. **Typo in `DictBlock.uniform` Return Shape Annotation**:
   - Line 261 of `dictblock.py` annotates return as `Float[Array, "... b"]`, whereas the method actually constructs and returns an array of shape `(b, self.h, self.k)`.
7. **Unused `Sparse.sd` Field**:
   - `Sparse` defines `sd: float = 0.0 # ignored; passed as argument to addnoise instead`.
8. **Type Annotation Inconsistency in `Linear.ghost`**:
   - `Linear.ghost` annotates `lbound: int = -10.0, ubound: int = 10.0` (annotated as `int`, given float defaults).
9. **`NLinearBlock.rev` Restricted to `n=1`**:
   - `NLinearBlock.rev` explicitly raises `NotImplementedError` for any `n != 1`, requiring `BilinearBlock` for `n=2`.
10. **`ontologize/main.py` is a Placeholder**:
    - The entrypoint in `ontologize/main.py` only prints a placeholder string and does not invoke the CLI commands described in older documentation.
11. **`OntologizerIntervention.method` Semantic Naming Oddity**:
    - The field `method` in `OntologizerIntervention` stores a `DictIntervention` dataclass instance, not a method/callable.
12. **`Hyperparams.update` Explicitly Unused**:
    - `Hyperparams.update` has a docstring stating: `"Unused. self.train calls ontologize.ontostate.update instead."`

---

## 5. Work Breakdown Recommendation for Workers

To parallelize docstring additions cleanly without git merge conflicts, the work should be partitioned across workers as follows:

| Batch / Worker | Target Subpackage / Files | Primary Task |
| :--- | :--- | :--- |
| **Worker 1 (Layers)** | `ontologize/layers/` (`dictblock.py`, `dictenc.py`, `linear.py`, `nlinear.py`, `sparse.py`, `__init__.py`) | Document core neural network layers, weight dimensions `(h, k, d)`, `(n, h, d_out, d_in)`, tensor transformations, and Gram matrix computations. Skip the 4 dead modules. |
| **Worker 2 (Training & Data)** | `ontologize/training/` (`config.py`, `ontostate.py`, `serialize.py`) + `ontologize/data/` (`langs.py`, `loaders.py`, `multilingual.py`, `pretrained.py`, `__init__.py`) | Document Grain data pipelines, tokenization, pooling `(batch, seq, d_model) -> (batch, d_model)`, `OntoState` train loops, annealing schedules, and serialization. |
| **Worker 3 (Fns, Inference, Viz, Root)** | `ontologize/fns/` (`classify.py`, `keys.py`, `loss.py`) + `ontologize/inference/` (`steerable.py`) + `ontologize/visualize/` (`loss.py`) + `chat.py` + `ontologizer.py` + `main.py` | Document loss math (aic, bic, ghost, cossim), interactive REPL `chat.py`, `Steerable` inference wrapper, and root `Ontologizer` multilayer architecture. |
