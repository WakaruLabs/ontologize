# Transcoder design: Ontologizer as a map between representation spaces

Goal: train the Ontologizer to map `x` from one representation space
(dim `d_x`) to a target `y` from another (dim `d_y`) -- e.g. one
encoder's sentence space to another's, or layer i to layer j of the same
model -- instead of reconstructing its own input. This document audits
exactly what the current code already supports, specifies the package
changes a real implementation needs (file by file), and marks what the
thin harness in this directory builds today without touching the
package.

## 1. What already supports transcoding (audit)

The stack is closer to pair-ready than it looks; the autoencoder
assumption lives in exactly one place.

- **`ontologize/ontologizer.py`** -- `Ontologizer` carries separate
  `d_in` and `d_out` fields; `setup()` builds
  `decoder = Linear(d_in=e_dec, d_out=self.d_out)`, so the decoder
  target dim is already independent of the input dim. `__call__`,
  `withStats`, `withGhost` never touch a target tensor *except* through
  residual forwarding (see section 3).
- **`ontologize/training/config.py`** -- `Hyperparams(d_in, d_out, ...)`
  keeps the dims separate; `Hyperparams.loss(X, Y, X_g, stats)` is
  target-agnostic (it scores whatever `Y` it is handed); `mse_weights`
  is a `(d_out,)` vector, i.e. already a *target-space* object.
- **`ontologize/training/ontostate.py`** -- the jitted step
  `update(state, lossfn, rng, X, Y_0, ...)` takes the input and the
  target as **separate arguments**. Nothing in the gradient path
  assumes `Y_0 is X`.
- **The one hardcode**: `ontostate.train()`:

  ```python
  # Assuming autoencoder where target Y_0 is exactly input X
  X = batch
  ...
  Y = X
  if decoder is not None:
      Y = decoder(Y)
  ```

  The `decoder` hook can only derive the target *from the input*, so
  targets that come from a second data stream cannot enter through the
  stock loop.

## 2. Required changes, file by file

### 2a. Loader pairing (`ontologize/data/loaders.py`, `ontologize/fns/keys.py`)

Add a paired source + loader:

- `PairedNpyDataSource(file_x, file_y)` -- two memory-mapped `.npy`
  caches aligned by row index; `__getitem__(i)` returns
  `{"x": row_x, "y": row_y}`. Same lazy-memmap `__getstate__` pickling
  pattern as the existing `NpyDataSource` so it survives grain worker
  processes. Assert equal lengths at construction.
- **Alignment contract**: row `i` of both caches must derive from the
  same underlying sample. `encode_corpus.py` is deterministic and
  resumable in corpus order, so two encode passes over the same corpus
  with two different encoders produce aligned caches; record the
  corpus + encoder ids of both files in a sidecar (`meta.json`) and
  check them at load.
- `EmbeddingPairLoader` = `SampleLoader` with no per-sample transforms
  (grain's `Batch` already collates dict leaves -- the text path batches
  `{"input_ids", "attention_mask"}` dicts today).
- `fns/keys.py::get_srctype` gains `"embedding_pair"`.
- `Metadata.data_path` for this srctype becomes a pair of paths (or a
  dir containing `x.npy`/`y.npy`; pick one and validate).

### 2b. Loss target (`ontologize/training/ontostate.py::train`)

Replace the hardcoded unpack with:

```python
if isinstance(batch, dict):
    X, Y = batch["x"], batch["y"]
else:
    X = batch
    Y = X
```

(then the existing `encoder`/`decoder` hooks apply to `X`/`Y`
respectively). `update()` needs no change. `TrainingEnv` needs no
change beyond passing the new srctype through.

### 2c. Decoder target dim (config only)

No code change: set `d_out = d_y` in the run config;
`hyper.ontologizer(d_x, d_y, e_dec, k, h, l, ...)` already plumbs it to
the decoder. `mse_weights` **must be recomputed over the target cache**
(`make_mse_weights.py cache_y.npy out_y.npy`); the shipped
`data/out/sonar/mse_weights.npy` is SONAR-space and is wrong for any
other target space.

### 2d. Layer-to-layer forwarding (the delicate part)

- `forward="labels"` **works unchanged**: layer i+1 consumes layer i's
  flattened classification; no target reference anywhere. It inherits
  the documented starvation risk under hard temperatures (sonar.py:
  "the starvation probe showed layers 2-4 receive constant input").
- `forward="resid"` is **not transcoder-valid as-is**, and must not be
  "fixed" by threading the target:
  - Dimensionally: `nextinput(X, R, K)` computes
    `X - decode(R)` from the model *input* `X` (`withStats` calls
    `self.nextinput(X, R, K)`), which requires `d_in == d_out` (stated
    in the field comment).
  - Semantically: substituting the target `y` for `X` would give upper
    layers the training-time residual `y - decode(R)` -- a signal that
    does not exist at inference. That is target leakage: the model
    would train on inputs it can never see again.
- The inference-safe analog is a new mode, `forward="input"`: every
  layer reclassifies the **same** encoded input `E`, contributions
  still sum into the shared `e_dec` residual, and `deepsup` (unchanged)
  gives each prefix its own y-reconstruction loss so later layers are
  still pushed toward refining what earlier layers left unexplained --
  the residual objective arrives through gradients instead of through
  the input. Exact touch points:
  - `Ontologizer.setup()`: `d_next = d_enc` branch for `"input"`
    (currently `k*h` for labels, `d_out + resid_const` for resid).
  - `Ontologizer.nextinput()`: return the layer-0 `E` (thread `E_0`
    through `classify`/`withStats`/`withGhost`, or recompute -- the
    encoder is cheap/passthrough in the live configs).
  - `decodeLayerEntries`/`decodeLayerUnif`: treat `"input"` like
    `"resid"` (no label path; entries decode directly through the
    shared decoder).
  - Checkpoint compat: changes upper-layer classifier input dims
    relative to *both* existing modes -- fresh run dir required (same
    caveat the resid_const migration carries in sonar.py).
- Future option (bigger change, noted but not specified): condition
  layer i+1 on the model's *own prediction so far*,
  `concat[E, sg(decode(R_i))]` -- inference-safe (the prediction exists
  at test time) and closest in spirit to RVQ refinement, but it changes
  per-layer classifier dims and adds a d_y-sized input to every upper
  layer.

### 2e. Downstream / eval implications

- `refit.onto_linear_model`'s exactness (`Y = P_flat @ G`) holds for
  any forwarding mode: it only needs the zero residual seed + linear
  decoder, both untouched.
- `autointerp.onto_acts_fn`'s probe calls `module.nextinput(X, R, ...)`;
  under `forward="input"` it keeps working if `nextinput` ignores its
  residual arguments in that mode.
- `textfid.py` / `steerfid.py` / `decode.py` assume the *output* space
  is SONAR-decodable (they push reconstructions through the M2M100
  decoder). They remain valid only when the transcoder's target space
  is SONAR; for other targets they are out of scope.
- `sae.py` baselines can be trained on the same pairs only after an
  equivalent pairing change; until then compare against the linear
  ridge baseline (below).

## 3. What the harness in this directory builds today

Without touching the package (see `transcode_train.py`):

- `PairedNpySource` (the loader-pairing prototype from 2a, living in
  the experiment dir) + the stock `SampleLoader` batching dicts;
- a training loop that mirrors `ontostate.train` but unpacks
  `X, Y = batch["x"], batch["y"]` and calls the **stock**
  `ontostate.update` / `ontostate.schedules` / `OntoState.save` --
  i.e. change 2b implemented outside the package;
- `forward="labels"` only (per 2d, the only mode that is currently
  both available and inference-safe);
- a closed-form ridge baseline `x -> y` fit on held-out-disjoint rows,
  reported as FVU next to the model's FVU on the cache tail: the floor
  any transcoder must beat before its dictionary structure means
  anything.

Not built here (needs the package changes above): `forward="input"`,
`srctype="embedding_pair"` integration into `Metadata`/`TrainingEnv`,
ghost-path support for pairs (`withGhost` mirrors `withStats`; the
harness sets `ghost=False`, which the bilinear config makes inert
anyway per sonar.py).
