# Reference fixtures

Small trained categorical dictionaries, with a few input rows and what this
repository's JAX forward computes on them at every layer. A reader that
reimplements the forward, in PyTorch for example, can check itself against
them on CPU from plain files. The first layer whose arrays disagree is where
the port went wrong.

`tests/test_reference.py` checks that the forward here still reproduces
them, so any change to the forward that changes its outputs fails there
until the fixtures are regenerated on purpose.

## The fixtures

| fixture | what it covers |
|---|---|
| `legacy_const0` | a checkpoint from before `const0` existed, shaped like `resid_nc`: soft selection at temperature 0.03, no gain-shape, `resid_norm`, and no constant coordinate at layer 0. Its spec lacks the `const0` field and the fields added after it, so a loader must infer `const0` from layer 0's classifier width (`migrate_spec`). 2 layers of 4 heads with 8 entries, `d` 16 |
| `direct_gain_ste` | the `headline` branch's `V_direct_gain_ste` design, small: signed entries in the output space with no decoder, every layer's residual gain-shaped including layer 0's (`resid_first`), each gain clipped at the row's norm (`gain_clip`), hard selection. It is positively homogeneous, so rows 6 and 7 keep row 0's codes at every layer and their reconstructions scale with them. 3 layers of 4 heads with 8 entries, `d` 16 |

Both were trained for 1,500 steps on planted rows. Each row is one of 8
random atoms plus one of 8 others, with noise and a gain between 0.5 and
1.5. The stock training noises were on.

## The files

Each fixture directory holds:

| file | contents |
|---|---|
| `spec.json` | the model's fields as a checkpoint stores them (`dataclasses.asdict`). In this repository, `Ontologizer(**migrate_spec(spec, params))` builds the model |
| `params.npz` | the parameter tree, with its keys joined by `/`, for example `dictencs_0/classifier/weight` |
| `inputs.npz` | `X`, the 8 raw input rows, `(8, d)`, before any normalization |
| `expected.npz` | per layer `l`, under `L<l>/`, the arrays below. `Y` is the model's output, equal to the last layer's `Y` |
| `meta.json` | the temperature, the tolerances, which row is which, the near tie's gap, and how the fixture was made (commit, seed, steps, versions) |

The arrays per layer:

| key | shape | what it is |
|---|---|---|
| `U` | `(n, D)` | the classifier's input: the unit shape of the layer's input, followed by the constant coordinate (1) where the layer has one. Without gain-shape it is the input itself |
| `G` | `(n, 1)` | the gain, the norm of the input's non-constant part. Present only with gain-shape |
| `K` | `(n, h, k)` | the logits, before the temperature |
| `P` | `(n, h, k)` | the selection at the fixture's temperature: one-hot in the forward under `ste`, a softmax otherwise |
| `top2_index`, `top2_logit` | `(n, h, 2)` | each head's two largest logits and their entries |
| `Y` | `(n, d)` | the reconstruction after this layer, decoded |

## The forward these follow

`build.py`'s `trace` computes them, step by step as `DictStack.classify`
does; the code is the definition, and this is a guide to it.
1. **The input.** The encoder is a passthrough, so the layer-0 input is the
   raw row. The residual starts at zero. Layer 0 gets the constant coordinate
   when `resid_const` and `const0` are set. Under `resid_first` layer 0
   instead reads the residual X − decode(0), built as in step 3.
2. **Per layer:**
   - **Shape and gain.** With gain-shape, the shape and gain are split off
     (`gainshape_in`).
   - **Logits.** The classifier is a product of `n = 2` linear maps of `U`,
     with weight `(2, h, k, D)`, so `K = (W[0]·U)(W[1]·U)`.
   - **Selection.** The selection takes `K / T` (`DictBlock.cluster`).
   - **Each head's write.** A head writes its selection `P` times its
     dictionary: the `abs` of the weights, or the weights themselves when
     `signed` (`hfwd`). The heads are summed (`combine`).
   - **The layer's write.** The sum is scaled by the gain (`gained`) and
     added to the residual, and the residual is decoded.
3. **The next layer's input.** It is `X` minus the decoded reconstruction,
   with its norm capped at the row's under `gain_clip`, unit-normalized
   under `resid_norm` (without gain-shape), with the constant coordinate
   appended (`nextinput`).

`logit_norm`, which neither fixture uses, standardizes each
head's logits across its entries at the classifier's output, before the
temperature.

Training-only parts don't apply to a fixture: the noises, winner dropout
and head dropout.

## The rows and the tolerances

| row | what it is |
|---|---|
| 0–3 | held-out rows from the training distribution |
| 4 | a clear winner: the pool row whose smallest top-two gap over layer 0's heads is largest |
| 5 | a near tie at layer 0, head 0: a point on the segment between two rows that head codes differently, where its top-two gap is 2 × 10⁻⁴ of its largest logit |
| 6, 7 | row 0 scaled by 0.5 and by 3. Under gain-shape their layer-0 codes equal row 0's, and their gains scale |

- **What a float32 reader must reproduce.** Every array, within `rtol` 10⁻⁴
  and an absolute tolerance of 10⁻⁵ times the array's largest magnitude.
  Every winning entry (`top2_index[..., 0]`) must match exactly.
- **The near tie and lower precision.** The near tie's gap is about 500
  times what float32's order of summation moves it. But it is below
  float16's resolution and far below bfloat16's. A reader in either may
  flip row 5's code, which is the expected sign of the precision, not a
  bug.

## Regenerating

```bash
uv run python tests/reference/build.py            # both fixtures
uv run python tests/reference/build.py legacy_const0
```

The build takes about 25 seconds on CPU. On one machine it reproduces every
file byte for byte, except `meta.json`'s commit after a new commit. Review
the diff before committing regenerated fixtures: a changed `expected.npz`
means the forward's outputs changed.
