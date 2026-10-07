# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ontologizer is a JAX/Flax implementation of dictionary learning for neural network interpretability. It trains sparse autoencoder-like models (`Ontologizer`) to decompose activations from pretrained transformer encoders (currently SONAR/M2M100 sentence embeddings, or raw MNIST pixels) into sparse combinations of learned "ontofeatures," and supports causal interventions on those features for mechanistic interpretability experiments.

**The codebase is research-grade and in active flux.** Expect dead code, half-migrated modules, and root-level scratch scripts alongside the real package — see "Known dead/stale code" below before assuming a file is load-bearing.

## Style guide

Annotate functions with `jaxtyping`, including shapes.
Code should be TPU-compatible. Avoid dynmamic shapes &c. Variables that would force recompilation should parameterize the relevant layer class.
In function and variable names, underscores should be read as subscripts rather than separators.

### Comments

Comments should reflect the current state of the code and not be used as a changelog.
Comments should not reference specific results or "live configurations" except when necessary to understand the code.
All notes to this effect should instead go in `scratch/` or `~/.claude/`.

### Experiments

Log results in a `notes.md` for each experiment in `experiments/`. 
Keep these separate from the `README.md` for an experiment,
which should be used for explaining code structure, usage, and rationale.

### Writeup

The writeup is compiled using `make`. Compilation uses `pdflatex` and `biber` for citations,
which are stored in `writeup/refs.bib`. The toolchain is provided by the root `flake.nix`;
the Makefiles run `pdflatex`/`biber` through `nix develop` themselves (skipped when already
inside a nix shell), so plain `make -C writeup` (or `make -C writeup brief`, `make -C description`) works.

Organize sections in `writeup/sections`.
Reserve `results.tex` for headline results. Keep other results in `appendix.tex`.

## Project Setup

- Requires Python >= 3.13, managed with `uv` from the repository root (there is no `pytorch/` subdirectory despite what older docs may say — the package lives at `ontologize/` in the repo root).
- Install: `uv sync`
- Stack: JAX/Flax (linen) for the model, `optax` for optimization, `orbax.checkpoint` for checkpointing, `grain` for data loading, PyTorch + HuggingFace `transformers` only for loading pretrained encoder/decoder models (SONAR/M2M100), bridged into JAX via `dlpack`.

## Common Commands

There is no real CLI. `python -m ontologize.main` is currently just a `print("Hello from ontologize!")` stub — do not treat it as the entry point. The actual entry points are root-level scripts, run with `uv run`:

```bash
# Precompute SONAR embeddings for the mC4 corpus into a .npy cache
# (resumable; sonar.py trains from the cache when its `cache` config is set,
# which keeps the frozen encoder out of the training loop entirely)
uv run python encode_corpus.py

# Harvest transformer residual-stream activations into the same cache layout
# (default gpt2 blocks.8.resid_post), for a model organism where sparse
# dictionary learning is known to work. Writes .index.npy (doc, pos, token),
# .docs.jsonl, .docstart.npy and its own .mse_weights.npy alongside. Rows are
# in DOCUMENT order so the tail holdout is a document-level split
uv run python encode_acts.py -h

# Train the Ontologizer on that cache. `--base` picks the config module every
# unset value falls back to: `sonar` (the live SONAR run) or `gpt2_l8`, which
# overrides only the widths, cache, output dir and MSE weighting. ONTO_ACTS
# names which harvest under data/activations to use
uv run python experiments/ste-arm/train_ste.py --base gpt2_l8 --dict-init-scale 1.0

# Train on SONAR multilingual sentence embeddings (mC4 data)
uv run python sonar.py
# or via tmux wrapper:
./sonar.sh

# Train an MNIST toy autoencoder (currently broken/stale, see below)
uv run python mnist.py

# Interactive reconstruction + intervention REPL against a trained checkpoint
uv run python decode.py

# Decode a trained model's learned dictionary tags to text
uv run python decode_tags.py

# Non-interactive smoke-test harnesses that drive decode.py / decode_tags.py
# with scripted stdin (useful as a template for manual testing changes to the REPL)
uv run python run_chat.py
uv run python run_decode.py
uv run python run_decode_tags.py

# SAE-baseline comparison suite (all score on the cache tail sae.py holds out):
uv run python sae.py           # train a field-standard top-k SAE on the same
                               # cache + whitened MSE (CLI flags; built to be swept).
                               # --groups/--group-fn/--prefixes add Ontologizer-style
                               # structure one rung at a time (heads / deepsup layers)
uv run python pareto.py        # reconstruction-vs-code-capacity Pareto table:
                               # Ontologizer top-m deviation codes vs trained SAEs
uv run python autointerp.py -h # 4-stage auto-interp pipeline (harvest/texts/
                               # describe/score) over Ontologizer tags & SAE latents
uv run python headstruct.py -h # post-hoc head-structure discovery: do SAE latents
                               # form exhaustive/exclusive groups? (split-half + nulls);
                               # --modularity also scores each group's sample partition
                               # by soft modularity on the input heat-kernel graph;
                               # --onto CKPT scores an Ontologizer's heads instead,
                               # each layer on its own input; --prefix-layers treats a
                               # Matryoshka SAE's blocks as layers (experiments/modularity)
uv run python headcoh.py -h    # head/group SEMANTIC coherence: within-head decode-
                               # direction similarity vs size-matched nulls (quota-
                               # free), plus description-text view over autointerp
                               # artifacts; --assignment scores discovered groups;
                               # --metric whitened = objective's inverse-variance
                               # inner product. The <out>_w suffix applies ONLY
                               # when --out is absent; with --out both metrics
                               # land on one path, so it now refuses to
                               # overwrite a run recorded under the other
uv run python compose.py -h    # intervention composition: do two-head set-
                               # interventions compose additively in decode space?
                               # (final layer exact by linearity = numerical floor;
                               # deviation above it = downstream reclassification)
uv run python splitting.py -h  # cross-model feature splitting via activation
                               # containment (parents/children/matches, absorption)
uv run python textfid.py -h    # downstream text fidelity ("loss recovered"): decode
                               # x vs recon(x) through M2M100, chrF + NLL vs floor
uv run python langprobe.py -h  # SAEBench-style k-sparse probing of language
                               # identity (langs sidecar); dense-probe ceiling
uv run python refit.py -h      # shrinkage: refit LS coefficients on each frozen
                               # support; selection-vs-magnitude error split
uv run python assignmap.py -h  # per-layer heatmaps of held-out samples x
                               # (head, entry) assignment probabilities, rows
                               # sliced by lang / GPT-2 token class / pos / doc;
                               # makes head collapse and dead entries visible
                               # per sample. --replot redraws from assign.npz
                               # (every script below has --replot too)
uv run python enrich.py -h     # entry x label enrichment dotplot (hypergeometric
                               # with correct tails, Haldane 2x2 log2OR, BH FDR) +
                               # head x label NMI over a shuffled null; --by as assignmap
uv run python headnmi.py -h    # within-model (l*h)x(l*h) head-partition NMI on the
                               # tail, chance-adjusted, layer-blocked or hclust-ordered;
                               # --ei <ckpt>/effinfo adds the EI panel + NMI-vs-EI scatter
uv run python entrydrift.py -h # (layer, head, entry) x condition heatmap: usage /
                               # decoded-direction drift / Hungarian-matched cos over one
                               # run's steps (--ckpt) or index-aligned fine-tunes (--runs;
                               # refuses independent seeds)
uv run python bilinspec.py -h  # classifier geometry from weights only: |cos(w,v)|,
                               # eigen-spectrum and resid_const odd share per feature,
                               # per layer and vs step
uv run python selection.py -h  # held-out noise2self partition score vs a sweep key
                               # with unpartitioned + size-matched random baselines
                               # drawn; --grid ROW COL = pareto.py's FVU over a grid
uv run python anatomy.py -h    # one (layer, head) as a block figure for the methods
                               # section: P_h, PWAK kernel, rowcos, support_overlap, W_h
uv run python steerfid.py -h   # steering fidelity: effect (cycle-consistency
                               # activation gain) vs collateral (1-chrF) at matched
                               # magnitude; W_dec / eigenfeature / withArgs steering
```

Seed stability: train a replica with `sae.py --seed 43` (run dirs get an
`_s43` suffix) and feed the pair to `splitting.py` — its "A matched"/"B
matched" columns are the reproducible-feature fractions.

Tests: the curated suite lives in `tests/`, wired via `[tool.pytest.ini_options] testpaths = ["tests"]` in `pyproject.toml`, so a bare `uv run pytest` runs only it (fast, CPU-only — `tests/conftest.py` forces `JAX_PLATFORMS=cpu`, so it is safe to run alongside a live GPU training run). It covers: `deepsup_sg` gradient semantics, `resid_norm`/`resid_const` conditioning (including bilinear sign-blindness), the `s_Hm`/`KL_m` batch mean-entropy bonus, top-k tag selection (`select="top<k>"`), the `s_L2pwak`/`L2_pwak` noise2self partition score, noise-path NaN regressions, checkpoint save/restore including the `n_stats` stats-width migration and the legacy-spec-key rename, `ConcatDictBlock`'s overridden statistics, the gated SAE encoder's gradient routing, `NLinearBlock.withL1`'s gate-factor L1, `L1_S`'s position in the stats row, `support_overlap` and the retired `cossim_h`/`cossim_flat` slots (`test_support`), fibers and head dropout (`test_fibers`), the SAE modules' layout converters and configuration checks (`test_sae_modules`), and a short end-to-end training loop per live run configuration. The suite's largest block now tests the root eval scripts' pure helpers (`test_sae`, `test_autointerp`, `test_pareto`, `test_headstruct`, `test_headcoh`, `test_compose`, `test_splitting`, `test_textfid`, `test_langprobe`, `test_refit`, `test_steerfid`, `test_encode_acts`, `test_assignmap`, `test_enrich`, `test_headnmi`, `test_entrydrift`, `test_bilinspec`, `test_selection`, `test_anatomy` import them directly) — breaking `sae.py` or its siblings breaks the suite, so keep those scripts import-safe under `__main__` guards.

```bash
uv run pytest -v
```

(A generation of root-level `test_*.py` ad hoc debugging scripts was deleted when this suite landed; if one resurfaces from git history, treat it as a scratch script, not a spec.)

No linter, formatter, or type checker is configured (no ruff/mypy/black config exists).

## Architecture

### Model stack (`ontologize/`)

Layers build on a common base in `layers/sparse.py`:

- **`Sparse`** (`layers/sparse.py`) — base `nn.Module` for nearly all layers. Resolves string-keyed config (`activation`, `noise`, `dtype_str`) into callables via `fns/keys.py`, and provides shared stats (`l1`, `cossim`, `bcossim`, `entropy`) that are `stop_gradient`-wrapped unless explicitly enabled as loss terms, plus noise injection and a `ghost()` hook for ghost-gradient dead-feature resurrection.
- **`Linear`** (`layers/linear.py`) — standard dense layer with `fwd`/`rev` (raw matmul, no bias/activation, used to compose ghost gradients across layers).
- **`NLinear`/`Bilinear`** and their multihead variants **`NLinearBlock`/`BilinearBlock`** (`layers/nlinear.py`) — generalized bilinear layers (`Bilinear` implements eigendecomposition-based analysis per Pearce et al. 2025-style bilinear MLPs). `NLinearBlock`/`BilinearBlock` are what `DictEnc`'s classifier actually instantiates.
- **`DictBlock`** (`layers/dictblock.py`) — the core dictionary/tag lookup. Holds a `(h, k, d)` weight tensor (heads × dict entries × feature dim; always `abs()`'d so features can't subtract). Given per-head classification logits, `cluster` turns them into probabilities per its `select` rule (`"softmax"` default; `"argmax"`/`"ste"` straight-through; `"top<k>"`, e.g. `"top4"`, softmaxes the top-k logits per head and zeroes the rest exactly — incompatible with `pwak_loss`), does a weighted sum over dictionary entries (`fwd`/`hfwd`), and exposes the intervention interface (`intervene`: uniform-ablate, zero-ablate, set, add, subtract, scale — see `DictIntervention` below).
- **`ConcatDictBlock`** (`layers/dictblock.py`) — `DictBlock` with heads writing **disjoint slices** of the output instead of each getting all of it and summing, selected by `concat=True` on `Ontologizer`/`DictEnc`. The weight narrows to `(h, k, d // h)`, so between-head disjoint support holds by construction (`support_overlap` is identically 0) and the dictionary costs `h` times fewer parameters at matched `d`. That disjointness is in the dictionary space only: the shared decoder maps the slices onto overlapping output subspaces, so decoded heads are not orthogonal. A subclass rather than a flag because `cossim`, `flatcos`, `support_overlap`, `tags` and `ghost` would otherwise compare or hand out vectors from different slices as if they shared a space; each is overridden, and its docstring has the reasoning. `train_ste.py --concat --d-head D` sets `e_dec = h * D`; the useful window is `d_out / h <= D <= d_out`, wide enough that `h` slices span the output and narrow enough that each head's decoder column block keeps full column rank.
- **`DictEnc`** (`layers/dictenc.py`) — composes an `NLinearBlock` classifier (activations → per-head logits), a `DictBlock` (logits → reconstructed features), and `Linear`/`NLinear` encoder/decoder/scaling. One `DictEnc` = one dictionary-learning layer. Also owns the two PWAK statistics (`pwak_kl`, `pwak_l2`), since both read the layer *input* alongside the classifications and neither touches the dictionary: `KL_pwak` pulls classifications toward a per-head neighbourhood consensus, while `L2_pwak` scores the partition by how well a sample's own (stop-gradiented) input is predicted from its co-classified neighbours — noise2self, so its only gradient path is the partition gate. See `fns/pwak.py`.
- **`Ontologizer`** (`ontologizer.py`) — stacks `l` `DictEnc` layers residually: with the live `forward="resid"`, layer *i+1* classifies the reconstruction residual `X − decode(R_i)` (note the dataclass default is still `forward="labels"` — flattened-classification forwarding — which every live config overrides because it starves upper layers). Holds the shared `encoder`/`decoder` `Linear` (or `Sparse` passthrough) around the stack. Key methods: `__call__` (plain forward), `withStats`/`withGhost`/`withArgs` (forward + validation stats / ghost-gradient / per-layer intervention list), `decodeEntries`/`decodeUniform` (decode the learned dictionary itself, rather than model input, into output space — used by `decode_tags.py`).
- **`DictIntervention`** / **`OntologizerIntervention`** (`ontologizer.py`) — dataclasses describing a causal intervention (scale a head, force/add/subtract specific dict entries, zero/uniform-ablate a head) applied per `Ontologizer` layer via `withArgs`.
- **`SAE`** / **`BilinearSAE`** (`sae.py`) and **`Grouped`** (`grouped.py`) — the SAE baselines as `Sparse` modules: a `Linear` (or `Bilinear`, or the gated encoder's `gate` plus `r_mag`/`b_mag`) encoder and a `Linear` decoder whose bias is pre-subtracted. Fields: `d`, `m`, `topk` (0 = ReLU + `s_l1`), `s_l1`, `aux_k`, `prefixes` (Matryoshka nested-prefix losses), `enc` (`"linear"`/`"gated"`/`"bilinear"`). `Grouped` adds `groups`/`group_fn` (`"top1"`/`"softmax"`; softmax frees the decoder-row norms) and requires `topk=0`. Methods used under `apply`: `preacts`, `gated_pre`, `encode`, `decode`, `loss` (aux-k, L1, prefixes, or `gated_loss`), `eigenfeatures`; on a parameter tree without `apply`: `initialize`, `renorm`, `resample`. The root `sae.py` is their trainer and keeps the flat `params.npz` key layout (`W_enc`, `W_dec`, `b_dec`, …) every downstream script reads; its module-level `encode`/`decode`/`preacts`/… take that layout and delegate to the classes through `from_legacy`/`to_legacy`, so there is one implementation. The gated encoder's training requirements (`s_l1 > 0`, `aux_k = 0`) are checked in `gated_loss`, not at construction, because consumers encode gated checkpoints without them.

`fns/keys.py` holds the string→callable lookup tables (activations, dtypes, losses, noise, data source type) that let all of the above be JSON-serializable dataclasses instead of holding raw callables. `fns/loss.py` has the actual math (mse/l1/l0/entropy/cossim/ghost-gradient). `fns/classify.py` has classification-specific activations (softmax, straight-through estimator).

### Training (`ontologize/training/`)

- **`config.py`** — `Hyperparams` (loss weights, optimizer, temperature/noise settings; builds the `Ontologizer` and drives `init`/`train`), `Metadata` (paths, checkpoint cadence, data source type; builds the orbax `CheckpointManager` and the `SampleLoader`), `TrainingEnv` (glues `Ontologizer` spec + `Hyperparams` + `Metadata`, handles resume-from-checkpoint and truncating `loss.csv`/`log.jsonl` on resume).
- **`ontostate.py`** — `OntoState` (a Flax `TrainState` subclass carrying the model instance itself as a static field, plus a rolling stats buffer). `state_init`, `update` (jitted optax step with optional global-norm grad clipping), `train` (the actual training loop, checkpointing + stats-flushing every `save_each` steps).
- **`serialize.py`** — two separable things. Its `save_model`/`load_model` pair (orbax `StandardSave`/`StandardRestore`) overlaps with `OntoState.save`/`Metadata.manager` and is **not** used by the main training path (`Hyperparams.load`/`TrainingEnv.init` use the manager/spec-item approach in `config.py`/`ontostate.py` instead) — don't assume both are equally live. But `migrate_spec` in the same module **is** live everywhere: a model is serialized as `dataclasses.asdict(model)` and rebuilt by splatting that back into `Ontologizer`, so renaming a field makes every earlier checkpoint unconstructable. `migrate_spec` maps known legacy names forward. Loaders build their model from `restore_spec(manager, step)`, which restores the spec and migrates it with the stored weights' shapes (orbax metadata, no arrays loaded) — `pareto.py` (and every script using its `load_onto`), `autointerp.py`, `decode.py`, `decode_tags.py`, `inference/steerable.py`, `Hyperparams.load`, `freeze_diag.py`, and `dictgeom.py`/`lastlayer.py`/`decodehead.py` in `experiments/ste-arm`. Add a rename to `LEGACY_SPEC_KEYS` when you make one. A field whose default is not what checkpoints from before the field did needs an entry in `ABSENT_SPEC_DEFAULTS` (absent ⇒ that value): `resid_gain` is there, because it was added at False and defaulted to True minutes later, so a spec without it trained without gain-shape. `const0` (whether `resid_const`'s coordinate reaches layer 0) cannot be read off a spec — every checkpoint before the field lacks it, with and without the coordinate — so `migrate_spec(spec, params)` infers it from the stored layer-0 classifier width; that is what makes `resid_nc`/`resid_nc_hm` (trained before layer 0 had the coordinate) loadable. Without shapes it falls to the default and such a checkpoint fails with a shape error rather than loading wrong. Beyond renames it drops only what cannot change the model: `headline`-branch fields at their no-op value (`HEADLINE_INERT_SPEC_KEYS`, raising if one is switched on) and the stats-only `fast_stats`. Any other unrecognized key still raises rather than being silently dropped.

Model serialization is plain `dataclasses.asdict(model)` (there is no `@model_spec()` decorator or dedicated spec-class pattern in this codebase — if you see references to one, it's stale).

### Data (`ontologize/data/`)

- **`loaders.py`** — Grain-based (`grain.python`) pipeline: `SampleLoader` (base), `TextLoader` (adds tokenization with SONAR BCP-47 language routing), `ImageLoader` (flattens/normalizes images, e.g. for MNIST), `EmbeddingLoader` (no-op transform over precomputed embedding vectors; the `srctype="embedding"` loader). `HFDataSource`/`JSONLDataSource`/`NpyDataSource` adapt HuggingFace `datasets`, raw JSONL, or an on-disk `.npy` (memory-mapped; e.g. the embedding cache written by root-level `encode_corpus.py`) into Grain data sources.
- **`pretrained.py`** — loads pretrained PyTorch/HuggingFace encoders (special-cased for SONAR: loads the `M2M100Encoder` directly from a `pytorch_model.bin`, bypassing `AutoModel`). `encode()`/`l2_pooling()` run the torch model, then hand the hidden state to JAX via `jnp.from_dlpack` and mean-pool + L2-normalize it (SONAR embeddings are unit-normed). `decode()` runs the paired M2M100 decoder's `.generate()` with an `Ontologizer` output substituted in as `last_hidden_state`.
- **`multilingual.py`** / **`langs.py`** — loads and interleaves per-language HuggingFace datasets (mC4/C4), with a static ISO→SONAR/NLLB BCP-47 language code map.

This is the load-bearing pattern to know: **JAX/Flax owns the `Ontologizer` model; PyTorch is used only to run the frozen pretrained SONAR encoder/decoder, with `dlpack` as the zero-copy bridge between the two frameworks.**

### Inference / interventions

- **`chat.py`** (`ChatEnv`) — interactive CLI REPL for setting per-layer `DictIntervention`s (`set <layer> <field> <values>`, `clear`, `clear_all`, `run <text>`), used by `decode.py`.
- **`inference/steerable.py`** (`Steerable`, and a *second, different* `ChatEnv` dataclass) — loads a trained `Ontologizer` checkpoint and runs `withArgs` with a list of interventions. Note the name collision: this `ChatEnv` is not the same class as `chat.py`'s — don't assume they're interchangeable.

### Visualization

`visualize/loss.py` reads `<output_dir>/loss.csv` and plots each column vs. step with matplotlib. Called from `sonar.py` after training. `ontologize/visualize/loss.py:COLUMNS` is authoritative and documents each column's provenance — **import that list, do not restate it**: local copies have gone stale and silently mislabelled columns (`experiments/devinterp-tracking/track.py` kept a 9-wide copy that put `KL_m` at index 8, which is `cossim_k`, so every tracked run recorded the wrong statistic under that name). It is currently 19 wide:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| loss | MSE | MSE_ghost | L1_K | L1_F | entropy | cossim_b | ~~cossim_h~~ | cossim_k | cossim_k_hmax |

| 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|
| KL_m | KL_pwak | L2_pwak | s_L1F | s_kcossim | cossim_k_max | ~~cossim_flat~~ | L1_S | support |

The layout is **append-only**, so a shorter row from an older run still aligns column for column and `read_loss` reads the missing tail as NaN. Columns 0–2 are added by `Hyperparams.loss`; 3–12 are the *first ten* of the per-layer `DictEnc.withStats` row summed over layers — that row is 13 wide, and its last three entries (`cossim_flat`, `L1_S`, `support`) are appended at 16–18 rather than inserted; 13–15 are the two setpoint-controlled multipliers as applied this step, plus the max of `cossim_k` over heads *and* layers.

**Retired columns keep their slot and read NaN** in new runs: `cossim_h` (7) and `cossim_flat` (16), both superseded by `support` (18) — `DictBlock.support_overlap`, the mean cosine between heads' usage-weighted coordinate profiles `sum_k pbar_hk |W_hk|` in the dictionary space, zero exactly when no two heads share a coordinate (signed dictionaries included). `Hyperparams.RETIRED_STATS` lists the retired positions in the weight vector, and `Hyperparams.loss` zeroes them before the weighted sum (NaN × 0 would otherwise reach the loss). Their weights `s_hcossim`/`s_flatcos` raise if set; the term is `s_support`, gated by `support_loss`. In tests, "the stats are finite" means the live columns are: use `conftest.finite_live` with `RETIRED_LAYER` (per-layer row) or `RETIRED_LOSS` (loss.csv row) rather than `jnp.isfinite(...).all()`. Retire a stat the same way — never delete or reuse its slot.

Three columns are the same statistic reduced three ways, and only the names distinguish them: `cossim_k` is the mean over a layer's heads summed over layers, `cossim_k_hmax` the per-head max within a layer summed over layers, and `cossim_k_max` a single max over heads and layers. The last exists because a summed `cossim_k` cannot see one layer's head collapse — the other layers improve faster than it degrades, so the sum falls while the max rises.

`n_stats` on `Hyperparams`/`OntoState` must equal the width. `Hyperparams.s_loss`'s weight vector is 14 wide and pairs positionally against `[MSE_ghost] + <the 13-wide per-layer row>`, **not** against the loss.csv order — so a stat inserted anywhere but the end makes every weight past it multiply the wrong quantity, which is why `DictEnc.withPWAK` says so in its docstring. Each stat keeps the position it was appended at, so `withPWAK` places `L1_S` (11) between `DictBlock`'s eighth entry and anything `DictBlock` appended later (`support`, 12). On resume, the setpoint controllers recover their multipliers from columns 13–14 of the last `loss.csv` row, which every row since they were added reaches, so a widened layout does not restart them.

`experiments/hsic-bottleneck/hsic_hyperparams.py` overrides `loss` but keeps this layout: it delegates to the stock loss and records its raw HSIC penalty in column 2 (`MSE_ghost`, otherwise 0 with the ghost path off).

## Experiments

### Scripts

Scripts and writeups are stored in `experiments`. 

### Training data

Cached SONAR embeddings are stored in `data/sonar_embeddings`.

### Output

Trained models, loss logs, and other experimental artifacts are stored in `data/out`.

## Known dead/stale code

- `ontologize/layers/dictblock_enhanced.py`, `dict_interpreter.py`, `integration_clean.py`, `vae_integration.py` all import from a nonexistent `ontologize.jax.layers.*` path (a leftover from a prior multi-backend `pytorch/`/`jax/`/`haskell/`/`julia/` layout that has since collapsed into the flat `ontologize/` package). They are never imported anywhere else and contain their own "placeholder, not implemented" docstrings. Treat them as dead — the real dictionary-learning code is `layers/dictblock.py`/`dictenc.py`.
- `mnist.py` is dead from constructor-signature drift, not truncation: the unused `ontologize.training.data.ImageLoader` import doesn't exist (`ImageLoader` lives in `ontologize/data/loaders.py`), the positional `Ontologizer(...)` and `Hyperparams(...)` calls misalign with the current field order (values land in the wrong fields), and `Metadata(epochs=...)` passes a field `Metadata` doesn't have (immediate `TypeError`). Do not use it as a template.
