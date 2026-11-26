# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ontologize` is a JAX/Flax implementation of multi-head categorical dictionary learning for neural network interpretability. It trains categorical dictionaries (the class `DictStack`, alias `Ontologizer`) to decompose activations from pretrained transformer encoders (currently SONAR/M2M100 sentence embeddings, or raw MNIST pixels) into sparse combinations of learned "ontofeatures," and supports causal interventions on those features for mechanistic interpretability experiments.

Below, and in the experiments' notes, "Ontologizer" still names the model, as the code's alias and the run names do. New prose (the writeup, READMEs) calls the method multi-head categorical dictionary learning and a model a categorical dictionary; `latex/CLAUDE.md`'s Terminology has the rules.

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

The findings writeup, its two-page brief and the project proposal live together in
`latex/` and share one style (`latex/wakaru.sty`), one set of macros, one bibliography
(`latex/refs.bib`), and the same diagrams and figures; `latex/CLAUDE.md` covers all three. They are compiled with
`make`, using `pdflatex` and `biber`. The toolchain is provided by `latex/nix/flake.nix`; the
Makefile runs `pdflatex`/`biber` through `nix develop` itself (skipped when already inside
a nix shell), so plain `make -C latex` (the findings), `make -C latex brief` and
`make -C latex proposal` work.

Organize the findings' sections in `latex/findings/sections` and the proposal's in
`latex/proposal/sections`.
Reserve the findings' `results.tex` for headline results. Keep other results in `appendix.tex`.

## Project Setup

- Requires Python >= 3.13, managed with `uv` from the repository root (there is no `pytorch/` subdirectory despite what older docs may say — the package lives at `ontologize/` in the repo root).
- Install: `uv sync`
- Stack: JAX/Flax (linen) for the model, `optax` for optimization, `orbax.checkpoint` for checkpointing, `grain` for data loading, PyTorch + HuggingFace `transformers` only for loading pretrained encoder/decoder models (SONAR/M2M100), bridged into JAX via `dlpack`.

## Common Commands

There is no real CLI. `python -m ontologize.main` is currently just a `print("Hello from ontologize!")` stub — do not treat it as the entry point. The actual entry points are root-level scripts, run with `uv run`:

```bash
# Precompute SONAR embeddings for the mC4 corpus into a .npy cache
# (resumable; sonar.py trains from the cache when its `cache` config is set,
# which keeps the frozen encoder out of the training loop entirely). New
# caches hold SONAR's own masked-mean embedding, not the unit vector
# mc4_4M holds; `pooling` in a cache's meta.json says which
uv run python encode_corpus.py

# Sentence-level SONAR cache (mc4_sent16): up to 16 consecutive sentences
# per mc4_4M document, split by SaT Capped and filtered by script and
# GlotLID, encoded under each config's own tag without the L2 step. Two
# stages; .index.npy (doc, pos, segment) and .docstart.npy key rows back to
# mc4_4M's documents, .kind.npy marks each row a sentence, a whole line or
# a piece the cap made, and meta.json records the package versions and model
# revisions. SaT and fastText are layered on, not dependencies
uv run --with wtpsplit --with fasttext==0.9.3 python encode_sentences.py split
uv run python encode_sentences.py encode

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

# Train toward the NEXT sentence of mc4_sent16 (x_t -> x_{t+1}), or toward
# the input over the same rows (`--objective self`), to compare how well each
# objective's code reproduces across seeds. train_ste.py's flags, defaulting
# to one layer of 380 heads, through its parser/finish/setup/train split,
# which takes any data source. eval_next.py scores held-out FVU_w against
# both targets beside mean, copy and ridge baselines. transitions.py measures,
# on the held-out tail, how often each head keeps its entry between sentences
# (adjacent, d apart, same language, random; Cohen's kappa) and how much closer
# adjacent sentences are than other pairs, with no model
# (experiments/next-sentence)
uv run python experiments/next-sentence/train_next.py --objective next
uv run python experiments/next-sentence/eval_next.py --ckpt <run dirs>
uv run python experiments/next-sentence/transitions.py --ckpt <run dirs> --out <dir>
# The pair objective (experiments/next-sentence/pair_objective.md). Each
# record is two sentences of one document, and each reconstructs itself. The
# first --pair-heads heads per layer are slow heads, weighed by --s-pair:
# --pair-term fano (default) rewards a Fano lower bound on each one's mutual
# information between the two sentences' entries; agree penalizes switches
# alone and collapses them onto one entry (pairloss.PairHyperparams, the
# HSIC experiment's ghost-slot pattern; the raw term is logged in loss.csv
# column 2). train_ste.setup takes the Hyperparams class through hp=, which
# must subclass the class its flags chose. --calibrate N prints the stock and
# pair gradient norms at init. pilot.sh [fano|agree] sweeps s_pair
uv run python experiments/next-sentence/train_pair.py --s-pair <s>
experiments/next-sentence/pilot.sh fano
# Linear slow-feature analysis, the slow heads' closed-form baseline: the
# directions two sentences of one document keep correlated (S v = rho C v,
# S the pair's symmetrized cross-covariance), fitted on the pair runs'
# training prefix with and without each language's mean, and scored on the
# held-out tail by transitions.py's pair types. --ckpt adds how much of the
# top directions each block of heads explains, and SFA on the block's code
uv run python experiments/next-sentence/sfa.py --ckpt <run dirs>
# Whether a head is an informationally closed level (Rosas 2026), one step
# ahead: the held-out cross-entropy of its next entry given its current one
# (a transition table) minus that given the current entry and the whole
# embedding (a softmax probe started at the table), I(x_t; z_t+1 | z_t).
# Fitted on adjacent pairs from rows no pair run trained on, scored on the
# held-out tail; groups: each run's slow heads and a sample of the rest,
# random-projection heads, and sfa.py's top directions cut into k bins
uv run python experiments/next-sentence/closure.py --ckpt <run dirs>
# Two seeds' slow blocks mode by mode (modeagree.py): the held-out R^2 of
# one block's code-covariance mode scores from the other seed's block,
# beside its next block of heads and shuffled rows; and per slow head, its
# redundancy in its own seed (a VIF analogue, the largest in-seed NMI)
# against its seed match and margin
uv run python experiments/next-sentence/modeagree.py --a <run> --b <run> --out <dir>
# The slow heads told apart (experiments/slow-order). train_order.py is
# train_pair.py's Fano arm plus --dec times the mean CKA (normalized
# unbiased HSIC under the "same entry" kernel) between slow heads, through
# orderloss.OrderedPairHyperparams. By default each head's earlier heads
# are stop-gradiented in the term, which orders the block;
# --dec-symmetric decorrelates without the order. order.py scores two
# seeds of one arm on the held-out tail: seedmatch.py's specificity of the
# slow block, whether the match keeps the head index, and how kappa and
# language NMI run with the index
uv run python experiments/slow-order/train_order.py --dec 1 --epochs 1
JAX_PLATFORMS=cpu uv run python experiments/slow-order/order.py --a <run> --b <run> --step <step>
# A code of sentence differences (experiments/pair-diff). diffs.py serves
# x_a - x_b for train_pair.py's pairs (DiffSource) and, run once, writes
# the differences' own whitening and the held-out tail's differences.
# train_diff.py is the stock trainer on that source. evaldiff.py scores two
# seeds' head match on the held-out differences beside two sentence models'
# on the pairs' first sentences, with what each head does when the
# difference is negated
uv run python experiments/pair-diff/diffs.py
uv run python experiments/pair-diff/train_diff.py --epochs 1
JAX_PLATFORMS=cpu uv run python experiments/pair-diff/evaldiff.py --a <run> --b <run> --step <step>

# Where a residual stack puts a planted hierarchy
# (experiments/planted-hierarchy). tree.py plants a two-level tree (11
# parents, 3 of them with 3 children each, a child only under its own
# parent) in dictenc-toy's setting and trains an l-layer stack on it, on
# CPU. Per head it reports the share of the parent's entropy the entry
# carries and the share of what the child adds beyond the parent; per
# layer the prefix's FVU beside the noise floor. --shuffle-tree,
# --child-scale, --l 1 --h 2, --no-deepsup and --k separate the hierarchy
# from variance, depth, the prefix loss and spare entries; grid.sh runs
# them all
uv run python experiments/planted-hierarchy/tree.py --seeds 5
experiments/planted-hierarchy/grid.sh
# Head disagreement with nothing missing (experiments/planted-circle).
# circle.py plants a circle (cuts fixed only up to rotation) and a segment
# (cuts pinned by its ends) and trains tree.py's model at many model seeds
# on the same data; per seed, which head carries each factor and where it
# cuts it; per pair of clean seeds (one head per factor), the distance
# between their cuts (0 same, ~0.5 independent) and the partition NMI,
# beside 1 - 1/(2 ln m), the NMI of equal arcs at independent offsets.
# --tilt makes the circle's density uneven, which should pin the cuts
uv run python experiments/planted-circle/circle.py --seeds 32 --tilt 0.9

# Whether the atoms a row selects together are less alike than the
# dictionary (experiments/ste-arm/coselect.py): the cosine between two
# heads' decoded atoms over the pairs a row selects, over all pairs, and
# over pairs drawn from each head's usage independently (the null, with
# its spread from permutations), within and across layers, as decoded and
# centered on each head's mean atom; --sae does the same for an SAE's
# active latents against sets drawn by firing frequency
uv run python experiments/ste-arm/coselect.py --ckpt <run dirs> --sae <params.npz>
# Whether two checkpoints' heads share the relations among their atoms,
# whichever way the atoms point (experiments/ste-arm/gramqap.py): each
# pair of heads' entries matched from the atom Grams alone, by scipy's
# QAP (FAQ) on the Grams' deviations, for every pair of a layer's heads,
# so the null is matched the same way; atomgram.py's ratio beside it, and
# chance from the halves of --a's own layer. Run it within one run first
# (--b = --a at an earlier step), where stored order says whether the
# Grams can find a correspondence that exists
JAX_PLATFORMS=cpu uv run python experiments/ste-arm/gramqap.py --a <run> --step-a <step> --b <run> --step-b <step>
# Whether a layer's atoms would gain from depending on an earlier layer's
# choice (experiments/ste-arm/condrefit.py, after QINCo): with every
# assignment and the other layers held fixed, the layer's decoded atoms
# are refit by least squares once for all rows, separately within each
# cell of an earlier head, and separately within a random partition of the
# same cell sizes (the null); fit on training rows, scored on mc4_fresh
uv run python experiments/ste-arm/condrefit.py --ckpt <run dir> --layer 1
# Where a code's held-out error sits over the whitened eigenbasis
# (experiments/ste-arm/errspectrum.py): per direction, against reverse
# water-filling at the code's rate (pareto.gaussian_reference's
# allocation), for each layer prefix with the gains pinned to their means,
# and what a per-direction affine rescaling of the reconstruction, fitted
# on half the rows, removes from the other half; --sae runs unquantized
JAX_PLATFORMS=cpu uv run python experiments/ste-arm/errspectrum.py --ckpt <run dir> --sae <run dir> --sae-topk <k>
# Whether a head's cells are convex cones, whether they are convex, and
# whether it ignores its input's sign (experiments/ste-arm/cellshape.py):
# on each layer's own classifier input, the share of means of 2, 8, 32 and
# all of an entry's rows that keep the entry, rescaled to their rows' norm
# (a cone test) and the same means as they are (_raw: convexity on the
# affine slice), with the share kept raw and lost rescaled (_radial), and
# of rows whose negated shape keeps theirs, beside random-projection heads
# (convex cones, so every mean keeps its entry); dir:temperature:step
# scores an earlier step
JAX_PLATFORMS=cpu uv run python experiments/ste-arm/cellshape.py --ckpt <run dir>:<temperature>[:<step>] ...
# Whether a head spends an entry on nothing (experiments/ste-arm/nullentry.py):
# used entries whose decoded atom is below a share of their head's median,
# uncentered and centered on the head's mean write, per layer; for each,
# its rows' gain and, with --index (a harvest's .index.npy), whether its
# rows are the corpus's token mix while the head's other entries are
# token classes
JAX_PLATFORMS=cpu uv run python experiments/ste-arm/nullentry.py --a <run dirs> --index data/activations/gpt2_l8.index.npy
# Where a head's vertices sit (experiments/vertices/place.py): per entry,
# its atom against its cell's least-squares atom and spread, both about
# their head's means (a component every entry shares is reported apart,
# as `common`), the share of the cell's rows past it along the ray from
# the head's centre, and a trimmed extreme with its hard-decode price;
# the nearest-row distances of rows with one head forced (frozen) to an
# atom or an extreme, and of a soft model's argmax decode; and the
# archetypal SAE's best-cosine score beside covariance-shaped random
# directions and an SAE's decoder (--sae)
JAX_PLATFORMS=cpu uv run python experiments/vertices/place.py --ckpt <run dir> --sae <sae.py run dir> --out <dir>
# The one-head seed test from a spherical k-means start
# (experiments/vertices/kmstart.py): `fit` runs spherical k-means per seed
# in the frame w * (u - ubar) of the shape the classifier reads, so the
# start is exact: one classifier factor on the constant coordinate, the
# other on a centroid, logits scaled to a trained stock head's median gap,
# and the decoder moved by the least change that decodes each entry to its
# cell's target; train_kmstart.py is train_eps.py with --km-start, which
# installs it in `init` and refuses to train if it did not take;
# kmstart_seeds.sh trains width_seeds.sh's h = 1 recipe from each seed's
# own fit and from a shared one, after an idle GPU, as a systemd unit
# (README.md), and `score` reads each run against its start and each pair
JAX_PLATFORMS=cpu uv run python experiments/vertices/kmstart.py fit --margin-from data/out/sonar/width_seeds/l1_h1_s42 --out <dir>
experiments/vertices/kmstart_seeds.sh
# Whether a head forced to its cell's trimmed extreme steers better than
# one forced to its vertex or its cell's centroid, in text
# (experiments/vertices/steer.py): at each layer, one head per held-out
# row forced (frozen) to another entry's vertex, extreme or centroid, at
# their own lengths and along each step at matched lengths beside a
# covariance-shaped random direction; each row decoded through M2M100,
# re-encoded by SONAR and read by the model (the forced entry's
# probability at the head, on the row given and after the round trip),
# with 1 - chrF against the reconstruction's decode, the re-encoding's
# cosine to its source row and to the row decoded, and the round trip's
# ceiling; steer.sh runs it on resid_nc after the unit AFTER (default
# kmstart-seeds) and an idle GPU, as a systemd unit (README.md)
JAX_PLATFORMS=cpu uv run python experiments/vertices/steer.py --ckpt <run dir> --device cuda --out <dir>
experiments/vertices/steer.sh
# The planted-simplex toy (experiments/vertices/simplex.py): rows are
# Dirichlet mixtures of planted atoms, so the atoms are the corners of the
# data's hull; one head trained under each selection rule (ste, softmax and
# sparsemax under the stock anneal, sparsemax at a fixed temperature and at
# a decided-share setpoint, and sparsemax-ste-sparsemax schedules that
# switch the model's select between phases, keeping parameters and
# optimizer state), and a Voronoi head whose logits are tied to its atoms
# (voronoi_ste, voronoi_softmax), beside Cutler & Breiman's archetypes; each matched
# entry's vertex, cell centroid and trimmed extreme placed along the ray
# from the atoms' mean through its atom, with place.py's measures, FVU
# against the noise floor, and the code's support. CPU. --h, --l (with
# --deepsup, --resid-norm), --no-const0 and --unit move the model toward
# the SONAR stack one knob at a time; with more than one head or layer only
# place.py's measures are read, per layer. simplex_grid.sh runs each knob
# alone and then all together
uv run python experiments/vertices/simplex.py --seeds 5
experiments/vertices/simplex_grid.sh

# Reference fixtures for readers that reimplement the forward
# (tests/reference, README.md there): small trained dictionaries
# (legacy_const0, shaped like resid_nc, whose spec lacks const0;
# direct_gain_ste, headline's direct gain-shape design with resid_first and
# gain_clip) with 8 input rows, among them a clear winner, a near tie decided
# only in float32 and two rescaled copies, and every layer's classifier input,
# gain, logits, selection and reconstruction, as plain npz/json;
# test_reference checks the forward still reproduces them.
# Regenerate on purpose, and review the diff
uv run python tests/reference/build.py

# A feedback anneal for the planted-code toy (experiments/dictenc-toy):
# --gap-target replaces the stepwise temperature schedule with the
# package's DualLoop on the hardness 1/T, watching the batch's
# hard-minus-soft FVU gap (after Agustsson et al. 2017). Compare with the
# stock schedule, with the stock schedule at the temperature the
# controller reached (--temperature-end), and with --select ste
uv run python experiments/dictenc-toy/toy_dictenc.py --seeds 5 --gap-target 0.01

# Which heads and SAE latents the SONAR source tag moves
# (experiments/lang-tags). tagflip.py reads retag.py's re-encoded rows:
# each head's keep rate, and each latent's stay rate, between a row under
# the cache's tag and under its intended one, beside two random shifts of
# the row's own size, with the noise an excess is read against
uv run python experiments/lang-tags/tagflip.py

# Whether forcing one entry moves decodes into a script while keeping their
# meaning (experiments/entry-force): every entry of the head is forced over
# the same held-out rows (the others are the null), decoded through
# M2M100, and scored by the share of letters in the script and by the
# cosine of the decode's SONAR re-encoding to its row, against another
# row's; usage.csv gives which languages' rows choose the entry unforced.
# English is forced in decoding unless --free-lang, and forced English
# translates, so a language test needs --free-lang; controls.json decodes
# the language's own rows as the positive control. --unit scales rows for a
# model trained on mc4_4M reading mc4_sent16. scan.py forces every entry of
# every head over two halves of the rows and reads the outputs with a
# language probe (ranked on one half, read on the other), then decodes the
# strongest, the entries the language's rows choose, and random ones
JAX_PLATFORMS=cpu uv run python experiments/entry-force/force.py --model <run dir> --layer 1 --head 8 --entry 17 --device cuda --free-lang
uv run python experiments/entry-force/scan.py --model <run dir> --cache data/sonar_embeddings/mc4_sent16.npy --lang my --device cuda

# Non-linear steering on GPT-2 small at the gpt2_l8 site
# (experiments/nonlinear-steering), after Hewitt (2026). rescale.py: how much
# GPT-2's next-token behaviour depends on the layer-8 residual's overall
# scale, which a gain-shape code cannot see, against orthogonal steps of the
# same length, and how Ontologizer and SAE codes respond to both. pilot.py:
# whether GPT-2 can host his constructions there -- one direction whose
# scale picks the task, and two vectors whose sum steers a third -- trained
# by NLL on task text plus KL on the others', beside one independent vector
# per task. Edits apply at positions >= 1 (BOS is never harvested)
uv run python experiments/nonlinear-steering/rescale.py
uv run python experiments/nonlinear-steering/pilot.py --seed 0

# Adam's eps (experiments/adam-eps). train_eps.py is train_ste.py with
# --adam-eps, handed to optax through a subclass of whichever Hyperparams
# class the flags choose (with_eps), so an arm and its control differ in eps
# alone. throttle.py prints, per layer, the median share of a full Adam step
# (sqrt(v_hat) / (sqrt(v_hat) + eps)) the classifier and the dictionary take,
# from the second moments a checkpoint saves, under the run's own eps
# (log.jsonl's adam_eps; absent = optax's 1e-8); given an sae.py run
# directory, per parameter from its state.npz, under meta.json's adam_eps.
# Beside each share it prints the realized relative step, the run's lr times
# |m_hat / (sqrt(v_hat) + eps)| over the block's norm (per unit lr for an
# sae.py run whose meta.json predates its `lr`). Hyperparams.opt keeps
# optax's default, at which the GPT-2 classifiers take about 0.1% of a step
uv run python experiments/adam-eps/train_eps.py --base gpt2_l8 --dict-init-scale 1.0 --adam-eps 1e-15
JAX_PLATFORMS=cpu uv run python experiments/adam-eps/throttle.py <run>:<step> ... [<sae.py run dir> ...]

# The step that holds other heads' margins (experiments/boundary-count).
# nullstep.py builds, per row and target head, the shortest input step that
# carries the head to its runner-up entry with other heads' margins held to
# first order (exactly, while the margins held are fewer than d; by a
# weighted least squares otherwise), and scores it beside the plain
# gradient step, a random step of its length, and `do`, the intervention
# itself. Its frozen_logits is the hard forward with the codes held and
# the residual and gains following the input, which the module's own
# forward stop-gradients; hard residual stacks without router or fibers
uv run python experiments/boundary-count/nullstep.py --model <run dir>

# Two weaker kinds of seed agreement (experiments/seed-probes).
# translate.py predicts each head of --a from --b's code on mc4_fresh, by
# a ridge on one-hot features (and --mlp, a hidden layer), beside the same
# probe from random-projection heads, the untrained model, --a's --earlier
# checkpoint and the input; a score is read as its margin over the
# controls. bandagree.py repeats headcontrib.py's matched cosine within
# bands of the whitened data spectrum
uv run python experiments/seed-probes/translate.py --a <run> --b <run> --earlier <step> --mlp
uv run python experiments/seed-probes/bandagree.py --a <run> --b <run> --earlier <step>
# complement.py asks how much of what --b's code says about the input --a's
# says too: a ridge from one-hot codes to the whitened input, from --a's
# code alone, from each other code alone, and from --a's concatenated with
# it, per layer and over the stack; overlap is the share of a code's
# explained variance --a's explains already, read against --a's --earlier
# checkpoint (redundant), random-projection heads and the untrained model.
# CPU; budget an estimated 25-30 GB at the whole stack
JAX_PLATFORMS=cpu uv run python experiments/seed-probes/complement.py --a <run> --b <run> --earlier <step>
# Seed agreement as subspaces against the data's own principal directions,
# in the whitened frame (experiments/seed-stability/subspace.py): --mode
# sae is Gerasimov et al.'s transfer of matched and unmatched latents'
# decoder subspaces; --mode stack is LoRA's phi(i, j) between two seeds'
# per-layer contribution eigenbases, each beside the data, r/d, and the
# principal subspaces of each seed's layer input (the residual it is fit to)
uv run python experiments/seed-stability/subspace.py --mode sae --a <params.npz> --b <params.npz>
uv run python experiments/seed-stability/subspace.py --mode stack --a <run> --b <run> --step <step>
# A Gaussian copy of mc4_sent16 as the baseline for seed agreement
# (experiments/seed-stability/gausscopy.py): prepare saves the training
# prefix's mean and covariance and writes held-out Gaussian rows in the
# cache layout, which the scorers read with --cache; train_gauss.py is
# train_eps.py on rows from that Gaussian (GaussianSource, a seeded stream
# generated in blocks, so a resume sees the same rows); gauss_seeds.sh
# trains the ladder's all-off arm on it at seeds 42 and 43 as a systemd
# unit and scores the pair as the cache pair was scored; gausscopy.py
# mixture keeps each language's mean and covariance too (mixture.npz, from
# the .langs.npy sidecar), for a copy with the data's language structure and
# nothing beyond it; gausscopy.py head writes the training stream's first
# rows (head.npy) for a scorer that reads a cache's head
uv run python experiments/seed-stability/gausscopy.py prepare
uv run python experiments/seed-stability/gausscopy.py mixture
experiments/seed-stability/gauss_seeds.sh
# The k32 SAE seed pair on a Gaussian copy of mc4_4M: sae_gauss.py is
# sae.py with --gauss DIR, which hands sae.train(cfg, rows=...) the copy's
# training stream, generated in memory, and the end of its eval.npy in
# place of a cache; sae_gauss_seeds.sh trains sae_conv/m5120_k32's settings
# at seeds 42 and 43 after the unit AFTER (default width-seeds) and an idle
# GPU, as a systemd unit (README.md), and scores the pair with subspace.py
# --mode sae and splitting.py
uv run python experiments/seed-stability/sae_gauss.py --gauss data/out/sonar/gausscopy_4M --epochs 150 --out <dir>
experiments/seed-stability/sae_gauss_seeds.sh
# How far a k-cell partition of the data reproduces at all
# (experiments/seed-stability/kmeans_seeds.py): k-means in the whitened
# frame at several seeds, fit on training rows and assigned on the tail,
# with partition.py's NMI between every pair of seeds beside a shuffled
# null; --gauss repeats it on the Gaussian copy, --mixture on the language
# mixture copy (each row drawn under the real row's language),
# --within-language on the cache's rows less their language's mean,
# --marginals pca ica on copies of the cache's rows with each principal
# direction's (pca) or each FastICA source's (ica) column permuted on its
# own, which keep every coordinate's marginal and nothing that ties them
# (coordinates_<frame>.csv gives each one's skew, kurtosis and bimodality,
# frame_<frame>.npz the fitted frame, --ica-patience damps FastICA's steps
# row by row when they stop improving, which they can do by cycling
# (--ica-shared-step: one step for all rows, which stalls from a random
# start but fitted frame_ica.iter800.npz), and --ica-init
# continues FastICA from a saved frame_ica.npz or the checkpoint
# frame_ica.partial.npz on the same rows, or with --ica-iters 0 uses it as
# it is), --groups TAU ... on copies that
# keep groups of FastICA sources whose magnitudes correlate (Spearman of
# |s|, average linkage cut at mean correlation TAU) and permute groups
# apart, --scale-copy on a copy that keeps each row's scale shared by its
# sources (the median of their magnitudes) and shuffles each source within
# bins of that scale, --kmeans spherical clusters the centred rows'
# directions (with FVU against each cell's mean row) and --trim-seed Q seeds
# k-means++ from rows within the Q quantile of distance from the mean, every
# table gives the largest cell's share, --no-cache leaves the cache's own rows unscored, and --ckpt adds
# each layer's heads' best NMI with the k-means cells at the model's k; the
# cache, the mixture and within also score each seed's cells against the
# rows' languages
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py --gauss data/out/sonar/gausscopy --ckpt <run>
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py --mixture data/out/sonar/gausscopy/mixture.npz --within-language data/out/sonar/gausscopy/mixture.npz --out data/out/sonar/kmeans_seeds_language
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py --marginals pca ica --out data/out/sonar/kmeans_seeds_marginals
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/kmeans_seeds.py --groups 0.2 0.1 --ica-init <frame_ica.npz> --ica-iters 0 --no-cache --out <dir>
# Whether FastICA's sources reproduce (experiments/seed-stability/ica_seeds.py):
# kmeans_seeds.py's FastICA on its training rows (--frame-a continues a saved
# fit for --iters-a more iterations, 0 using it as it is) and on as many
# disjoint rows for --iters, each of A's sources matched one to one
# with B's on the tail by |correlation| and by the |cosine| of their
# whitened directions, beside B rotated at random and the principal frames
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/ica_seeds.py --frame-a <frame_ica.npz> --iters-a 0 --iters 800 --out <dir>
# Whether a stack's atoms point along FastICA's sources
# (experiments/seed-stability/atomsources.py): per layer, each source's best
# |cosine| with the layer's centered decoded atoms, by class from
# ica_seeds.py's sources.csv (skewed, symmetric, other), beside the sources
# rotated at random and the principal directions; with --b, the sources
# both runs have an atom along, beside independence
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/atomsources.py --frame <frame_ica.npz> --sources <sources.csv> --a <run> --b <run>_s43 --out <dir>
# Which heads reproduce as heads (experiments/seed-stability/headscreen.py),
# a multitrait-multimethod screen: matched across seeds on a third of the
# tail rows, each head's Miller-Madow-corrected NMI with its partner must,
# on the other two thirds, beat zero, both sides' runner-up partners and
# both sides' closest heads within their own seed, each by --z SDs of the
# two thirds' disagreement
JAX_PLATFORMS=cpu uv run python experiments/seed-stability/headscreen.py --a <run> --b <run>_s43 --rows 65536 --out <dir>
# Seed agreement against width in the trained stack
# (experiments/seed-stability/width_seeds.sh): one-layer stacks of h heads
# (the ladder's all-off arm, WIDTHS default "1 4 16 76") at seeds 42 and
# 43, each pair scored by partition.py and ratecurve.py; waits for an idle
# GPU, run as a systemd unit (README.md), beside kmeans_structure.py's LSQ
# curve
WIDTHS="1 4 16 76" experiments/seed-stability/width_seeds.sh
# Summed heads against a cascade, with quantizers alone
# (experiments/seed-stability/kmeans_structure.py): --mode aq sums h
# codebooks at one stage (FAISS's LSQ++, not a project dependency, hence
# --with faiss-cpu) over restarts, matching heads between runs by NMI and
# reading agreement against each run's error; --mode rvq is a cascade of
# k-means stages, run per seed and with stage 0 shared
JAX_PLATFORMS=cpu uv run --with faiss-cpu python experiments/seed-stability/kmeans_structure.py --mode aq --gauss data/out/sonar/gausscopy

# The recipe ladder (experiments/recipe-ladder): both Ontologizers on
# mc4_sent16 with one training noise (--sd-k, --sd-f, --p-drop) or one
# auxiliary weight (--s-bcossim, --s-support, --s-l1f, --s-hm) removed at a
# time against a control and its seed-43 replica, 50k steps each, keeping
# only the final checkpoint (train_ste.py --max-to-keep 1
# --checkpoint-each 50000); and the k32 SAE with the Ontologizer's noises
# added (sae.py --sd-pre/--p-drop/--sd-out). WEIGHTS is no837 or med837;
# ARM names arms from ladder.sh's table (default: the one-change arms),
# which also removes or rescales --s-bcossim and --s-support together and
# apart, removes --s-l1f with both, and --s-hm with all three (that arm also
# at seed 43, and without training noise at seeds 42 and 43); ARCHS in the
# environment limits it to the architectures named. sae_k256_run.sh trains
# the k160 SAE with --topk 256. All
# three run as systemd units that wait for an idle GPU (README.md).
# rowbudget.py reads SAE codes held to a per-row budget at exact budgets,
# where quantrate.py's rowbudget.csv samples a grid and interpolates: rate
# control, each row's longest code that fits, one step for every row at the
# budget's mean, and the one step at which every row fits; and the least
# budget at which rate control reaches each --targets FVU
experiments/recipe-ladder/ladder.sh WEIGHTS [ARM ...]
experiments/recipe-ladder/sae_noise_runs.sh
experiments/recipe-ladder/sae_k256_run.sh
JAX_PLATFORMS=cpu uv run python experiments/recipe-ladder/rowbudget.py --runs <sae.py runs> --targets <FVUs>
# noiseshare.py replays a stack's training forward with each training
# noise alone (sd_F, sd_K, p_drop, at the run's values or the flags') and
# together, beside the clean forward: per layer, the noise's share of the
# input residual, the shape's angle and the heads whose entry flips
JAX_PLATFORMS=cpu uv run python experiments/recipe-ladder/noiseshare.py --ckpt <run dirs>

# The hard stack trained on its entropy-coded rate (experiments/rate-penalty).
# A negative s_Hm charges each head's batch-mean usage entropy, -s_Hm/h per
# bit, which is entropy-constrained VQ per head; write it --s-hm=-2e-8, since
# argparse reads a bare -2e-8 as an option. arms.sh trains the ladder's
# all-weights-off hard stack at three such weights, one at a time once the
# GPU is idle, as a systemd unit, then scores them. ratecurve.py reads, per
# layer prefix on the tail, FVU over the free coordinates against the
# realized bits (each head's plug-in usage entropy), with the gains pinned
# to their means or as computed. headdrop.py reads the rates the first run
# reaches without retraining: each prefix with its last layer's heads
# dropped, cheapest per bit first (a dropped head writes its mean output),
# and the lower convex hull of those points, against which every later run's
# points are read at the same rate; headbits.csv gives each head's bits and
# entries used
experiments/rate-penalty/arms.sh
JAX_PLATFORMS=cpu uv run python experiments/rate-penalty/ratecurve.py --ckpt <run dirs>
JAX_PLATFORMS=cpu uv run python experiments/rate-penalty/headdrop.py --ckpt <control> <runs>
# plainquant.py: FAISS's product and residual quantizers fit in the same
# whitened frame and scored on the same tail rows, at the stack's rates (a
# residual quantizer of 380 stages of 32 gives all five prefix rates); FAISS
# is not a project dependency, hence --with faiss-cpu
JAX_PLATFORMS=cpu uv run --with faiss-cpu python experiments/rate-penalty/plainquant.py

# This repository's jobs on a Cloud TPU VM (experiments/cloud). tpu.sh, on
# the laptop, stages code (git archive of HEAD), data and queue files in a
# bucket, then requests, starts, checks and deletes one node: flex-start,
# spot or on-demand. Its settings come from ~/.config/ontologize/cloud.env,
# outside the repository (cloud.env.example). On the VM, restart.sh pulls
# everything (first refusing a stage/data whose pull would not leave
# ONTO_DATA_SPARE_GB free under data/, where outputs land too), builds
# uv.lock's environment without the CUDA stack plus jax[tpu] (bootstrap.sh,
# tpureqs.py), and starts one runq.py per queue
# (q<i>.txt: chip i alone; q.txt: the whole VM). runq.py pulls each job's
# output directory from the bucket, so a resuming trainer continues, and
# mirrors it back while the job runs; it writes a marker per job and passes
# SIGTERM on. Jobs default to JAX matmul precision `high`, since a TPU's
# float32 default is one bfloat16 pass. Never `uv run` on the VM
experiments/cloud/tpu.sh stage code
experiments/cloud/tpu.sh up && experiments/cloud/tpu.sh wait && experiments/cloud/tpu.sh start

# A Base-LCM as a model organism (experiments/lcm-organism: plan.md, README.md).
# lcm.py is the model (robust scaler, PreNet, a causal pre-norm transformer
# with RoPE and a gated MLP from NLinear and Linear in float32, PostNet) and
# its padded document batches. --gate silu (default) is SwiGLU; --gate none
# builds the MLP's up-projection as Bilinear, so its weights can be read.
# train_lcm.py trains it on the documents before the last 262,144 rows,
# scoring a prediction only against the next segment. It resumes, keeps
# step 0 (the untrained init), saves on SIGTERM and exits 75, and logs
# held-out FVU_w at `highest` beside the ridge and context-ridge baselines.
# harvest_lcm.py writes a block's states as a cache train_pair.py and
# seedmatch.py read: the prefix keeps its row numbers and the tail follows,
# with the sidecars sliced and matched MSE weights. stateagree.py compares
# two harvests of the same rows: CKA, Procrustes (rotation), whitened
# Procrustes and ridge R^2 (linear map); --prefix-rows 0 harvests the tail
# alone for it. stateinput.py is Gate B: the held-out R^2 of each block's
# state from x_n, from x_n with the run's earlier mean, and from the
# model's own state for x_n read alone
uv run python experiments/lcm-organism/stateinput.py --run data/out/lcm-organism/lcm_l8_d1024_s42 --blocks 4 8
uv run python experiments/lcm-organism/train_lcm.py --seed 42 --epochs 1
uv run python experiments/lcm-organism/harvest_lcm.py --run data/out/lcm-organism/lcm_l8_d1024_s42 --block 4
uv run python experiments/lcm-organism/stateagree.py --a <harvest> --b <harvest>
# spectrum.py: each arm's covariance spectrum in the frame its loss weighs
# (participation ratio, effective rank, directions per share of the
# variance, pareto.gaussian_reference's FVU at --bits rates, the share
# between languages), of the total and the pooled within-language
# covariance, on the same prefix and tail rows of every arm. recolour.py is
# the control for effective dimension: SONAR's rows through one linear map
# that gives them a harvest's spectrum rank for rank, with a random rotation
# so the trainer's inverse-variance weights stay near uniform, laid out as
# that harvest with its sidecars linked and its own matched weights
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/spectrum.py
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/recolour.py
# readout.py is the control for context averaging: Gate B's ridge readout of
# a harvest's state from x_n, the mean of its document's earlier rows and an
# indicator of n = 0, written at every row and laid out as the harvest
# (--features row: from x_n alone, Gate B's `x`, into lcm_readout_x_b4.npy).
# phase2.sh trains and scores arms on one GPU, waiting for it to be idle;
# ARMS names the arms by cache (<arm>_b4.npy). ctxmean.py is the fixed mix:
# SONAR's row plus --weight times its document's earlier mean, with no LCM,
# laid out as the harvest and weighted as the SONAR arm (no837, rescaled).
# recolour.py reads a source already laid out as its target row for row,
# so --source <ctxmean>.npy recolours the fixed mix, and with --about-mean maps
# each row's variation about the source's mean, leaving the mean where the
# source's weighted frame has it, scaled by c, or with --center maps that
# variation and leaves the mean out. weightframe.py writes a
# cache's rows in the frame its loss weighs, c * sqrt(w) * y, unrotated, with
# --scale-like taking c from a recoloured cache: recolouring a cache to its own
# spectrum is these rows rotated, so the two arms differ by the rotation alone.
# --rotate-except J keeps coordinates J on their axes and mixes the other live
# coordinates among themselves with one random rotation. Within J,
# --rotate-block K mixes K among themselves by a rotation of their own, apart
# from the rest (with --block-basis principal, onto their principal axes over
# the prefix), --whiten-block K whitens them among themselves by ZCA, keeping
# their mean and total variance, and --gaussianize K maps each coordinate in K
# onto a Gaussian with its mean and SD through its empirical distribution over
# the prefix, keeping ties and the rows' order; --weights-like takes another
# cache's loss weights, refused unless they are equal across the block;
# --center subtracts the mapped rows' prefix mean as the last step; and with it
# --move-mean J|random puts the mean back at its own norm on the coordinates J
# (equal shares, each with the mean's sign there) or along a direction drawn
# over the live coordinates (--mean-seed), so the arm differs from the
# uncentered rows only in where the mean lies. axes.py reads what
# each arm's axes carry that a rotation would remove, in that frame: coordinate
# marginals and magnitude dependence beside random directions, the correlations'
# tail beside random rotations of the covariance, the group of coordinates with
# a strong partner (--disjoint rereads it on other rows), and --draw K
# coordinates outside it, for --rotate-except. templates.py flags the held-out
# tail's templated rows (a sentence another document of the cache repeats,
# digits aside; a SONAR near-twin in another tail document; a skeleton, the
# sentence with names, numbers and URLs masked, that another document
# repeats; masked word 3-grams overlapping another tail document's row, by
# MinHash and an exact Jaccard check) and writes masks for seedmatch.py's
# --row-mask, which leaves a mask's False rows out of every score, with a
# control that drops as many rows of each language at random. meanplace.py
# reads where each arm's mean lies, as the gain-shape layer reads the rows,
# on the same rows of every arm: its share of a row's squared norm, how it lies
# against the covariance (the variance along it, its principal directions, its
# Mahalanobis distance) and against the coordinate axes (how many it spreads
# over, its largest coordinate, how far each sits from zero in SDs), what a
# row's norm says about its unit shape, whether centering changes the cells (the
# held-out FVU of an affine map between the centered and uncentered unit
# shapes, and spherical k-means seed agreement on each, overall and within each
# language, beside a shuffled null), and the languages' own offsets
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/readout.py
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/ctxmean.py --weight 1
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/weightframe.py --scale-like <recoloured>.npy
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/axes.py --disjoint --draw 19
uv run python experiments/lcm-organism/templates.py
JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/meanplace.py --arms lcm_wf sonar_wf lcm_readout_x_wf
ARMS=lcm_readout experiments/lcm-organism/phase2.sh train
# A float16 copy of a float32 cache, for the upload (experiments/lcm-organism).
# Writes <stem>_f16.npy in chunks through a .partial file, links the
# sidecars under the copy's stem, and records in its meta.json what
# rounding cost (each row's relative error, whitened FVU per --weights).
# --bulk DIR writes the array on another disk with a symlink in its place.
# train_pair.py's StackedPairSource and seedmatch.py read it as float32;
# NpyDataSource returns float16 rows
uv run python experiments/lcm-organism/f16cache.py --cache data/sonar_embeddings/mc4_sent16.npy --bulk <dir>

# Train on SONAR multilingual sentence embeddings (mC4 data)
uv run python sonar.py

# Interactive reconstruction + intervention REPL against a trained checkpoint
uv run python decode.py

# Decode a trained model's learned dictionary tags to text
uv run python decode_tags.py

# SAE-baseline comparison suite (all score on the cache tail sae.py holds out):
uv run python sae.py           # train a field-standard top-k SAE on the same
                               # cache + whitened MSE (CLI flags; built to be swept).
                               # --groups/--group-fn/--prefixes add Ontologizer-style
                               # structure one rung at a time (heads / deepsup layers).
                               # --block B --topk L --aux-k 0 trains the vanilla
                               # block-sparse featurizer (Fel et al. 2026): the L
                               # largest-norm of m/B blocks of B signed latents.
                               # --sd-pre/--p-drop/--sd-out add the Ontologizer's
                               # training noises (selection noise, winner dropout,
                               # output noise) to a flat linear top-k SAE; they
                               # apply only in training (SAE.loss with an rng)
uv run python pareto.py        # reconstruction-vs-code-capacity Pareto table:
                               # Ontologizer top-m deviation codes vs trained SAEs;
                               # a model whose selection is not softmax also gets
                               # a softmax(K/t) row (under ste its "soft" row is
                               # the argmax); --surrogate gaussian[_unit] scores
                               # every model on Gaussian rows with the tail's
                               # mean and covariance instead of the tail
uv run python autointerp.py -h # auto-interp pipeline (harvest/texts/describe/
                               # score) over Ontologizer tags & SAE latents;
                               # `language` stratifies detection by how many
                               # languages a feature's top rows span; `match`
                               # draws language-matched rows for
                               # --mode cacts_lang / --negatives lang
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
                               # containment (parents/children/matches, absorption);
                               # beside it the field's seed-agreement statistics,
                               # MPPC, Leask et al.'s decoder-cosine share and
                               # PW-MCC (agreement.json)
uv run python textfid.py -h    # downstream text fidelity ("loss recovered"): decode
                               # x vs recon(x) through M2M100, chrF + NLL vs floor;
                               # its SonarDecoder is the shared decode/score helper
uv run python textstrip.py -h  # per-sentence text strips: layer prefixes, and per
                               # head live/frozen uniform + zero ablation, a null
                               # step of equal length, the head's write alone;
                               # ablations scored by dNLL against decode(x_hat)
                               # and decode(x) (chrF flips on rounding), with a
                               # per-layer live/frozen table, HTML out.
                               # --removals: the live/frozen split in embedding
                               # space for unif/mean/runnerup/resample
                               # replacements of random heads (added error vs
                               # departure from x_hat), no decoder
uv run python langprobe.py -h  # SAEBench-style k-sparse probing of language
                               # identity (langs sidecar); dense-probe ceiling;
                               # --only-langs restricts rows to listed languages
uv run python refit.py -h      # shrinkage: refit LS coefficients on each frozen
                               # support; selection-vs-magnitude error split,
                               # each share beside a random support's of the
                               # same size (random_share) and |S|/d
uv run python quantrate.py -h  # operational rate: quantize every continuous number
                               # a code sends (SAE coefficients, soft assignments,
                               # gains), entropy-code the symbols, sweep the step;
                               # FVU vs total bits per code, one axis for all.
                               # Rates are means over rows; rowbits.csv and
                               # rowbudget.csv give each SAE's per-row lengths
                               # and its FVU held to a per-row budget
uv run python assignmap.py -h  # per-layer heatmaps of held-out samples x
                               # (head, entry) assignment probabilities, rows
                               # sliced by lang / GPT-2 token class / pos / doc;
                               # makes head collapse and dead entries visible
                               # per sample. --sae adds a block of a sae.py
                               # run's codes on the same rows, in each page's
                               # row order (trained groups as heads, a flat
                               # code cut to its most used latents).
                               # --replot redraws from assign.npz
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
                               # per layer and vs step; per head, the share of the
                               # forms in their head-mean form, which the assignment
                               # cannot see, and whether centering keeps each
                               # feature's dominant-eigenvalue sign
uv run python selection.py -h  # held-out noise2self partition score vs a sweep key
                               # with unpartitioned + size-matched random baselines
                               # drawn; --grid ROW COL = pareto.py's FVU over a grid;
                               # --jinv adds the J-invariant split score, the
                               # bandwidth it selects (--taus) and random-projection
                               # (LSH) heads as the null, in jinv.csv
uv run python anatomy.py -h    # one (layer, head) as a block figure for the methods
                               # section: P_h, PWAK kernel, rowcos, support_overlap, W_h
uv run python steerfid.py -h   # steering fidelity: effect (cycle-consistency
                               # activation gain) vs collateral (1-chrF) at matched
                               # magnitude; W_dec / eigenfeature / withArgs steering
```

Seed stability: train a replica with `sae.py --seed 43` (run dirs get an
`_s43` suffix) and feed the pair to `splitting.py` — its "A matched"/"B
matched" columns are the reproducible-feature fractions.

Tests: the curated suite lives in `tests/`, wired via `[tool.pytest.ini_options] testpaths = ["tests"]` in `pyproject.toml`, so a bare `uv run pytest` runs only it (fast, CPU-only — `tests/conftest.py` forces `JAX_PLATFORMS=cpu`, so it is safe to run alongside a live GPU training run). It covers: `deepsup_sg` gradient semantics, `resid_norm`/`resid_const` conditioning (including bilinear sign-blindness), the `s_Hm`/`KL_m` batch mean-entropy bonus, top-k tag selection (`select="top<k>"`), sparsemax selection (`select="sparsemax"`: the projection's optimality conditions, its support against the temperature, its Jacobian on the support, a dropped winner, and finite training steps; `test_sparsemax_select`), the `s_L2pwak`/`L2_pwak` noise2self partition score and the `s_pwak`/`KL_pwak` consensus KL with the kernel the two share (`test_pwak_kl`), and the walk both take, which re-zeroes the self-edge at every depth (a hard cell's s=2 score equals its s=1 score, where a plain G² would scale it by ((n−2)/(n−1))²), noise-path NaN regressions, the reference fixtures (`test_reference`: the forward reproduces every stored array of `tests/reference`, winning entries exactly; the legacy fixture loads through `migrate_spec`'s inference of `const0`; the near tie is decided in float32 and not at float16's resolution; rescaled rows keep their layer-0 codes under gain-shape, and the direct gain-shape fixture is positively homogeneous at every layer), the flags ported from `headline` (`test_headline_flags`: `logit_norm` standardizes each head's logits, keeps their order and makes a softmax blind to their scale; `resid_first` is the plain input without a decoder and the row about decode(0) with one, with that residual's norm as layer 0's gain; `gain_clip` caps the residual's norm at the row's and keeps its direction; contradictory settings and the ghost path refuse), checkpoint save/restore including the `n_stats` stats-width migration, the legacy-spec-key rename and a checkpoint that stores neither of the class's names, `ConcatDictBlock`'s overridden statistics, the gated SAE encoder's gradient routing, `NLinearBlock.withL1`'s gate-factor L1, `L1_S`'s position in the stats row, `support_overlap` (a single head reads 0, not NaN) and the retired `cossim_h`/`cossim_flat` slots (`test_support`), fibers and head dropout (`test_fibers`), the layers' float32 parameter defaults (`test_layer_dtypes`), the SAE modules' layout converters and configuration checks (`test_sae_modules`), `TokenizeTransform`'s source-language tags (`test_tokenize`), `NpyDataSource` and the batching of `(input, target)` records (`test_loaders`), `l2_pooling` with and without its L2 step (`test_l2_pooling`), sentence segmentation, the script check and run selection (`test_segment`) and `encode_sentences.py`'s choice of a document's sentences and token-capped batches (`test_encode_sentences`), and a short end-to-end training loop per live run configuration. The suite's largest block now tests the root eval scripts' pure helpers (`test_sae`, `test_autointerp`, `test_pareto`, `test_headstruct`, `test_headcoh`, `test_compose`, `test_splitting`, `test_textfid`, `test_langprobe`, `test_refit`, `test_steerfid`, `test_encode_acts`, `test_assignmap`, `test_enrich`, `test_headnmi`, `test_entrydrift`, `test_bilinspec`, `test_selection`, `test_anatomy`, `test_textstrip`, `test_quantrate` import them directly), and some `experiments/ste-arm` analyses' the same way (`test_steerembed`, `test_headlang`, `test_logitscale`, `test_atomgram`, `test_reconseeds`, `test_headrange`, `test_coordfvu`, `test_coordtrace`, `test_auxpull`, `test_train_ste` (the recipe flags reach what `setup` builds), `test_coselect` (heads planted to avoid one another's directions read below the usage null and independent heads at it, for head pairs and for an SAE's active sets), `test_condrefit` (a per-cell refit beats a random partition's only when the planted atoms depend on the cell, and an entry a cell never selects keeps its prior atom), `test_errspectrum` (the water level reproduces `pareto.gaussian_reference`, per-direction errors add up to the total, and the rescaling recovers a planted scale), `test_cellshape` (linear heads keep every mean of their own rows, rescaled or not, and give negated rows another entry; an affine cell that is not a cone keeps its raw means and loses rescaled ones; squared logits are sign-blind and split a cell into opposite lobes), `test_nullentry` (a used near-zero atom is quiet, a rare one and a uniformly small head are not, an entry at the head's mean write is quiet only centered, and token concentration reads 1 for random rows), `test_gramqap` (the QAP recovers a relabelled Gram, finds heads whose atoms were relabelled and rotated where the match on atoms misses them, reads independent heads near chance, and its best order never reads above the atom match), `experiments/cloud`'s `test_cloud` (with a local directory as the bucket, the queue runner skips a finished job, a stopped job's progress reaches the bucket and a fresh VM resumes it rather than restarting it, a failing job is retried and then given up, and the chip and precision reach the job; the TPU requirements drop the CUDA and torch stack; restart.sh's fit check counts only what a pull of stage/data adds and refuses a stage that leaves too little free), `experiments/lcm-organism`'s `test_lcm_organism` (the float16 copy is the cache rounded, across chunks that do not divide it and with `--bulk` on another disk; the sidecars resolve under the copy's stem; values beyond float16 and a copy that would fill the disk are refused; an LCM state never sees later rows, padding or other documents; its MLP is SwiGLU from `NLinear` and `Linear` with every parameter float32, and with `--gate none` its weight loads into `Bilinear`, whose interaction tensor reproduces the layer; the scaler round-trips; only the next segment is a target; the LCM learns a planted dynamic that x_t cannot predict; a SIGTERM saves the step and a rerun resumes; a harvest keeps its row numbers and aligned sidecars, and a tail-only one has no weights; Gate B reads a state with attention zeroed as wholly its own row's; the agreement statistics tell a rotation from a linear map from nothing; the spectrum reads planted eigenvalues in the weighted frame, leaves a rare label out of the within-language covariance, and reads a harvest at the cache rows it keeps; and recolouring gives the target spectrum rank for rank with every coordinate's variance near 1; the readout's features are Gate B's and stay in their document, and on states planted as an affine map of them it reproduces the states row for row, which a readout from x_n alone cannot; the fixed mix adds the weighted earlier mean and keeps the source's weights, scaled to its whitened base, and recolouring reads a source laid out as its target row for row, about the mean moves every rotated row by one offset that puts the mean where the unrotated rows have it, with the rotated rows' weights, centered moves the rotated and the unrotated rows each by one offset to a prefix mean of zero, the centered rotated rows being the centered unrotated ones rotated with no offset and both keeping their uncentered weights, and to its own spectrum is a scaled rotation, of the rows weightframe.py writes, whose own weights are flat and zero what the source's zero, and which with `--rotate-except` keep the listed columns and map the rest by an orthogonal matrix, with `--rotate-block` map a kept block by an orthogonal matrix of its own and leave the rest as without it, or with `--block-basis principal` turn it onto uncorrelated coordinates whose variances are its eigenvalues, with `--whiten-block` give the block equal variances and no correlation and keep its mean, and with `--gaussianize` keep each kept coordinate's row order, mean and spread and give it a Gaussian's shape (the map fitted on the prefix, keeping ties and taking the end quantiles beyond its range), and `--weights-like` takes another cache's weights only when they are equal across the block, and after centering `--move-mean` moves every row by one offset of the mean's norm, on the coordinate named with the mean's sign there or spread over the live coordinates, and keeps the centered rows' weights; on the axes, a skewed coordinate is outlying and not after a rotation, one random scale per row reads the same magnitude excess in any basis and a Gaussian none, and a correlated group is found in the cache's numbering and not after a rotation, with the draw avoiding it; a sentence is a template only when another document repeats it, case, spacing and digits aside, the hashes read every line once however the file is split, and a twin is the closest row of another document, with the last chunk padded; a skeleton masks names, numbers and URLs and counts only with two words left, MinHash signatures agree about as often as the sets' Jaccard index, a shingle twin is another document's, and the random control drops as many rows of each language; and where a mean lies, read as one axis when it is on one coordinate, at that coordinate's offset in SDs, and as n axes in equal shares over n, with the variance along a planted direction, a planted share between languages, an affine map fitted exactly and independent rows not at all, and separated directions reproducing across k-means seeds where the shuffled null does not, on an arm read as a harvest), `experiments/planted-hierarchy`'s `test_planted_hierarchy` (a child appears only under its parent unless shuffled, the conditional score does not credit what the parent implies, and the probe's last prefix is the model's output), `experiments/lang-tags`'s `test_retag` and `test_tagflip` (the keep and stay rates, and the noise an excess over the random shifts is read against), `experiments/entry-force`'s `test_entry_force` (the script share counts letters only, the null pairing has no fixed point, the forced rows avoid the script's language, and the scan's AUROC, affine probe read-back and usage log-odds), `experiments/sentence-split`'s `test_cachestats`, `experiments/recipe-ladder`'s `test_rowbudget` (rate control's FVU does not rise with the budget, which its least-budget search relies on) and `test_noiseshare` (layer 0's input never carries noise, feature noise reaches every later layer's, and the clean replay is deterministic), `experiments/rate-penalty`'s `test_ratecurve` (an evenly used head reads log2(k) realized bits, a constant one none and a skewed one its entropy, and the prefix FVU drops zero-weight coordinates as `coordfvu.py` does) and `test_headdrop` (heads are dropped by FVU per bit saved, silent heads first, and the hull passes under a point above its chord) and `test_plainquant` (the FVU's ends, and a residual code's prefixes are partial sums of its chosen entries), `experiments/next-sentence`'s `test_next_sentence`, which also covers `train`'s `(input, target)` batches, `sfa.py`'s pair matrix, moments and fits, and `seedmatch.py`'s row mask (the mask's last rows, and refusing one that covers fewer than are scored), and `test_closure` (a lumpable chain's coarse-graining reads a zero closure gap and a non-lumpable one the planted 1 − H(0.9) bits), `experiments/slow-order`'s `test_slow_order` (the dependence term reads 1 for copied heads and 0 for independent ones with no small-batch floor, its ordered gradient spares the earlier head, and `order.py` tells a match that keeps the index from a permuted one), `experiments/pair-diff`'s `test_pair_diff` (the source serves row minus partner, the differences' weights keep the sentence weights' mask and sum, and the seed-match statistics), `experiments/nonlinear-steering`'s `test_nonlinear_steering` (the equal-length steps, the next-token metrics, the hook that leaves BOS alone, the task windows, and the pilot's objective and pass rule), `experiments/adam-eps`'s `test_adam_eps`, which checks that `--adam-eps` reaches optax — Adam, AdamW, and `DictWDHyperparams`'s own `opt` — the throttle arithmetic, the relative step (Adam's own last update over the block's norm), and its reading of `sae.py`'s saved state, `experiments/transcoder`'s `test_transcoder` (the labels-forwarding transcoder builds from its own flags and trains on a cache pair), `experiments/boundary-count`'s `test_boundary_count` (the constrained and weighted steps meet and minimize what they say, no step exists once the constraints span the target, and the frozen forward's Jacobian matches finite differences above layer 0), and `experiments/seed-probes`'s `test_seed_probes` (the ridge reads a copied head and not noise or a joint function of two heads, which the hidden-layer probe does read; band agreement is found only in the band it is planted in) and `test_complement` (a relabelled copy overlaps fully and a code of independent information not at all, the block-assembled centered system is the ridge on centered features, and the penalty is chosen on validation rows), `experiments/seed-stability`'s `test_subspace` (φ is 1 for nested subspaces, 0 for orthogonal ones and r/d for random ones, and two seeds sharing a planted subspace transfer at the optimum; contributions that follow inputs sharing their top directions read the input agreement) and `test_gausscopy` (the chunked moments match numpy's under a large mean, the covariance's square root holds at reduced rank, and the Gaussian source's rows have the planted moments, survive a pickle and a block boundary, differ between streams, match the held-out file under its stream, and are generated once per block under concurrent readers; per-language moments match numpy's and recover the pooled covariance, and mixture rows follow their language's moments, depend only on its stream and refuse an unknown language; `sae_gauss.py`'s rows are the training stream's and its held-out rows the end of the held-out file) and `test_kmeans_seeds` (the assignment is the nearest centre; separated clusters and a dominant direction reproduce across seeds and an isotropic Gaussian does not; every cell ends up used; on rays at widely varying lengths spherical k-means finds the directions where Euclidean k-means lumps most rows into one cell, and trimmed seeding keeps the farthest rows out; the principal-frame marginal copy keeps each coordinate's values and the covariance and loses a planted dependence, FastICA recovers planted sources, with damped steps too, per row or shared, and stays at its fixed point when started there, a saved ICA fit continues, or is used as it is, only on its own rows, and the ICA copy keeps a non-orthogonal mix of uniform sources inside its parallelogram where the principal copy spills out of it; sources planted to share a scale are grouped by their magnitude correlations, and the group copy keeps that correlation within a group and breaks it between groups; the scale copy keeps a scale shared by every source, which the ICA copy breaks, and adds next to none without one; a second draw of training rows avoids the first), `test_atomsources` (copied directions are found along and others are not, sources are classed by kurtosis and skew, the rotated reference keeps the spectrum, centering drops unused entries, and both runs' coverage reads against independence), `test_ica_seeds` (a relabelled sign-flipped copy matches exactly; non-Gaussian sources reproduce across disjoint samples, a random rotation does not, and Gaussian sources land in a new rotation of their plane) and `test_headscreen` (independent partitions read zero once Miller-Madow-corrected; a shared head passes, a shared head with a copy in its own seed fails its within-seed check, and independent heads fail) and `test_kmeans_structure` (the cascade's prefix error falls and a shared first stage is kept, a relabelled and reordered copy of a code matches at NMI 1 and independent codes near 0, and FAISS's packed codes unpack least significant bit first across bytes), `experiments/planted-circle`'s `test_planted_circle` (the tilted sampler's mean cosine, cuts recovered from planted arcs and intervals and from a flickering boundary, the cut distance's ends, the NMI's ends, and equal arcs at random offsets reading 1 − 1/(2 ln m)), `experiments/next-sentence`'s `test_modeagree` (relabelled copies of a block predict its modes almost exactly while an independent block and shuffled rows do not, and a duplicated head reads a VIF R² near 1), `experiments/vertices`'s `test_vertices` (a cell's least-squares atom and a hard decode's excess over it, the share of a planted cell's rows past an atom at its centroid, edge and beyond, the atom's reach and the trimmed extreme, a planted layer's centroid atoms read as centroids, a gain-proportional residual shared by every entry set apart as `common` and an outward push read at its size, with the centroids it returns at the planted atoms the push started from, and the nearest-row, best-cosine and covariance-shaped random-direction helpers), `test_kmstart` (the start's two-factor classifier selects spherical k-means' nearest centre whatever a row's norm, its decoder decodes each entry to the cell's gain-weighted least-squares target by the least change, and other classifier shapes are refused), `test_steer` (a row's forced head and entry are drawn among the eligible ones, never the row's own; the forced rows lie along each step at its own length and at matched lengths, the extreme's own step named once; the per-decode measures; and the contrasts are paired differences) and `test_simplex` (the planted rows are convex mixtures that concentrate at small alpha, the ray position reads 1 at an atom and 1.5 pushed out, the cell matching recovers a relabelling, the numpy projection is sparsemax, the mixing weights and hull point are recovered, the archetypes find the corners one to one, the schedule's phases switch where they should, switching a model's selection rule keeps its parameters, and the readout covers every layer and reads the corners only for one head at one layer; a Voronoi head's logits pick the nearest atom, one placed at the corners reads them, and from one displaced atom per cluster the Voronoi STE moves its atoms to the cluster means)) — breaking `sae.py` or its siblings breaks the suite, so keep those scripts import-safe under `__main__` guards.

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
- **`DictBlock`** (`layers/dictblock.py`) — the core dictionary/tag lookup. Holds a `(h, k, d)` weight tensor (heads × dict entries × feature dim; always `abs()`'d so features can't subtract). Given per-head classification logits, `cluster` turns them into probabilities per its `select` rule (`"softmax"` default; `"argmax"`/`"ste"` straight-through; `"sparsemax"`, the projection onto the simplex, with exact zeros, a support that varies by row and shrinks with the temperature, and no gradient through a one-entry support; `"top<k>"`, e.g. `"top4"`, softmaxes the top-k logits per head and zeroes the rest exactly — incompatible with `pwak_loss`), does a weighted sum over dictionary entries (`fwd`/`hfwd`), and exposes the intervention interface (`intervene`: uniform-ablate, zero-ablate, set, add, subtract, scale — see `DictIntervention` below).
- **`ConcatDictBlock`** (`layers/dictblock.py`) — `DictBlock` with heads writing **disjoint slices** of the output instead of each getting all of it and summing, selected by `concat=True` on `Ontologizer`/`DictEnc`. The weight narrows to `(h, k, d // h)`, so between-head disjoint support holds by construction (`support_overlap` is identically 0) and the dictionary costs `h` times fewer parameters at matched `d`. That disjointness is in the dictionary space only: the shared decoder maps the slices onto overlapping output subspaces, so decoded heads are not orthogonal. A subclass rather than a flag because `cossim`, `flatcos`, `support_overlap`, `tags` and `ghost` would otherwise compare or hand out vectors from different slices as if they shared a space; each is overridden, and its docstring has the reasoning. `train_ste.py --concat --d-head D` sets `e_dec = h * D`; the useful window is `d_out / h <= D <= d_out`, wide enough that `h` slices span the output and narrow enough that each head's decoder column block keeps full column rank.
- **`DictEnc`** (`layers/dictenc.py`) — composes an `NLinearBlock` classifier (activations → per-head logits), a `DictBlock` (logits → reconstructed features), and `Linear`/`NLinear` encoder/decoder/scaling. One `DictEnc` = one dictionary-learning layer. Also owns the two PWAK statistics (`pwak_kl`, `pwak_l2`), since both read the layer *input* alongside the classifications and neither touches the dictionary: `KL_pwak` pulls classifications toward a per-head neighbourhood consensus, while `L2_pwak` scores the partition by how well a sample's own (stop-gradiented) input is predicted from its co-classified neighbours, so its only gradient path is the partition gate. It is noise2self-style leave-one-out but not J-invariant: the sample's own input still sets its weights. See `fns/pwak.py`.
- **`DictStack`** (`ontologizer.py`) — `Ontologizer` is the same class under its earlier name, kept as an alias, so code naming it (here, on the `headline` branch, and in copies of the package elsewhere) runs unchanged. Checkpoints store fields and attribute-named parameters, never the class name, so every checkpoint loads under either name (`test_checkpoint`). The module, the package and the related names (`OntologizerIntervention`, `load_onto`, `--onto`) keep theirs. It stacks `l` `DictEnc` layers residually: with the live `forward="resid"`, layer *i+1* classifies the reconstruction residual `X − decode(R_i)` (note the dataclass default is still `forward="labels"` — flattened-classification forwarding — which every live config overrides because it starves upper layers). Holds the shared `encoder`/`decoder` `Linear` (or `Sparse` passthrough) around the stack. Three flags come from the `headline` branch: `resid_first` (layer 0 reads X − decode(0) as the later layers read their residuals, built inside `constinput`), `gain_clip` (`nextinput` caps the residual's norm at the row's, so every layer's gain is clipped) and `logit_norm` (`NLinearBlock.standardize`: each head's logits standardized at the classifier's output). Each lives where the scripts that rebuild the forward already call, so they pick it up unchanged. Key methods: `__call__` (plain forward), `withStats`/`withGhost`/`withArgs` (forward + validation stats / ghost-gradient / per-layer intervention list), `decodeEntries`/`decodeUniform` (decode the learned dictionary itself, rather than model input, into output space — used by `decode_tags.py`).
- **`DictIntervention`** / **`OntologizerIntervention`** (`ontologizer.py`) — dataclasses describing a causal intervention (scale a head, force/add/subtract specific dict entries, zero/uniform-ablate a head) applied per `Ontologizer` layer via `withArgs`.
- **`SAE`** / **`BilinearSAE`** (`sae.py`) and **`Grouped`** (`grouped.py`) — the SAE baselines as `Sparse` modules: a `Linear` (or `Bilinear`, or the gated encoder's `gate` plus `r_mag`/`b_mag`) encoder and a `Linear` decoder whose bias is pre-subtracted. Fields: `d`, `m`, `topk` (0 = ReLU + `s_l1`), `s_l1`, `aux_k`, `prefixes` (Matryoshka nested-prefix losses), `enc` (`"linear"`/`"gated"`/`"bilinear"`). `Grouped` adds `groups`/`group_fn` (`"top1"`/`"softmax"`; softmax frees the decoder-row norms) and requires `topk=0`. Methods used under `apply`: `preacts`, `gated_pre`, `encode`, `decode`, `loss` (aux-k, L1, prefixes, or `gated_loss`), `eigenfeatures`; on a parameter tree without `apply`: `initialize`, `renorm`, `resample`. The root `sae.py` is their trainer and keeps the flat `params.npz` key layout (`W_enc`, `W_dec`, `b_dec`, …) every downstream script reads; its module-level `encode`/`decode`/`preacts`/… take that layout and delegate to the classes through `from_legacy`/`to_legacy`, so there is one implementation. The gated encoder's training requirements (`s_l1 > 0`, `aux_k = 0`) are checked in `gated_loss`, not at construction, because consumers encode gated checkpoints without them.
- **`BlockSparse`** (`blocksparse.py`) — the vanilla block-sparse featurizer of Fel et al. (2026), an `SAE` whose code keeps the `topk` blocks of `block` signed latents with the largest L2 norm (so `topk` counts blocks), with each decoder block on the unit Frobenius ball and the runner-up blocks' auxiliary loss in place of aux-k (training needs `aux_k=0`, `s_l1=0`). Its encoder is held by block, as `W_blk` (d, m/block, block) and `b_blk`, in the tree and in `params.npz` alike, so a checkpoint carries its block size (`legacy_block`). `sae.py`'s `model_for` refuses a block-sparse checkpoint unless the caller passes the same `block`: a consumer that reads `topk` as latents, or prices the support per latent, would otherwise mis-encode it silently. `decode` needs no `block`. `pareto.py` prices the support per block, `refit.py` and `quantrate.py` skip such runs, and every other consumer raises on one until it is taught blocks.

`fns/keys.py` holds the string→callable lookup tables (activations, dtypes, losses, noise, data source type) that let all of the above be JSON-serializable dataclasses instead of holding raw callables. `fns/loss.py` has the actual math (mse/l1/l0/entropy/cossim/ghost-gradient). `fns/classify.py` has classification-specific activations (softmax, straight-through estimator).

### Training (`ontologize/training/`)

- **`config.py`** — `Hyperparams` (loss weights, optimizer, temperature/noise settings; builds the `Ontologizer` and drives `init`/`train`), `Metadata` (paths, checkpoint cadence, data source type; builds the orbax `CheckpointManager` and the `SampleLoader`), `TrainingEnv` (glues `Ontologizer` spec + `Hyperparams` + `Metadata`, handles resume-from-checkpoint and truncating `loss.csv`/`log.jsonl` on resume).
- **`ontostate.py`** — `OntoState` (a Flax `TrainState` subclass carrying the model instance itself as a static field, plus a rolling stats buffer). `state_init`, `update` (jitted optax step with optional global-norm grad clipping), `train` (the actual training loop, checkpointing + stats-flushing every `save_each` steps). A batch that is an `(input, target)` tuple trains the input toward the target (`experiments/next-sentence`); any other batch is its own target. `SampleLoader` batches tuple records into such a tuple on both of its paths: grain's `Batch` for a source with a length, `loaders.collate` for an iterable one.
- **`serialize.py`** — two separable things. Its `save_model`/`load_model` pair (orbax `StandardSave`/`StandardRestore`) overlaps with `OntoState.save`/`Metadata.manager` and is **not** used by the main training path (`Hyperparams.load`/`TrainingEnv.init` use the manager/spec-item approach in `config.py`/`ontostate.py` instead) — don't assume both are equally live. But `migrate_spec` in the same module **is** live everywhere: a model is serialized as `dataclasses.asdict(model)` and rebuilt by splatting that back into `Ontologizer`, so renaming a field makes every earlier checkpoint unconstructable. `migrate_spec` maps known legacy names forward. Loaders build their model from `restore_spec(manager, step)`, which restores the spec and migrates it with the stored weights' shapes (orbax metadata, no arrays loaded) — `pareto.py` (and every script using its `load_onto`), `autointerp.py`, `decode.py`, `decode_tags.py`, `inference/steerable.py`, `Hyperparams.load`, `freeze_diag.py`, and `dictgeom.py`/`lastlayer.py`/`decodehead.py` in `experiments/ste-arm`. `load_onto` restores every array as numpy and then places it on the default device, so a checkpoint saved on the GPU loads under `JAX_PLATFORMS=cpu`; orbax's default restore reuses the saved sharding, which names a device a CPU-only process lacks, and fails there. Add a rename to `LEGACY_SPEC_KEYS` when you make one. A field whose default is not what checkpoints from before the field did needs an entry in `ABSENT_SPEC_DEFAULTS` (absent ⇒ that value): `resid_gain` is there, because it was added at False and defaulted to True minutes later, so a spec without it trained without gain-shape. `const0` (whether `resid_const`'s coordinate reaches layer 0) cannot be read off a spec — every checkpoint before the field lacks it, with and without the coordinate — so `migrate_spec(spec, params)` infers it from the stored layer-0 classifier width; that is what makes `resid_nc`/`resid_nc_hm` (trained before layer 0 had the coordinate) loadable. Without shapes it falls to the default and such a checkpoint fails with a shape error rather than loading wrong. Beyond renames it drops only what cannot change the model: `headline`-branch fields at their no-op value (`HEADLINE_INERT_SPEC_KEYS`, now only `head_sparse`, raising if one is switched on) and the stats-only `fast_stats`. `headline`'s `resid_first`, `gain_clip` and `logit_norm` are implemented here and load as they are. A `headline` checkpoint with `resid_gain` but without both `resid_first` and `resid_norm` is refused, because there the two branches' gain-shape computes different things (`headline` leaves layer 0 unscaled without `resid_first`, and classifies the raw residual without `resid_norm`). Any other unrecognized key still raises rather than being silently dropped.

Model serialization is plain `dataclasses.asdict(model)` (there is no `@model_spec()` decorator or dedicated spec-class pattern in this codebase — if you see references to one, it's stale).

### Data (`ontologize/data/`)

- **`loaders.py`** — Grain-based (`grain.python`) pipeline: `SampleLoader` (base), `TextLoader` (adds tokenization through `TokenizeTransform`, which sets each row's SONAR source-language tag from `langs.MC4_TO_SONAR` and raises on a language it lacks; the tag is the first token and moves the embedding), `ImageLoader` (flattens/normalizes images, e.g. for MNIST), `EmbeddingLoader` (no-op transform over precomputed embedding vectors; the `srctype="embedding"` loader). `HFDataSource`/`JSONLDataSource`/`NpyDataSource` adapt HuggingFace `datasets`, raw JSONL, or an on-disk `.npy` (memory-mapped; e.g. the embedding cache written by root-level `encode_corpus.py`) into Grain data sources.
- **`pretrained.py`** — loads pretrained PyTorch/HuggingFace encoders (special-cased for SONAR: loads the `M2M100Encoder` directly from a `pytorch_model.bin`, bypassing `AutoModel`). `encode()`/`l2_pooling()` run the torch model, then hand the hidden state to JAX via `jnp.from_dlpack` and mean-pool it, then L2-normalize unless `normalize=False`. The masked mean is SONAR's embedding; SONAR never normalizes (norms ~0.2 for sentences, ~0.3 for 512-token documents), so the L2 step is this pipeline's. `decode()` runs the paired M2M100 decoder's `.generate()` with an `Ontologizer` output substituted in as `last_hidden_state`.
- **`multilingual.py`** / **`langs.py`** — loads and interleaves per-language HuggingFace datasets (mC4/C4), with a static map from mC4 config names to the NLLB/FLORES-200 codes SONAR takes as source tags (`MC4_TO_SONAR`). **`mc4_4M` was not encoded with it**: it used `MC4_4M_TAGS`, which tags 80 of the 86 configs `eng_Latn`, so its embeddings are SONAR's under the English tag (whitened FVU 0.31 from the correctly tagged ones; `experiments/lang-tags`). `encode_corpus.py` records `src_tags` in a new cache's `meta.json`; a cache without the field was tagged the old way, and `encode_corpus.py` will not resume it.
- **`segment.py`** — sentence segmentation for the sentence cache, as pure functions over a text and SaT's boundary probabilities (per XLM-R subword; wtpsplit's per-character array is zero except at token ends): SaT's threshold split at input newlines (`threshold_spans`; `default_threshold` by model, 0.25 for the `-sm` models), the LCM's 200-character cap applied only inside longer segments (`cap_spans`, `capped_spans`; wtpsplit's own `max_length` instead packs sentences into each piece), what bounds each segment and whether it is a sentence, a whole line (a title, label or list entry) or a piece the cap made (`bounded_spans`, `kind`), the dominant-script check against a config's tag (`dominant_script`, `expected_scripts`; romanized configs expect Latin), and a document's run of consecutive usable segments from a random start (`run`). `experiments/sentence-split` has the tests that chose these settings.

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

The layout is **append-only** from 16 columns on, so a row of 16–18 columns still aligns column for column and `read_loss` reads the missing tail as NaN. Rows narrower than 16 come from runs that predate the layout and do not align past column 7 — the selection sweep's 11-wide rows hold `KL_m` at column 8, which `read_loss` labels `cossim_k` — so read those by run. Columns 0–2 are added by `Hyperparams.loss`; 3–12 are the *first ten* of the per-layer `DictEnc.withStats` row summed over layers — that row is 13 wide, and its last three entries (`cossim_flat`, `L1_S`, `support`) are appended at 16–18 rather than inserted; 13–15 are the two setpoint-controlled multipliers as applied this step, plus the max of `cossim_k` over heads *and* layers.

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
