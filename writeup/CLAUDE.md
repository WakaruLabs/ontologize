# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This directory is the LaTeX source for the **findings writeup**, "Dictionary Learning as
Unsupervised Steering-Vector Generation: Ontologizers vs. Sparse Autoencoders on SONAR
Sentence Embeddings" (`findings.tex`). It is the results record of the research campaign
in the parent `ontologize` repo; see `../CLAUDE.md` for the JAX/Flax implementation it
describes.

It is distinct from `../description/`, the project-proposal paper, which cites this
document as its working notes and has its own `CLAUDE.md`, `vars.tex` and build. The
proposal carries only headline methods and results, restated from this document's
headline sections; its former supplementary material (head freezing, the HSIC
bottleneck, development over training, the star sweep, REINFORCE, head-level auto-interp)
lives in this document's appendix, with its figures in `figures/`. When a headline
number changes here, change it in `../description/sections/results.tex` too.

Every number in the prose comes from a script's output in the parent repo, almost all of
it from `../experiments/ste-arm/` and the root eval scripts (`pareto.py`, `steerfid.py`,
`autointerp.py`, ...). Their running record is `../experiments/ste-arm/notes.md` (which
absorbed the former `writeup/ste-arm.md`); check claims against it or rerun the script,
and do not invent numbers.

`brief.tex` is a separate two-page summary for readers new to the project (`make brief`),
written in plain language for a reviewer who found `findings.tex` too long and too dense
with project vocabulary. Keep it free of that vocabulary (no heads/entries/atoms/
realization/collateral without a plain definition) and organized as what the method does,
the problem, its assumptions, how it works, how it compares with existing steering
methods, and what does not work. It covers both substrates and leads with GPT-2: GPT-2
numbers come from the `headline` branch's held-out diagnostics as summarized in its
`docs/OVERVIEW.md` (and `docs/COMPOSITIONAL_CELLS.md`), SONAR numbers from `findings.tex`;
change them at their source first. It does not use "label" for a classifier's choice,
since `docs/OVERVIEW.md` does and the writeup reserves the word for ground truth.

## Build

```bash
make            # pdflatex -> biber -> pdflatex x2, moves build/findings.pdf to ./findings.pdf
make clean      # rm -rf build
```

Toolchain is **pdflatex + biber/biblatex** (`backend=biber,style=numeric,sorting=none`).
The house style is `flockpaper.sty`, loaded as `\usepackage[biblatex]{flockpaper}` so it
does not pull in natbib; biblatex must load after it. The style already loads geometry,
hyperref, xcolor, graphicx, booktabs, amsmath, tikz and the newpx fonts, so do not load
those again (geometry and hyperref with other options are an option clash) and do not
add amssymb, whose symbols clash with newpxmath's. It also provides the author block's
`\mutedlabel` and the optional `\eyebrow` and `\runningtitle`.
A failed run can leave a malformed `build/findings.bcf` that makes biber refuse every
later run; `make clean && make` fixes it.

The system TeX Live install is unusable (no `texmf.cnf`, format files or `biber`), so the
Makefile runs `pdflatex`/`biber` through the root flake's `nix develop`; plain `make` works. For a quick
check without a full build, after any structural change run:

```bash
F="findings.tex sections/*.tex tikz/*.tex algorithms/*.tex config/*.tex"
comm -13 <(grep -ohP '\\label\{\K[^}]+' $F | sort -u) \
         <(grep -ohP '\\(?:ref|eqref)\{\K[^}]+' $F | tr ',' '\n' | sort -u)   # undefined refs
grep -ohP '\\label\{\K[^}]+' $F | sort | uniq -d                              # duplicate labels
```

plus a begin/end and brace balance check per edited file, and confirm every
`\includegraphics` path exists.

After a full build, count problems with `grep -a`: the log contains bytes that make
`grep` treat it as binary, and without `-a` a count prints nothing rather than 0, which
reads as clean. `grep -a -c Overfull build/findings.log` is 28 as of 2026-10-06, all
predating the notation pass (mostly the `config/` tables, 84--160pt too wide); an edit
should not raise it.

## Document structure

`findings.tex` holds the preamble, title, abstract, the evaluation frame and related work
inline, then inputs the rest:

```
findings.tex
├── (inline) abstract, Evaluation frame, Related work
├── sections/methods.tex       architecture, router and fibers; inputs every tikz/ diagram
├── (inline) Data, models and evaluation: a short overview pointing into the appendix
├── sections/results.tex       headline results only
├── sections/discussion.tex    formal reading, conclusions, follow-ups
├── (inline) Reproducibility, \printbibliography
└── sections/appendix.tex      starts with \appendix: supplementary methods and results
```

`algorithms/` holds the `DictBlock`, `DictEnc` and `Ontologizer` training passes
(`algorithm2e` floats, `\input` from `sections/methods.tex`) and the `steerembed.py`
evaluation (`\input` from the appendix). The appendix's Supplementary methods
(`app:methods`) is the single full description of data, models and evaluation
protocols, with each evaluation's defining equation; the main body keeps only an
overview, so add protocol detail there, not inline. `config/` holds one table per
model family (softmax, SONAR straight-through, GPT-2 straight-through, `headline`-branch
GPT-2, SAEs), `\input` from the appendix's "Model configurations" (`app:configs`). Model
configurations live there, not in the main body: the body names a model's shape where an
argument needs it and points to the table for hyperparameters. The tables are built from
each run's recorded configuration (`log.jsonl`'s last `env_config`, the checkpoint spec,
`config.json`/`meta.json`), never from defaults, and flag values that changed at a resume.

`sections/introduction.tex` is a restructured draft of the inline evaluation frame and
related work under a single Introduction heading. It is not `\input` yet, so
`findings.tex`'s inline copy is the live one; edit both or switch over.

**Results versus appendix.** `results.tex` carries the headline result of each line of
work: its main table and the conclusion. Eliminations, per-measure detail, ablations and
secondary analyses go in `appendix.tex`, grouped into sections that mirror the main ones
(softmax vs. SAEs; training, reproducibility and steering of the softmax model; the
straight-through arm; GPT-2 hard codes; dictionary geometry), with a pointer from the main
text. Labels move with their content and keep their names, so references survive moves.
Main-text references into the appendix use `Appendix~\ref{...}`.

**Results are stated as current, not as revisions.** When a measurement is rerun (for
example on the fixed initialization), rewrite the passage with the new numbers instead of
adding "previously ... now ..." qualifications. Note a run's provenance only where a
reader needs it to interpret the number.

### Figures and tikz

Diagram bodies live in `tikz/` as bare `tikzpicture` environments with no float, caption
or label; `sections/methods.tex` holds the floats and captions. Only `tikz` with
`arrows.meta` is loaded, so diagrams are written to that budget. Result figures are PDFs in
`figures/`, regenerated in `../description/figures/` by `make_figs.py`.

A `\caption` containing a blank line needs the optional short argument
(`\caption[short]{...}`), or `caption`+`hyperref` aborts.

## Macros

- `\fvu` (defined in `findings.tex`) is whitened FVU, $\mathrm{FVU}_w$; use it rather than
  spelling the subscript out.
- `vars.tex` has the operator shorthands (`\txtop{...}`: `\relu`, `\topk`, `\softmax`,
  `\classifier`, `\dictblock`, ...) and the `algorithm2e` keywords, which mirror class
  names in `../ontologize/`. Add a keyword there rather than hardcoding `\texttt{}` in an
  algorithm block.
- `\todo{...}` marks open items in red.

## Notation

**Fonts by rank.** Lowercase italic for scalars, `\mathbf` for vectors, uppercase italic
for matrices, `\mathsf` for tensors of rank three or more. A subscripted object takes the
font of what it *is*, not of its parent: $p_{nij}$ is a scalar entry of $\mathsf{P}$,
$\mathbf{p}_{ni}$ a vector slice, $D_i := \mathsf{D}_{i\cdot\cdot}$ a matrix slice. Name a
slice that is used repeatedly rather than writing the parent with dots. Batching adds a
rank: one sample's assignments $P \in \mathbb{R}^{h\times k}$, a batch's
$\mathsf{P} \in \mathbb{R}^{b\times h\times k}$. Vectors are columns: a matrix acts on the
left ($W_{\mathrm{dec}}\mathbf{r}$, $W_{\mathrm{enc}}(\mathbf{x}-\mathbf{b}_{\mathrm{dec}})$,
$J_i\mathbf{v}$), shapes read (output, input), and a decoder's directions are its
*columns* ("decoder directions" in prose, never "decoder rows"). A batch stacks samples as
rows, so a batched product carries a transpose ($\hat X = RW_{\mathrm{dec}}^\top$) or is
written as function application ($\mathrm{dec}(R)$). This is the package's convention
(`Linear` stores (out, in)); `sae.py`'s on-disk `params.npz` stores `W_dec` as $m\times d$
rows, which `from_legacy` transposes, so do not transcribe formulas from that layout. Use
`\tilde{}` only for noised variables (not normalized ones: the normalized layer input is
the shape $\mathbf{u}_\ell$) and `\hat{}` only for reconstructed variables, `\odot` for
elementwise multiplication and `\oplus` for concatenation. `../description/CLAUDE.md` defers to this section. Where a lowercase entry
would read as a reserved count ($k$, $m$), write it bracketed: $[M^{\mathrm{part}}]_{ii'}$,
$[\mathsf{K}]_{nij}$. Pearson's $r$, $R^2$ and $t$-statistics are written in words, since
$r$, $R$ and $t$ are reserved.

Exceptions: sets are calligraphic ($\mathcal{S}_n$, $\mathcal{G}$); named statistics and
losses are roman ($\mathrm{FVU}$ via `\fvu`, $\mathrm{MSE}$, $\mathrm{KL}_m$, $\mathrm{L1}$,
$\mathrm{cossim}_b$), as are $H$ (entropy) and $L$ (the loss); and the realized-bits
paragraph uses the information-theory convention of uppercase random variables
($A_{\ell i}$, $\mathbf{A}$, $\mathbf{X}$), which it says where it starts.

**Indices and counts.** Sample $n$, layer $\ell$ (0-indexed, $\ell = 0,\dots,l-1$, matching
"layer 0" in prose, tables and code), head $i$ and $i'$, entry $j$ and $j'$, coordinate
$q$.

**Layers versus states.** A layer is indexed by its position, $\ell = 0,\dots,l-1$; a
state of the stack by the number of layers applied, $0,\dots,l$. Layer $\ell$ reads state
$\ell$ and writes state $\ell+1$, so subscript 0 always means the initial state and no
index is ever $-1$:
$\mathbf{r}_0 := \mathbf{0}$, $\mathbf{r}_{\ell+1} = \mathbf{r}_\ell + g_\ell\mathbf{f}'_\ell$,
$\hat{\mathbf{x}}_\ell = \mathrm{dec}(\mathbf{r}_\ell)$,
$\mathbf{y}_\ell = \mathrm{sg}[\mathbf{x} - \hat{\mathbf{x}}_\ell]$ (so $\mathbf{y}_0 = \mathbf{x}$),
$g_\ell = \mathrm{sg}[\mathrm{L2}(\mathbf{y}_\ell)]$, the shape
$\mathbf{u}_\ell = \mathbf{y}_\ell/\mathrm{L2}(\mathbf{y}_\ell) \oplus [1]$ (what the
classifier, router and fibers read; $\mathbf{y}_\ell = g_\ell\mathbf{u}_\ell$ on the residual
coordinates), and the final reconstruction is $\hat{\mathbf{x}}_l$. Layer-owned objects
($\dictenc_\ell$, $\mathbf{f}'_\ell$, $g_\ell$, $\mathbf{u}_\ell$, $\boldsymbol{\psi}_\ell$)
carry the layer index; the
deep-supervision prefixes are $\hat{\mathbf{x}}_1,\dots,\hat{\mathbf{x}}_l$. This is
TransformerLens's convention (block $\ell$ reads `resid_pre` $\ell$). The one off-by-one
it leaves is at the tables: a row "layer $\ell$" reports the prefix through that layer,
$\hat{\mathbf{x}}_{\ell+1}$.

**Sites.** Say which side of a block a GPT-2 activation is on, in words: master's cache is
`blocks.8.resid_post`, the residual *leaving* block 8; the `headline` branch's is
`hidden_states[8]`, the residual *entering* block 8, one block earlier. In plain
language, entering 0-indexed block 8 is "after the first eight blocks", not "entering the
eighth".

Counts: batch $b$, layers $l$, heads $h$, entries $k$, input dimension $d$, dictionary
width $e$, fiber rank $r$, SAE width $m$. Inline settings use `{=}`: $k{=}32$, $e{=}1536$.

**Reserved symbols.** Each has one meaning; do not reuse them. The appendix's Notation
section (`app:notation`, the first section of `sections/appendix.tex`) is the reader's
version of these rules and this table, with where each symbol is defined; when a symbol
is added, renamed or retired, change both.

| symbol | meaning |
|---|---|
| $t$ | classifier temperature (`softmax`$_k(K/t)$) |
| $\tau$ | thresholds (containment, judge) |
| $\omega$, $s_\omega$ | between-head support overlap and its loss weight (code: `support`, `s_support`) |
| $\boldsymbol{\pi}_i$ | head $i$'s usage-weighted coordinate profile, from which $\omega$ is computed |
| $\sigma_X, \sigma_K, \sigma_F$ | noise scales; $\sigma$ appears only with these subscripts |
| $\varepsilon$, $\varepsilon_K$, $\varepsilon_F$ | noise draws and terms |
| $\epsilon$ | floating-point epsilon |
| $\rho$ | fiber norm cap |
| $\beta$ | fiber bound in $\mathbf{c}_i = \beta\tanh(A_i\mathbf{u}/\beta)$ |
| $\mathbf{c}_i$, $C$, $\mathsf{C}$ | fiber coordinates (one head, one sample's $h\times r$, a batch) |
| $\mathsf{A}$, $A_i$ | fiber readout tensor and its per-head matrix |
| $\mathsf{D}$, $D_i$ | dictionary tensor $(h,k,e)$ and head $i$'s $(k,e)$ matrix |
| $\mathbf{a}_{ij}$ | atom: $\lvert\mathsf{D}_{ij\cdot}\rvert$, or $\mathsf{D}_{ij\cdot}$ when signed |
| $\mathbf{a}^{\mathrm{dec}}_{ij}$ | decoded atom; $\bar{\mathbf{a}}^{\mathrm{dec}}_i$ its usage-weighted mean |
| $G_i$ | head $i$'s dictionary Gram $D_iD_i^\top$ |
| $g_\ell$, $g_n$ | gain (scalar), per layer or per sample |
| $\boldsymbol{\gamma}_n$, $\Gamma$ | router scales, one sample $(h)$ or a batch $(b,h)$; $\gamma_{ni}$ one head |
| $\mathbf{f}_i$, $\mathbf{f}'$ | one head's output, and a layer's pooled output ($e$); $F$, $\mathsf{F}$ when stacked over heads, or heads and samples |
| $\mathbf{r}_\ell$, $R$ | accumulator after $\ell$ layers (one sample), and batched; $\mathbf{r}_0 = \mathbf{0}$ |
| $\hat{\mathbf{x}}_\ell$, $\mathbf{y}_\ell$ | reconstruction and residual after $\ell$ layers; $\mathbf{y}_\ell$ is layer $\ell$'s input |
| $\kappa_i$, $\bar\kappa$ | within-head mean atom cosine, and its mean over heads (formerly row collinearity $c$) |
| $\boldsymbol{\psi}_\ell$ | layer $\ell$'s statistics row (Eq.~stats); $\mathbf{s}$ the loss-weight vector |
| $\boldsymbol{\phi}_{ni}$ | head $i$'s output contribution on sample $n$ (`headcontrib.py`) |
| $H_{\mathrm{real}}$ | realized bits, $\sum_{\ell,i} H(A_{\ell i})$ (not $\hat H$: a hat means a reconstruction) |
| $\mathbf{u}_\ell$, $\mathbf{u}$ | layer $\ell$'s shape: its input's unit direction with the constant coordinate (code: `U` from `gainshape_in`); $\mathbf{u}$ within one layer. Replaces the former $\tilde{\mathbf{y}}_\ell$ |
| $n_{\mathrm{bits}}$ | bit budget in the Gaussian reference $\mathrm{FVU}_{w,\mathrm{G}}$ |
| $n_{\mathrm{const}}$ | number of trailing constant input coordinates |
| $k_{\mathrm{SAE}}$ | an SAE's active-latent count; $k$ alone is always Ontologizer entries |
| $\topk$ | the SAE family, written "$\topk$ SAE" (never "top-$k$ SAE", whose $k$ would read as entries); $\mathrm{L0}$ its sparsity |
| $k_{\mathrm{sel}}$ | entries kept by the `top`$k_{\mathrm{sel}}$ selection rule |
| $\nu$ | Pareto deviations per head (`pareto.py --ms`), support set $\mathcal{S}_\nu$ |
| $\zeta$ | $z$-scores ($\zeta_{\cos}$); $\mathbf{z}$ is reserved for SAE latents |
| $\mathcal{G}$ | a head or latent group; $\kappa(\mathcal{G})$ its within-group cosine |
| $\alpha$ | steering step strength |
| $\lambda_j$ | eigenvalues (Gaussian reference, eigen-steering); an SAE's L1 weight is $s_{\mathrm{L1}}$ like every other loss weight |
| $\mathrm{assign}$ | the abstraction map from an input to its code, in the formal reading |
| $h'$ | a number of heads kept, probed or conjoined (out of $h$) |
| $k_{\mathrm{probe}}$ | the sparse-probing budget (SAEBench) |
| $W_{\mathrm{flat}}$ | the flat code-to-output map of a one-layer model |
| $\tau_{\mathrm{rev}}$ | the dead-entry revival threshold |
| $^{(1)}, ^{(2)}$ | the two models in a pairwise comparison (not $A$/$B$) |

Run names in `\texttt{}` keep their spelling (`m5120\_k32`), even where the notation
would now write $k_{\mathrm{SAE}}{=}32$.

## Bibliography

`refs.bib`, biblatex with biber. Put URLs in a `url=` field, not `note=`.

## Terminology

- **head**: anything with a head axis, `sae.py --groups` included.
- **entry**: the slot a head selects, index $k$ of head $h$. Not "tag", "label" or "class".
- **atom**: the vector an entry writes; **decoded atom** once mapped into output space.
- **cell**: the set of inputs a head assigns to one entry, one piece of its partition.
- **assignment**: $P$, the soft or hard distribution over a head's entries. **label** is
  reserved for ground truth (language, script, topic).
- **arm**: a trained model variant, including sweep and factorial settings.
- **condition**: a setting within one measurement on a fixed model (direction x strength,
  language x magnitude).
- **group**: only a post hoc discovered grouping (`headstruct.py`, co-firing clusters).

Code identifiers keep their names (`tags()`, `decode_tags.py`, `forward="labels"`,
`--groups`).

## Conventions

- Run and script names are `\texttt{}` with escaped underscores
  (`\texttt{ste\_h76\_init01}`).
- Every comparison states its null or control; a result without a null of the same shape
  is not reported as a result.
- One seed per cell is said explicitly where it applies.
