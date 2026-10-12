# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This directory holds the LaTeX sources of three documents that share one house style,
one set of macros, one bibliography, and the same diagrams and figures:

- `findings.tex`, the **findings writeup**, "Dictionary Learning as Unsupervised
  Steering-Vector Generation: Ontologizers vs. Sparse Autoencoders on SONAR Sentence
  Embeddings". It is the results record of the research campaign in the parent
  `ontologize` repo; see `../CLAUDE.md` for the JAX/Flax implementation it describes. Its
  sections are in `findings/sections/`.
- `brief.tex`, a two-page summary for readers new to the project (below).
- `proposal.tex`, the **project proposal**, "A proposal for interpreting dense networks",
  which cites the findings as its working notes. Its sections are in
  `proposal/sections/`; see "The proposal" below.

All three use `flockpaper.sty`, `vars.tex`, `refs.bib`, `tikz/` and `figures/`;
`algorithms/` and `config/` are the findings' alone. There is one copy of each, so a change
to a shared file reaches every document that uses it.

The proposal carries only headline methods and results, restated from the findings'
headline sections; its former supplementary material (head freezing, the HSIC
bottleneck, development over training, the star sweep, REINFORCE, head-level auto-interp)
lives in the findings' appendix, with its figures in `figures/`. When a headline number
changes in the findings, change it in `proposal/sections/results.tex` too.

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
make            # the findings: pdflatex -> biber -> pdflatex x2, moves build/findings.pdf here
make brief      # brief.pdf, the same way
make proposal   # proposal.pdf, the same way
make all        # all three
make clean      # rm -rf build
```

The final pdflatex runs twice after biber because biblatex asks for a rerun once the
bibliography changes page breaks; cut that back to one and `\ref`s to figures go stale.

Toolchain is **pdflatex + biber/biblatex** (`backend=biber,style=numeric,sorting=none`).
The house style is `flockpaper.sty`, loaded as `\usepackage[biblatex]{flockpaper}` so it
does not pull in natbib; biblatex must load after it. The style already loads geometry,
hyperref, xcolor, graphicx, booktabs, amsmath, tikz and the newpx fonts, so do not load
those again (geometry and hyperref with other options are an option clash) and do not
add amssymb, whose symbols clash with newpxmath's. It also provides the author block's
`\mutedlabel` and the optional `\eyebrow` and `\runningtitle`.
A failed run can leave a malformed `build/<document>.bcf` that makes biber refuse every
later run (`ERROR - build/proposal.bcf is malformed`, say); `make clean` and a rebuild fix
it, so do not chase the biber error itself.

The system TeX Live install is unusable (no `texmf.cnf`, format files or `biber`), so the
Makefile runs `pdflatex`/`biber` through `nix develop` on `flake.nix` in this directory;
plain `make` works. A flake inside a git repository sees only the files git tracks, so
`flake.nix` must be tracked (or at least `git add`ed) before `nix develop` can find it.
For a quick check without a full build, after any structural change run:

```bash
F="findings.tex findings/sections/*.tex tikz/*.tex algorithms/*.tex config/*.tex"
# for the proposal: F="proposal.tex proposal/sections/*.tex tikz/*.tex"
comm -13 <(grep -ohP '\\label\{\K[^}]+' $F | sort -u) \
         <(grep -ohP '\\(?:ref|eqref)\{\K[^}]+' $F | tr ',' '\n' | sort -u)   # undefined refs
grep -ohP '\\label\{\K[^}]+' $F | sort | uniq -d                              # duplicate labels
```

plus a begin/end and brace balance check per edited file, and confirm every
`\includegraphics` path exists.

After a full build, count problems with `grep -a`: the log contains bytes that make
`grep` treat it as binary, and without `-a` a count prints nothing rather than 0, which
reads as clean. `grep -a -c Overfull build/findings.log` is 0, and an edit should keep it
there. The 5 underfull warnings are the Reproducibility paragraph, set loose under
`sloppypar` because it is mostly unbreakable script names. A preamble
`\emergencystretch` gives other paragraphs a third line-breaking pass instead. The brief
and the proposal build with no overfull or underfull boxes and no undefined references;
keep them that way.

**Tables that would run wide.** Give a free-text column a ragged-right paragraph column,
`>{\raggedright\arraybackslash}p{0.2\linewidth}` (`array` is loaded), rather than letting
`l` columns run into the margin; put a long numeric header on two lines with `\stackhead`;
tighten a nearly-fitting table with `\setlength{\tabcolsep}{4pt}` inside its group. The
`config/` tables use a ragged-right description column in the shared block and
`\footnotesize` for the per-run block. A long path in `\texttt` takes an `\allowbreak`
after a slash, and a display equation too wide for a line goes in `gathered`. The text
width is 390pt, and `\plate`'s hairline frame adds 0.8pt, so a full-width plate image is
`\dimexpr\textwidth-0.8pt\relax`.

## Document structure: the findings

`findings.tex` holds the preamble, title and abstract inline, then inputs the rest:

```
findings.tex
├── (inline) abstract
├── findings/sections/introduction.tex Introduction (+ teaser figure), Background, Evaluation frame
├── findings/sections/methods.tex      architecture, router and fibers; inputs every tikz/ diagram
├── (inline) Data, models and evaluation: a short overview pointing into the appendix
├── findings/sections/results.tex      headline results only
├── findings/sections/discussion.tex   formal reading, conclusions, follow-ups
├── (inline) Reproducibility, \printbibliography
└── findings/sections/appendix.tex     starts with \appendix: supplementary methods and results
```

Paths in `\input` and `\includegraphics` are relative to this directory, where pdflatex
runs, not to the section file that contains them: a section inputs `tikz/dict.tex` and
`algorithms/dictblock.tex`, not `../../tikz/dict.tex`.

`algorithms/` holds the `DictBlock`, `DictEnc` and `Ontologizer` training passes
(`algorithm2e` floats, `\input` from `findings/sections/methods.tex`) and the `steerembed.py`
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

`findings/sections/introduction.tex` holds three sections. The **Introduction** tells
the proposal's Background (`proposal/sections/background.tex`) for the findings:
- an unheaded opening on supervised against unsupervised steering;
- *Problems with SAE features*, one paragraph each;
- *Contrastive classifiers*: the MoEfication conjecture and the architecture in plain
  terms, closing on the teaser figure, where entry swaps are first called steering
  moves;
- *Disjoint support*: how concatenated heads bear on each of those problems, in theory
  and as measured;
- *This document*: what the document compares and where each part is.

When one of the two documents changes this argument, check whether the other should
follow. *Disjoint support* restates results from the methods' disjoint-support
paragraph and the appendix's support and concatenated-heads sections (`sec:support`,
`sec:concatgeom`, `tab:orthant`), so when those numbers change, change it too. **Background** is the related work, and the Introduction leaves its details
(Matryoshka and Tree SAEs, Switch and Scale SAEs) to it. **Evaluation frame** states the
questions (a)–(d) that the results answer.

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
or label; each document's `methods.tex` holds the floats and captions. The proposal inputs
`ontologizer.tex`, `dictenc.tex` and `dict.tex`, and the findings input every diagram. The
findings load only `tikz` with `arrows.meta` (the proposal adds `bending` and
`positioning`), so a diagram is written to that smaller budget. The diagrams share a
style vocabulary (`blk`/`dec`/`cond`/`hook`/`lab`/`sig`/`aux`, Stealth arrowheads,
`black!8` fills, a `\footnotesize` base); reuse those names rather than inventing new
ones. `lab` carries `align=center` in `dictenc.tex` but not in `ontologizer.tex` or
`dict.tex`, so a `\\` inside a `lab` node there needs `align=center` at the use site, or
LaTeX stops with "Paragraph ended before ...".

Result figures are PDFs in `figures/`, drawn by `figures/make_figs.py` from the CSVs in
`figures/data/`. The proposal uses only `figures/fig-teaser.pdf` and the two plates,
`frontis-orbit.png` and `end-ridge.png`. The CSVs are copied out of `../data/out` by
`figures/extract_results.py`, which reads eval-script outputs on CPU, and
`figures/extract_curves.py`, which scores checkpoints on GPU. Extract a figure's numbers
rather than retyping them from a table; the teaser and the GPT-2 seed figure are the
exceptions, restated from outputs their scripts print rather than save.

A `\caption` containing a blank line needs the optional short argument
(`\caption[short]{...}`), or `caption`+`hyperref` aborts with "Paragraph ended before
\caption@prepareanchor was complete". The short argument is also what lands in a list
of figures.

## The proposal

**The proposal carries only headline methods and results.** Each results subsection is
one line of work's headline number, its null and its conclusion, mirroring the headline
sections of `findings/sections/results.tex`. Eliminations, ablations, training
diagnostics and secondary analyses belong in the findings. Every number must match the
findings, and numbers, hyperparameters and architecture claims are checked against the
code, not invented.

`proposal.tex` holds the preamble, title block, abstract and plates, and inputs the body:

```
proposal.tex
├── proposal/sections/introduction.tex   Introduction (+ teaser figure), Assumptions, Concerns
├── proposal/sections/motivation.tex     Threat model, Theory of change
├── proposal/sections/background.tex     SAEs and their problems, MoE routing
├── proposal/sections/methods.tex        architecture, selection rules, disjoint support,
│                                        loss, evaluation; holds the three diagram floats
│   ├── tikz/ontologizer.tex             the l-layer residual stack
│   ├── tikz/dictenc.tex                 inside one DictEnc
│   └── tikz/dict.tex                    per-head simplex geometry
├── proposal/sections/results.tex        Preliminary results: headlines only
└── proposal/sections/conclusion.tex     Future work
```

The Concerns in `introduction.tex` each point to the results subsection that answers
them, and "Against the desiderata" at the end of `results.tex` scores the introduction's
three desiderata; keep both in step with the results.

The proposal is meant for arXiv, which runs pdflatex and no biber; that is why the house
style is pdflatex-only. `flockpaper.sty` replaced an earlier `arxiv.sty` and provides the
`\eyebrow`, `\runningtitle` and `\imprint` macros `proposal.tex` uses; the author block
is a plain `\author{...}` set by its `\maketitle` (no `authblk`).
`proposal/SUBMISSION.md` lists the files an arXiv upload needs, including the `.bbl`.

Its notation is the findings' (below): it restates the findings' headline methods, so a
symbol that changes there changes in the proposal too.

## Macros

- `\fvu` (defined in `findings.tex`) is whitened FVU, $\mathrm{FVU}_w$; use it rather than
  spelling the subscript out.
- `vars.tex` has the operator shorthands (`\txtop{...}`: `\relu`, `\topk`, `\softmax`,
  `\classifier`, `\dictblock`, ...) and the `algorithm2e` keywords, which mirror class
  names in `../ontologize/`. Add a keyword there rather than hardcoding `\texttt{}` in an
  algorithm block.
- `\stackhead[r]{loss\\recovered}` (in `vars.tex`) sets a table header cell on two lines;
  the optional argument is its alignment (default `l`).
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
elementwise multiplication and `\oplus` for concatenation. Cosine similarity is
`\cos\angle(\mathbf{a}, \mathbf{b})`, the cosine of the angle between the vectors, never
`\cos(\mathbf{a}, \mathbf{b})`: `\cos` takes one argument. `$\lvert\cos\angle\rvert$` alone
is its absolute value for the pair in context; the Notation appendix's Conventions
paragraph defines it. This section governs all three documents. Where a lowercase entry
would read as a reserved count ($k$, $m$), write it bracketed: $[M^{\mathrm{part}}]_{ii'}$,
$[\mathsf{K}]_{nij}$. Pearson's $r$, $R^2$ and $t$-statistics are written in words, since
$r$, $R$ and $t$ are reserved.

Exceptions: sets are calligraphic ($\mathcal{S}_n$, $\mathcal{G}$); named statistics and
losses are roman ($\mathrm{FVU}$ via `\fvu`, $\mathrm{MSE}$, $\mathrm{KL}_m$, $\mathrm{L1}$,
$\mathrm{cossim}_b$), as are $H$ (entropy) and $L$ (the loss); and the realized-bits
paragraph uses the information-theory convention of uppercase random variables
($A_{\ell i}$, $\mathbf{A}$, $\mathbf{X}$), which it says where it starts.

**Averages and probabilities.** $\mathbb{E}$ is an average over what its subscript names:
$\mathbb{E}_n$ the samples in context (a batch in a training statistic, the scored rows in
an evaluation), $\mathbb{E}_i$ heads, $\mathbb{E}_{i\neq i'}$ pairs of distinct heads,
$\mathbb{E}_{j\sim\bar{\mathbf{p}}_i}$ a distribution, $\mathbb{E}_{\mathcal{G}'}$ a null's
random draws. Without a subscript it is over the data, so an average over anything else
takes one. $\Pr$ is a fraction of samples; in display math write `\Pr\nolimits_n`, since
amsmath's `\Pr` otherwise sets the subscript underneath, unlike $\mathbb{E}_n$'s. The
indicator is `\mathbf{1}[\cdot]`, with square brackets, and `\mathrm{sd}` and `\mathrm{var}`
take subscripts as $\mathbb{E}$ does. The data mean is $\bar{\mathbf{x}}$ and the mean
assignment $\mathbf{o}$, not $\mathbb{E}[\mathbf{x}]$ and $\mathbb{E}[\mathbf{p}]$. The
Notation appendix's Conventions paragraph says the same for the reader.

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
section (`app:notation`, the first section of `findings/sections/appendix.tex`) is the reader's
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
| $\Delta$, $\Delta_j$ | quantizer step (`quantrate.py --steps`), shared and per coefficient |
| $H_{\mathrm{op}}$ | operational rate: the summed per-slot entropy of a code's quantized symbols |
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

`refs.bib`, biblatex with biber, serves all three documents; each prints only the entries
it cites. Put URLs in a `url=` field, not `note=`: a bare URL in `note` bypasses
biblatex's URL line-breaking. A URL with a long unbreakable segment (a proceedings hash)
can still overrun the bibliography's line; check `Overfull` after adding entries. The
proposal's compute sentence in `proposal/sections/background.tex` (SAE forward passes
computing the whole feature space) cites `gao2024` and `mudide2025`.

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
  (`\texttt{ste\_h76\_init01}`). Command-line flags are typed as they are
  (`\texttt{--trained-groups}`): `flockpaper.sty` turns off the `--` ligature in
  typewriter text, which Plex Mono would otherwise set as an en dash. The same applies to
  every label the style sets in typewriter (`\mutedlabel`, the plates' captions), so an
  intended en dash there is written `\textendash{}`.
- Every comparison states its null or control; a result without a null of the same shape
  is not reported as a result.
- One seed per cell is said explicitly where it applies.
- Cross-references: equations are always `Eq.~\eqref{...}` (`Eqs.` for several), printed
  "Eq. (5)", including at the start of a sentence; never "Equation" or a bare `\ref` to
  an `eq:` label. Sections and algorithms are "Section~\ref" and "Algorithm~\ref" in
  prose and captions; "§" and "Alg." only inside tables.
- "autointerp" in prose for automated interpretability (the script is
  `\texttt{autointerp.py}`); not "auto-interp" or "auto-interpretability".
- American spelling (normalize, centered, labeled, neighbor, behavior, favor).
- A citation is tied to its word with `~\cite{...}`, never glued (`word\cite`).
- Quotation marks are ``...''; a straight `"` prints as a closing quote.
- Floats take `[tp]` or `[htbp]`, never `[h]` alone (LaTeX warns and changes it to `[ht]`).
- Multi-panel figures carry bold panel letters (`letters()` in `make_figs.py`, the
  `letter`/`place_letters` pair in `anatomy.py`), and captions name panels as (a), (b),
  never "left", "right", "top" or "centre".
