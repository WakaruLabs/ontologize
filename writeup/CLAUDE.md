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

TeX is not installed on the machine Claude Code usually runs on, so edits are checked
statically. After any structural change, run:

```bash
F="findings.tex sections/*.tex tikz/*.tex algorithms/*.tex config/*.tex"
comm -13 <(grep -ohP '\\label\{\K[^}]+' $F | sort -u) \
         <(grep -ohP '\\(?:ref|eqref)\{\K[^}]+' $F | tr ',' '\n' | sort -u)   # undefined refs
grep -ohP '\\label\{\K[^}]+' $F | sort | uniq -d                              # duplicate labels
```

plus a begin/end and brace balance check per edited file, and confirm every
`\includegraphics` path exists.

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
