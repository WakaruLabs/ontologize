# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope

This directory is the LaTeX source for the Ontologizer **project-proposal paper**
("A proposal for mixture-of-experts-based interpretability"). It is a subtree of the
`ontologize` research repo — see `../../CLAUDE.md` for the JAX/Flax implementation that
this paper describes. Numbers, hyperparameters, and architecture claims in the prose
should be checked against that code, not invented.

The sibling `../findings.tex` is a separate, self-contained working-notes record of the
SAE-comparison campaign. It is not `\input` by this paper and does not share `vars.tex`
or `arxiv.sty`; don't cross-edit them as one document.

## Build

```bash
make            # xelatex → biber → xelatex, then moves build/paper.pdf to ./paper.pdf
make clean      # rm -rf build
```

Toolchain is **xelatex + biber/biblatex** (not pdflatex, not bibtex).

Two build gotchas, both worth knowing before debugging a failure:

- **A failed run leaves a malformed `build/paper.bcf`, and biber then refuses to run on
  every subsequent `make`** (`ERROR - build/paper.bcf is malformed`). The fix is always
  `make clean && make`. Do not chase the biber error itself.
- `$(SRC)` covers `paper.tex`, `vars.tex`, `refs.bib`, `sections/*.tex` and `tikz/*.tex`,
  so ordinary edits do trigger a rebuild. The final target runs xelatex twice after biber
  (three passes total) because biblatex asks for a rerun once the bibliography changes
  page breaks; if you cut that back to one, `\ref`s to the figures go stale.

`build/` and `paper.pdf` are gitignored (see the `paper/description/**/*.pdf` etc. rules
at the bottom of `../../.gitignore`); source under `paper/description/` is explicitly
un-ignored against the blanket `paper/**` rule.

## Document structure

`paper.tex` is a thin shell: preamble, `\input{vars.tex}`, title block, and a single
`\input{sections/introduction.tex}`. Everything else is reached transitively:

```
paper.tex
└── sections/introduction.tex      ← the whole body lives here (Assumptions → Future work)
    └── sections/fig_architecture.tex   ← figure floats + captions
        ├── tikz/ontologizer.tex        ← the l-layer residual stack
        ├── tikz/dictenc.tex            ← inside one DictEnc
        └── tikz/dict.tex               ← per-head simplex geometry
```

- **`sections/threatmodel.tex` is currently orphaned** — complete prose, but not
  `\input` anywhere. Wire it into `introduction.tex` if the alignment framing is wanted.
- `sections/introduction.tex` is a working draft, not a finished paper: it opens and
  closes with raw unstructured notes (bullet dumps, a "freyavoice/jadevoice" dialogue,
  a bare `\subsection{Background}` topic list) surrounding the sections that are actually
  written. Treat those blocks as scratch to be absorbed, not as prose to preserve verbatim.

### Figures and tikz

Diagram bodies live in `tikz/` as bare `tikzpicture` environments with **no** float,
caption, or label; the figure floats, `subfigure` nesting, captions and `\label`s all live
in `sections/fig_architecture.tex`. Keep that split — a caption belongs with the float,
not with the drawing.

The three tikz files share a common style vocabulary (`blk`/`dec`/`cond`/`hook`/`lab`/
`sig`/`aux`, Stealth arrowheads, `black!8` fills, `\footnotesize` base). Reuse those names
rather than inventing new ones. `lab` carries `align=center` in `dictenc.tex` but **not**
in `ontologizer.tex` or `dict.tex`; using `\\` inside a `lab` node without it raises
`Paragraph ended before ...` / emergency stop, so add `align=center` at the use site.

**A `\caption` containing a blank line needs the optional short argument** —
`\caption[short]{para one … para two}`. Without it, `caption`+`hyperref` aborts with
`Paragraph ended before \caption@prepareanchor was complete`. All three multi-paragraph
figure captions rely on this. The short argument is also what lands in the LOF.

Only `tikz`, `arrows.meta`, `bending`, `positioning`, `amsmath`, `caption`, and
`subcaption` are available (loaded by `paper.tex`); the tikz files are written to that
budget deliberately.

The header comment in `sections/fig_architecture.tex` records the live SONAR
configuration the diagrams are annotated with (`d=1024, e=2048, k=32, h=32, l=5`,
`deepsup`/`resid_norm`/`resid_const` on). If `../../sonar.py` changes, those annotations
and the captions' "Live values" lines go stale together.

## Macros

`vars.tex` defines two families and both are load-bearing:

- Math operator shorthands via `\txtop{...}` (`\relu`, `\topk`, `\softmax`, `\encoder`,
  `\classifier`, `\partitioner`, …) — use these instead of `\mathrm{}` so operator styling
  stays uniform.
- `algorithm2e` keywords (`\SetKwFunction{DictBlock}`, `{DictEnc}`, `{Ontologizer}`,
  `{NLinearBlock}`, …) mirroring the class names in `../../ontologize/`. When adding an
  algorithm block, add the keyword here rather than hardcoding `\texttt{}`.

## Bibliography

`refs.bib` (biblatex, `backend=biber`). One known gap remains: `sections/introduction.tex`
contains an empty `\cite{}` (the claim about SAE forward passes computing the whole
feature space) that still needs a source, and prints a bold `[]` in the PDF until one is
supplied. Add the entry rather than deleting the `\cite`.

Put URLs in a `url=` field, not in `note=` — a bare URL in `note` bypasses biblatex's
URL line-breaking and produces underfull-hbox warnings in the bibliography.

A clean build is otherwise warning-free: no undefined references, no over/underfull
boxes. Keep it that way — `make clean && make` and check `build/paper.log` after
editing figures or captions.
