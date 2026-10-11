# arXiv submission checklist

Paths are relative to `latex/`, which also holds the findings and the brief, so
upload only the files below and keep their paths: `proposal.tex` inputs
`proposal/sections/`, `tikz/` and `figures/` from the upload's top level.

- proposal.tex           (\pdfoutput=1 is line 1; pdflatex)
- proposal.bbl           (REQUIRED: arXiv runs pdflatex only, no biber;
                         after `make proposal`, copy build/proposal.bbl
                         next to proposal.tex, and again after any refs.bib
                         change)
- refs.bib               (inert on arXiv once the .bbl ships; keep for
                         humans. It is the shared file, so it also holds
                         entries only the findings cite)
- flockpaper.sty, vars.tex
- proposal/sections/*.tex
- tikz/ontologizer.tex, tikz/dictenc.tex, tikz/dict.tex (the three the
                         proposal inputs; the others belong to the findings)
- figures/fig-teaser.pdf (the one result figure; the other PDFs in
                         figures/ belong to the findings)
- figures/frontis-orbit.png, figures/end-ridge.png (plates)

figures/make_figs.py and figures/data/ reproduce every result figure; they
are not needed in the upload but are kept in the repo.

Checked 2026-10-11, after the move into the shared directory: 0 undefined
references, 0 overfull or underfull boxes, 0 missing characters, all 23
fonts embedded (pdffonts). The fully offline compile and the absence of
shell-escape were last verified on 2026-08-26, under the old
single-directory layout.
