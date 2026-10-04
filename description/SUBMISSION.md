# arXiv submission checklist (verified 2026-08-26)

Upload the contents of this directory EXCLUDING build/, preview/,
Makefile, and this file. The essential list:

- paper.tex            (\pdfoutput=1 is line 1; pdflatex)
- paper.bbl            (REQUIRED: arXiv runs pdflatex only, no biber;
                        regenerate with `make` and re-copy from build/
                        after any refs.bib change)
- refs.bib             (inert on arXiv once .bbl ships; keep for humans)
- flockpaper.sty, vars.tex
- sections/*.tex, tikz/*.tex
- figures/*.pdf        (8 result figures, vector)
- figures/frontis-orbit.png, figures/end-ridge.png (plates)

figures/make_figs.py + figures/data/ reproduce every result figure;
they are not needed in the upload but are kept in the repo.

Verified this build: 0 undefined references, 0 overfull hboxes > 10pt,
0 missing characters, all fonts embedded (pdffonts), fully offline
compile, no shell-escape.
