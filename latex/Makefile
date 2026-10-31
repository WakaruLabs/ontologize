FIGURES = $(wildcard figures/*.pdf figures/*.png)
SHARED = vars.tex refs.bib flockpaper.sty $(FIGURES)
FINDINGS_SRC = findings.tex $(SHARED) $(wildcard findings/sections/*.tex) \
	$(wildcard tikz/*.tex) $(wildcard algorithms/*.tex) $(wildcard config/*.tex)
PROPOSAL_SRC = proposal.tex $(SHARED) $(wildcard proposal/sections/*.tex) \
	$(wildcard tikz/*.tex)

# TeX tools run inside this directory's flake dev shell, unless already in one or nix is absent
ifndef IN_NIX_SHELL
NIX := $(if $(shell command -v nix),nix develop $(CURDIR) --command)
endif

# Every pdflatex pass rewrites the .bcf. The final passes leave its contents
# as biber read them, so each PDF rule sets the .bcf's time back to the
# .bbl's; otherwise the next make would rerun biber and both passes.

findings.pdf: build/findings.bbl $(FINDINGS_SRC)
	$(NIX) pdflatex -output-directory=build findings.tex
	$(NIX) pdflatex -output-directory=build findings.tex
	mv build/findings.pdf .
	touch -r build/findings.bbl build/findings.bcf

build/findings.bbl: build/findings.bcf
	$(NIX) biber build/findings

build/findings.bcf: $(FINDINGS_SRC) | build
	$(NIX) pdflatex -output-directory=build findings.tex

brief: brief.pdf

brief.pdf: build/brief.bbl brief.tex $(SHARED)
	$(NIX) pdflatex -output-directory=build brief.tex
	$(NIX) pdflatex -output-directory=build brief.tex
	mv build/brief.pdf .
	touch -r build/brief.bbl build/brief.bcf

build/brief.bbl: build/brief.bcf
	$(NIX) biber build/brief

build/brief.bcf: brief.tex $(SHARED) | build
	$(NIX) pdflatex -output-directory=build brief.tex

proposal: proposal.pdf

proposal.pdf: build/proposal.bbl $(PROPOSAL_SRC)
	$(NIX) pdflatex -output-directory=build proposal.tex
	$(NIX) pdflatex -output-directory=build proposal.tex
	mv build/proposal.pdf .
	touch -r build/proposal.bbl build/proposal.bcf

build/proposal.bbl: build/proposal.bcf
	$(NIX) biber build/proposal

build/proposal.bcf: $(PROPOSAL_SRC) | build
	$(NIX) pdflatex -output-directory=build proposal.tex

all: findings.pdf brief.pdf proposal.pdf

build:
	mkdir build

clean:
	rm -rf build

.PHONY: all clean brief proposal
