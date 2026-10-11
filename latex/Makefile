SHARED = vars.tex refs.bib flockpaper.sty
FINDINGS_SRC = findings.tex $(SHARED) $(wildcard findings/sections/*.tex) \
	$(wildcard tikz/*.tex) $(wildcard algorithms/*.tex) $(wildcard config/*.tex)
PROPOSAL_SRC = proposal.tex $(SHARED) $(wildcard proposal/sections/*.tex) \
	$(wildcard tikz/*.tex)

# TeX tools run inside this directory's flake dev shell, unless already in one or nix is absent
ifndef IN_NIX_SHELL
NIX := $(if $(shell command -v nix),nix develop $(CURDIR) --command)
endif

findings.pdf: build/findings.bbl $(FINDINGS_SRC)
	$(NIX) pdflatex -output-directory=build findings.tex
	$(NIX) pdflatex -output-directory=build findings.tex
	mv build/findings.pdf .

build/findings.bbl: build/findings.bcf
	$(NIX) biber build/findings

build/findings.bcf: build $(FINDINGS_SRC)
	$(NIX) pdflatex -output-directory=build findings.tex

brief: brief.pdf

brief.pdf: build/brief.bbl brief.tex $(SHARED)
	$(NIX) pdflatex -output-directory=build brief.tex
	$(NIX) pdflatex -output-directory=build brief.tex
	mv build/brief.pdf .

build/brief.bbl: build/brief.bcf
	$(NIX) biber build/brief

build/brief.bcf: build brief.tex $(SHARED)
	$(NIX) pdflatex -output-directory=build brief.tex

proposal: proposal.pdf

proposal.pdf: build/proposal.bbl $(PROPOSAL_SRC)
	$(NIX) pdflatex -output-directory=build proposal.tex
	$(NIX) pdflatex -output-directory=build proposal.tex
	mv build/proposal.pdf .

build/proposal.bbl: build/proposal.bcf
	$(NIX) biber build/proposal

build/proposal.bcf: build $(PROPOSAL_SRC)
	$(NIX) pdflatex -output-directory=build proposal.tex

all: findings.pdf brief.pdf proposal.pdf

build:
	mkdir build

clean:
	rm -rf build

.PHONY: all clean brief proposal
