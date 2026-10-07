SRC = findings.tex vars.tex refs.bib $(wildcard sections/*.tex) $(wildcard tikz/*.tex)

# TeX tools run inside the root flake's dev shell, unless already in one or nix is absent
ifndef IN_NIX_SHELL
NIX := $(if $(shell command -v nix),nix develop $(CURDIR)/.. --command)
endif

findings.pdf: build/findings.bbl $(SRC)
	$(NIX) pdflatex -output-directory=build findings.tex
	$(NIX) pdflatex -output-directory=build findings.tex
	mv build/findings.pdf .

build/findings.bbl: build/findings.bcf
	$(NIX) biber build/findings

build/findings.bcf: build $(SRC)
	$(NIX) pdflatex -output-directory=build findings.tex

brief: brief.pdf

brief.pdf: build/brief.bbl brief.tex refs.bib
	$(NIX) pdflatex -output-directory=build brief.tex
	$(NIX) pdflatex -output-directory=build brief.tex
	mv build/brief.pdf .

build/brief.bbl: build/brief.bcf
	$(NIX) biber build/brief

build/brief.bcf: build brief.tex refs.bib
	$(NIX) pdflatex -output-directory=build brief.tex

build:
	mkdir build

clean:
	rm -rf build

.PHONY: clean brief
