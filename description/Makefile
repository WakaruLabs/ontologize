SRC = paper.tex vars.tex refs.bib flockpaper.sty $(wildcard sections/*.tex) $(wildcard tikz/*.tex)

# TeX tools run inside the root flake's dev shell, unless already in one or nix is absent
ifndef IN_NIX_SHELL
NIX := $(if $(shell command -v nix),nix develop $(CURDIR)/.. --command)
endif

paper.pdf: build/paper.bbl $(SRC)
	$(NIX) pdflatex -output-directory=build paper.tex
	$(NIX) pdflatex -output-directory=build paper.tex
	mv build/paper.pdf .

build/paper.bbl: build/paper.bcf
	$(NIX) biber build/paper

build/paper.bcf: build $(SRC)
	$(NIX) pdflatex -output-directory=build paper.tex

build:
	mkdir build

clean:
	rm -rf build

.PHONY: clean
