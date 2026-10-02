SRC = paper.tex vars.tex refs.bib $(wildcard sections/*.tex) $(wildcard tikz/*.tex)

paper.pdf: build/paper.bbl $(SRC)
	xelatex -output-directory=build paper.tex
	xelatex -output-directory=build paper.tex
	mv build/paper.pdf .

build/paper.bbl: build/paper.bcf
	biber build/paper

build/paper.bcf: build $(SRC)
	xelatex -output-directory=build paper.tex

build:
	mkdir build

clean:
	rm -rf build

.PHONY: clean
