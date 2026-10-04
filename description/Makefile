SRC = paper.tex vars.tex refs.bib flockpaper.sty $(wildcard sections/*.tex) $(wildcard tikz/*.tex)

paper.pdf: build/paper.bbl $(SRC)
	pdflatex -output-directory=build paper.tex
	pdflatex -output-directory=build paper.tex
	mv build/paper.pdf .

build/paper.bbl: build/paper.bcf
	biber build/paper

build/paper.bcf: build $(SRC)
	pdflatex -output-directory=build paper.tex

build:
	mkdir build

clean:
	rm -rf build

.PHONY: clean
