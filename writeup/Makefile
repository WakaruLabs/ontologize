SRC = findings.tex vars.tex refs.bib $(wildcard sections/*.tex) $(wildcard tikz/*.tex)

findings.pdf: build/findings.bbl $(SRC)
	pdflatex -output-directory=build findings.tex
	pdflatex -output-directory=build findings.tex
	mv build/findings.pdf .

build/findings.bbl: build/findings.bcf
	biber build/findings

build/findings.bcf: build $(SRC)
	pdflatex -output-directory=build findings.tex

brief: brief.pdf

brief.pdf: build/brief.bbl brief.tex refs.bib
	pdflatex -output-directory=build brief.tex
	pdflatex -output-directory=build brief.tex
	mv build/brief.pdf .

build/brief.bbl: build/brief.bcf
	biber build/brief

build/brief.bcf: build brief.tex refs.bib
	pdflatex -output-directory=build brief.tex

build:
	mkdir build

clean:
	rm -rf build

.PHONY: clean brief
