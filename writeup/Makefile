SRC = findings.tex vars.tex refs.bib $(wildcard sections/*.tex) $(wildcard tikz/*.tex)

findings.pdf: build/findings.bbl $(SRC)
	pdflatex -output-directory=build findings.tex
	pdflatex -output-directory=build findings.tex
	mv build/findings.pdf .

build/findings.bbl: build/findings.bcf
	biber build/findings

build/findings.bcf: build $(SRC)
	pdflatex -output-directory=build findings.tex

build:
	mkdir build

clean:
	rm -rf build

.PHONY: clean
