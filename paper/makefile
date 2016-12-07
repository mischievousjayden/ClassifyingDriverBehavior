FILE=paper

build:
	latex ${FILE}
	bibtex ${FILE}
	latex ${FILE}
	latex ${FILE}
	pdflatex ${FILE}

clean:
	rm -rf *.log *.aux *.bbl *.blg *.dvi *.pdf

