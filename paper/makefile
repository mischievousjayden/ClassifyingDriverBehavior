FILE=paper

build:
	pdflatex ${FILE}
	bibtex ${FILE}
	pdflatex ${FILE}
	pdflatex ${FILE}

clean:
	rm -rf *.log *.aux *.bbl *.blg *.dvi *.pdf

