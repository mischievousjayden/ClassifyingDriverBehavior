FILE=presentation

build:
	pdflatex ${FILE}
	bibtex ${FILE}
	pdflatex ${FILE}
	pdflatex ${FILE}

clean:
	rm -rf *.aux *.bbl *.blg *.lof *.log *.lot *.out *.gz *.toc *.nav *.snm

