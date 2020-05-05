SOURCE=$(wildcard *.tex)
TARGET=${SOURCE:.tex=.pdf}

all: ${TARGET}

%.pdf : %.tex
	latexmk -pdf $<

clean:
	rm -f *.synctex.gz *.aux *.log *.fdb_latexmk *.fls *.nav *.out *.snm *.toc *.bbl *.blg *.vrb
