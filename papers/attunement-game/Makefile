
LATEX=pdflatex
BIBTEX=bibtex
TARGET=attunement_game

all: pdf

pdf:
	$(LATEX) $(TARGET).tex
	$(BIBTEX) $(TARGET)
	$(LATEX) $(TARGET).tex
	$(LATEX) $(TARGET).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc

veryclean: clean
	rm -f $(TARGET).pdf
