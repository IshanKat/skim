# Build SKIM paper. All intermediate artifacts go in ./build;
# only main.tex (+ sources) and main.pdf stay at the top level.
.PHONY: all clean

all:
	latexmk -pdf -bibtex -outdir=build main.tex
	mv -f build/main.pdf main.pdf

clean:
	latexmk -outdir=build -C
	rm -f main.pdf
