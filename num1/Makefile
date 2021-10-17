.ONESHELL:
SHELL	= /bin/bash
TAR		:= tar -czvf
RM		:= rm -rf
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

#=-----------------------------------------------------------------------------

all: tar 

tar: tex *.py
	$(TAR) out.tar.gz *.py *.pdf

tex: *.tex py
	pdflatex *.tex

py: program.py
	$(CONDA_ACTIVATE)
	python3 $^
