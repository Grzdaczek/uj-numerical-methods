.ONESHELL:
SHELL	= /bin/bash
TAR		:= tar -czvf
RM		:= rm -rf
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

#=-----------------------------------------------------------------------------

TARGET = num1.tar.gz
PROGOUT = f32.pgf f64.pgf

all: $(TARGET)

$(TARGET): *.pdf *.py
	$(TAR) $(TARGET) *.py *.pdf

%.pdf: %.tex $(PROGOUT)
	pdflatex $^

$(PROGOUT): program.py
	$(CONDA_ACTIVATE)
	python3 $^
