SOURCES := $(wildcard *.md)
PDFS := $(SOURCES:.md=.pdf)

.PHONY: all clean

all: $(PDFS)

%.pdf: %.md
	podman run --rm \
		-v $(CURDIR)/../../..:/data \
		-w /data/user/anthony/presentation \
		pandoc/extra:latest \
		$< -t beamer \
		--pdf-engine=xelatex \
		-V theme=metropolis \
		-V aspectratio=169 \
		-o $@

clean:
	rm -f $(PDFS)
