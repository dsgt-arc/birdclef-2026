# Build Beamer presentations from Markdown via Pandoc in Podman.
#
# On PACE, set XDG_RUNTIME_DIR before running:
#   export XDG_RUNTIME_DIR=/tmp/runtime-$(id -u) && mkdir -p "$XDG_RUNTIME_DIR"
#   make

SOURCES := $(wildcard *.md)
PDFS := $(SOURCES:.md=.pdf)

XDG_RUNTIME_DIR ?= /tmp/runtime-$(shell id -u)
export XDG_RUNTIME_DIR

.PHONY: all clean

all: $(PDFS)

%.pdf: %.md
	@mkdir -p $(XDG_RUNTIME_DIR)
	podman run --rm \
		-v $(CURDIR)/../../..:/data \
		-w /data/user/anthony/presentation \
		docker.io/pandoc/extra:latest \
		$< -t beamer \
		--pdf-engine=xelatex \
		-V theme=metropolis \
		-V aspectratio=169 \
		-o $@

clean:
	rm -f $(PDFS)
