# ===== File: Makefile =====

# Cognitive Stability Daemon (CSD)
# Paper build + packaging targets
# Usage: make [pdf|watch|open|clean|distclean|archive|zenodo|fmt|check|help]

SHELL := /bin/sh
MAIN ?= main.tex
BASENAME := $(basename $(MAIN))
DATE := $(shell date +%Y%m%d)
DIST_DIR := dist

.DEFAULT_GOAL := pdf
.PHONY: all pdf watch open clean distclean archive zenodo fmt check help

all: pdf

pdf:
	@echo "[CSD] Building $(MAIN) → $(BASENAME).pdf"
	@if command -v latexmk >/dev/null; then \
	  latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN); \
	else \
	  if ! command -v biber >/dev/null; then \
	    echo "ERROR: Install latexmk or biber (biblatex backend requires biber)."; exit 2; \
	  fi; \
	  pdflatex -interaction=nonstopmode $(MAIN) || exit $$?; \
	  biber $(BASENAME) || exit $$?; \
	  pdflatex -interaction=nonstopmode $(MAIN) || exit $$?; \
	  pdflatex -interaction=nonstopmode $(MAIN) || exit $$?; \
	fi
	@echo "[CSD] Done: $(BASENAME).pdf"

watch:
	@if ! command -v latexmk >/dev/null; then \
	  echo "ERROR: watch mode requires latexmk."; exit 2; \
	fi
	@echo "[CSD] Watching $(MAIN) (latexmk -pvc)…"
	@latexmk -pdf -pvc -interaction=nonstopmode $(MAIN)

open: pdf
	@if command -v open >/dev/null; then \
	  open $(BASENAME).pdf; \
	elif command -v xdg-open >/dev/null; then \
	  xdg-open $(BASENAME).pdf; \
	else \
	  echo "Open '$(BASENAME).pdf' manually."; \
	fi

clean:
	@echo "[CSD] Cleaning auxiliaries…"
	@-latexmk -c $(MAIN) >/dev/null 2>&1 || true
	@rm -f $(BASENAME).{aux,bbl,bcf,blg,lof,log,lot,out,run.xml,toc,fls,fdb_latexmk,xdv,nav,snm}

distclean: clean
	@echo "[CSD] Removing outputs…"
	@rm -f $(BASENAME).pdf references.bib

archive: pdf
	@echo "[CSD] Creating source archive…"
	@mkdir -p $(DIST_DIR)
	@tar czf $(DIST_DIR)/$(BASENAME)-$(DATE).tar.gz \
		$(MAIN) Makefile README.md || true
	@echo "[CSD] Wrote $(DIST_DIR)/$(BASENAME)-$(DATE).tar.gz"

zenodo: pdf
	@echo "[CSD] Preparing Zenodo bundle…"
	@mkdir -p $(DIST_DIR)
	@tar czf $(DIST_DIR)/csd-zenodo-$(DATE).tar.gz \
		$(MAIN) Makefile README.md \
		LICENSE CITATION.cff 2>/dev/null || true
	@echo "[CSD] Wrote $(DIST_DIR)/csd-zenodo-$(DATE).tar.gz"

fmt:
	@if command -v latexindent >/dev/null; then \
	  latexindent -l -w $(MAIN); \
	else \
	  echo "latexindent not found; skipping format."; \
	fi

check:
	@if command -v chktex >/dev/null; then \
	  chktex -q $(MAIN); \
	else \
	  echo "chktex not found; skipping checks."; \
	fi

help:
	@printf "\nCSD Make targets:\n"
	@printf "  pdf        Build the paper (default).\n"
	@printf "  watch      Rebuild on change (latexmk required).\n"
	@printf "  open       Open the resulting PDF.\n"
	@printf "  clean      Remove aux files.\n"
	@printf "  distclean  Clean + remove PDF and generated bib.\n"
	@printf "  archive    Create dist/<basename>-<date>.tar.gz of sources.\n"
	@printf "  zenodo     Create dist/csd-zenodo-<date>.tar.gz bundle.\n"
	@printf "  fmt        Format with latexindent (if installed).\n"
	@printf "  check      Lint with chktex (if installed).\n\n"
