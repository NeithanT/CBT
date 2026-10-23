#!/usr/bin/env bash
# Two-phase build: (1) pdflatex + bibtex resolve references, (2) two pdflatex passes settle numbering.
set -euo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode -halt-on-error RESEARCH_PAPER.tex
bibtex RESEARCH_PAPER
pdflatex -interaction=nonstopmode -halt-on-error RESEARCH_PAPER.tex
pdflatex -interaction=nonstopmode -halt-on-error RESEARCH_PAPER.tex

cp RESEARCH_PAPER.pdf ../RESEARCH_PAPER.pdf
echo "[OK] Built RESEARCH_PAPER.pdf"
