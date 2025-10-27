#!/bin/bash
set -e

SRC="presentation_grand_public-pdf.md"
OUT_HTML="temp_grand_public_reveal.html"
OUT_PDF="presentation_grand_public.pdf"
CSS="print_styles_grand_public.css"

pandoc "$SRC" -t revealjs -s -o "$OUT_HTML" -V revealjs-url=https://cdn.jsdelivr.net/npm/reveal.js@5
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-margins \
  --print-to-pdf="$OUT_PDF" --print-to-pdf-no-header --landscape "file://$(pwd)/$OUT_HTML"
