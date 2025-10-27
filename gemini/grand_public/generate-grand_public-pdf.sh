#!/bin/bash
set -e

HTML_FILE="presentation_grand_public.html"
PDF_FILE="presentation_grand_public.pdf"
CSS_FILE="print_styles_grand_public.css"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

if [ ! -f "$CHROME" ]; then
  echo "❌ Chrome introuvable à $CHROME"
  exit 1
fi

if [ ! -f "$HTML_FILE" ]; then
  echo "❌ Fichier HTML introuvable : $HTML_FILE"
  exit 1
fi

echo "➡️ Génération du PDF fidèle au HTML ($HTML_FILE)..."

"$CHROME"   --headless   --disable-gpu   --print-to-pdf="$PDF_FILE"   --print-to-pdf-no-header   --disable-web-security   --allow-file-access-from-files   --virtual-time-budget=10000   --css-file="$CSS_FILE"   "file://$(pwd)/$HTML_FILE"

echo "✅ PDF généré avec succès : $PDF_FILE"
