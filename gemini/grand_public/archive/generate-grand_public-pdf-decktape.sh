#!/bin/bash
set -e

SRC="presentation_grand_public-pdf.md"
OUT_HTML="temp_grand_public_reveal.html"
OUT_PDF="presentation_grand_public.pdf"
TEMPLATE="reveal_template.html"

if ! command -v pandoc &> /dev/null; then
  echo "❌ Erreur : Pandoc n'est pas installé."
  exit 1
fi

if ! command -v decktape &> /dev/null; then
  echo "❌ Erreur : Decktape n'est pas installé. Installe-le avec : npm install -g decktape"
  exit 1
fi

echo "➡️ Conversion du Markdown vers Reveal.js (A4, Mermaid, CSS custom)…"
pandoc "$SRC" -t revealjs -s   -o "$OUT_HTML"   --template="$TEMPLATE"   --lua-filter=filters.lua   -V revealjs-url=https://cdn.jsdelivr.net/npm/reveal.js@5   -V theme=white   -V transition=fade

echo "➡️ Génération du PDF A4 paysage via Decktape…"
decktape reveal "file://$(pwd)/$OUT_HTML" "$OUT_PDF"   --size 2480x1754   --slides 1-999   --pause 4000   --chrome-arg=--no-sandbox   --chrome-arg=--disable-web-security   --chrome-arg=--allow-file-access-from-files   --chrome-arg=--enable-unsafe-webgl   --chrome-arg=--force-device-scale-factor=1

echo "✅ PDF A4 paysage généré : $OUT_PDF"
