#!/bin/bash
set -e

echo "🧹 Nettoyage et réorganisation du dossier 'grand_public'..."

# ========================
# 1️⃣ Création de la nouvelle arborescence
# ========================
mkdir -p "src" "html" "styles" "output" "scripts" "archive"

# ========================
# 2️⃣ Déplacement des fichiers utiles
# ========================

# ---- Markdown (sources)
for f in "presentation_grand_public.md" "presentation_grand_public-pdf.md" "filters.lua"; do
  [ -f "$f" ] && mv "$f" "src/"
done

# ---- HTML (affichage et imprimable)
for f in "presentation_grand_public.html" "presentation_grand_public_printable.html"; do
  [ -f "$f" ] && mv "$f" "html/"
done

# ---- CSS
[ -f "print_styles_grand_public.css" ] && mv "print_styles_grand_public.css" "styles/"

# ---- Résultats de sortie (PDF / PPTX)
for f in "presentation_grand_public.pdf" "presentation_grand_public.pptx"; do
  [ -f "$f" ] && mv "$f" "output/"
done

# ---- Script PDF principal
if [ -f "generate-grand_public-pdf.sh" ]; then
  mv "generate-grand_public-pdf.sh" "scripts/generate-grand_public-html-to-pdf.sh"
fi

# ========================
# 3️⃣ Archivage des fichiers obsolètes
# ========================
echo "📦 Archivage des fichiers obsolètes..."

obsolete_items=(
  "generate-grand_public-pdf-decktape.sh"
  "generate-grand_public-pdf.sh.old.sh"
  "reveal_template.html"
  "temp_grand_public_reveal.html"
  "grand_public_template.pptx"
  "temp_grand_public_reveal.html"
  "generate-grand_public-pdf.sh.old"
  "presentation_grand_public_reveal.html"
)

for item in "${obsolete_items[@]}"; do
  if [ -e "$item" ]; then
    mv "$item" "archive/"
    echo "   → Archivé : $item"
  fi
done

# Archive aussi tous les anciens scripts de test (.sh) non déplacés
find . -maxdepth 1 -type f -name "*.sh" ! -name "clean_grand_public.sh" ! -path "./scripts/*" -exec mv {} "archive/" \; 2>/dev/null || true

# Archive tout fichier temporaire (HTML ou MD) commençant par "temp_"
find . -maxdepth 1 -type f \( -name "temp_*.html" -o -name "temp_*.md" \) -exec mv {} "archive/" \; 2>/dev/null || true

# ========================
# 4️⃣ Vérification de la nouvelle structure
# ========================
echo ""
echo "✅ Nouvelle arborescence :"
if command -v tree &>/dev/null; then
  tree -L 2
else
  echo "ℹ️ La commande 'tree' n'est pas installée."
  echo "Voici la liste des dossiers :"
  find . -maxdepth 2 -type d
fi

echo ""
echo "🎯 Nettoyage terminé avec succès !"
echo "Tu peux maintenant générer ton PDF fidèle avec :"
echo "   bash \"scripts/generate-grand_public-html-to-pdf.sh\""
