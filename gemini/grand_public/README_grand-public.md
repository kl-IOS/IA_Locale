📘 Documentation du projet : IA Locale Grand Public

---

1️⃣ 🎯 Objectif du projet

Le projet **Grand Public** sert de modèle de présentation visuelle et imprimable, conçu pour transformer des fichiers **Markdown** en différents formats de communication professionnels :

- 🌐 **HTML interactif** — version dynamique avec transitions et navigation.  
- 🖨️ **HTML imprimable / PDF** — version statique et fidèle, adaptée au format A4.  
- 📊 **PPTX (PowerPoint)** — export textuel depuis Markdown pour diffusion ou édition.  

L’objectif principal est d’assurer une **cohérence visuelle et technique** entre tous les formats produits, tout en maintenant une **structure claire et maintenable**.

---

2️⃣ 🧩 Prérequis techniques

Avant toute utilisation, installez les outils suivants sur macOS (testé sur macOS 12+).  
Ces dépendances sont nécessaires à la génération des fichiers HTML, PDF et PPTX.

| Outil | Version minimale | Rôle principal | Commande d’installation |
|--------|------------------|----------------|--------------------------|
| 🧭 **Google Chrome** | ≥ 115 | Génération PDF via le mode Headless | `brew install --cask google-chrome` |
| 📄 **Pandoc** | ≥ 3.1 | Conversion Markdown → HTML / PPTX | `brew install pandoc` |
| 📈 **Mermaid CLI** | ≥ 10 | Export des diagrammes (RAG, etc.) en SVG/PNG | `npm install -g @mermaid-js/mermaid-cli` |
| ⚙️ **Node.js / npm** *(optionnel)* | ≥ 18 | Automatisation de tâches et scripts (`npm run pdf`) | `brew install node` |
| 🌳 **tree** | — | Visualisation rapide de la structure du projet | `brew install tree` |
| 🧰 **jq** *(optionnel)* | — | Manipulation JSON dans les scripts avancés | `brew install jq` |
| 🖋️ **Fonts : Segoe UI / Fira Code** | — | Cohérence visuelle avec la charte graphique | `brew install --cask font-fira-code` ; `brew install --cask font-segoe-ui` |

💡 **Vérification rapide des installations :**
```bash
chrome --version
pandoc --version
node -v
npm -v
mmdc -v


3️⃣ 🧱 Structure du dossier final

Après exécution du script de nettoyage (clean_grand_public.sh), la structure standard est :
grand_public/
├── src/
│   ├── filters.lua
│   ├── presentation_grand_public.md
│   └── presentation_grand_public-pdf.md
│
├── html/
│   ├── presentation_grand_public.html
│   └── presentation_grand_public_printable.html
│
├── styles/
│   └── print_styles_grand_public.css
│
├── scripts/
│   └── generate-grand_public-html-to-pdf.sh
│
├── output/
│   ├── presentation_grand_public.pdf
│   └── presentation_grand_public.pptx
│
└── archive/
    └── anciens scripts et fichiers Reveal.js

| Dossier    | Contenu            | Description                                 |
| ---------- | ------------------ | ------------------------------------------- |
| `src/`     | `.md`, `.lua`      | Fichiers sources textuels et filtres Pandoc |
| `html/`    | `.html`            | Présentations web interactives et statiques |
| `styles/`  | `.css`             | Feuilles de style globales et print         |
| `scripts/` | `.sh`              | Scripts de génération et automatisation     |
| `output/`  | `.pdf`, `.pptx`    | Résultats exportés                          |
| `archive/` | fichiers obsolètes | Scripts Reveal.js et anciennes versions     |


4️⃣ ⚙️ Flux de production complet

4.1 ✍️ Rédaction du contenu

Les contenus sont écrits dans les fichiers Markdown du dossier src/.
	presentation_grand_public.md → version de base (contenu principal)
	presentation_grand_public-pdf.md → version enrichie pour les exports imprimables ou PPTX

4.2 🌐 Génération du HTML interactif

Création de la version web de la présentation (avec transitions et navigation JavaScript).

pandoc "src/presentation_grand_public-pdf.md" \
  -t html5 -s -o "html/presentation_grand_public.html" \
  --css "styles/print_styles_grand_public.css"


  🟢 Résultat :
	html/presentation_grand_public.html
	Interactif, utilisable en ligne ou localement avec transitions, classes .slide, et fond dégradé.


4.3 🖨️ Génération du HTML imprimable

Création du fichier html/presentation_grand_public_printable.html (statique).
Toutes les slides sont visibles, sans effets de transition ni navigation JS.

Ce fichier est utilisé comme base pour la génération PDF.


4.4 🧾 Export PDF fidèle (via Chrome Headless)

Le script scripts/generate-grand_public-html-to-pdf.sh génère un PDF identique au rendu HTML.
📘 Commande :
bash "scripts/generate-grand_public-html-to-pdf.sh"


⚙️ Détails techniques :
	Format : A4 paysage
	Marges : 1 cm
	1 slide = 1 page
	Styles CSS identiques au HTML
	Couleurs et dégradés conservés
	Effets et animations supprimés

📄 Résultat :
Le fichier est exporté vers :	
	output/presentation_grand_public.pdf


4.5 📊 Export PowerPoint (PPTX)

Conversion depuis le Markdown avec Pandoc :
pandoc "src/presentation_grand_public-pdf.md" \
  -o "output/presentation_grand_public.pptx"

💡 Option avancée :
Appliquer un thème PowerPoint personnalisé :

pandoc "src/presentation_grand_public-pdf.md" \
  -o "output/presentation_grand_public.pptx" \
  --reference-doc="reference.pptx"


5️⃣ 🎨 Design Guidelines appliquées

5.1 🎨 Couleurs principales

| Élément          | Couleur    | Code HEX  |
| ---------------- | ---------- | --------- |
| Accent principal | Rouge vif  | `#FE4447` |
| Secondaire       | Vert d’eau | `#5EA8A7` |
| Fond global      | Gris clair | `#f0f2f5` |
| Texte            | Gris foncé | `#333333` |


5.2 ✍️ Typographie

Police principale : 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif
Police pour le code : 'Fira Code', 'Consolas'

5.3 📐 Hiérarchie typographique

| Élément   | Taille | Couleur   | Utilisation     |
| --------- | ------ | --------- | --------------- |
| h1        | 48 px  | `#5EA8A7` | titre principal |
| h2        | 28 px  | `#FE4447` | sous-titre      |
| h3        | 22 px  | `#5EA8A7` | section         |
| p / liste | 16 px  | `#555`    | texte courant   |


5.4 🧱 Mise en page

	Structure : .presentation-container contenant plusieurs .slide
	Centrage : display:flex; justify-content:center; align-items:center;
	Fond des slides :
		background: radial-gradient(circle, rgba(94,168,167,0.05), rgba(254,68,71,0.05));
	Ombres : box-shadow: 0 8px 30px rgba(0,0,0,0.15);

	Arrondis : border-radius: 15px;
	Padding : 40px;	

5.5 🖨️ Impression (@media print)

	Format : A4 paysage
	Marge : 1 cm
	.slide = 1 page (page-break-after: always;)
	Pas d’ombre ni d’animation
	Centrage vertical via flex (justify-content:center; align-items:center;)


6️⃣ 📁 Scripts et commandes utiles

🧹 Nettoyage et réorganisation
bash "clean_grand_public.sh"
➡️ Crée la structure standard et archive les fichiers obsolètes.


🖨️ Génération du PDF
bash "scripts/generate-grand_public-html-to-pdf.sh"
➡️ Produit un PDF fidèle dans output/.


📊 Génération du PPTX
pandoc "src/presentation_grand_public-pdf.md" \
  -o "output/presentation_grand_public.pptx"

🔍 Vérification de la structure
tree -L 2
ou si tree n’est pas installé :
find . -maxdepth 2 -type d


7️⃣ ⚙️ Contraintes et bonnes pratiques

| Aspect                    | Recommandation                                        |
| ------------------------- | ----------------------------------------------------- |
| 🎨 **Cohérence visuelle** | Respecter la palette et les typographies définies     |
| 🧱 **Structure HTML**     | Ne pas modifier `.presentation-container` ni `.slide` |
| 🖨️ **Centrage PDF**      | Vérifier la section `@media print` dans le CSS        |
| 📏 **Slides longues**     | Scinder plutôt que forcer un débordement              |
| 🗃️ **Archive**           | Ne jamais supprimer `archive/` (historique utile)     |
| ⚙️ **Compatibilité**      | Chrome Headless ≥ v115, Pandoc ≥ 3.1                  |



8️⃣ 🧠 Adaptation à d’autres projets

Pour adapter ce modèle à un nouveau projet (ex. corporate) :
	Copier le dossier grand_public/ → corporate/
	Renommer les fichiers (presentation_corporate.*)
	Adapter le CSS (print_styles_corporate.css)
	Modifier les scripts :
		generate-corporate-html-to-pdf.sh
		clean_corporate.sh
	Renommer ce document : README_corporate.md


9️⃣ ✅ Résumé rapide des commandes

| Étape                 | Commande                                                                               | Résultat         |
| --------------------- | -------------------------------------------------------------------------------------- | ---------------- |
| 🧹 Nettoyage          | `bash clean_grand_public.sh`                                                           | Structure propre |
| 🖨️ Génération PDF    | `bash scripts/generate-grand_public-html-to-pdf.sh`                                    | PDF fidèle       |
| 📊 Génération PPTX    | `pandoc src/presentation_grand_public-pdf.md -o output/presentation_grand_public.pptx` | PowerPoint       |
| 📂 Vérifier structure | `tree -L 2`                                                                            | Vue arborescente |


🔚Informations complémentaires

🧩 Auteur : Karim Laurent
📅 Version : 2025-10
🧰 Compatibilité : macOS 12+, Chrome Headless, Pandoc, Mermaid CLI, Node.js
💡 Licence interne : Réutilisable pour tout projet