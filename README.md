# Guide et Présentations sur l'IA Locale

Ce dépôt contient un guide détaillé et des présentations conçues pour expliquer la création et l'implémentation d'une Intelligence Artificielle fonctionnant entièrement en local.

## Contenu Généré

Tous les fichiers générés sont disponibles dans le répertoire `gemini/`.

### 1. Guide Technique Détaillé

*   **Source :** `gemini/IA_Locale_Guide_Technique_Detaille.md` (fichier Markdown complet)
*   **Format Office :** `gemini/guide_technique_detaille.docx` (document Word avec table des matières, glossaire, etc.)

### 2. Présentation Grand Public

*   **Source :** `gemini/grand_public/presentation_grand_public.md` (fichier Markdown pour Pandoc)
*   **Format Office :** `gemini/grand_public/presentation_grand_public.pptx` (présentation PowerPoint moderne)
*   **Format Web :** `gemini/grand_public/presentation_grand_public.html` (présentation interactive HTML standalone)

### 3. Présentation Corporate

*   **Source :** `gemini/corporate/presentation_corporate.md` (fichier Markdown pour Pandoc)
*   **Format Office :** `gemini/corporate/presentation_corporate.pptx` (présentation PowerPoint professionnelle)
*   **Format Web :** `gemini/corporate/presentation_corporate.html` (présentation interactive HTML standalone)

## Structure du Projet

```
./
├── README.md
├── prompt_gemini.txt (Journal de progression et instructions pour l'IA)
├── design-guidelines-corporate.md
├── design-guidelines-grand-public.md
└── gemini/
    ├── IA_Locale_Guide_Technique_Detaille.md
    ├── guide_technique_detaille.docx
    ├── corporate/
    │   ├── presentation_corporate.md
    │   ├── presentation_corporate.pptx
    │   └── presentation_corporate.html
    └── grand_public/
        ├── presentation_grand_public.md
        ├── presentation_grand_public.pptx
        └── presentation_grand_public.html
```

## Comment Utiliser les Fichiers

*   **Présentations HTML :** Ouvrez simplement les fichiers `.html` (`gemini/grand_public/presentation_grand_public.html` et `gemini/corporate/presentation_corporate.html`) dans votre navigateur web préféré.
*   **Présentations PowerPoint :** Ouvrez les fichiers `.pptx` avec Microsoft PowerPoint ou un logiciel compatible.
*   **Guide Technique DOCX :** Ouvrez le fichier `.docx` avec Microsoft Word ou un logiciel compatible.
*   **Fichiers Markdown (.md) :** Ces fichiers sont les sources. Ils peuvent être lus avec n'importe quel éditeur de texte ou visualiseur Markdown.

## Génération des Fichiers (Pandoc)

Les documents `.docx` et `.pptx` sont générés à partir des fichiers Markdown correspondants en utilisant [Pandoc](https://pandoc.org/).

### Prérequis

Assurez-vous que Pandoc est installé sur votre système.

### Commandes de Génération

Pour régénérer les fichiers, exécutez les commandes suivantes depuis la racine du projet :

```bash
# Générer le guide DOCX
pandoc "./gemini/IA_Locale_Guide_Technique_Detaille.md" -o "./gemini/guide_technique_detaille.docx" --toc --toc-depth=3

# Générer les présentations PPTX
pandoc "./gemini/grand_public/presentation_grand_public.md" -o "./gemini/grand_public/presentation_grand_public.pptx" -t pptx
pandoc "./gemini/corporate/presentation_corporate.md" -o "./gemini/corporate/presentation_corporate.pptx" -t pptx
```

