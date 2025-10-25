# Storyboard Hyper-Détaillé : IA Locale - Guide Stratégique (Corporate Moderne)

Ce document décrit chaque diapositive de la présentation Corporate avec un niveau de détail élevé pour le contenu, le design et les animations, en s'appuyant sur le "Guide de Design - Version Corporate".

---

## Diapositive 1 : Titre

*   **Titre :** IA Locale : La Prochaine Révolution pour votre Entreprise
*   **Sous-titre :** Un guide stratégique pour une mise en œuvre réussie
*   **Contenu additionnel :** "Document confidentiel" (en bas à gauche)
*   **Design (CSS Inline / Classes) :**
    *   **Conteneur de la slide (`.slide`) :**
        *   `background: linear-gradient(135deg, var(--color-primary-dark) 0%, var(--color-primary) 100%);` (Dégradé bleu marine foncé vers bleu marine)
        *   `padding: 4rem;`
        *   `justify-content: flex-start; align-items: flex-start;` (Alignement en haut à gauche)
    *   **Titre (`h1`) :**
        *   `font-size: 4.5rem;` (text-7xl)
        *   `color: white;`
        *   `font-weight: 700;`
        *   `line-height: 1.2;`
        *   `margin: 0;`
    *   **Sous-titre (`h2`) :**
        *   `font-size: 1.875rem;` (text-3xl)
        *   `color: var(--color-muted);` (Gris très clair)
        *   `font-weight: 400;`
        *   `margin: 0;`
    *   **Ligne de séparation (`div.highlight-bar`) :**
        *   `height: 2px;`
        *   `width: 120px;`
        *   `background: var(--color-highlight);` (Bleu accent)
        *   `margin: 2rem 0;`
    *   **Texte confidentiel (`p`) :**
        *   `font-size: 1.25rem;` (text-xl)
        *   `color: var(--color-muted);`
        *   `position: absolute; bottom: 4rem; left: 4rem;`
*   **Animations :**
    *   Titre : `opacity: 0; transform: translateY(-20px); animation: fadeInSlideDown 0.8s forwards;`
    *   Sous-titre : `opacity: 0; transform: translateY(-20px); animation: fadeInSlideDown 0.8s 0.3s forwards;`
    *   Ligne : `width: 0; animation: expandWidth 0.6s 0.6s forwards;`
    *   Texte confidentiel : `opacity: 0; animation: fadeIn 0.8s 0.9s forwards;`

---

## Diapositive 2 : Pourquoi l'IA Locale est un Impératif Stratégique

*   **Titre :** Pourquoi l'IA Locale est un Impératif Stratégique
*   **Contenu :** Quatre points clés, chacun dans une "carte" (`.info-card`).
    *   **Sécurité Renforcée :** "Vos données ne quittent jamais votre infrastructure." (Icône : bouclier 🛡️)
    *   **Maîtrise des Coûts :** "Éliminez les abonnements et maîtrisez votre budget." (Icône : graphique de tendance à la baisse 📉)
    *   **Souveraineté Technologique :** "Réduisez votre dépendance aux fournisseurs tiers." (Icône : drapeau 🚩)
    *   **Conformité Assurée :** "Alignement natif avec le RGPD et les normes ISO." (Icône : document certifié ✅)
*   **Design (CSS Inline / Classes) :**
    *   **Conteneur de la slide (`.slide`) :** `background-color: var(--color-surface);`
    *   **Titre de la slide (`.slide-title`) :**
        *   `font-size: 2.25rem;` (text-4xl)
        *   `color: var(--color-primary-dark);`
        *   `font-weight: 700;`
        *   `border-bottom: 3px solid var(--color-highlight);`
        *   `width: 100%; text-align: left; margin-bottom: 2rem;`
    *   **Conteneur des cartes (`.slide-content`) :**
        *   `display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem;`
    *   **Carte d'information (`.info-card`) :**
        *   `background: var(--color-muted);` (Gris très clair)
        *   `border: 1px solid var(--color-border);`
        *   `border-radius: 8px;`
        *   `padding: 1.5rem;`
        *   `box-shadow: 0 4px 12px rgba(0,0,0,0.05);`
    *   **Titre de carte (`.info-card h3`) :**
        *   `font-size: 1.25rem;` (text-xl)
        *   `color: var(--color-primary);`
        *   `margin-top: 0;`
    *   **Texte de carte (`.info-card p`) :**
        *   `font-size: 1rem;` (text-base)
        *   `color: var(--color-surface-foreground);`
*   **Animations :**
    *   Titre de la slide : `opacity: 0; transform: translateY(-20px); animation: fadeInSlideDown 0.8s forwards;`
    *   Chaque `.info-card` : `opacity: 0; transform: scale(0.9); animation: fadeInScale 0.6s forwards;` (avec un délai progressif pour chaque carte).

---

## Diapositive 3 : Feuille de Route Accélérée

*   **Titre :** Feuille de Route Accélérée
*   **Contenu :** Une frise chronologique en 4 étapes.
    *   **1. Analyse (1-2 sem.) :** Audit des données et cas d'usage prioritaires.
    *   **2. Préparation (1-2 sem.) :** Nettoyage et structuration du corpus.
    *   **3. POC (3-6 sem.) :** Déploiement d'un pilote RAG.
    *   **4. Production (continu) :** Industrialisation et formation.
*   **Design (CSS Inline / Classes) :**
    *   **Conteneur de la slide (`.slide`) :** `background-color: var(--color-surface);`
    *   **Titre de la slide (`.slide-title`) :** (Identique à la slide 2)
    *   **Conteneur de la feuille de route (`.roadmap-container`) :**
        *   `display: flex; justify-content: space-between; width: 100%; position: relative; padding-top: 2rem;`
        *   `::before` (ligne de progression) : `content: ''; position: absolute; top: 2.5rem; left: 5%; width: 90%; height: 4px; background-color: var(--color-border);`
    *   **Étape de la feuille de route (`.roadmap-step`) :**
        *   `display: flex; flex-direction: column; align-items: center; z-index: 1;`
    *   **Point de progression (`.roadmap-step .dot`) :**
        *   `width: 20px; height: 20px; background-color: var(--color-highlight); border-radius: 50%; border: 4px solid var(--color-surface);`
    *   **Titre d'étape (`.roadmap-step h4`) :**
        *   `margin-top: 1rem; font-weight: 700; font-size: 1.125rem;` (text-lg)
    *   **Description d'étape (`.roadmap-step p`) :**
        *   `text-align: center; color: var(--color-muted-foreground); font-size: 0.875rem;` (text-sm)
*   **Animations :**
    *   Titre de la slide : (Identique à la slide 2)
    *   Ligne de progression : `width: 0; animation: expandWidth 1s forwards;`
    *   Chaque `.roadmap-step` : `opacity: 0; transform: translateY(20px); animation: fadeInSlideUp 0.8s forwards;` (avec un délai progressif).

---

## Diapositive 4 : Étape 2 : La Préparation des Données

*   **Titre :** Étape 2 : La Préparation des Données
*   **Contenu :** Trois points clés, chacun dans une "carte" (`.info-card`) avec icône.
    *   **Nettoyage et Structuration :** "Conversion des documents bruts (PDF, Word, etc.) en texte propre et structuré." (Icône : balai 🧹)
    *   **Anonymisation des PII :** "Masquage des **Informations d'Identification Personnelle** (PII) comme les noms, adresses ou numéros de téléphone pour garantir la conformité avec le RGPD." (Icône : masque 🎭)
    *   **Chunking :** "Division des longs documents en petits morceaux (chunks) pour que l'IA puisse les analyser efficacement." (Icône : ciseaux ✂️)
*   **Design (CSS Inline / Classes) :**
    *   **Conteneur de la slide (`.slide`) :** `background-color: var(--color-surface);`
    *   **Titre de la slide (`.slide-title`) :** (Identique à la slide 2)
    *   **Conteneur des cartes (`.slide-content`) :**
        *   `display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem;`
    *   **Carte d'information (`.info-card`) :** (Identique à la slide 2)
    *   **Icônes :** Utiliser des emojis ou des icônes SVG simples.
*   **Animations :**
    *   Titre de la slide : (Identique à la slide 2)
    *   Chaque `.info-card` : (Identique à la slide 2, avec délai progressif).

---

## Diapositive 5 : L'Approche Recommandée : Le RAG

*   **Titre :** L'Approche Recommandée : Le RAG (Retrieval-Augmented Generation)
*   **Contenu :** Un grand schéma central expliquant le RAG. 
    1.  **Source :** Une "bibliothèque" de documents (icônes de PDF, Word, etc. 📚).
    2.  **Processus :** Un "moteur de recherche intelligent" (icône de loupe avec un cerveau 🧠) qui extrait les informations pertinentes.
    3.  **Résultat :** Un "générateur de réponses" (icône de LLM ou de robot 🤖) qui rédige la réponse finale en se basant sur les extraits.
*   **Texte sous le schéma :** "Le RAG transforme vos documents internes en une base de connaissances interrogeable, offrant des réponses factuelles et sourcées."
*   **Design (CSS Inline / Classes) :**
    *   **Conteneur de la slide (`.slide`) :** `background-color: var(--color-surface);`
    *   **Titre de la slide (`.slide-title`) :** (Identique à la slide 2)
    *   **Conteneur du schéma (`.slide-content`) :** `display: flex; flex-direction: column; align-items: center; text-align: center;`
    *   **Éléments du schéma (`.rag-step`) :**
        *   `background: var(--color-muted); border: 1px solid var(--color-border); border-radius: 8px; padding: 1rem 2rem; margin-bottom: 1rem; width: 60%;`
        *   `font-size: 1.125rem;` (text-lg)
    *   **Flèches (`.arrow`) :**
        *   `font-size: 2rem; color: var(--color-highlight); margin: 0.5rem 0;`
*   **Animations :**
    *   Titre de la slide : (Identique à la slide 2)
    *   Chaque élément du schéma et les flèches : `opacity: 0; transform: translateY(20px); animation: fadeInSlideUp 0.8s forwards;` (avec délai progressif).

---

## Diapositive 6 : Analyse Coûts-Bénéfices

*   **Titre :** Analyse Coûts-Bénéfices
*   **Contenu :** Deux sections principales.
    *   **Investissements initiaux :** (Liste des coûts)
    *   **Gains annuels nets :** (Liste des gains)
*   **Texte clé :** "ROI potentiel dès la première année", "+25% de productivité sur les tâches de recherche".
*   **Design (CSS Inline / Classes) :**
    *   **Conteneur de la slide (`.slide`) :** `background-color: var(--color-surface);`
    *   **Titre de la slide (`.slide-title`) :** (Identique à la slide 2)
    *   **Conteneur des graphiques (`.slide-content`) :**
        *   `display: grid; grid-template-columns: 1fr 1fr; gap: 3rem;`
    *   **Section de coûts/bénéfices (`.cost-benefit-section`) :**
        *   `background: var(--color-muted); border-radius: 8px; padding: 1.5rem;`
    *   **Titres de section (`.cost-benefit-section h3`) :**
        *   `font-size: 1.5rem; color: var(--color-primary); margin-top: 0;`
    *   **Listes (`ul`) :** `list-style: none; padding: 0; margin: 0;`
    *   **Items de liste (`li`) :** `font-size: 1rem; margin-bottom: 0.5rem;`
*   **Animations :**
    *   Titre de la slide : (Identique à la slide 2)
    *   Chaque section de coûts/bénéfices : `opacity: 0; transform: translateX(-20px); animation: fadeInSlideRight 0.8s forwards;` (avec délai progressif).

---

## Diapositive 7 : KPIs de Succès

*   **Titre :** KPIs de Succès
*   **Contenu :** Trois indicateurs clés, chacun dans une "carte" (`.kpi-card`).
    *   **< 2 secondes :** Temps de réponse moyen
    *   **> 70% :** Taux d'adoption à 6 mois
    *   **500+ :** Requêtes traitées par jour
*   **Design (CSS Inline / Classes) :**
    *   **Conteneur de la slide (`.slide`) :** `background-color: var(--color-surface);`
    *   **Titre de la slide (`.slide-title`) :** (Identique à la slide 2)
    *   **Conteneur des KPIs (`.slide-content`) :**
        *   `display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem;`
    *   **Carte KPI (`.kpi-card`) :**
        *   `background: var(--color-muted); border: 2px solid var(--color-highlight); border-radius: 8px; padding: 1.5rem; text-align: center;`
    *   **Chiffre KPI (`.kpi-card .value`) :**
        *   `font-size: 3rem; font-weight: 700; color: var(--color-primary-dark); margin: 0;`
    *   **Description KPI (`.kpi-card .label`) :**
        *   `font-size: 1rem; color: var(--color-muted-foreground); margin-top: 0.5rem;`
*   **Animations :**
    *   Titre de la slide : (Identique à la slide 2)
    *   Chaque `.kpi-card` : `opacity: 0; transform: translateY(20px); animation: fadeInSlideUp 0.8s forwards;` (avec délai progressif).

---

## Diapositive 8 : Glossaire

*   **Titre :** Glossaire
*   **Contenu :** Liste de termes techniques avec leurs définitions.
    *   **IA Locale :** Intelligence Artificielle exécutée sur l'infrastructure interne de l'entreprise.
    *   **RAG (Retrieval-Augmented Generation) :** Modèle qui s'appuie sur une base de connaissances externe pour générer des réponses factuelles.
    *   **LLM (Large Language Model) :** Modèle de langage de grande taille, moteur de l'IA générative.
    *   **PII (Personally Identifiable Information) :** Informations d'Identification Personnelle. Données permettant d'identifier un individu.
    *   **POC (Proof of Concept) :** Preuve de concept. Projet pilote pour valider la faisabilité et la valeur d'une solution.
    *   **Chunking :** Processus de segmentation des documents en morceaux.
    *   **Embeddings :** Représentations vectorielles de l'information, permettant à l'IA de comprendre les similarités sémantiques.
*   **Design (CSS Inline / Classes) :**
    *   **Conteneur de la slide (`.slide`) :** `background-color: var(--color-surface);`
    *   **Titre de la slide (`.slide-title`) :** (Identique à la slide 2)
    *   **Liste du glossaire (`.slide-content ul`) :**
        *   `list-style: none; padding: 0; margin: 0;`
        *   `display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;`
    *   **Item du glossaire (`.slide-content li`) :**
        *   `font-size: 1rem; margin-bottom: 0.5rem;`
        *   `b { color: var(--color-primary); }`
*   **Animations :**
    *   Titre de la slide : (Identique à la slide 2)
    *   Chaque item du glossaire : `opacity: 0; animation: fadeIn 0.6s forwards;` (avec délai progressif).

---

## Diapositive 9 : Prochaines Étapes & Partenariat

*   **Titre :** Prochaines Étapes & Partenariat
*   **Contenu :**
    *   **Atelier de Cadrage :** Définir ensemble le périmètre de votre POC.
    *   **Lancement du Pilote :** Démarrer le projet avec une équipe dédiée.
    *   **Contact :** [Votre Nom/Département], [Votre Email]
*   **Design (CSS Inline / Classes) :**
    *   **Conteneur de la slide (`.slide`) :** `background-color: var(--color-surface);`
    *   **Titre de la slide (`.slide-title`) :** (Identique à la slide 2)
    *   **Conteneur du contenu (`.slide-content`) :** `display: flex; flex-direction: column; align-items: center; text-align: center;`
    *   **Items (`li`) :** `font-size: 1.25rem; margin-bottom: 1rem;`
    *   **Contact (`p`) :** `font-size: 1.125rem; color: var(--color-primary-dark);`
*   **Animations :**
    *   Titre de la slide : (Identique à la slide 2)
    *   Chaque point de contenu : `opacity: 0; transform: translateX(-20px); animation: fadeInSlideRight 0.8s forwards;` (avec délai progressif).
