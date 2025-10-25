# Plan de Présentation : Créer votre IA Locale (Style Grand Public Moderne)

Ce document est un storyboard pour la présentation PowerPoint, aligné avec le guide de design "Grand Public Moderne".

---

### Diapositive 1 : Titre

*   **Titre :** Créez votre IA Locale
*   **Sous-titre :** Le Guide Complet de A à Z pour les non-techniciens
*   **Design (selon guidelines grand public) :**
    *   **Fond :** Dégradé `linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%)`.
    *   **Titre :** `text-7xl`, `color: white`, `text-align: center`.
    *   **Sous-titre :** `text-3xl`, `color: white`, `opacity: 0.95`.
*   **Animation :** Le titre apparaît avec un léger rebond (`fadeInSlideDown`). Le sous-titre apparaît en fondu en dessous (`fadeInSlideDown` avec délai).

---

### Diapositive 2 : Qu'est-ce qu'une IA locale ? 🤔

*   **Titre :** Qu'est-ce qu'une IA locale ? 🤔
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Trois "cartes d'information" empilées verticalement, chacune avec :

        *   `background-color: var(--color-muted)`, `border-radius: 12px`, `padding: 1.5rem`.

        *   Une icône circulaire (`width: 50px; height: 50px; background-color: var(--color-primary); border-radius: 50%; color: white; font-size: 1.5rem; display: flex; justify-content: center; align-items: center; flex-shrink: 0;`).

        *   Un `h3` pour le titre (`color: var(--color-primary-dark)`) et un `p` pour la description.

        *   Ex: "Confidentialité totale : Vos données restent chez vous" (Icône : 🔒).

*   **Animation :** Chaque carte apparaît l'une après l'autre avec un effet de zoom léger (`fadeInScale`).

---

### Diapositive 3 : Que peut faire votre IA locale ?

*   **Titre :** Que peut faire votre IA locale ?
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Quatre "cartes d'information" (grid 2x2) avec les caractéristiques suivantes :

        *   `background-color: var(--color-muted)`, `border-radius: 12px`, `padding: 1.5rem`, `box-shadow: 0 4px 8px rgba(0,0,0,0.05)`.

        *   Chaque carte contient un `h3` pour le titre (ex: "Assistant Personnel") et un `p` pour la description.

        *   Icônes pertinentes pour chaque point (ex: cerveau, loupe, classeur, stylo).

*   **Animation :** Chaque carte apparaît l'une après l'autre avec un effet de zoom léger (`fadeInScale`).

---

### Diapositive 4 : De quoi avez-vous besoin ? 💻

*   **Titre :** De quoi avez-vous besoin ? 💻
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Deux "cartes d'information" côte à côte (grid 1x2) :

        *   **Cartes :** `background-color: var(--color-muted)`, `border-radius: 12px`, `padding: 1.5rem`.

        *   Chaque carte contient un `h3` pour le titre (ex: "Matériel") (`color: var(--color-primary-dark)`) et une liste (`ul`, `li`) de points.

    *   **Phrase de réconfort :** "Rassurez-vous : nous verrons tout pas à pas !" (`p`, `color: var(--color-muted-foreground)`).

*   **Animation :** Les cartes apparaissent en glissant depuis les côtés (`fadeInSlideUp`).

---

### Diapositive 5 : Étape 1 - Définir votre besoin

*   **Titre :** 1. Définir votre besoin
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Phrase d'introduction :** `h3` (`margin-bottom: 1rem`).

    *   **Contenu :** Liste (`ul`, `li`) de questions.

        *   Chaque question est un `li` (`margin-bottom: 1rem`, `<b>` pour la question).

*   **Animation :** Chaque question apparaît en fondu (`fadeIn`).

---

### Diapositive 6 : Étape 2 - Préparer vos données

*   **Titre :** 2. Préparer vos données
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Deux colonnes (grid 1x2) :

        *   **Colonne "Sources possibles" :** `h3` (`color: var(--color-primary-dark)`) pour le titre, suivi d'une liste (`ul`, `li`) des sources.

        *   **Colonne "Organisation" :** `h3` (`color: var(--color-primary-dark)`) pour le titre, suivi d'une liste (`ul`, `li`) des points d'organisation.

    *   **Section "Utiliser vos données YouTube" :** `h3` (`color: var(--color-primary-dark)`) pour le titre, `p` pour la description (ex: "Extrayez votre historique de visionnage et vos sous-titres via Google Takeout pour créer une base de connaissances personnalisée sur vos contenus préférés. 🎬").

*   **Animation :** Les colonnes apparaissent en glissant depuis les côtés (`fadeInSlideRight`), puis la section YouTube apparaît en fondu (`fadeIn`).

---

### Diapositive 7 : Étape 3 - Deux approches principales

*   **Titre :** 3. Deux approches principales
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Deux blocs colorés côte à côte (grid 1x2) :

        *   **Bloc RAG :** `background-color: var(--color-primary)`, `border-radius: 12px`, `padding: 1.5rem`, `color: white`.

            *   `h3` (`color: white`) pour le titre.

            *   `p` (`color: white`) pour la description.

        *   **Bloc Fine-tuning :** `background-color: var(--color-accent)`, `border-radius: 12px`, `padding: 1.5rem`, `color: white`.

            *   `h3` (`color: white`) pour le titre.

            *   `p` (`color: white`) pour la description.

*   **Animation :** Les boîtes apparaissent en glissant depuis les côtés (`fadeInSlideUp`).

---

### Diapositive 8 : Comment fonctionne le RAG ?

*   **Titre :** Comment fonctionne le RAG ?
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Un schéma simple en 3 étapes avec des blocs verticaux et des flèches :

        *   **Blocs :** `background-color: var(--color-muted)`, `border-radius: 12px`, `padding: 1.5rem`, `width: 60%`.

            *   **1. Indexation :** "Vos documents sont convertis en vecteurs mathématiques."

            *   **2. Recherche :** "L'IA trouve les passages pertinents pour votre question."

            *   **3. Génération :** "L'IA formule une réponse basée sur ces passages."

        *   **Flèches :** `font-size: 2rem`, `color: var(--color-primary)`.

*   **Animation :** Les étapes apparaissent en séquence (`fadeInSlideUp`).

---

### Diapositive 9 : Choisir la Bonne Approche pour votre Projet

*   **Titre :** Choisir la Bonne Approche pour votre Projet
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Texte introductif (`p`, `font-size: 1.125rem`, `color: var(--color-surface-foreground)`), suivi d'une liste (`ul`, `li`) de scénarios et d'approches recommandées.

        *   Ex: "<b>Vous avez des documents (PDF, notes) ?</b> &rarr; Le <b>RAG</b> est idéal pour des réponses précises."

    *   **Conseil pratique :** "<b>Conseil :</b> Commencez simple avec le RAG, puis explorez d'autres options si besoin !" (`p`, `font-size: 1.125rem`, `color: var(--color-surface-foreground)`).

*   **Animation :** Le texte introductif apparaît en fondu (`fadeIn`), suivi des points de la liste (`fadeIn`) et du conseil (`fadeIn`).

---

### Diapositive 10 : Étape 4 - Outils à installer

*   **Titre :** 4. Outils à installer
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :**

        *   **Outil principal : Ollama :** `h3` (`color: var(--color-primary-dark)`) pour le titre, `p` pour la description.

        *   **Outils complémentaires :** `h3` (`color: var(--color-primary-dark)`) pour le titre, liste (`ul`, `li`) des outils.

*   **Animation :** Le bloc Ollama apparaît en premier (`fadeIn`), puis les outils complémentaires (`fadeIn`).

---

### Diapositive 11 : Étape 5 - Mise en pratique

*   **Titre :** 5. Mise en pratique
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Une liste verticale de 5 étapes numérotées, chacune avec un badge numérique circulaire.

        *   **Badge numérique circulaire :** `width: 50px; height: 50px; background: var(--color-primary); color: white; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 1.5rem; font-weight: 700;`.

        *   **Badge de l'étape finale (5) :** `background: var(--color-accent)`.

        *   **Texte de l'étape :** `p` (`margin: 0`).

*   **Animation :** Chaque étape apparaît l'une après l'autre (`fadeInSlideUp`).

---

### Diapositive 12 : Avantages et Limites

*   **Titre :** Avantages et Limites
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Deux colonnes (grid 1x2) :

        *   **Colonne "Avantages" :** `h3` (`color: var(--color-primary-dark)`) pour le titre, suivi d'une liste (`ul`, `li`) des avantages.

        *   **Colonne "À considérer" :** `h3` (`color: var(--color-primary-dark)`) pour le titre, suivi d'une liste (`ul`, `li`) des points à considérer.

*   **Animation :** Les colonnes apparaissent en glissant depuis les côtés (`fadeInSlideRight`).

---

### Diapositive 13 : Glossaire

*   **Titre :** Glossaire
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Deux colonnes de listes (`ul`, `li`) de termes et définitions.

        *   **Terme :** `<b>` pour le terme (ex: `<b>IA Locale :</b>`).

        *   **Définition :** Texte normal (`margin-bottom: 0.5rem`).

*   **Animation :** Chaque élément de la liste apparaît progressivement (`fadeIn`).

---

### Diapositive 14 : Sécurité et Confidentialité de vos Données

*   **Titre :** Sécurité et Confidentialité de vos Données
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Quatre "cartes d'information" (grid 2x2) avec les caractéristiques suivantes :

        *   `background-color: var(--color-muted)`, `border-radius: 12px`, `padding: 1.5rem`, `box-shadow: 0 4px 8px rgba(0,0,0,0.05)`.

        *   Chaque carte contient un `h3` pour le titre (ex: "Vos données restent chez vous") et un `p` pour la description.

        *   Icônes pertinentes pour chaque point (ex: cadenas, masque, manette, coche).

*   **Animation :** Chaque carte apparaît l'une après l'autre avec un effet de zoom léger (`fadeInScale`).

---

### Diapositive 15 : Conseils pour Réussir votre Projet IA

*   **Titre :** Conseils pour Réussir votre Projet IA
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Quatre "cartes d'information" (grid 2x2) avec les caractéristiques suivantes :

        *   `background-color: var(--color-muted)`, `border-radius: 12px`, `padding: 1.5rem`, `box-shadow: 0 4px 8px rgba(0,0,0,0.05)`.

        *   Chaque carte contient un `h3` pour le titre (ex: "Commencez simple (MVP)") et un `p` pour la description.

        *   Icônes pertinentes pour chaque point (ex: fusée, flèches circulaires, coche, graphique).

*   **Animation :** Chaque carte apparaît l'une après l'autre avec un effet de zoom léger (`fadeInScale`).

---

### Diapositive 16 : Prochaines étapes

*   **Titre :** Prochaines étapes
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Phrase d'accroche :** "Vous êtes prêt à commencer !" (`h2`, `font-size: 2rem`, `color: var(--color-primary-dark)`).

    *   **Sous-titre :** "Commencez simplement avec le RAG et Ollama." (`p`, `font-size: 1.25rem`, `color: var(--color-surface-foreground)`).

    *   **Contenu :** Trois étapes numérotées, chacune avec un badge numérique circulaire.

        *   **Badge numérique circulaire :** `width: 50px; height: 50px; background: var(--color-primary); color: white; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 1.5rem; font-weight: 700;`.

        *   **Texte de l'étape :** `p` (`margin: 0`).

*   **Animation :** Les boîtes apparaissent l'une après l'autre (`fadeInSlideUp`).

---

### Diapositive 17 : Questions ?

*   **Titre :** Questions ?
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Une grande icône de point d'interrogation (`font-size: 10rem`, `color: var(--color-primary)`).

*   **Animation :** Le point d'interrogation apparaît en fondu (`fadeIn`).

---

### Diapositive 18 : Les 5 grandes étapes

*   **Titre :** Les 5 grandes étapes
*   **Design (selon guidelines grand public) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`.

    *   **Contenu :** Une liste verticale d'étapes, chacune avec un badge numérique circulaire.

        *   **Badge numérique circulaire :** `width: 50px; height: 50px; background: var(--color-primary); color: white; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 1.5rem; font-weight: 700;`.

        *   **Badge de l'étape finale (5) :** `background: var(--color-accent)`.

        *   **Texte de l'étape :** `p` (`margin: 0`, `<b>` pour le titre).

*   **Animation :** Chaque étape apparaît l'une après l'autre (`fadeInSlideUp`).