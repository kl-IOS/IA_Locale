# Plan de Présentation : IA Locale - Guide Stratégique (Style Corporate Moderne)

Ce document est un storyboard pour une présentation qui allie un contenu stratégique et un design moderne et épuré.

---

### Diapositive 1 : Titre

*   **Titre :** IA Locale : La Prochaine Révolution pour votre Entreprise
*   **Sous-titre :** Un guide stratégique pour une mise en œuvre réussie
*   **Design (selon guidelines corporate) :**
    *   **Fond :** Dégradé `linear-gradient(135deg, var(--color-primary-dark) 0%, var(--color-primary) 100%)`.
    *   **Titre :** `text-7xl`, `color: white`, `font-weight: 700`.
    *   **Sous-titre :** `text-3xl`, `color: var(--color-muted)`.
    *   **Séparateur :** Ligne de 2px de hauteur, 120px de largeur, `background: var(--color-highlight)`, marges verticales de 2rem.
    *   **Texte confidentiel :** `text-xl`, `color: var(--color-muted)`, positionné en bas à gauche.
*   **Animation :** Fondu très rapide pour le titre (`fadeInSlideDown`), suivi d'une apparition progressive du sous-titre (`fadeInSlideDown` avec délai), puis de la ligne (`expandWidth`) et du texte confidentiel (`fadeIn`).

---

### Diapositive 2 : Pourquoi l'IA Locale est un Impératif Stratégique

*   **Titre :** Pourquoi l'IA Locale est un Impératif Stratégique
*   **Design (selon guidelines corporate) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`, `border-bottom: 3px solid var(--color-highlight)`, `padding-bottom: 0.5rem`.
    *   **Contenu :** Quatre "cartes d'information" (grid 2x2) avec les caractéristiques suivantes :
        *   `background: var(--color-muted)`, `border: 1px solid var(--color-border)`, `border-radius: 8px`, `padding: 1.5rem`, `box-shadow: 0 4px 12px rgba(0,0,0,0.05)`.
        *   Chaque carte contient un `h3` pour le titre (ex: "Sécurité Renforcée") et un `p` pour la description (ex: "Vos données ne quittent jamais votre infrastructure.").
        *   Icônes pertinentes pour chaque point (ex: bouclier, graphique de tendance à la baisse, drapeau, document certifié).
*   **Animation :** Les cartes apparaissent en cascade avec un léger zoom avant (`fadeInScale`).

---

### Diapositive 3 : Feuille de Route Accélérée

*   **Titre :** Feuille de Route Accélérée
*   **Design (selon guidelines corporate) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`, `border-bottom: 3px solid var(--color-highlight)`, `padding-bottom: 0.5rem`.
    *   **Frise chronologique :** Une frise horizontale avec des points circulaires pour chaque étape.
        *   **Ligne de progression :** `height: 4px`, `background-color: var(--color-border)`, `width: 90%`.
        *   **Points d'étape :** `width: 20px`, `height: 20px`, `background-color: var(--color-highlight)`, `border-radius: 50%`, `border: 4px solid var(--color-surface)`.
        *   **Titre de l'étape :** `h4`, `font-weight: 700`, `font-size: 1.125rem`.
        *   **Description :** `p`, `color: var(--color-muted-foreground)`, `font-size: 0.875rem`.
*   **Animation :** La ligne de la frise se dessine (`expandWidth`), puis chaque point apparaît en séquence (`fadeInSlideUp`).

---

### Diapositive 4 : Étape 2 : La Préparation des Données

*   **Titre :** Étape 2 : La Préparation des Données
*   **Design (selon guidelines corporate) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`, `border-bottom: 3px solid var(--color-highlight)`, `padding-bottom: 0.5rem`.
    *   **Contenu :** Trois "cartes d'information" (grid 3x1) avec les caractéristiques suivantes :
        *   `background: var(--color-muted)`, `border: 1px solid var(--color-border)`, `border-radius: 8px`, `padding: 1.5rem`, `box-shadow: 0 4px 12px rgba(0,0,0,0.05)`.
        *   Chaque carte contient un `h3` pour le titre (ex: "Nettoyage et Structuration") et un `p` pour la description.
        *   Icônes pertinentes pour chaque point (ex: balai pour nettoyage, masque pour anonymisation, ciseaux pour chunking).
*   **Animation :** Apparition successive de chaque carte (`fadeInScale`).

---

### Diapositive 5 : L'Approche Recommandée : Le RAG

*   **Titre :** L'Approche Recommandée : Le RAG (Retrieval-Augmented Generation)
*   **Design (selon guidelines corporate) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`, `border-bottom: 3px solid var(--color-highlight)`, `padding-bottom: 0.5rem`.
    *   **Contenu :** Un schéma central expliquant le RAG avec trois blocs verticaux et des flèches :
        *   **Blocs :** `background: var(--color-muted)`, `border: 1px solid var(--color-border)`, `border-radius: 8px`, `padding: 1rem 2rem`, `width: 60%`.
            *   **1. Source :** "Une \"bibliothèque\" de documents 📚"
            *   **2. Processus :** "Un \"moteur de recherche intelligent\" 🧠"
            *   **3. Résultat :** "Un \"générateur de réponses\" 🤖"
        *   **Flèches :** `font-size: 2rem`, `color: var(--color-highlight)`.
    *   **Texte explicatif :** "Le RAG transforme vos documents internes en une base de connaissances interrogeable, offrant des réponses factuelles et sourcées." (`p`, `font-size: 1.125rem`, `color: var(--color-surface-foreground)`).
*   **Animation :** Les blocs et flèches apparaissent successivement (`fadeInSlideUp`, `fadeIn`).

---

### Diapositive 6 : Analyse Coûts-Bénéfices

*   **Titre :** Analyse Coûts-Bénéfices
*   **Design (selon guidelines corporate) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`, `border-bottom: 3px solid var(--color-highlight)`, `padding-bottom: 0.5rem`.
    *   **Contenu :** Deux blocs côte à côte (grid 1x2) :
        *   **Bloc "Investissements initiaux" :** `background: var(--color-muted)`, `border-radius: 8px`, `padding: 1.5rem`.
            *   `h3` (`font-size: 1.5rem`, `color: var(--color-primary)`) pour le titre.
            *   Liste (`ul`, `li`, `font-size: 1rem`) des investissements (Développement, Infrastructure, Formation).
        *   **Bloc "Gains annuels nets" :** `background: var(--color-muted)`, `border-radius: 8px`, `padding: 1.5rem`.
            *   `h3` (`font-size: 1.5rem`, `color: var(--color-primary)`) pour le titre.
            *   Liste (`ul`, `li`, `font-size: 1rem`) des gains (ROI potentiel, productivité).
*   **Animation :** Les blocs apparaissent en glissant depuis la gauche (`fadeInSlideRight`).

---

### Diapositive 7 : KPIs de Succès

*   **Titre :** KPIs de Succès
*   **Design (selon guidelines corporate) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`, `border-bottom: 3px solid var(--color-highlight)`, `padding-bottom: 0.5rem`.
    *   **Contenu :** Trois "cartes d'information" (grid 1x3) avec les caractéristiques suivantes :
        *   `background: var(--color-muted)`, `border: 2px solid var(--color-highlight)`, `border-radius: 8px`, `padding: 1.5rem`, `text-align: center`.
        *   **Chiffre principal :** `p` (`font-size: 3rem`, `font-weight: 700`, `color: var(--color-surface-foreground)`).
        *   **Libellé :** `p` (`font-size: 1rem`, `color: var(--color-muted-foreground)`).
*   **Animation :** Les cartes apparaissent en glissant depuis le bas (`fadeInSlideUp`).

---

### Diapositive 8 : Glossaire

*   **Titre :** Glossaire
*   **Design (selon guidelines corporate) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`, `border-bottom: 3px solid var(--color-highlight)`, `padding-bottom: 0.5rem`.
    *   **Contenu :** Deux colonnes de listes (`ul`, `li`) de termes et définitions.
        *   **Terme :** `<b>` pour le terme (ex: `<b>IA Locale :</b>`).
        *   **Définition :** Texte normal (`font-size: 1rem`, `margin-bottom: 0.5rem`).
*   **Animation :** Chaque élément de la liste apparaît progressivement (`fadeIn`).

---

### Diapositive 9 : Prochaines Étapes & Partenariat

*   **Titre :** Prochaines Étapes & Partenariat
*   **Design (selon guidelines corporate) :**
    *   **Titre de slide :** `text-4xl`, `color: var(--color-primary-dark)`, `border-bottom: 3px solid var(--color-highlight)`, `padding-bottom: 0.5rem`.
    *   **Contenu :** Liste (`ul`, `li`) des prochaines étapes.
        *   Chaque point est un `li` (`font-size: 1.25rem`, `margin-bottom: 1rem`).
        *   Ex: "Atelier de Cadrage : Définir ensemble le périmètre de votre POC."
*   **Animation :** Chaque point apparaît en glissant depuis la gauche (`fadeInSlideRight`).