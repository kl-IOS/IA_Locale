# Plan de Présentation : Créer votre IA Locale (Style Grand Public Moderne)

Ce document est un storyboard pour la présentation PowerPoint, aligné avec le guide de design "Grand Public Moderne".

---

### Diapositive 1 : Titre

*   **Titre :** Créez votre IA Locale
*   **Sous-titre :** Le Guide Complet de A à Z pour les non-techniciens
*   **Graphisme :** Fond en dégradé du teal (`#5EA8A7`) au teal foncé (`#277884`). Titre en blanc, police Arial, 72px (text-7xl). Sous-titre en blanc (opacité 95%), 30px (text-3xl).
*   **Animation :** Le titre apparaît avec un léger rebond. Le sous-titre apparaît en fondu en dessous.

---

### Diapositive 2 : Qu'est-ce qu'une IA locale ? 🤔

*   **Titre :** Qu'est-ce qu'une IA locale ? 🤔
*   **Phrase d'accroche :** Une IA locale fonctionne entièrement sur votre ordinateur, sans connexion Internet.
*   **Contenu :** Trois points clés, chacun dans une carte avec un fond `muted` (`#F8F9FA`), des coins arrondis (12px) et une icône circulaire colorée.
    *   **Confidentialité totale :** Vos données restent chez vous (Icône : 🔒 dans un cercle `primary` `#5EA8A7`)
    *   **Contrôle complet :** Vous maîtrisez tout le système (Icône : 🎮 dans un cercle `primary` `#5EA8A7`)
    *   **Sans frais d'abonnement :** Pas de coûts récurrents (Icône : 💸 dans un cercle `primary` `#5EA8A7`)
*   **Animation :** Chaque carte apparaît l'une après l'autre avec un effet de zoom léger.

---

### Diapositive 3 : De quoi avez-vous besoin ? 💻

*   **Titre :** De quoi avez-vous besoin ? 💻
*   **Contenu :** Deux cartes avec fond `muted` (`#F8F9FA`) et coins arrondis.
    *   **Carte 1 : Matériel**
        *   Ordinateur moderne (Windows, Mac ou Linux)
        *   16 à 32 Go de RAM
        *   Carte graphique recommandée (NVIDIA idéalement)
    *   **Carte 2 : Logiciels**
        *   Python (langage de programmation)
        *   Outils d'IA (Ollama ou LM Studio)
        *   Bibliothèques spécialisées
*   **Phrase de réconfort en bas :** Rassurez-vous : nous verrons tout pas à pas ! (Texte en `muted-foreground` `#6C757D`)
*   **Animation :** Les cartes apparaissent en glissant depuis les côtés.

---

### Diapositive 4 : Les 5 grandes étapes

*   **Titre :** Les 5 grandes étapes
*   **Graphisme :** Une liste verticale. Chaque étape a un badge numérique circulaire (fond `primary` `#5EA8A7`, texte blanc). L'étape finale (5) a un badge de couleur `accent` (`#FE4447`).
    1.  **Définir votre besoin**
    2.  **Préparer vos données**
    3.  **Choisir la bonne méthode**
    4.  **Installer et configurer**
    5.  **Tester et utiliser !**
*   **Animation :** Chaque étape apparaît l'une après l'autre.

---

### Diapositive 5 : Étape 1 - Définir votre besoin

*   **Titre :** 1. Définir votre besoin
*   **Phrase d'intro :** Posez-vous ces questions :
*   **Contenu :** Trois sections avec des icônes de questions.
    *   **Que voulez-vous faire ?** (Répondre à des questions, résumer des documents, analyser du texte...)
    *   **Quelles données avez-vous ?** (Documents PDF, notes, emails, historique YouTube...)
    *   **Quelles sont vos contraintes ?** (Vitesse, confidentialité, budget matériel...)
*   **Animation :** Chaque question apparaît en fondu.

---

### Diapositive 6 : Étape 2 - Préparer vos données

*   **Titre :** 2. Préparer vos données
*   **Contenu :**
    *   **Sources possibles :**
        *   Documents personnels (PDF, Word)
        *   Notes et transcriptions
        *   Historique YouTube (via Google Takeout)
    *   **Organisation :**
        *   Nettoyer le texte
        *   Supprimer les doublons
        *   **Protéger les informations sensibles :** On s'assure de masquer les informations personnelles comme les noms ou les adresses email.
        *   Découper en sections
*   **Graphisme :** Des icônes pour chaque type de source. Des cases à cocher pour l'organisation.
*   **Animation :** Les sources apparaissent, puis la liste d'organisation avec un effet de "check".

---

### Diapositive 7 : Étape 3 - Deux approches principales

*   **Titre :** 3. Deux approches principales
*   **Contenu :** Deux boîtes colorées.
    *   **Boîte 1 : RAG (Recherche + Génération)** (Couleur : `primary` `#5EA8A7`)
        *   Rapide à mettre en place
        *   Idéal pour des documents
        *   **Recommandé pour débuter**
    *   **Boîte 2 : Fine-tuning (Entraînement personnalisé)** (Couleur : `accent` `#FE4447`)
        *   Plus de contrôle
        *   Style personnalisé
        *   **Plus technique**
*   **Animation :** Les boîtes apparaissent avec un effet de "flip".

---

### Diapositive 8 : Comment fonctionne le RAG ?

*   **Titre :** Comment fonctionne le RAG ?
*   **Graphisme :** Un schéma très simple en 3 étapes avec des icônes.
    1.  **Indexation :** Vos documents sont convertis en vecteurs mathématiques.
    2.  **Recherche :** L'IA trouve les passages pertinents pour votre question.
    3.  **Génération :** L'IA formule une réponse basée sur ces passages.
*   **Animation :** Les étapes apparaissent en séquence.

---

### Diapositive 9 : Étape 4 - Outils à installer

*   **Titre :** 4. Outils à installer
*   **Contenu :**
    *   **Outil principal : Ollama**
        *   Interface simple pour faire fonctionner des modèles d'IA.
        *   **Recommandé pour débuter.**
    *   **Outils complémentaires :**
        *   Python, FAISS ou Chroma, Transformers
*   **Graphisme :** Logo d'Ollama bien visible.
*   **Animation :** Le bloc Ollama apparaît en premier, puis les outils complémentaires.

---

### Diapositive 10 : Étape 5 - Mise en pratique

*   **Titre :** 5. Mise en pratique
*   **Graphisme :** Une liste de 5 étapes numérotées.
    1.  Installer Ollama
    2.  Télécharger un modèle (Llama 3.1)
    3.  Indexer vos documents
    4.  Créer votre système Q&R
    5.  Tester et affiner !
*   **Animation :** Chaque étape s'allume ou change de couleur lorsqu'elle est "terminée".

---

### Diapositive 11 : Avantages et Limites

*   **Titre :** Avantages et Limites
*   **Contenu :** Deux listes côte à côte.
    *   **✓ Avantages**
        *   Confidentialité maximale
        *   Pas de frais récurrents
        *   Personnalisation totale
    *   **▲ À considérer**
        *   Investissement matériel
        *   Courbe d'apprentissage
        *   Maintenance
*   **Animation :** La liste des avantages apparaît, puis celle des limites.

---

### Diapositive 12 : Glossaire

*   **Titre :** Glossaire
*   **Contenu :**
    *   **IA Locale :** Une intelligence artificielle qui fonctionne directement sur votre ordinateur, sans passer par Internet.
    *   **RAG :** Une méthode qui permet à l'IA de rechercher dans vos documents pour vous donner une réponse. C'est comme lui donner une bibliothèque à lire.
    *   **LLM (Grand Modèle de Langage) :** Le "cerveau" de l'IA, qui comprend et génère du texte.
    *   **Fine-tuning :** Entraîner l'IA pour qu'elle adopte un style ou des connaissances spécifiques.
    *   **Ollama :** Un outil simple et populaire pour installer et utiliser des IA locales.
    *   **Anonymisation :** Le processus de suppression des informations personnelles (noms, adresses...) d'un document.
    *   **Chunking :** Le fait de découper de grands documents en petits morceaux pour que l'IA puisse les "digérer" plus facilement.

---

### Diapositive 13 : Prochaines étapes

*   **Titre :** Prochaines étapes
*   **Phrase d'accroche :** Vous êtes prêt à commencer !
*   **Sous-titre :** Commencez simplement avec le RAG et Ollama.
*   **Graphisme :** Trois boîtes numéotées.
    1.  Installez Ollama
    2.  Préparez vos données
    3.  Créez votre premier système RAG
*   **Animation :** Les boîtes apparaissent l'une après l'autre.

---

### Diapositive 14 : Questions ?

*   **Titre :** Questions ?
*   **Graphisme :** Une grande icône de point d'interrogation.
*   **Animation :** Le point d'interrogation peut avoir une légère animation de pulsation.