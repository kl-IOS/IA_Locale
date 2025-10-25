---
title: "Créez votre IA Locale"
subtitle: "Le Guide Complet de A à Z pour les Non-Techniciens"
author: "Guide IA Locale"
date: "Octobre 2025"
theme: "Madrid"
colortheme: "seahorse"
fonttheme: "structurebold"
aspectratio: 169
---

# Créez votre IA Locale 🚀

**Le Guide Complet de A à Z**

*Pour les non-techniciens*

::: notes
Présentation simplifiée, focus sur les concepts et pas le code.
Version accessible pour comprendre l'IA locale sans être développeur.
:::

---

# Qu'est-ce qu'une IA locale ? 🤔

Une IA locale fonctionne **entièrement sur votre ordinateur**, sans connexion Internet.

:::::::::::::: {.columns}
::: {.column width="33%"}
## 🔒 Confidentialité

Vos données restent **chez vous**

- Aucune transmission externe
- Contrôle total
- Zéro fuite de données
:::

::: {.column width="33%"}
## 🎮 Contrôle complet

Vous maîtrisez **tout le système**

- Personnalisation totale
- Pas de limitations
- Votre infrastructure
:::

::: {.column width="33%"}
## 💸 Sans abonnement

**Pas de coûts récurrents**

- Investissement unique
- Pas de surprise
- Économies long terme
:::
::::::::::::::

::: notes
Les 3 piliers : confidentialité, contrôle, économies.
ChatGPT Plus = 20€/mois = 240€/an = 720€ sur 3 ans.
:::

---

# De quoi avez-vous besoin ? 💻

:::::::::::::: {.columns}
::: {.column width="50%"}
## Matériel

✅ **Ordinateur moderne**
- Windows, Mac ou Linux
- 16 à 32 Go de RAM
- Carte graphique (idéalement NVIDIA)
- SSD 256 Go minimum

💡 *Pas besoin de super-ordinateur !*
:::

::: {.column width="50%"}
## Logiciels

✅ **Outils gratuits et open-source**
- **Ollama** (l'outil principal)
- Python (optionnel pour aller plus loin)

💡 *Tout est gratuit et téléchargeable !*

**💰 Budget recommandé**
- Minimum : 500-800€
- Optimal : 1200-1800€
:::
::::::::::::::

::: notes
Rassurer : un bon PC gaming ou MacBook Pro récent suffit.
Ollama sera notre outil principal, très simple à utiliser.
:::

---

# Les 5 grandes étapes 🗺️

**Du téléchargement à l'utilisation**

1️⃣ **Définir votre besoin** - Qu'est-ce que je veux faire ?

2️⃣ **Préparer vos données** - Organiser mes documents

3️⃣ **Installer Ollama** - L'outil magique (2 minutes)

4️⃣ **Télécharger un modèle d'IA** - Choisir le cerveau

5️⃣ **Utiliser votre IA !** - Poser vos questions

::: notes
Processus simple en 5 étapes.
Peut se faire en un week-end.
:::

---

# Étape 1 : Définir votre besoin 🎯

**Posez-vous ces questions :**

:::::::::::::: {.columns}
::: {.column width="33%"}
### ❓ Que voulez-vous faire ?

- Répondre à des questions
- Résumer des documents
- Analyser du texte
- Générer du contenu
:::

::: {.column width="33%"}
### 📁 Quelles données avez-vous ?

- Documents PDF, Word
- Notes personnelles
- Emails archivés
- Historique YouTube
:::

::: {.column width="33%"}
### ⚡ Vos contraintes ?

- Vitesse nécessaire
- Niveau de confidentialité
- Budget matériel
- Complexité acceptable
:::
::::::::::::::

::: notes
Exemples concrets :
- Assistant pour chercher dans documentation personnelle
- Résumeur automatique d'articles de veille
- Chatbot pour répondre sur notes de cours
:::

---

# Étape 2 : Préparer vos données 📊

:::::::::::::: {.columns}
::: {.column width="50%"}
## Sources possibles

**📄 Documents**
- PDF, Word, PowerPoint
- Fichiers texte

**📝 Notes**
- Markdown, Notion
- Obsidian, Evernote

**📧 Autres**
- Emails exportés
- Transcriptions
:::

::: {.column width="50%"}
## Organisation simple

**1. Nettoyer** 🧹
- Supprimer doublons
- Corriger erreurs évidentes

**2. Protéger** 🔐
- Masquer infos personnelles (noms, adresses)

**3. Regrouper** 📂
- Tout dans un dossier
- Sous-dossiers par thème

💡 *Plus c'est organisé, mieux c'est !*
:::
::::::::::::::

::: notes
Insister sur l'importance de la qualité des données.
"Garbage in, garbage out".
Pas besoin de perfection, juste un minimum d'organisation.
:::

---

# C'est quoi le RAG ? 🔍

**RAG = Retrieval Augmented Generation**
(Génération Augmentée par Recherche)

## 📚 Analogie simple : La bibliothèque

**Sans RAG (IA classique)**
- L'IA répond de mémoire
- Comme réciter un cours appris par cœur
- ❌ Ne connaît pas VOS documents

**Avec RAG (IA locale personnalisée)**
- L'IA **cherche d'abord** dans vos documents
- Comme consulter un livre pendant un examen
- ✅ Répond en citant VOS sources

::: notes
Analogie clé : RAG = livre ouvert pendant l'examen.
L'IA ne devine pas, elle cherche dans vos docs puis répond.
C'est pour ça que c'est précis et fiable.
:::

---

# Comment fonctionne le RAG ? 🔄

**Processus en 3 étapes simples**

:::::::::::::: {.columns}
::: {.column width="33%"}
### 1️⃣ Indexation 📇

**Une seule fois au début**

Vos documents sont "découpés" et "numérotés" comme un index de livre

💡 *Comme créer la table des matières*
:::

::: {.column width="33%"}
### 2️⃣ Recherche 🔍

**À chaque question**

L'IA cherche les passages pertinents dans l'index

💡 *Comme chercher dans un dictionnaire*
:::

::: {.column width="33%"}
### 3️⃣ Réponse ✅

**Génération intelligente**

L'IA lit les passages trouvés et formule une réponse

💡 *Comme synthétiser ses notes*
:::
::::::::::::::

**Résultat : Réponses précises basées sur VOS documents** 🎯

::: notes
Trois étapes faciles à comprendre.
L'indexation = une seule fois, automatique.
Ensuite c'est automatique : question → recherche → réponse.
:::

---

# RAG vs Fine-tuning : Quelle différence ? ⚖️

:::::::::::::: {.columns}
::: {.column width="50%"}
## RAG 🔍📝
**L'IA avec un livre ouvert**

✅ **Avantages**
- ⚡ Rapide à mettre en place
- 📄 Idéal pour documents
- 🎯 **Recommandé pour débuter**
- 💰 Pas d'entraînement coûteux

**Cas d'usage**
- Documentation entreprise
- Notes de cours
- Archives personnelles
:::

::: {.column width="50%"}
## Fine-tuning 🎓
**L'IA qui apprend par cœur**

✅ **Avantages**
- 🎨 Style très personnalisé
- 🧠 Connaissances intégrées

⚠️ **Mais...**
- 🔧 **Beaucoup plus technique**
- 💾 Nécessite des milliers d'exemples
- ⏱️ Temps d'entraînement (heures/jours)

**Pour utilisateurs avancés**
:::
::::::::::::::

**Notre recommandation : Commencez par RAG !** 🎯

::: notes
RAG = accessible, rapide, efficace pour 90% des cas.
Fine-tuning = pour plus tard, si vraiment nécessaire.
:::

---

# Étape 3 : Installer Ollama ⭐

**L'outil qui fait tout le travail !**

## ✅ Pourquoi Ollama ?

- 🚀 Installation en **2 minutes**
- 🎮 Interface ultra-simple
- 💻 Windows, Mac, Linux
- 🆓 Gratuit et open-source
- 🔒 100% local et privé

## 📥 Installation

**Sur tous les systèmes : [ollama.com](https://ollama.com)**

1. Télécharger l'installeur
2. Double-cliquer
3. C'est fait ! ✅

::: notes
Ollama est THE outil à installer.
Ultra simple, pas besoin de compétences techniques.
Installation en 2 clics comme n'importe quel logiciel.
:::

---

# Étape 4 : Choisir et télécharger un modèle 🧠

**Le "cerveau" de votre IA**

| Modèle | Taille | RAM nécessaire | Qualité | Recommandation |
|--------|--------|----------------|---------|----------------|
| **Llama 3.1 8B** | 4.7 GB | 8 GB | ⭐⭐⭐⭐ | ✅ **Débutants** |
| **Mistral 7B** | 4.1 GB | 8 GB | ⭐⭐⭐⭐ | ✅ **Excellent français** |
| **Phi-3 Mini** | 2.3 GB | 4 GB | ⭐⭐⭐ | PC modestes |
| **Llama 13B** | 7.4 GB | 16 GB | ⭐⭐⭐⭐⭐ | Utilisateurs avancés |

## 💡 Notre recommandation

**Llama 3.1 8B** - Équilibre parfait qualité/performance

::: notes
Tableau clair pour aider au choix.
Llama 3.1 8B : le meilleur compromis pour débuter.
Mistral excellent pour le français.
:::

---

# Étape 5 : Utiliser votre IA ! 🎬

**3 commandes essentielles à connaître**

## 1️⃣ Télécharger un modèle

```bash
ollama pull llama3.1:8b
```

## 2️⃣ Discuter avec l'IA

```bash
ollama run llama3.1:8b
```

## 3️⃣ Lister vos modèles installés

```bash
ollama list
```

**C'est tout ! Vous êtes opérationnel** ✅

::: notes
Seulement 3 commandes à retenir.
Pas de script Python complexe dans cette présentation.
Le guide d'installation détaillé aura plus d'info.
:::

---

# Exemple concret : Assistant de révision 🎯

**Situation** : 50 PDFs de cours universitaires

## 📋 Processus simple

1. **Regrouper** vos PDFs dans un dossier
2. **Lancer Ollama** avec votre modèle
3. **Poser vos questions** en langage naturel

## 💬 Exemples de questions

- *"Résume le chapitre sur les réseaux de neurones"*
- *"Quelle est la différence entre CNN et RNN ?"*
- *"Donne-moi 5 points clés sur la régression linéaire"*

## ✅ Résultat

- ⚡ Réponses en **1-3 secondes**
- 📚 Basées sur **vos documents**
- ⏱️ **Économie de temps : -82%** (45min → 8min)

::: notes
Exemple concret et relatable.
Montrer la valeur : gagner du temps dans révisions.
Pas de code technique, juste l'usage.
:::

---

# Avantages de l'IA locale ✅

:::::::::::::: {.columns}
::: {.column width="50%"}
## 🎉 Les gros plus

**Confidentialité maximale** 🔒
- Vos données restent chez vous
- Aucune fuite possible
- Conforme RGPD automatiquement

**Pas de frais récurrents** 💰
- Investissement unique
- Pas d'abonnement mensuel
- Amortissement rapide

**Personnalisation totale** 🎨
- Adapté à vos besoins exacts
- Aucune limite d'utilisation
- Fonctionne hors ligne
:::

::: {.column width="50%"}
## 💼 Cas d'usage concrets

**🏢 Entreprise**
- Documentation interne
- Analyse de contrats
- Support client niveau 1

**👨‍🎓 Éducation**
- Assistant de révisions
- Résumés de cours
- Q&A sur notes de lecture

**🏥 Santé**
- Dossiers médicaux
- Protocoles de soins

**🔬 Recherche**
- Analyse de littérature
- Veille scientifique
:::
::::::::::::::

::: notes
Tous bénéficient de la confidentialité.
Cas d'usage variés, tous secteurs.
:::

---

# Les limites à connaître ⚠️

**Soyons honnêtes !**

:::::::::::::: {.columns}
::: {.column width="50%"}
## 💸 Investissement initial

- PC performant : **500-2000€**
- Mais amorti en 1-2 ans vs cloud

## 📚 Courbe d'apprentissage

- **Quelques heures** pour les bases
- **Quelques jours** pour maîtriser
- Mais guides disponibles !

## 🔧 Maintenance

- Mises à jour manuelles
- Gestion de l'espace disque
:::

::: {.column width="50%"}
## ⚡ Performance variable

- Dépend de votre matériel
- Plus lent qu'IA cloud sur petit PC
- **Mais** : souvent suffisant !

## 🤓 Un peu technique

- Ligne de commande au début
- Puis interfaces graphiques existent
- **Ce guide est là pour vous aider !**

**Verdict : Les avantages l'emportent !** 🎉
:::
::::::::::::::

::: notes
Être transparent sur les contraintes.
Mais montrer que c'est gérable et que ça vaut le coup.
:::

---

# Comparaison : Local vs Cloud ☁️

| Critère | IA Locale 🏠 | IA Cloud ☁️ |
|---------|-------------|------------|
| **Confidentialité** | ✅ Totale | ❌ Partielle |
| **Coût mensuel** | ✅ 0€ | ❌ 20-100€/mois |
| **Coût initial** | ⚠️ 500-2000€ | ✅ 0€ |
| **Performance** | ⚠️ Selon matériel | ✅ Très élevée |
| **Personnalisation** | ✅ Totale | ❌ Limitée |
| **Hors ligne** | ✅ Oui | ❌ Non |
| **Complexité** | ⚠️ Moyenne | ✅ Simple |
| **Données sensibles** | ✅ Parfait | ❌ Risqué |

## 🎯 Quand choisir le local ?

- 🏥 **Données sensibles** (entreprise, santé, finance)
- 💪 **Usage intensif** (amortissement rapide)
- 🎨 **Besoin de personnalisation**
- 🌐 **Pas de connexion Internet fiable**

::: notes
Calcul d'amortissement : ChatGPT Plus 20$/mois = 720$ sur 3 ans.
PC avec GPU RTX 3060 à 1000€ amorti en < 2 ans.
:::

---

# Prochaines étapes 🚀

## 📖 Ce que vous avez appris aujourd'hui

✅ Qu'est-ce qu'une IA locale et ses avantages
✅ C'est quoi le RAG (livre ouvert vs par cœur)
✅ Le matériel et logiciels nécessaires
✅ Les 5 étapes pour créer votre IA
✅ Ollama : l'outil principal

## 🎯 Votre plan d'action

**Ce week-end** 🏁
1. Installer Ollama (10 min)
2. Télécharger Llama 3.1 (15 min)
3. Faire vos premiers tests (1h)

**La semaine prochaine** 📅
- Consulter le **Guide d'Installation détaillé**
- Organiser vos documents
- Créer votre premier système RAG

## 📚 Ressources disponibles

- **Guide Installation Ollama** (pas-à-pas avec screenshots)
- **Guide Technique Complet** (pour aller plus loin)
- Communauté : r/LocalLLaMA sur Reddit

::: notes
Fournir plan d'action clair et réaliste.
Mentionner le guide d'installation qui sera créé.
:::

---

# Questions fréquentes ❓

**Quel budget minimum ?**
→ 500-800€ pour débuter, 1200-1800€ pour l'optimal

**Faut-il être développeur ?**
→ Non ! Ollama s'utilise comme n'importe quel logiciel

**Combien de temps pour être opérationnel ?**
→ Week-end pour les tests, 1-2 semaines pour maîtriser

**Puis-je installer sur plusieurs PC ?**
→ Oui ! Chaque PC aura sa propre IA locale (voir guide installation)

**C'est vraiment 100% local ?**
→ Oui avec Ollama, aucune donnée ne sort de votre machine

**Quelle taille de modèle choisir ?**
→ Débutant : Llama 3.1 8B / Avancé : Llama 13B

**Puis-je utiliser mes documents ?**
→ Oui ! C'est tout l'intérêt du RAG

::: notes
Réponses concises aux questions fréquentes.
Orienter vers le guide détaillé pour les questions techniques.
:::

---

# Conclusion : Lancez-vous ! 🎉

## 🌟 Pourquoi vous êtes prêt

- ✅ Vous comprenez les avantages de l'IA locale
- ✅ Vous savez ce qu'est le RAG (recherche + réponse)
- ✅ Vous connaissez Ollama, l'outil principal
- ✅ Vous avez un plan d'action clair

## 🚀 Premier pas CONCRET

**Aujourd'hui même :**
1. Allez sur **ollama.com**
2. Téléchargez Ollama
3. Installez-le (2 clics)
4. Testez votre première IA !

## 📧 Ressources et contact

- **Guide Installation Ollama** : à venir
- **Guide Technique Complet** : guide_technique_detaille.pdf
- **Email** : karim.laurent@gmail.com
- **Code source** : github.com/kl-IOS/IA_Locale

**Vous avez tout ce qu'il faut pour réussir !** 💪

---

# Merci ! Questions ? 🙋

:::::::::::::: {.columns}
::: {.column width="50%"}
## 📚 Documentation

- Guide d'installation Ollama (détaillé)
- Guide technique complet
- Scripts d'exemple

## 🔗 Liens utiles

- **ollama.com** - L'outil principal
- **huggingface.co** - Catalogue de modèles
- **r/LocalLLaMA** - Communauté d'entraide
:::

::: {.column width="50%"}
## 💬 Support

**Email** : karim.laurent@gmail.com

## 🎯 Fichiers du projet

- `presentation_grand_public.pdf`
- `guide_installation_ollama.pdf`
- `guide_technique_detaille.pdf`

## 🌐 Communauté

- Discord LangChain
- Reddit r/LocalLLaMA
- Hugging Face Forums
:::
::::::::::::::

**N'hésitez pas à poser vos questions !** 😊
