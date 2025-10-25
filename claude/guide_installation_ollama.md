# Guide d'Installation Ollama - Pas à Pas

**Pour débutants - Avec captures d'écran et explications détaillées**

---

## 📋 Table des matières

1. [Installation d'Ollama](#1-installation-dollama)
   - Windows
   - macOS
   - Linux
2. [Premiers pas avec Ollama](#2-premiers-pas-avec-ollama)
3. [Télécharger et utiliser un modèle](#3-télécharger-et-utiliser-un-modèle)
4. [Exemples d'utilisation concrète](#4-exemples-dutilisation-concrète)
5. [Installation sur plusieurs PC](#5-installation-sur-plusieurs-pc)
6. [Comprendre l'indexation (expliqué simplement)](#6-comprendre-lindexation)
7. [Problèmes courants et solutions](#7-problèmes-courants-et-solutions)

---

## 1. Installation d'Ollama

### 🪟 Windows (Windows 10/11)

**Étape 1 : Télécharger Ollama**

1. Ouvrez votre navigateur web
2. Allez sur **[https://ollama.com](https://ollama.com)**
3. Cliquez sur le bouton **"Download for Windows"**

> 📸 *Capture d'écran recommandée : Page d'accueil ollama.com avec le bouton de téléchargement*

**Étape 2 : Installer Ollama**

1. Ouvrez le fichier téléchargé `OllamaSetup.exe`
2. Windows peut afficher un avertissement de sécurité → Cliquez sur **"Exécuter quand même"**
3. Suivez l'assistant d'installation (cliquez sur "Suivant" → "Installer")
4. L'installation prend environ **30 secondes**
5. À la fin, cliquez sur **"Terminer"**

> 📸 *Capture d'écran recommandée : Fenêtre d'installation Windows*

**Étape 3 : Vérifier l'installation**

1. Appuyez sur `Windows + R`
2. Tapez `cmd` et appuyez sur Entrée
3. Dans la fenêtre noire (invite de commandes), tapez :
   ```bash
   ollama --version
   ```
4. Vous devriez voir : `ollama version 0.x.x`

✅ **C'est installé !**

---

### 🍎 macOS (Mac)

**Étape 1 : Télécharger Ollama**

1. Ouvrez Safari ou Chrome
2. Allez sur **[https://ollama.com](https://ollama.com)**
3. Cliquez sur **"Download for macOS"**
4. Un fichier `.zip` sera téléchargé

> 📸 *Capture d'écran recommandée : Page d'accueil ollama.com*

**Étape 2 : Installer Ollama**

1. Ouvrez le fichier `.zip` téléchargé (double-clic)
2. Glissez l'application **Ollama** dans votre dossier **Applications**
3. Ouvrez le dossier **Applications**
4. Double-cliquez sur **Ollama**
5. macOS peut afficher "Impossible d'ouvrir car provient d'un développeur non identifié"
   - Allez dans **Préférences Système** → **Sécurité et confidentialité**
   - Cliquez sur **"Ouvrir quand même"**

> 📸 *Capture d'écran recommandée : Fenêtre de sécurité macOS*

**Étape 3 : Vérifier l'installation**

1. Ouvrez **Terminal** (Cmd + Espace → tapez "Terminal")
2. Tapez :
   ```bash
   ollama --version
   ```
3. Vous devriez voir : `ollama version 0.x.x`

✅ **C'est installé !**

---

### 🐧 Linux (Ubuntu/Debian)

**Étape 1 : Installation en une ligne**

Ouvrez un Terminal et exécutez :

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Cette commande :
- Télécharge le script d'installation
- Installe Ollama automatiquement
- Configure tout pour vous

**Attente : environ 1-2 minutes**

**Étape 2 : Vérifier l'installation**

```bash
ollama --version
```

Vous devriez voir : `ollama version 0.x.x`

✅ **C'est installé !**

---

## 2. Premiers pas avec Ollama

### 🎯 Les 5 commandes essentielles

Voici les **seules commandes** que vous devez connaître :

#### 1️⃣ Télécharger un modèle

```bash
ollama pull llama3.1:8b
```

**Explication** :
- `pull` = télécharger
- `llama3.1:8b` = nom du modèle (8 milliards de paramètres)

**Temps** : 5-10 minutes selon votre connexion

---

#### 2️⃣ Lancer une conversation avec l'IA

```bash
ollama run llama3.1:8b
```

**Résultat** : Une interface de chat s'ouvre

```
>>> Bonjour !
Bonjour ! Comment puis-je vous aider aujourd'hui ?

>>> Explique-moi ce qu'est une IA locale
Une IA locale est...
```

Pour **quitter** : tapez `/bye` ou `Ctrl+D`

---

#### 3️⃣ Lister les modèles installés

```bash
ollama list
```

**Résultat** :
```
NAME                ID              SIZE    MODIFIED
llama3.1:8b        a1b2c3d4        4.7GB   2 hours ago
mistral:7b         e5f6g7h8        4.1GB   1 day ago
```

---

#### 4️⃣ Supprimer un modèle

```bash
ollama rm llama3.1:8b
```

Utile pour libérer de l'espace disque.

---

#### 5️⃣ Voir les modèles disponibles

Allez sur **[https://ollama.com/library](https://ollama.com/library)**

Liste complète de tous les modèles téléchargeables.

---

## 3. Télécharger et utiliser un modèle

### 🧠 Quel modèle choisir ?

**Pour débuter, nous recommandons :**

| Modèle | Commande | Taille | RAM nécessaire | Qualité |
|--------|----------|--------|----------------|---------|
| **Llama 3.1 8B** ⭐ | `ollama pull llama3.1:8b` | 4.7 GB | 8 GB | ⭐⭐⭐⭐ Excellent |
| **Mistral 7B** 🇫🇷 | `ollama pull mistral:7b` | 4.1 GB | 8 GB | ⭐⭐⭐⭐ Parfait français |
| **Phi-3 Mini** 💻 | `ollama pull phi3:mini` | 2.3 GB | 4 GB | ⭐⭐⭐ PC modestes |

**Notre conseil** : Commencez par **Llama 3.1 8B** ou **Mistral 7B**

---

### 📥 Télécharger votre premier modèle

**Étape par étape :**

1. Ouvrez votre **Terminal** (ou **Invite de commandes** sur Windows)

2. Tapez et appuyez sur Entrée :
   ```bash
   ollama pull llama3.1:8b
   ```

3. Vous verrez :
   ```
   pulling manifest
   pulling 8934d96d3f08... 100% ▕████████████▏ 4.7 GB
   pulling 8c17c2ebb0ea... 100% ▕████████████▏ 7.0 KB
   verifying sha256 digest
   writing manifest
   success
   ```

4. **Attente** : 5-10 minutes selon votre connexion Internet

✅ **Modèle téléchargé et prêt à l'emploi !**

---

### 💬 Lancer une conversation

```bash
ollama run llama3.1:8b
```

**Interface interactive :**

```
>>> Bonjour, peux-tu te présenter ?
Bonjour ! Je suis un assistant IA fonctionnant localement sur votre ordinateur.
Je peux vous aider avec diverses tâches : répondre à des questions, résumer des
textes, écrire du contenu, etc. Comment puis-je vous aider aujourd'hui ?

>>> Résume-moi ce qu'est le RAG en 3 lignes
Le RAG (Retrieval Augmented Generation) est une technique qui combine :
1. La recherche d'informations pertinentes dans une base de documents
2. L'utilisation d'un modèle d'IA pour générer une réponse basée sur ces documents
3. Le tout permet d'obtenir des réponses précises et sourcées, sans hallucinations.
```

**Pour quitter** : tapez `/bye`

---

## 4. Exemples d'utilisation concrète

### Exemple 1 : Résumer un texte

```bash
ollama run llama3.1:8b
```

```
>>> Résume ce texte en 3 points clés :
L'intelligence artificielle locale présente de nombreux avantages par rapport
aux solutions cloud. Elle garantit la confidentialité totale des données car
rien ne quitte votre ordinateur. Elle permet aussi de réaliser des économies
substantielles en évitant les abonnements mensuels coûteux. Enfin, elle offre
une personnalisation complète selon vos besoins spécifiques.

Voici les 3 points clés :
1. Confidentialité totale - aucune donnée ne quitte votre ordinateur
2. Économies importantes - pas d'abonnement mensuel
3. Personnalisation complète selon vos besoins
```

---

### Exemple 2 : Poser des questions sur un sujet

```
>>> Explique-moi la différence entre RAM et stockage SSD

La RAM (Random Access Memory) et le stockage SSD ont des rôles différents :

RAM :
- Mémoire volatile (données effacées à l'extinction)
- Très rapide (accès en nanosecondes)
- Utilisée pour les programmes en cours d'exécution
- Plus chère par Go

SSD :
- Stockage permanent (données conservées)
- Rapide mais moins que la RAM
- Utilisé pour stocker fichiers et programmes
- Moins cher par Go

Analogie : La RAM est comme votre bureau de travail (rapide mais temporaire),
le SSD est comme votre armoire de rangement (permanent mais un peu plus lent).
```

---

### Exemple 3 : Générer du contenu

```
>>> Écris-moi un email professionnel pour demander un rendez-vous

Objet : Demande de rendez-vous

Bonjour [Nom],

J'espère que ce message vous trouve en bonne forme.

Je souhaiterais solliciter un rendez-vous avec vous afin de discuter de
[sujet précis]. Je pense que nos échanges pourraient être mutuellement
bénéfiques concernant [contexte].

Seriez-vous disponible la semaine prochaine pour un entretien de 30 minutes ?
Je m'adapterai volontiers à vos disponibilités.

Dans l'attente de votre retour,
Cordialement,
[Votre nom]
```

---

## 5. Installation sur plusieurs PC

### 🖥️ Question : Puis-je installer Ollama sur plusieurs ordinateurs ?

**Réponse : OUI ! Et c'est même recommandé.**

Chaque PC aura sa **propre IA locale indépendante**.

---

### 📦 Méthode 1 : Installation manuelle (recommandée pour 2-3 PC)

**Sur chaque PC :**

1. Téléchargez Ollama depuis [ollama.com](https://ollama.com)
2. Installez-le (2 minutes)
3. Téléchargez les modèles souhaités :
   ```bash
   ollama pull llama3.1:8b
   ollama pull mistral:7b
   ```

**Avantages** :
- ✅ Simple et sûr
- ✅ Chaque PC est indépendant
- ✅ Pas de compétences techniques requises

**Temps** : 10-15 minutes par PC

---

### 🚀 Méthode 2 : Script d'installation automatique (Windows)

**Pour installer sur 5+ PC de manière automatisée**

Créez un fichier `install_ollama.bat` :

```batch
@echo off
echo ========================================
echo Installation automatique d'Ollama
echo ========================================

REM Télécharger Ollama
echo Téléchargement d'Ollama...
curl -L https://ollama.com/download/OllamaSetup.exe -o %TEMP%\OllamaSetup.exe

REM Installer Ollama en mode silencieux
echo Installation en cours...
%TEMP%\OllamaSetup.exe /S

REM Attendre la fin de l'installation
timeout /t 10

REM Télécharger les modèles
echo Téléchargement des modèles IA...
ollama pull llama3.1:8b
ollama pull mistral:7b

echo ========================================
echo Installation terminée avec succès !
echo ========================================
pause
```

**Utilisation** :
1. Copiez ce fichier sur une clé USB
2. Sur chaque PC : double-cliquez sur `install_ollama.bat`
3. Attendez 15-20 minutes (téléchargement + installation)

---

### 🐧 Méthode 3 : Script pour Linux

Créez un fichier `install_ollama.sh` :

```bash
#!/bin/bash

echo "========================================"
echo "Installation automatique d'Ollama"
echo "========================================"

# Installation d'Ollama
echo "Installation d'Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Téléchargement des modèles
echo "Téléchargement des modèles..."
ollama pull llama3.1:8b
ollama pull mistral:7b

echo "========================================"
echo "Installation terminée avec succès !"
echo "========================================"
```

**Utilisation** :

```bash
chmod +x install_ollama.sh
./install_ollama.sh
```

---

### 🔄 Synchroniser les modèles entre PC (optionnel)

**Problème** : Les modèles sont gros (4-7 GB), les télécharger 10 fois est long.

**Solution** : Télécharger une fois, copier sur les autres PC.

**Où se trouvent les modèles ?**

- **Windows** : `C:\Users\VotreNom\.ollama\models`
- **macOS** : `~/.ollama/models`
- **Linux** : `~/.ollama/models`

**Procédure** :

1. Sur le PC 1 : téléchargez tous les modèles
2. Copiez le dossier `.ollama/models` sur une clé USB
3. Sur les autres PC :
   - Installez Ollama (sans télécharger de modèles)
   - Remplacez le dossier `.ollama/models` par celui de la clé USB

**Gain de temps** : Téléchargement 1 fois au lieu de N fois !

---

## 6. Comprendre l'indexation

### 🤔 C'est quoi l'indexation exactement ?

**Explication très simple :**

Imaginez que vous avez 50 livres et qu'on vous demande : *"Où est-il parlé de recettes de pâtes ?"*

**Sans index** : Vous devez lire les 50 livres en entier (des heures !)

**Avec index** : Vous consultez l'index de chaque livre, qui dit :
- Livre 3 : pages 45, 78, 120
- Livre 12 : pages 23, 56

Vous allez **directement** aux bonnes pages ! (quelques minutes)

---

### 📇 L'indexation pour l'IA = créer un index

**Processus :**

1. **Vos documents** (PDFs, Word, textes)
   - Exemple : 50 PDFs de cours universitaires

2. **Découpage en morceaux** (chunks)
   - Chaque PDF est découpé en petits passages de ~500 mots
   - Exemple : PDF de 10 pages → 20 morceaux

3. **Transformation en nombres** (embeddings/vecteurs)
   - Chaque morceau est converti en une série de nombres
   - Ces nombres représentent le "sens" du texte
   - Exemple : "recette de pâtes" → `[0.2, -0.5, 0.8, ...]` (768 nombres)

4. **Stockage dans une base de données** (vectorstore)
   - Tous les morceaux numérotés sont rangés
   - Comme un gros index de bibliothèque

---

### 🔍 Comment ça marche quand vous posez une question ?

**Vous demandez** : *"C'est quoi un réseau de neurones ?"*

**Étapes automatiques :**

1. **Votre question est transformée en nombres**
   - "réseau de neurones" → `[0.3, -0.4, 0.7, ...]`

2. **L'IA cherche dans l'index**
   - Elle compare vos nombres avec tous les morceaux indexés
   - Elle trouve les 3-5 morceaux les plus proches (similaires)

3. **L'IA lit les passages pertinents**
   - Passage 1 : "Un réseau de neurones est..."
   - Passage 2 : "Les réseaux de neurones se composent de..."

4. **L'IA formule une réponse**
   - Elle combine les informations trouvées
   - Elle génère une réponse claire et précise

**Résultat** : Réponse basée sur **vos documents**, pas inventée !

---

### ⏱️ Quand fait-on l'indexation ?

**Une seule fois au début !**

- Vous indexez vos 50 PDFs : **une fois** (2-5 minutes)
- Ensuite, vous posez 1000 questions : **instantané** (1-3 secondes par réponse)

**C'est comme :**
- Créer l'index d'un livre : long (une fois)
- Consulter l'index : rapide (à chaque fois)

---

### 🛠️ Comment indexer vos documents ?

**Outils qui font tout automatiquement :**

1. **AnythingLLM** ([useanything.com](https://useanything.com))
   - Interface graphique (pas de code)
   - Glisser-déposer vos PDFs
   - Indexation automatique

2. **Langflow** ([langflow.org](https://www.langflow.org))
   - Interface visuelle
   - Connexion facile à Ollama

3. **Open WebUI** ([github.com/open-webui](https://github.com/open-webui/open-webui))
   - Interface type ChatGPT
   - Upload de documents
   - Indexation en 1 clic

**Avec ces outils : pas besoin de coder !**

---

### 📊 Exemple concret d'indexation

**Situation** : 50 PDFs de cours (environ 2000 pages)

**Processus avec AnythingLLM** :

1. **Télécharger AnythingLLM** (5 min)
2. **Connecter à Ollama** (1 min)
3. **Importer vos PDFs** (glisser-déposer, 2 min)
4. **Lancer l'indexation** (cliquer sur "Index", attendre 3-5 min)
5. **C'est prêt !** Posez vos questions

**Temps total : 10-15 minutes**

**Ensuite** :
- Question 1 : *"Résume le chapitre 3"* → Réponse en 2 secondes ✅
- Question 2 : *"Différence entre CNN et RNN ?"* → Réponse en 3 secondes ✅

---

## 7. Problèmes courants et solutions

### ❌ Problème 1 : "ollama: command not found"

**Sur Windows :**
1. Redémarrez l'Invite de commandes
2. Si ça ne marche toujours pas :
   - Cherchez "Modifier les variables d'environnement système"
   - Ajoutez `C:\Program Files\Ollama` au PATH

**Sur Mac/Linux :**
```bash
# Ajoutez à votre ~/.bashrc ou ~/.zshrc
export PATH=$PATH:/usr/local/bin
```

Puis redémarrez le Terminal.

---

### ❌ Problème 2 : "Out of memory"

**Causes** : Votre PC manque de RAM pour le modèle choisi.

**Solutions** :

1. **Utilisez un modèle plus petit**
   ```bash
   ollama pull phi3:mini  # Seulement 2.3 GB
   ```

2. **Fermez les autres applications**
   - Navigateur web
   - Applications lourdes

3. **Ajoutez de la RAM** (solution à long terme)
   - 8 GB → 16 GB pour confort
   - 16 GB → 32 GB pour modèles avancés

---

### ❌ Problème 3 : Téléchargement très lent

**Causes** : Connexion Internet lente ou serveurs surchargés.

**Solutions** :

1. **Réessayez plus tard** (le soir, moins de trafic)

2. **Téléchargez depuis un autre réseau**
   - WiFi au lieu d'Ethernet (ou inverse)
   - Hotspot mobile

3. **Vérifiez votre connexion**
   ```bash
   # Test de vitesse
   speedtest-cli
   ```

---

### ❌ Problème 4 : Réponses lentes (>10 secondes)

**Causes** : Pas d'accélération GPU ou CPU trop lent.

**Diagnostics** :

**Sur Windows/Linux avec NVIDIA :**
```bash
nvidia-smi
```

Si ça affiche votre carte graphique → OK
Si erreur → GPU pas détecté

**Solutions** :

1. **Installer les drivers NVIDIA** ([nvidia.com/drivers](https://www.nvidia.com/drivers))

2. **Installer CUDA Toolkit**
   - [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

3. **Utiliser un modèle plus petit**
   - Phi-3 Mini au lieu de Llama 13B

---

### ❌ Problème 5 : Réponses imprécises ou "hallucinations"

**Causes** : Le modèle invente des réponses.

**Solutions** :

1. **Utilisez le RAG !**
   - Avec RAG : réponses basées sur vos documents (précises)
   - Sans RAG : le modèle devine (imprécis)

2. **Soyez plus précis dans vos questions**
   - ❌ Mauvais : "Explique-moi ça"
   - ✅ Bon : "Explique-moi la différence entre RAG et fine-tuning en 3 points"

3. **Ajustez la température** (pour utilisateurs avancés)
   ```bash
   ollama run llama3.1:8b --temperature 0.1
   ```
   - 0.1 = très factuel, peu créatif
   - 0.7 = équilibré
   - 1.0 = très créatif, peut inventer

---

## 🎓 Récapitulatif : Vos prochaines étapes

**Vous savez maintenant :**

✅ Installer Ollama sur Windows, Mac ou Linux
✅ Télécharger et utiliser un modèle d'IA
✅ Les 5 commandes essentielles
✅ Installer Ollama sur plusieurs PC (avec ou sans scripts)
✅ Ce qu'est l'indexation et comment ça marche
✅ Résoudre les problèmes courants

**Plan d'action immédiat :**

1. **Installez Ollama** (10 min)
2. **Téléchargez Llama 3.1 8B** (10 min)
3. **Testez avec quelques questions** (30 min)
4. **Explorez AnythingLLM pour l'indexation** (1h)
5. **Indexez vos propres documents** (1h)

**Dans 2-3 heures, vous aurez une IA locale fonctionnelle ! 🎉**

---

## 📚 Ressources complémentaires

**Documentation officielle :**
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/README.md)
- [Liste des modèles](https://ollama.com/library)

**Outils graphiques (pas de code) :**
- [AnythingLLM](https://useanything.com) - Interface complète
- [Open WebUI](https://github.com/open-webui/open-webui) - Interface type ChatGPT
- [Langflow](https://www.langflow.org) - Builder visuel

**Communautés d'entraide :**
- Reddit : [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
- Discord : LangChain, Ollama
- Hugging Face Forums

**Contact :**
- Email : karim.laurent@gmail.com

---

**Bonne chance dans votre aventure avec l'IA locale ! 🚀**
