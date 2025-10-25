# Guide de Design - Version Corporate

## 🏢 Palette de couleurs

### Couleurs principales
```css
--color-primary: #2E4053;           /* Bleu marine - Couleur corporative principale */
--color-primary-dark: #1C2833;      /* Bleu marine très foncé - Profondeur */
--color-accent: #5D6D7E;            /* Gris-bleu - Éléments secondaires */
--color-surface: #FFFFFF;           /* Blanc de fond - Clarté */
--color-surface-foreground: #1C2833;/* Texte principal foncé - Autorité */
--color-muted: #F4F6F6;             /* Gris très clair - Zones neutres */
--color-muted-foreground: #5D6D7E;  /* Gris moyen - Informations secondaires */
--color-border: #D5D8DC;            /* Bordures grises - Structure */
--color-highlight: #3498DB;         /* Bleu accent - Points d'attention */
```

### Usage des couleurs
| Couleur | Utilisation | Exemple |
|---------|-------------|---------|
| **Bleu marine** | Titres, badges, zones importantes | Headers, KPIs |
| **Bleu marine foncé** | Arrière-plans, dégradés | Slide de titre, footers |
| **Gris-bleu** | Bordures, cadres secondaires | Encadrés d'information |
| **Bleu highlight** | Accents, bordures importantes | Soulignement de titres |
| **Gris clair** | Fond de cartes, tableaux | Boxes d'information |

### Principes d'utilisation
- **Sobriété** : Privilégier le bleu marine et les gris
- **Contraste** : Blanc sur bleu foncé, texte foncé sur fond clair
- **Hiérarchie** : Le bleu highlight (#3498DB) pour guider l'œil
- **Professionnalisme** : Éviter les couleurs vives ou flashy

---

## 📝 Typographie

### Polices
```css
--font-family-display: Arial, sans-serif;   /* Titres - Universelle et professionnelle */
--font-family-content: Arial, sans-serif;   /* Corps de texte - Lisibilité */
--font-weight-display: 700;                 /* Gras pour titres - Impact */
--font-weight-content: 400;                 /* Normal pour texte - Clarté */
```

### Hiérarchie des tailles (réduites pour meilleure lisibilité)
```css
/* Très grands titres (slide de couverture uniquement) */
.text-6xl { font-size: 3rem; }      /* 48px - réduit de 72px */

/* Grands titres (titres de slides) */
.text-3xl { font-size: 1.75rem; }   /* 28px - réduit de 36px */

/* Sous-titres importants */
.text-2xl { font-size: 1.375rem; }  /* 22px - réduit de 30px */
.text-xl { font-size: 1.125rem; }   /* 18px - réduit de 24px */

/* Titres de sections */
.text-lg { font-size: 1rem; }       /* 16px - réduit de 20px */
.text-base { font-size: 0.938rem; } /* 15px - réduit de 18px */

/* Corps de texte */
.text-sm { font-size: 0.875rem; }   /* 14px - maintenu */

/* Petits textes / notes */
.text-xs { font-size: 0.75rem; }    /* 12px */
```

### Règles d'utilisation (mise à jour pour PowerPoint)
- **Titres de slides** : text-3xl (28px), couleur primary-dark, avec bordure inférieure
- **Titres de sections** : text-xl (18px), couleur primary, gras
- **Corps de texte** : text-sm (14px), couleur surface-foreground
- **Légendes et notes** : text-xs (12px), couleur muted-foreground
- **Cohérence stricte** : Pas plus de 3 tailles par slide
- **Limite de contenu** : Maximum 7-8 points par slide, données chiffrées privilégiées

---

## 🎯 Caractéristiques design

### Style général
- **Ambiance** : Professionnel, formel, corporate
- **Ton visuel** : Sobre, épuré, structuré
- **Public cible** : Décideurs, managers, professionnels

### Formes et éléments

#### En-têtes de slides avec bordure
```html
<header class="fit px-10 pt-8 pb-6">
  <div style="border-bottom: 3px solid var(--color-highlight); 
              padding-bottom: 0.5rem;">
    <h2 class="text-4xl" style="color: var(--color-primary-dark); margin: 0;">
      Titre Corporate
    </h2>
  </div>
</header>
```

#### Badges numérotés (carrés/rectangles)
```html
<div style="width: 60px; height: 60px; 
            background: var(--color-primary); 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            border-radius: 4px;">
  <p class="text-2xl" style="margin: 0; color: white; font-weight: bold;">1</p>
</div>
```

#### Cartes d'information avec bordures
```html
<div style="border: 2px solid var(--color-highlight); 
            border-radius: 8px; 
            padding: 1.5rem; 
            background: var(--color-muted);">
  <h3 class="text-xl" style="color: var(--color-primary); margin: 0;">
    <b>Titre Section</b>
  </h3>
  <p class="text-sm text-muted-foreground" style="margin: 0.5rem 0 0 0;">
    Description professionnelle et concise
  </p>
</div>
```

#### Bordure accent gauche
```html
<div style="border-left: 4px solid var(--color-highlight); 
            padding-left: 1.5rem; 
            margin-bottom: 1.5rem;">
  <h3 class="text-2xl" style="color: var(--color-primary); margin: 0;">
    Titre avec accent
  </h3>
  <p class="text-base text-muted-foreground">Description</p>
</div>
```

#### Dégradé pour slide de titre
```html
<body style="background: linear-gradient(135deg, 
             var(--color-primary-dark) 0%, 
             var(--color-primary) 100%);">
```

### Espacements
```css
--gap: 1.5rem;  /* Espacement standard */
```

- **Entre sections** : 1.5-2rem
- **Padding des cartes** : 1-1.5rem
- **Marges de slide** : 2.5rem (px-10, py-8)
- **Gap entre items** : 0.5-1rem

### Coins arrondis
- **Subtils** : 4-8px (border-radius: 4px ou 8px)
- **Jamais circulaires** (sauf exceptions)
- **Rectangulaire dominant** pour aspect professionnel

### Bordures
- **Épaisses et structurantes** : 2-3px
- **Couleurs** : highlight (#3498DB) pour important, border (#D5D8DC) pour neutre
- **Usage** : Souligner les titres, encadrer les contenus importants

---

## 💡 Tips d'implémentation

### ✅ À FAIRE

1. **Bordures épaisses pour structurer**
   ```html
   <!-- Bordure en bas des titres -->
   <div style="border-bottom: 3px solid var(--color-highlight);">
   
   <!-- Bordure à gauche pour accent -->
   <div style="border-left: 4px solid var(--color-highlight);">
   
   <!-- Cadres pour informations importantes -->
   <div style="border: 2px solid var(--color-primary);">
   ```

2. **Formes rectangulaires et carrés**
   - Badges numérotés : border-radius: 4px (pas 50%)
   - Cartes : border-radius: 8px maximum
   - Structure géométrique claire

3. **Hiérarchie par les bordures**
   - Bleu highlight pour éléments importants
   - Bleu marine pour sections principales
   - Gris pour éléments secondaires

4. **Tableaux et données structurées**
   - Privilégier les tableaux pour les KPIs
   - Alignement strict des colonnes
   - Fond alterné pour lisibilité

5. **Espaces blancs calculés**
   - Padding uniforme (1rem, 1.5rem)
   - Marges cohérentes
   - Aération professionnelle

6. **Typographie sobre**
   - Gras pour les titres et KPIs
   - Normal pour le corps de texte
   - Jamais de fantaisie

### ❌ À ÉVITER

1. **Couleurs vives ou flashy** - Rester dans les bleus et gris
2. **Trop de cercles** - Préférer rectangles et carrés
3. **Emojis excessifs** - Les utiliser avec parcimonie
4. **Texte centré partout** - Privilégier l'alignement à gauche
5. **Dégradés multiples** - Un seul pour la slide de titre
6. **Manque de structure** - Toujours cadrer et border

---

## 📐 Templates de slides types

### Slide de titre
```html
<body style="background: linear-gradient(135deg, 
             var(--color-primary-dark) 0%, 
             var(--color-primary) 100%); 
             padding: 4rem 4rem 6rem 4rem;">
  <div class="col gap-lg" style="max-width: 700px;">
    <h1 class="text-7xl" style="color: white; margin: 0; line-height: 1.2;">
      Intelligence Artificielle
    </h1>
    <h2 class="text-3xl" style="color: var(--color-muted); margin: 0;">
      Guide stratégique
    </h2>
    <div style="height: 2px; width: 120px; 
                background: var(--color-highlight); 
                margin: 2rem 0;"></div>
    <p class="text-xl" style="color: var(--color-muted); margin: 0;">
      Document confidentiel
    </p>
  </div>
</body>
```

### Slide de contenu standard
```html
<body class="col">
  <header class="fit px-10 pt-8 pb-6">
    <div style="border-bottom: 3px solid var(--color-highlight); 
                padding-bottom: 0.5rem;">
      <h2 class="text-4xl" style="color: var(--color-primary-dark); margin: 0;">
        Titre de la slide
      </h2>
    </div>
  </header>
  <main class="fill-height px-10 pb-24">
    <!-- Contenu ici -->
  </main>
</body>
```

### Slide avec 2 colonnes
```html
<main class="fill-height row gap-xl px-10 pb-12">
  <section class="fill-width col gap">
    <div style="border-left: 4px solid var(--color-highlight); 
                padding-left: 1.5rem;">
      <h3 class="text-2xl" style="color: var(--color-primary);">
        Section A
      </h3>
    </div>
    <!-- Contenu colonne A -->
  </section>
  <section class="fill-width col gap">
    <div style="border-left: 4px solid var(--color-primary); 
                padding-left: 1.5rem;">
      <h3 class="text-2xl" style="color: var(--color-primary);">
        Section B
      </h3>
    </div>
    <!-- Contenu colonne B -->
  </section>
</main>
```

### Slide avec feuille de route (étapes)
```html
<main class="fill-height px-10 pb-24">
  <div class="col gap-sm">
    <div class="row gap items-center">
      <div style="width: 60px; text-align: center;">
        <div style="width: 55px; height: 55px; 
                    background: var(--color-primary); 
                    border-radius: 4px;">
          <p class="text-xl" style="margin: 0; color: white; font-weight: bold;">
            1
          </p>
        </div>
      </div>
      <div class="fill-width bg-muted p-3 rounded">
        <p class="text-base" style="margin: 0;"><b>Phase 1</b></p>
        <p class="text-sm text-muted-foreground">Description</p>
      </div>
    </div>
    <!-- Répéter pour autres étapes -->
  </div>
</main>
```

### Slide avec KPIs
```html
<main class="fill-height row gap-xl px-10 pb-12">
  <section class="fill-width col gap">
    <h3 class="text-2xl" style="color: var(--color-primary);">
      Performance technique
    </h3>
    <div class="col gap-sm">
      <div class="bg-muted p-3 rounded">
        <p class="text-sm text-muted-foreground" style="margin: 0;">
          Temps de réponse moyen
        </p>
        <p class="text-lg" style="margin: 0.25rem 0 0 0; 
                           color: var(--color-primary); 
                           font-weight: bold;">
          &lt; 2 secondes
        </p>
      </div>
    </div>
  </section>
</main>
```

---

## 🎨 Exemples de combinaisons réussies

### Combinaison 1 : Titre avec bordure + Liste structurée
- **Titre** : text-4xl, primary-dark, bordure inférieure highlight
- **Items** : bg-muted, padding uniforme, texte base
- **Structure** : Espaces blancs généreux

### Combinaison 2 : Feuille de route
- **Badges** : Carrés bleu marine (55x55px)
- **Descriptions** : Cards avec fond muted
- **Progression** : Badge highlight pour dernière étape

### Combinaison 3 : Comparaison de sections
- **Séparation** : 2 colonnes avec bordures gauches colorées
- **Titres** : text-2xl, primary
- **Contenu** : Listes ou paragraphes courts

### Combinaison 4 : Analyse risques/mitigation
- **Structure** : Cards avec bordure primary
- **Contenu** : Risque en gras, mitigation en muted-foreground
- **Layout** : Gap-sm pour densité d'information

---

## 📊 Checklist de validation

Avant de finaliser une slide, vérifiez :

- [ ] Palette corporate respectée (bleus, gris)
- [ ] Bordures structurantes présentes
- [ ] Formes majoritairement rectangulaires
- [ ] Contraste suffisant pour projection
- [ ] Typographie sobre et hiérarchisée
- [ ] Alignement à gauche (sauf titre centré)
- [ ] Espaces blancs équilibrés
- [ ] Cohérence avec slides précédentes
- [ ] Pas de couleurs vives ou flashy
- [ ] Données chiffrées mises en avant

---

## 📈 Spécificités Corporate

### Présentation de données
```html
<!-- KPI avec highlight -->
<div class="bg-highlight p-4 rounded">
  <p class="text-sm" style="margin: 0; color: white;">Métrique clé</p>
  <p class="text-3xl" style="margin: 0.5rem 0 0 0; 
                         color: white; 
                         font-weight: bold;">
    85%
  </p>
</div>

<!-- Tableau de données -->
<div class="row gap-lg">
  <p class="text-base" style="margin: 0; width: 200px; 
                         color: var(--color-accent);">
    Label
  </p>
  <p class="text-base" style="margin: 0;"><b>Valeur</b></p>
</div>
```

### Sections avec importance variable
```html
<!-- Important : Bordure highlight -->
<div style="border: 2px solid var(--color-highlight); 
            background: var(--color-muted);">

<!-- Normal : Bordure primary -->
<div style="border: 2px solid var(--color-primary); 
            background: var(--color-muted);">

<!-- Secondaire : Bordure accent -->
<div style="border: 2px solid var(--color-accent); 
            background: var(--color-muted);">
```

### En-têtes professionnels
- **Toujours** : Bordure inférieure 3px solid highlight
- **Couleur titre** : primary-dark
- **Padding** : 0.5rem en bas
- **Fond** : Transparent (blanc de la slide)

---

## 🔐 Éléments spécifiques métier

### Conformité et sécurité
```html
<div style="border: 2px solid var(--color-highlight); 
            border-radius: 8px; 
            padding: 1rem; 
            background: var(--color-muted);">
  <h3 class="text-base" style="color: var(--color-primary);">
    <b>Conformité RGPD</b>
  </h3>
  <p class="text-sm text-muted-foreground">
    Détails de conformité
  </p>
</div>
```

### Budget et ROI
```html
<div style="background: var(--color-primary-dark); 
            padding: 1.5rem; 
            border-radius: 8px; 
            text-align: center;">
  <p class="text-lg" style="color: white; margin: 0;">
    Budget prévisionnel : <b>50-80 k€</b>
  </p>
</div>
```

### Phases de projet
- **Phase en cours** : Badge highlight
- **Phases futures** : Badge primary
- **Phases terminées** : Badge accent (gris-bleu)

---

## 🔗 Ressources complémentaires

### Standards corporate
- **Inspiration** : Présentations McKinsey, BCG
- **Style** : Professional, data-driven, structured
- **Format** : 16:9, slides denses mais aérées

### Outils de vérification
- **Contraste** : Minimum WCAG AA (4.5:1)
- **Lisibilité** : Test projection à 3 mètres
- **Cohérence** : Revue complète du deck

### Bonnes pratiques
- **Une idée par slide** pour clarté
- **Données sources** en notes de bas de page
- **Numérotation** des slides recommandée
- **Logo entreprise** si applicable

---

*Guide créé pour la présentation "IA Locale - Version Corporate"*
