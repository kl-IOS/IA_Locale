# Guide de Design - Version Grand Public Moderne

## 🎨 Palette de couleurs

### Couleurs principales
```css
--color-primary: #5EA8A7;           /* Teal principal - Couleur dominante */
--color-primary-dark: #277884;      /* Teal foncé - Contrastes et ombres */
--color-accent: #FE4447;            /* Coral/Rouge accent - Points d'attention */
--color-surface: #FFFFFF;           /* Blanc de fond - Arrière-plans */
--color-surface-foreground: #2C3E50;/* Texte principal - Lisibilité maximale */
--color-muted: #F8F9FA;             /* Gris très clair - Zones secondaires */
--color-muted-foreground: #6C757D;  /* Gris moyen - Texte secondaire */
--color-border: #DEE2E6;            /* Bordures - Séparateurs subtils */
```

### Usage des couleurs
| Couleur | Utilisation | Exemple |
|---------|-------------|---------|
| **Teal principal** | Titres, icônes, badges principaux | Headers, boutons CTA |
| **Teal foncé** | Arrière-plans, dégradés | Slide de titre |
| **Coral/Rouge** | Accents, étapes finales, alertes | Badge "Terminé", points importants |
| **Gris clair** | Cartes, zones de contenu | Boxes d'information |
| **Gris moyen** | Descriptions, sous-titres | Texte explicatif |

---

## 📝 Typographie

### Polices
```css
--font-family-display: Arial, sans-serif;   /* Titres et headers */
--font-family-content: Arial, sans-serif;   /* Corps de texte */
--font-weight-display: 700;                 /* Gras pour titres */
--font-weight-content: 400;                 /* Normal pour texte */
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
- **Titres principaux** : text-3xl (28px), couleur primary-dark, gras
- **Sous-titres** : text-xl (18px), couleur primary, gras
- **Titres de sections** : text-lg (16px), couleur primary
- **Corps de texte** : text-sm (14px), couleur surface-foreground
- **Descriptions/notes** : text-xs (12px), couleur muted-foreground
- **Maximum 3 tailles** de texte par slide pour la cohérence
- **Limite de contenu** : Maximum 6-7 points par slide, utiliser des puces courtes

---

## 🎯 Caractéristiques design

### Style général
- **Ambiance** : Moderne, dynamique, accueillant, accessible
- **Ton visuel** : Chaleureux et friendly
- **Public cible** : Non-techniciens, grand public

### Formes et éléments

#### Badges numérotés (cercles)
```html
<div style="width: 60px; height: 60px; 
            background: var(--color-primary); 
            color: white; 
            border-radius: 50%; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            font-size: 28px; 
            font-weight: bold;">
  <p style="margin: 0; color: white;">1</p>
</div>
```

#### Cartes d'information
```html
<div class="bg-muted p-4 rounded">
  <p class="text-lg" style="margin: 0;">
    <b style="color: var(--color-primary);">Titre</b>
  </p>
  <p class="text-sm text-muted-foreground" style="margin: 0.5rem 0 0 0;">
    Description claire et concise
  </p>
</div>
```

#### Icônes circulaires colorées
```html
<div style="width: 60px; height: 60px; 
            background: var(--color-primary); 
            border-radius: 50%;"></div>
```

#### Dégradés pour slides importantes
```html
<body style="background: linear-gradient(135deg, 
             var(--color-primary) 0%, 
             var(--color-primary-dark) 100%);">
```

### Espacements
```css
--gap: 1.5rem;  /* Espacement standard entre éléments */
```

- **Entre sections** : 1.5-2rem
- **Padding des cartes** : 1-1.5rem
- **Marges de slide** : 2-3rem (px-8, py-8)
- **Gap entre éléments d'une liste** : 0.5-1rem

### Coins arrondis
- **Standard** : 8-12px (rounded, border-radius: 8px)
- **Cercles** : 50% (border-radius: 50%)
- **Pills** : 999px pour boutons allongés

---

## 💡 Tips d'implémentation

### ✅ À FAIRE

1. **Utilisez des emojis** pour illustrer les concepts
   - 💻 Matériel, 🛠️ Outils, 📁 Données, ⚡ Performance
   - Placez-les dans des textes (pas dans des divs seuls)

2. **Badges numérotés circulaires** pour les étapes
   ```html
   <!-- Étape 1, 2, 3... en cercles teal -->
   <!-- Étape finale en cercle coral/rouge -->
   ```

3. **Dégradés pour impact visuel**
   - Slide de titre
   - Slides de transitions importantes
   - Blocs de conclusion

4. **Hiérarchie visuelle forte**
   - Gros titres (text-4xl)
   - Sous-titres plus petits (text-xl)
   - Corps de texte lisible (text-base)

5. **Contrastes de couleurs**
   - Texte foncé sur fond clair
   - Texte blanc sur fonds colorés (teal, coral)

6. **Illustrations visuelles**
   - Cercles colorés pour représenter des concepts
   - Flèches (→, ↓) pour montrer des flux
   - Lignes de séparation colorées

### ❌ À ÉVITER

1. **Trop de couleurs différentes** - Limitez-vous à la palette
2. **Texte trop dense** - Maximum 5 points par slide
3. **Formes carrées strictes** - Préférez les arrondis
4. **Langage technique** - Vulgarisez au maximum
5. **Plus de 3 tailles de texte** par slide
6. **Manque d'espace blanc** - Aérez le contenu

---

## 📐 Templates de slides types

### Slide de titre
```html
<body style="background: linear-gradient(135deg, 
             var(--color-primary) 0%, 
             var(--color-primary-dark) 100%);">
  <div class="col center gap-lg">
    <h1 class="text-7xl" style="color: white; text-align: center;">
      Titre Principal
    </h1>
    <h2 class="text-3xl" style="color: white; opacity: 0.95;">
      Sous-titre
    </h2>
  </div>
</body>
```

### Slide de contenu standard
```html
<body class="col">
  <header class="fit p-8 pb-4">
    <h2 class="text-4xl" style="color: var(--color-primary-dark);">
      Titre de la slide
    </h2>
  </header>
  <main class="fill-height col gap-lg px-8 pb-16">
    <!-- Contenu ici -->
  </main>
</body>
```

### Slide avec étapes numérotées
```html
<div class="col gap">
  <div class="row gap items-center">
    <div style="width: 60px; height: 60px; 
                background: var(--color-primary); 
                color: white; 
                border-radius: 50%;">
      <p style="margin: 0; color: white;">1</p>
    </div>
    <div class="fill-width">
      <p class="text-xl" style="margin: 0;"><b>Titre de l'étape</b></p>
      <p class="text-base text-muted-foreground">Description</p>
    </div>
  </div>
</div>
```

### Slide de comparaison (2 colonnes)
```html
<main class="fill-height row gap-xl px-8 pb-12">
  <section class="fill-width" style="background: var(--color-primary); 
                                     padding: 2rem; 
                                     border-radius: 12px;">
    <h3 class="text-3xl" style="color: white;">Option A</h3>
    <p style="color: white;">Description</p>
  </section>
  <section class="fill-width" style="background: var(--color-accent); 
                                     padding: 2rem; 
                                     border-radius: 12px;">
    <h3 class="text-3xl" style="color: white;">Option B</h3>
    <p style="color: white;">Description</p>
  </section>
</main>
```

---

## 🎨 Exemples de combinaisons réussies

### Combinaison 1 : Titre + Liste
- **Titre** : text-4xl, primary-dark
- **Items de liste** : text-base avec icons circulaires teal
- **Fond** : blanc

### Combinaison 2 : Étapes progressives
- **Numéros** : Cercles teal (étapes 1-4), cercle coral (étape finale)
- **Titres d'étapes** : text-xl, bold
- **Descriptions** : text-sm, muted-foreground

### Combinaison 3 : Cards d'information
- **Fond de card** : muted (#F8F9FA)
- **Titre de card** : text-lg, primary
- **Corps** : text-sm, muted-foreground
- **Padding** : 1rem, border-radius: 8px

---

## 📊 Checklist de validation

Avant de finaliser une slide, vérifiez :

- [ ] Les couleurs proviennent de la palette définie
- [ ] Maximum 3 tailles de texte différentes
- [ ] Contraste suffisant (texte lisible)
- [ ] Espaces blancs généreux
- [ ] Coins arrondis (pas de carrés stricts)
- [ ] Emojis ou icônes pour illustrer
- [ ] Contenu concis (5 points max par slide)
- [ ] Hiérarchie visuelle claire
- [ ] Cohérence avec les autres slides

---

## 🔗 Ressources complémentaires

### Outils de vérification
- **Contraste de couleurs** : WebAIM Contrast Checker
- **Palette** : Coolors.co pour variations
- **Accessibilité** : WCAG 2.1 niveau AA minimum

### Inspiration
- Style : Moderne, startup tech, design thinking
- Références : Presentations Canva "Modern & Colorful"
- Ambiance : Accessible, engageant, non-intimidant

---

*Guide créé pour la présentation "IA Locale - Version Grand Public Moderne"*
