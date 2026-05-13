# Présentation : Détection de Motifs ADN (Gibbs Sampling)

Ce dossier contient les sources LaTeX pour la présentation du projet de Processus Stochastiques.

## Contenu
- `presentation.tex` : Fichier source Beamer.
- `images/` (référencé depuis la racine) : Illustrations utilisées.

## Instructions de Compilation
Pour générer le PDF de la présentation, utilisez `pdflatex` :
```bash
pdflatex presentation.tex
```
(Il peut être nécessaire de le lancer deux fois pour les références et la table des matières).

## Structure de la Présentation (8min 30s)
1. **Introduction (1 min)** : Le problème de découverte de motifs et les inconnues ($\Theta, \mathcal{A}$).
2. **Théorie (2 min)** : Dérivation des lois conditionnelles (Dirichlet et Ratios de vraisemblance).
3. **Architecture (1 min)** : Fonctionnement de la boucle de Gibbs.
4. **Implémentation (2 min)** : Analyse des extraits de code critiques.
5. **Optimisation (1 min)** : Phase Shift et Redémarrages multiples.
6. **Résultats (1 min)** : Comparaison Artificiel vs PurR.
7. **Conclusion (30 s)** : Bilan et perspectives.
