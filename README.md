# GraphBench Challenge
## Réfutation automatique de conjectures en théorie des graphes

**Master 1 MIAGE – Dépôt final : 9 mai 2026**  
**Étudiantes :** NADA BOUSSETTA & Lilia Kaci  
**GitHub :** https://github.com/nadaBoussetta/Projet_OC

---

### Résultats finaux Heuristique Simple 

| Métrique | Valeur |
|----------|--------|
| Conjectures réfutées | **96 / 100** |
| Score total | **867.9** |
| Temps moyen | 4.04 s |

### Résultats finaux FunSearch 

| Métrique | Valeur |
|----------|--------|
| Conjectures réfutées | **96 / 100** |
| Score total | **778.3** |
| Temps moyen | 3.11 s |


---

## Description

Ce projet implémente un système automatisé de réfutation de conjectures en théorie des graphes. Il combine :
- **Partie 1** : Heuristique de recherche locale avec mutations, recuit simulé et redémarrages
- **Partie 2** : Architecture FunSearch — évolution de fonctions de score via **Groq LLM** (Llama-3.3-70B)

## Structure du projet

```
Projet_OC/
├── src/                          # Code source
│   ├── conjecture.py             # Chargement et évaluation des conjectures
│   ├── invariants.py             # Calcul des 25 invariants (lazy, algorithmes exacts)
│   ├── mutations.py              # Générateurs initiaux et mutations locales
│   ├── heuristic.py              # Partie 1 : heuristique multi-stratégie (SA)
│   ├── funsearch.py              # Partie 2 : architecture FunSearch avec Groq LLM
│   ├── main.py                   # Orchestration CLI
│   ├── validate.py               # Validation des contre-exemples
│   └── visualize.py              # Génération des figures (matplotlib)
├── benchmark/
│   └── benchmark.xlsx            # Benchmark des 100 conjectures
├── results/
│   ├── results_final.json
│   ├── results_funsearch.json
│   ├── best_score_function.py
│   ├── groq_generated_functions  
│   └── figures/                  # Graphiques générés par visualize.py
├── requirements.txt
├── .env                          # GROQ_API_KEY (optionnel, mode funsearch)
├── README.md
├── report.pdf
```

## Installation

```bash
pip install -r requirements.txt
```

Créez un fichier `.env` à la racine du projet :

```
GROQ_API_KEY=gsk_votre_cle_ici
```

## Utilisation

### Partie 1 – Heuristique simple
```bash
python src/main.py --mode simple --time 60
```

### Partie 2 – Architecture FunSearch (Groq LLM)
```bash
python src/main.py --mode funsearch --time 60
```

### Validation des résultats
```bash
python src/validate.py results/results_final.json
```

### Génération des figures
```bash
python src/visualize.py
# With results:
python src/visualize.py --results results/results_final.json
```

### Options `main.py`
```
--mode       simple|funsearch   Mode de recherche (défaut: simple)
--time       60                 Limite de temps par conjecture (secondes)
--benchmark  path               Chemin vers le benchmark xlsx
--output     path               Fichier de sortie JSON
--verbose                       Afficher les logs détaillés
```

## Architecture

### Partie 1 : Heuristique multi-stratégie

La recherche se déroule en 3 phases :

1. **Phase 0 – Graphes spéciaux (~1 s)** : étoiles, chemins, graphes complets, bipartis, Petersen, grilles…  
2. **Phase 1 – Exploration exhaustive (n=3..11, ~5 s)** : tous les graphes de petite taille  
3. **Phase 2 – Hill-climbing + Recuit simulé** : population de 15 graphes, sélection par tournoi, liste tabu, redémarrages

Chaque phase est adaptée aux classes de la conjecture (arbre, graphe sans griffe, connexe…).  
La fonction de score est `violation × 10 + bonus_structurels` avec correction flottante `ε = 1e-9`.

### Partie 2 : FunSearch avec Groq LLM

Le système fait **évoluer automatiquement la fonction de score** en 3 étapes :

1. **Initialisation (Phase 1 – Seeds)** : 10 fonctions de score artisanales couvrant différentes stratégies
2. **Évolution par LLM (Phase 2 – Groq)** : le modèle `llama-3.3-70b-versatile` génère des variantes en voyant les meilleures fonctions courantes + le contexte de la conjecture
3. **Combinaisons offline (Phase 3)** : combinaisons pondérées des meilleures fonctions

**Prompt engineering** : le prompt système présente le contexte expert (25 invariants, classes de graphes, format de violation). Le prompt utilisateur inclut les 3 meilleures fonctions courantes, les métriques de la conjecture et les résultats d'évaluation.

**Sandbox sécurisé** : le code généré est exécuté dans un espace restreint sans accès aux modules dangereux (`import`, `open`, `exec`, `subprocess`, `socket`, `os`, `sys`…).

Les fonctions générées sont sauvegardées dans :
- `results/best_score_function.py` — meilleure fonction retenue
- `results/groq_generated_functions.py` — tout l'historique Groq

## Invariants supportés

`n, m, minimum_degree, maximum_degree, average_degree, density, diameter, radius, triangle_number, clique_number, domination_number, total_domination_number, independence_number, vertex_cover_number, independent_domination_number, matching_number, randic_index, harmonic_index, proximity, remoteness, largest_eigenvalue, second_smallest_laplace_eigenvalue, largest_distance_eigenvalue`

## Classes de graphes supportées

`connected, tree, bipartite, planar, claw_free`

## Résultats

Les résultats sont sauvegardés dans `results/results_simple.json` ou `results/results_funsearch.json`.

Format :
```json
{
  "mode": "simple",
  "total_score": 38.5,
  "n_found": 100,
  "n_total": 100,
  "results": [
    {
      "conjecture_id": 980,
      "found": true,
      "time": 0.05,
      "cost": 0.05,
      "violation": 0.0882,
      "graph6": "ShCGGC@...",
      "n": 19,
      "m": 24
    }
  ]
}
```

## Score

- **Contre-exemple trouvé en t secondes** : coût = t  
- **Non trouvé** : coût = 120  
- **Score total** = somme des coûts (à minimiser, minimum théorique = 0)

## Figures générées (`python visualize.py`)

| Fichier | Description |
|---------|-------------|
| `results/figures/fig1_benchmark_overview.png` | Distribution des classes, signes, degrés et invariants |
| `results/figures/fig2_funsearch_evolution.png` | Courbe d'évolution FunSearch (seed → Groq → combos) |
| `results/figures/fig3_invariant_network.png` | Réseau de dépendances entre invariants |
| `results/figures/fig4_results_simple.png` | Analyse des résultats mode simple |
| `results/figures/fig4_results_funsearch.png` | Analyse des résultats mode FunSearch |
| `results/figures/fig5_comparison.png` | Comparaison simple vs FunSearch |
| `results/figures/fig6_funsearch_architecture.png` | Schéma de l'architecture FunSearch |
