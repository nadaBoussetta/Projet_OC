# GraphBench Challenge
## Réfutation automatique de conjectures en théorie des graphes

**Master 1 MIAGE – Dépôt final : 9 mai 2026**

---

## Description

Ce projet implémente un système automatisé de réfutation de conjectures en théorie des graphes. Il combine :
- **Partie 1** : Heuristique de recherche locale avec mutations et redémarrages
- **Partie 2** : Architecture FunSearch – évolution de fonctions de score via LLM (Claude)

## Structure du projet

```
graphbench/
├── src/
│   ├── conjecture.py      # Chargement et évaluation des conjectures
│   ├── invariants.py      # Calcul de tous les invariants de graphes
│   ├── mutations.py       # Générateurs initiaux et mutations locales
│   ├── heuristic.py       # Partie 1 : heuristique simple
│   ├── funsearch.py       # Partie 2 : architecture FunSearch
│   ├── main.py            # Script principal
│   └── validate.py        # Validation des contre-exemples
├── benchmark/
│   └── benchmark.xlsx     # Benchmark des 100 conjectures
├── results/               # Résultats JSON et fonctions évoluées
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Partie 1 – Heuristique simple
```bash
cd src
python main.py --mode simple --time 60
```

### Partie 2 – Architecture FunSearch
```bash
cd src
python main.py --mode funsearch --time 60
```

### Validation des résultats
```bash
cd src
python validate.py ../results/results_simple.json
```

### Options
```
--mode     simple|funsearch   Mode de recherche (défaut: simple)
--time     60                 Limite de temps par conjecture (secondes)
--benchmark path              Chemin vers le benchmark xlsx
--output   path               Fichier de sortie JSON
```

## Architecture

### Partie 1 : Heuristique simple

1. **Génération initiale** : graphes adaptés à la classe (arbres, graphes sans griffe, connexes...)
2. **Sélection** : tournoi sur une population de graphes scorés
3. **Mutations** : ajout/suppression d'arêtes/sommets, subdivision, ajout de cliques...
4. **Réparation** : reconnexion, suppression de cycles (arbres), correction des griffes
5. **Score** : `violation × 10 + bonus_structurels`
6. **Redémarrage** : si bloqué après 80 itérations sans amélioration

### Partie 2 : FunSearch

Le système fait évoluer automatiquement la fonction de score :
1. Initialisation avec une fonction de base
2. Appel à Claude (API) pour proposer des variantes améliorées
3. Évaluation des nouvelles fonctions sur des cas de test
4. Sélection des meilleures, répétition (5 itérations)

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
  "total_score": 1234.5,
  "n_found": 85,
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
- **Score total** = somme des coûts (à minimiser)
