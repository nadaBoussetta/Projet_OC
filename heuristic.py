"""
heuristic.py – Partie 1 : heuristique simple de recherche de contre-exemples.
Algorithme : recherche locale + redémarrage + tabou léger.
"""
import time
import random
import copy
import networkx as nx

from invariants import compute_invariants, satisfies_class
from mutations import generate_initial_graphs, mutate, repair


class Counterexample:
    def __init__(self, G, invariants, violation, conjecture_id):
        self.G = G
        self.invariants = invariants
        self.violation = violation
        self.conjecture_id = conjecture_id
        self.graph6 = nx.to_graph6_bytes(G, header=False).decode().strip()

    def __repr__(self):
        return (f"Counterexample(conj={self.conjecture_id}, "
                f"violation={self.violation:.4f}, "
                f"n={self.G.number_of_nodes()}, "
                f"graph6={self.graph6})")


def heuristic_score(G, invariants, conjecture):
    """
    Fonction de score de base (Partie 1).
    Maximise la violation tout en guidant vers des graphes prometteurs.
    """
    violation = conjecture.violation(invariants)
    n = invariants.get("n", 0)
    m = invariants.get("m", 0)
    delta = invariants.get("minimum_degree", 0)
    Delta = invariants.get("maximum_degree", 0)
    diam = invariants.get("diameter", 0)
    gamma = invariants.get("domination_number", 0)
    alpha = invariants.get("independence_number", 0)
    tau = invariants.get("vertex_cover_number", 0)
    triangles = invariants.get("triangle_number", 0)
    density = invariants.get("density", 0)

    # Score de base = violation fortement pondérée
    score = 10.0 * violation

    # Bonus structurels : encourager des graphes de taille raisonnable
    # et des propriétés structurelles qui tendent vers la violation
    if n > 0:
        score += 0.3 * diam
        score += 0.2 * Delta
        score += 0.1 * triangles
        score -= 0.05 * n
        score -= 0.2 * abs(density - 0.4)

    return score


def search_counterexample(conjecture, time_limit=60.0, verbose=False):
    """
    Heuristique principale : recherche locale avec population + redémarrages.
    Retourne un Counterexample ou None.
    """
    start_time = time.time()
    classes = conjecture.graph_classes

    # Paramètres
    pop_size = 8
    max_no_improve = 80
    tabu_size = 20

    best_score = float("-inf")
    best_result = None
    tabu = []

    # Générer la population initiale
    population = generate_initial_graphs(conjecture, n_graphs=pop_size)
    if not population:
        # Fallback: graphes aléatoires simples
        for _ in range(pop_size):
            G = nx.path_graph(random.randint(4, 12))
            population.append(G)

    # Calculer scores initiaux
    scored = []
    for G in population:
        try:
            inv = compute_invariants(G)
            sc = heuristic_score(G, inv, conjecture)
            scored.append((sc, G, inv))
        except Exception:
            pass

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_G, best_inv = scored[0]

    # Vérifier si déjà contre-exemple
    if conjecture.violation(best_inv) > 0:
        return Counterexample(best_G, best_inv, conjecture.violation(best_inv), conjecture.id)

    no_improve = 0
    iteration = 0

    while time.time() - start_time < time_limit:
        iteration += 1

        # Sélectionner un candidat (tournoi)
        candidates = random.sample(scored, min(3, len(scored)))
        _, parent_G, _ = max(candidates, key=lambda x: x[0])

        # Appliquer N mutations consécutives
        n_mutations = random.randint(1, 3)
        H = parent_G.copy()
        for _ in range(n_mutations):
            H = mutate(H, classes)

        # Réparation
        H = repair(H, classes)

        # Vérifier classe
        if not satisfies_class(H, classes):
            continue

        # Vérifier taille minimale
        if H.number_of_nodes() < 2:
            continue

        # Calculer invariants et score
        try:
            inv_H = compute_invariants(H)
            sc_H = heuristic_score(H, inv_H, conjecture)
        except Exception:
            continue

        # Vérifier si contre-exemple
        viol = conjecture.violation(inv_H)
        if viol > 0:
            elapsed = time.time() - start_time
            if verbose:
                print(f"  [FOUND] Contre-exemple en {elapsed:.2f}s | violation={viol:.4f}")
            return Counterexample(H, inv_H, viol, conjecture.id)

        # Tabou: éviter les graphes déjà vus
        g6 = nx.to_graph6_bytes(H, header=False).decode().strip()
        if g6 in tabu:
            continue
        tabu.append(g6)
        if len(tabu) > tabu_size:
            tabu.pop(0)

        # Mise à jour population
        if sc_H > best_score:
            best_score = sc_H
            best_G = H
            best_inv = inv_H
            no_improve = 0
        else:
            no_improve += 1

        # Mise à jour population avec remplacement du pire
        scored.append((sc_H, H, inv_H))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:pop_size * 2]

        # Redémarrage si bloqué
        if no_improve >= max_no_improve:
            if verbose:
                print(f"  [RESTART] iter={iteration}, best_score={best_score:.4f}")
            new_graphs = generate_initial_graphs(conjecture, n_graphs=pop_size)
            for G_new in new_graphs:
                try:
                    inv_new = compute_invariants(G_new)
                    sc_new = heuristic_score(G_new, inv_new, conjecture)
                    scored.append((sc_new, G_new, inv_new))
                    viol_new = conjecture.violation(inv_new)
                    if viol_new > 0:
                        elapsed = time.time() - start_time
                        return Counterexample(G_new, inv_new, viol_new, conjecture.id)
                except Exception:
                    pass
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:pop_size * 2]
            no_improve = 0

    return None
