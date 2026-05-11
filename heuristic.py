"""
heuristic.py – Heuristique optimisée avec score spécialisé par conjecture et calcul paresseux.
"""
import time
import random
import networkx as nx

from invariants import compute_invariants, satisfies_class, get_needed_invariants
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
    Fonction de score spécialisée selon les invariants de la conjecture.
    Guide intelligemment la recherche selon le type d'invariants impliqués.
    """
    violation = conjecture.violation(invariants)
    n = invariants.get("n", 1) or 1
    m = invariants.get("m", 0)
    x_name = conjecture.x_name
    y_name = conjecture.y_name

    # Score de base : violation très fortement pondérée
    score = 20.0 * violation

    # ── Bonus spécialisés selon les invariants en jeu ──────────────────────

    # Conjectures sur le diamètre/rayon → favoriser des graphes longs
    if "diameter" in (x_name, y_name) or "radius" in (x_name, y_name):
        diam = invariants.get("diameter", 0)
        score += 0.5 * diam - 0.02 * n

    # Conjectures sur la domination → favoriser des graphes épars
    elif "domination" in y_name or "domination" in x_name:
        gamma = invariants.get("domination_number", 0)
        delta = invariants.get("minimum_degree", 0)
        score += 0.4 * gamma - 0.1 * delta

    # Conjectures sur l'indépendance/couverture → favoriser des graphes bipartis-like
    elif "independence" in y_name or "independence" in x_name:
        alpha = invariants.get("independence_number", 0)
        score += 0.4 * alpha - 0.02 * n

    # Conjectures sur le couplage
    elif "matching" in y_name or "matching" in x_name:
        mu = invariants.get("matching_number", 0)
        score += 0.3 * mu

    # Conjectures sur les valeurs propres → favoriser des graphes hétérogènes
    elif "eigenvalue" in y_name or "eigenvalue" in x_name:
        lam = invariants.get("largest_eigenvalue", 0)
        Delta = invariants.get("maximum_degree", 0)
        score += 0.3 * lam + 0.1 * Delta - 0.03 * n

    # Conjectures sur proximity/remoteness → favoriser de longs chemins
    elif "proximity" in (x_name, y_name) or "remoteness" in (x_name, y_name):
        rem = invariants.get("remoteness", 0)
        diam = invariants.get("diameter", 0)
        score += 0.5 * rem + 0.3 * diam - 0.05 * n

    # Conjectures sur les degrés / Zagreb / Randic
    elif any(k in (x_name, y_name) for k in ["zagreb", "randic", "harmonic"]):
        Delta = invariants.get("maximum_degree", 0)
        delta = invariants.get("minimum_degree", 0)
        score += 0.2 * (Delta - delta)  # hétérogénéité des degrés

    # Conjectures sur les triangles/cliques
    elif "triangle" in (x_name, y_name) or "clique" in (x_name, y_name):
        tri = invariants.get("triangle_number", 0)
        Delta = invariants.get("maximum_degree", 0)
        score += 0.2 * tri + 0.1 * Delta

    else:
        # Bonus génériques
        Delta = invariants.get("maximum_degree", 0)
        diam = invariants.get("diameter", 0)
        score += 0.2 * Delta + 0.1 * diam - 0.02 * n

    return score


def search_counterexample(conjecture, time_limit=60.0, verbose=False):
    """
    Heuristique principale optimisée :
    - Calcul paresseux des invariants (seulement ceux nécessaires)
    - Score spécialisé par conjecture
    - Population + redémarrages + diversification progressive
    """
    start_time = time.time()
    classes = conjecture.graph_classes

    # Invariants nécessaires seulement → calcul beaucoup plus rapide
    needed = get_needed_invariants(conjecture)

    # Paramètres
    pop_size = 10
    max_no_improve = 60
    tabu_size = 30

    best_score = float("-inf")
    tabu = set()
    n_restarts = 0

    # Population initiale
    population = generate_initial_graphs(conjecture, n_graphs=pop_size)
    if not population:
        population = [nx.path_graph(random.randint(4, 12)) for _ in range(pop_size)]

    scored = []
    for G in population:
        try:
            inv = compute_invariants(G, needed)
            sc = heuristic_score(G, inv, conjecture)
            scored.append((sc, G, inv))
            if conjecture.violation(inv) > 0:
                return Counterexample(G, inv, conjecture.violation(inv), conjecture.id)
        except Exception:
            pass

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0]
    no_improve = 0
    iteration = 0

    while time.time() - start_time < time_limit:
        iteration += 1

        # Sélection par tournoi parmi les meilleurs
        pool = scored[:max(4, len(scored)//2)]
        candidates = random.sample(pool, min(3, len(pool)))
        _, parent_G, _ = max(candidates, key=lambda x: x[0])

        # Nombre de mutations adaptatif : plus de mutations si bloqué
        n_mutations = random.randint(1, 2 + min(3, n_restarts))
        H = parent_G.copy()
        for _ in range(n_mutations):
            H = mutate(H, classes)

        H = repair(H, classes)

        if not satisfies_class(H, classes) or H.number_of_nodes() < 2:
            continue

        # Éviter les graphes déjà vus (tabou sur graph6)
        try:
            g6 = nx.to_graph6_bytes(H, header=False).decode().strip()
        except Exception:
            continue
        if g6 in tabu:
            continue
        tabu.add(g6)
        if len(tabu) > tabu_size * 10:
            # Vider partiellement le tabou pour ne pas bloquer
            tabu.clear()

        try:
            inv_H = compute_invariants(H, needed)
            sc_H = heuristic_score(H, inv_H, conjecture)
        except Exception:
            continue

        viol = conjecture.violation(inv_H)
        if viol > 0:
            return Counterexample(H, inv_H, viol, conjecture.id)

        if sc_H > best_score:
            best_score = sc_H
            no_improve = 0
        else:
            no_improve += 1

        scored.append((sc_H, H, inv_H))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:pop_size * 2]

        # Redémarrage si bloqué
        if no_improve >= max_no_improve:
            n_restarts += 1
            new_graphs = generate_initial_graphs(conjecture, n_graphs=pop_size)
            for G_new in new_graphs:
                try:
                    inv_new = compute_invariants(G_new, needed)
                    sc_new = heuristic_score(G_new, inv_new, conjecture)
                    scored.append((sc_new, G_new, inv_new))
                    if conjecture.violation(inv_new) > 0:
                        return Counterexample(G_new, inv_new, conjecture.violation(inv_new), conjecture.id)
                except Exception:
                    pass
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:pop_size * 2]
            no_improve = 0

    return None
