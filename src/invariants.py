"""
invariants.py – Calcul des invariants avec algorithmes exacts pour petits graphes.
Optimisé pour la performance: calcul paresseux et algorithmes adaptés à la taille.
"""
import math
import networkx as nx
import numpy as np
from itertools import combinations


def get_needed_invariants(conjecture) -> set:
    """Retourne les invariants nécessaires pour évaluer une conjecture."""
    needed = {"n", "m", "minimum_degree", "maximum_degree", "average_degree", "density"}
    needed.add(conjecture.x_name)
    needed.add(conjecture.y_name)
    return needed


def compute_invariants(G: nx.Graph, needed: set = None) -> dict:
    """Calcul optimisé des invariants - ne calcule que ce qui est nécessaire."""
    inv = {}
    n = G.number_of_nodes()
    m = G.number_of_edges()
    inv["n"] = n
    inv["m"] = m

    if n == 0:
        return {k: 0 for k in _all_keys()}

    degrees = [d for _, d in G.degree()]
    inv["minimum_degree"] = min(degrees)
    inv["maximum_degree"] = max(degrees)
    inv["average_degree"] = (2 * m / n) if n > 0 else 0
    inv["density"] = (2 * m / (n * (n - 1))) if n > 1 else 0

    _need = needed if needed is not None else set(_all_keys())

    # Diameter and radius
    if "diameter" in _need or "radius" in _need:
        try:
            if nx.is_connected(G):
                ecc = nx.eccentricity(G)
                inv["diameter"] = max(ecc.values())
                inv["radius"] = min(ecc.values())
            else:
                gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                ecc = nx.eccentricity(gcc)
                inv["diameter"] = max(ecc.values())
                inv["radius"] = min(ecc.values())
        except Exception:
            inv["diameter"] = 0
            inv["radius"] = 0
    else:
        inv["diameter"] = 0
        inv["radius"] = 0

    # Triangle number
    if "triangle_number" in _need:
        inv["triangle_number"] = sum(nx.triangles(G).values()) // 3
    else:
        inv["triangle_number"] = 0

    # Clique number
    if "clique_number" in _need:
        inv["clique_number"] = _clique_number(G)
    else:
        inv["clique_number"] = 1

    # Domination number (exact for small graphs)
    if "domination_number" in _need:
        inv["domination_number"] = _domination_number_exact(G)
    else:
        inv["domination_number"] = 0

    # Total domination number
    if "total_domination_number" in _need:
        inv["total_domination_number"] = _total_domination_number_exact(G)
    else:
        inv["total_domination_number"] = 0

    # Independence number (exact for small graphs)
    if "independence_number" in _need or "vertex_cover_number" in _need:
        alpha = _independence_number_exact(G)
        inv["independence_number"] = alpha
        inv["vertex_cover_number"] = n - alpha
    else:
        inv["independence_number"] = 0
        inv["vertex_cover_number"] = 0

    # Independent domination number
    if "independent_domination_number" in _need:
        inv["independent_domination_number"] = _independent_domination_number_exact(G)
    else:
        inv["independent_domination_number"] = 0

    # Matching number
    if "matching_number" in _need:
        inv["matching_number"] = len(nx.max_weight_matching(G, maxcardinality=True))
    else:
        inv["matching_number"] = 0

    # Zagreb indices
    if "first_zagreb_index" in _need:
        inv["first_zagreb_index"] = sum(d ** 2 for d in degrees)
    else:
        inv["first_zagreb_index"] = 0

    if "second_zagreb_index" in _need:
        inv["second_zagreb_index"] = sum(G.degree(u) * G.degree(v) for u, v in G.edges())
    else:
        inv["second_zagreb_index"] = 0

    # Randic index
    if "randic_index" in _need:
        inv["randic_index"] = sum(
            1.0 / math.sqrt(G.degree(u) * G.degree(v))
            for u, v in G.edges() if G.degree(u) > 0 and G.degree(v) > 0
        )
    else:
        inv["randic_index"] = 0

    # Harmonic index
    if "harmonic_index" in _need:
        inv["harmonic_index"] = sum(
            2.0 / (G.degree(u) + G.degree(v))
            for u, v in G.edges() if G.degree(u) + G.degree(v) > 0
        )
    else:
        inv["harmonic_index"] = 0

    # Proximity and remoteness
    # Definition used by benchmark authors:
    #   proximity  = 1 / max_v(avg_dist_from_v)   -- reciprocal of the most-peripheral vertex's avg-dist
    #   remoteness = 1 / min_v(avg_dist_from_v)   -- reciprocal of the most-central vertex's avg-dist
    # Both values lie in (0, 1] for connected graphs with n >= 2.
    if "proximity" in _need or "remoteness" in _need:
        try:
            if nx.is_connected(G) and n > 1:
                dist = dict(nx.all_pairs_shortest_path_length(G))
                avg_dist = {v: sum(dist[v].values()) / (n - 1) for v in G.nodes()}
                max_avg = max(avg_dist.values())
                min_avg = min(avg_dist.values())
                inv["proximity"] = (1.0 / max_avg) if max_avg > 0 else 0.0
                inv["remoteness"] = (1.0 / min_avg) if min_avg > 0 else 0.0
            else:
                inv["proximity"] = 0.0
                inv["remoteness"] = 0.0
        except Exception:
            inv["proximity"] = 0.0
            inv["remoteness"] = 0.0
    else:
        inv["proximity"] = 0.0
        inv["remoteness"] = 0.0

    # Eigenvalues
    _eigen_needed = {"largest_eigenvalue", "second_smallest_laplace_eigenvalue",
                     "largest_distance_eigenvalue"}
    if _eigen_needed & _need:
        try:
            if n > 1:
                A = nx.to_numpy_array(G)
                eigenvalues = np.linalg.eigvalsh(A)
                inv["largest_eigenvalue"] = float(np.max(eigenvalues))

                L = nx.laplacian_matrix(G).toarray().astype(float)
                lap_eigs = np.sort(np.linalg.eigvalsh(L))
                inv["second_smallest_laplace_eigenvalue"] = float(lap_eigs[1]) if len(lap_eigs) > 1 else 0.0

                if "largest_distance_eigenvalue" in _need and nx.is_connected(G):
                    D = _distance_matrix(G)
                    inv["largest_distance_eigenvalue"] = float(np.max(np.linalg.eigvalsh(D)))
                else:
                    inv["largest_distance_eigenvalue"] = 0.0
            else:
                inv["largest_eigenvalue"] = 0.0
                inv["second_smallest_laplace_eigenvalue"] = 0.0
                inv["largest_distance_eigenvalue"] = 0.0
        except Exception:
            inv["largest_eigenvalue"] = 0.0
            inv["second_smallest_laplace_eigenvalue"] = 0.0
            inv["largest_distance_eigenvalue"] = 0.0
    else:
        inv["largest_eigenvalue"] = 0.0
        inv["second_smallest_laplace_eigenvalue"] = 0.0
        inv["largest_distance_eigenvalue"] = 0.0

    return inv


def _all_keys():
    return [
        "n", "m", "minimum_degree", "maximum_degree", "average_degree", "density",
        "diameter", "radius", "triangle_number", "clique_number", "domination_number",
        "total_domination_number", "independence_number", "vertex_cover_number",
        "independent_domination_number", "matching_number", "first_zagreb_index",
        "second_zagreb_index", "randic_index", "harmonic_index", "proximity",
        "remoteness", "largest_eigenvalue", "second_smallest_laplace_eigenvalue",
        "largest_distance_eigenvalue"
    ]


# ─────────────────────────────────────────────
#  ALGORITHMES EXACTS POUR PETITS GRAPHES
# ─────────────────────────────────────────────

def _clique_number(G):
    """Nombre de clique - exact via NetworkX."""
    try:
        cliques = list(nx.find_cliques(G))
        return max(len(c) for c in cliques) if cliques else 1
    except Exception:
        return 1


def _domination_number_exact(G):
    """Nombre de domination exact pour n <= 22, greedy sinon."""
    n = G.number_of_nodes()
    if n == 0:
        return 0
    if n == 1:
        return 1

    nodes = list(G.nodes())
    closed_neighborhoods = {}
    for v in nodes:
        closed_neighborhoods[v] = frozenset(G.neighbors(v)) | {v}

    if n <= 22:
        all_nodes = frozenset(nodes)
        for k in range(1, n + 1):
            for subset in combinations(nodes, k):
                dominated = set()
                for v in subset:
                    dominated |= closed_neighborhoods[v]
                if dominated >= all_nodes:
                    return k
        return n
    else:
        return _domination_greedy(G, closed_neighborhoods)


def _domination_greedy(G, closed_neighborhoods=None):
    """Approximation greedy du nombre de domination."""
    nodes = list(G.nodes())
    n = len(nodes)
    if closed_neighborhoods is None:
        closed_neighborhoods = {}
        for v in nodes:
            closed_neighborhoods[v] = frozenset(G.neighbors(v)) | {v}

    all_nodes = set(nodes)
    dominated = set()
    dominating = []
    available = set(nodes)

    while dominated != all_nodes:
        best_v = max(available, key=lambda v: len(closed_neighborhoods[v] - dominated))
        dominating.append(best_v)
        dominated |= closed_neighborhoods[best_v]
        available.discard(best_v)
        if not available:
            break

    return len(dominating)


def _total_domination_number_exact(G):
    """Nombre de domination totale exact pour petits graphes."""
    n = G.number_of_nodes()
    if n <= 1:
        return n

    nodes = list(G.nodes())
    if min(d for _, d in G.degree()) == 0:
        return n

    open_neighborhoods = {}
    for v in nodes:
        open_neighborhoods[v] = frozenset(G.neighbors(v))

    if n <= 20:
        all_nodes = frozenset(nodes)
        for k in range(2, n + 1):
            for subset in combinations(nodes, k):
                s = frozenset(subset)
                all_dominated = True
                for v in nodes:
                    if not open_neighborhoods[v] & s:
                        all_dominated = False
                        break
                if all_dominated:
                    return k
        return n
    else:
        # Greedy
        dominated = set()
        dominating = []
        sorted_nodes = sorted(nodes, key=lambda v: G.degree(v), reverse=True)
        for v in sorted_nodes:
            if v not in dominated and G.degree(v) > 0:
                dominating.append(v)
                dominated |= open_neighborhoods[v]
            if len(dominated) == n:
                break
        return max(len(dominating), 2)


def _independence_number_exact(G):
    """Nombre d'indépendance exact pour n <= 30, amélioré sinon."""
    n = G.number_of_nodes()
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n <= 30:
        return _max_independent_set_bb(G)
    else:
        return _independence_greedy_improved(G)


def _max_independent_set_bb(G):
    """Branch and bound pour le maximum independent set."""
    nodes = list(G.nodes())
    adj = {v: set(G.neighbors(v)) for v in nodes}
    # Sort by degree descending for better pruning
    nodes_sorted = sorted(nodes, key=lambda v: len(adj[v]), reverse=True)
    best = [0]

    def _branch(candidates, current_size):
        if current_size + len(candidates) <= best[0]:
            return
        if not candidates:
            best[0] = max(best[0], current_size)
            return

        # Pick vertex with max degree in subgraph (more pruning)
        v = max(candidates, key=lambda u: len(adj[u] & set(candidates)))
        rest = [u for u in candidates if u != v]

        # Branch 1: include v
        new_candidates = [u for u in rest if u not in adj[v]]
        _branch(new_candidates, current_size + 1)

        # Branch 2: exclude v
        _branch(rest, current_size)

    _branch(nodes_sorted, 0)
    return best[0]


def _independence_greedy_improved(G):
    """Multiple random greedy runs for independence number."""
    import random
    nodes = list(G.nodes())
    best_size = 0
    for _ in range(10):
        ind_set = set()
        excluded = set()
        order = list(nodes)
        random.shuffle(order)
        order.sort(key=lambda v: G.degree(v))
        for v in order:
            if v not in excluded:
                ind_set.add(v)
                excluded.add(v)
                excluded.update(G.neighbors(v))
        best_size = max(best_size, len(ind_set))
    return best_size


def _independent_domination_number_exact(G):
    """Nombre de domination indépendante exact pour petits graphes."""
    n = G.number_of_nodes()
    if n == 0:
        return 0
    if n == 1:
        return 1

    nodes = list(G.nodes())
    adj = {v: set(G.neighbors(v)) for v in nodes}
    closed = {v: adj[v] | {v} for v in nodes}

    if n <= 20:
        for k in range(1, n + 1):
            for subset in combinations(nodes, k):
                s = set(subset)
                # Check independent
                is_ind = True
                for v in s:
                    if adj[v] & s:
                        is_ind = False
                        break
                if not is_ind:
                    continue
                # Check dominating
                dominated = set()
                for v in s:
                    dominated |= closed[v]
                if len(dominated) == n:
                    return k
        return n
    else:
        # Greedy
        ind_dom = set()
        excluded = set()
        sorted_nodes = sorted(nodes, key=lambda v: G.degree(v), reverse=True)
        for v in sorted_nodes:
            if v not in excluded:
                ind_dom.add(v)
                excluded.add(v)
                excluded.update(adj[v])
        return len(ind_dom)


def _distance_matrix(G):
    """Matrice de distances."""
    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    D = np.zeros((n, n))
    sp = dict(nx.all_pairs_shortest_path_length(G))
    for u in nodes:
        for v in nodes:
            D[node_idx[u]][node_idx[v]] = sp[u].get(v, 0)
    return D


# ─────────────────────────────────────────────
#  VÉRIFICATION DE CLASSE
# ─────────────────────────────────────────────

def is_claw_free(G):
    """Vérifie qu'un graphe est sans griffe (pas de K_{1,3} induit)."""
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) < 3:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    a, b, c = neighbors[i], neighbors[j], neighbors[k]
                    if not G.has_edge(a, b) and not G.has_edge(b, c) and not G.has_edge(a, c):
                        return False
    return True


def is_tree(G):
    return nx.is_tree(G)


def satisfies_class(G, graph_classes):
    """Vérifie que G satisfait toutes les classes requises."""
    if G.number_of_nodes() < 2:
        return False
    if "connected" in graph_classes and not nx.is_connected(G):
        return False
    if "tree" in graph_classes and not nx.is_tree(G):
        return False
    if "bipartite" in graph_classes and not nx.is_bipartite(G):
        return False
    if "planar" in graph_classes and not nx.check_planarity(G)[0]:
        return False
    if "claw_free" in graph_classes and not is_claw_free(G):
        return False
    return True
