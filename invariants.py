"""
invariants.py – Calcul de tous les invariants de graphes nécessaires au benchmark.
"""
import math
import networkx as nx
import numpy as np


def compute_invariants(G: nx.Graph) -> dict:
    """Calcule tous les invariants nécessaires pour un graphe G."""
    inv = {}
    n = G.number_of_nodes()
    m = G.number_of_edges()
    inv["n"] = n
    inv["m"] = m

    if n == 0:
        return {k: 0 for k in _all_keys()}

    degrees = [d for _, d in G.degree()]
    inv["minimum_degree"] = min(degrees) if degrees else 0
    inv["maximum_degree"] = max(degrees) if degrees else 0
    inv["average_degree"] = (2 * m / n) if n > 0 else 0
    inv["density"] = (2 * m / (n * (n - 1))) if n > 1 else 0

    # Diamètre et rayon (graphe connexe supposé)
    try:
        if nx.is_connected(G):
            inv["diameter"] = nx.diameter(G)
            inv["radius"] = nx.radius(G)
        else:
            # Composante géante
            gcc = G.subgraph(max(nx.connected_components(G), key=len))
            inv["diameter"] = nx.diameter(gcc)
            inv["radius"] = nx.radius(gcc)
    except Exception:
        inv["diameter"] = 0
        inv["radius"] = 0

    # Triangles
    tri = sum(nx.triangles(G).values()) // 3
    inv["triangle_number"] = tri

    # Clique maximum (approximation pour grands graphes)
    try:
        cliques = list(nx.find_cliques(G))
        inv["clique_number"] = max(len(c) for c in cliques) if cliques else 1
    except Exception:
        inv["clique_number"] = 1

    # Domination (gamma) – NP-hard, approx gloutonne
    inv["domination_number"] = _domination_number(G)

    # Domination totale
    inv["total_domination_number"] = _total_domination_number(G)

    # Ensemble indépendant maximum (alpha)
    inv["independence_number"] = _independence_number(G)

    # Couverture par sommets (tau = n - alpha par König / loi générale)
    inv["vertex_cover_number"] = n - inv["independence_number"]

    # Domination indépendante
    inv["independent_domination_number"] = _independent_domination_number(G)

    # Couplage maximum (mu)
    inv["matching_number"] = len(nx.max_weight_matching(G, maxcardinality=True))

    # Indices de Zagreb
    inv["first_zagreb_index"] = sum(d ** 2 for d in degrees)
    inv["second_zagreb_index"] = sum(G.degree(u) * G.degree(v) for u, v in G.edges())

    # Indice de Randic
    inv["randic_index"] = sum(
        1.0 / math.sqrt(G.degree(u) * G.degree(v))
        for u, v in G.edges()
        if G.degree(u) > 0 and G.degree(v) > 0
    )

    # Indice harmonique
    inv["harmonic_index"] = sum(
        2.0 / (G.degree(u) + G.degree(v))
        for u, v in G.edges()
        if G.degree(u) + G.degree(v) > 0
    )

    # Proximité et éloignement (proximity, remoteness)
    try:
        if nx.is_connected(G) and n > 1:
            dist = dict(nx.all_pairs_shortest_path_length(G))
            avg_dist = {v: sum(dist[v].values()) / (n - 1) for v in G.nodes()}
            inv["proximity"] = min(avg_dist.values())
            inv["remoteness"] = max(avg_dist.values())
        else:
            inv["proximity"] = 0.0
            inv["remoteness"] = 0.0
    except Exception:
        inv["proximity"] = 0.0
        inv["remoteness"] = 0.0

    # Valeurs propres spectrales
    try:
        if n > 1:
            A = nx.to_numpy_array(G)
            eigenvalues = np.linalg.eigvalsh(A)
            eigenvalues_sorted = np.sort(eigenvalues)[::-1]
            inv["largest_eigenvalue"] = float(eigenvalues_sorted[0])

            # Laplacien
            L = nx.laplacian_matrix(G).toarray().astype(float)
            lap_eigs = np.sort(np.linalg.eigvalsh(L))
            inv["second_smallest_laplace_eigenvalue"] = float(lap_eigs[1]) if len(lap_eigs) > 1 else 0.0

            # Distance matrix eigenvalue
            if nx.is_connected(G):
                D = _distance_matrix(G)
                dist_eigs = np.sort(np.linalg.eigvalsh(D))[::-1]
                inv["largest_distance_eigenvalue"] = float(dist_eigs[0])
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

    return inv


def _all_keys():
    return [
        "n", "m", "minimum_degree", "maximum_degree", "average_degree",
        "density", "diameter", "radius", "triangle_number", "clique_number",
        "domination_number", "total_domination_number", "independence_number",
        "vertex_cover_number", "independent_domination_number", "matching_number",
        "first_zagreb_index", "second_zagreb_index", "randic_index",
        "harmonic_index", "proximity", "remoteness", "largest_eigenvalue",
        "second_smallest_laplace_eigenvalue", "largest_distance_eigenvalue",
    ]


def _domination_number(G):
    """Approximation gloutonne du nombre de domination."""
    dominated = set()
    dominating = set()
    nodes = list(G.nodes())
    # Trier par degré décroissant
    nodes.sort(key=lambda v: G.degree(v), reverse=True)
    for v in nodes:
        if v not in dominated:
            dominating.add(v)
            dominated.add(v)
            dominated.update(G.neighbors(v))
    return len(dominating)


def _total_domination_number(G):
    """Approximation du nombre de domination totale."""
    if G.number_of_nodes() == 0:
        return 0
    dominated = set()
    dominating = set()
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    for v in nodes:
        if v not in dominated and G.degree(v) > 0:
            dominating.add(v)
            dominated.update(G.neighbors(v))
        if len(dominated) == G.number_of_nodes():
            break
    # Si des sommets isolés existent, total dom peut ne pas exister
    return max(len(dominating), 2) if G.number_of_nodes() >= 2 else len(dominating)


def _independence_number(G):
    """Taille d'un ensemble indépendant maximum (approx)."""
    try:
        # Utilise la complémentarité avec la couverture (König pour bipartis)
        # Sinon: algo glouton
        ind_set = set()
        excluded = set()
        nodes = sorted(G.nodes(), key=lambda v: G.degree(v))
        for v in nodes:
            if v not in excluded:
                ind_set.add(v)
                excluded.update(G.neighbors(v))
        return len(ind_set)
    except Exception:
        return 0


def _independent_domination_number(G):
    """Nombre de domination indépendante (= plus petit ensemble indépendant dominant)."""
    # Un ensemble indépendant dominant = indépendant + dominant
    # Approx: on cherche un ensemble indépendant qui domine tout
    dominated = set()
    ind_dom = set()
    excluded = set()
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)
    for v in nodes:
        if v not in excluded:
            ind_dom.add(v)
            excluded.add(v)
            excluded.update(G.neighbors(v))
            dominated.add(v)
            dominated.update(G.neighbors(v))
    return len(ind_dom)


def _distance_matrix(G):
    """Matrice des distances pour un graphe connexe."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    D = np.zeros((n, n))
    sp = dict(nx.all_pairs_shortest_path_length(G))
    for u in nodes:
        for v in nodes:
            D[node_idx[u]][node_idx[v]] = sp[u].get(v, 0)
    return D


def is_claw_free(G):
    """Vérifie si le graphe est sans griffe (K_{1,3} induit absent)."""
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) < 3:
            continue
        # Chercher 3 voisins 2 à 2 non adjacents
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    a, b, c = neighbors[i], neighbors[j], neighbors[k]
                    if (not G.has_edge(a, b) and
                            not G.has_edge(b, c) and
                            not G.has_edge(a, c)):
                        return False
    return True


def is_tree(G):
    return nx.is_tree(G)


def satisfies_class(G, graph_classes):
    """Vérifie si G appartient aux classes demandées."""
    if "connected" in graph_classes:
        if not nx.is_connected(G):
            return False
    if "tree" in graph_classes:
        if not nx.is_tree(G):
            return False
    if "bipartite" in graph_classes:
        if not nx.is_bipartite(G):
            return False
    if "planar" in graph_classes:
        if not nx.check_planarity(G)[0]:
            return False
    if "claw_free" in graph_classes:
        if not is_claw_free(G):
            return False
    return True
