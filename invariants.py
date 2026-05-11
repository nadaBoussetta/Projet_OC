"""
invariants.py – Calcul des invariants avec calcul paresseux des invariants coûteux.
"""
import math
import networkx as nx
import numpy as np


def get_needed_invariants(conjecture) -> set:
    needed = {"n", "m", "minimum_degree", "maximum_degree", "average_degree", "density"}
    needed.add(conjecture.x_name)
    needed.add(conjecture.y_name)
    return needed


def compute_invariants(G: nx.Graph, needed: set = None) -> dict:
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
    _need = needed if needed is not None else set(_all_keys())

    if "diameter" in _need or "radius" in _need:
        try:
            if nx.is_connected(G):
                inv["diameter"] = nx.diameter(G)
                inv["radius"] = nx.radius(G)
            else:
                gcc = G.subgraph(max(nx.connected_components(G), key=len))
                inv["diameter"] = nx.diameter(gcc)
                inv["radius"] = nx.radius(gcc)
        except Exception:
            inv["diameter"] = 0
            inv["radius"] = 0
    else:
        inv["diameter"] = 0
        inv["radius"] = 0

    inv["triangle_number"] = sum(nx.triangles(G).values()) // 3 if "triangle_number" in _need else 0

    if "clique_number" in _need:
        try:
            cliques = list(nx.find_cliques(G))
            inv["clique_number"] = max(len(c) for c in cliques) if cliques else 1
        except Exception:
            inv["clique_number"] = 1
    else:
        inv["clique_number"] = 1

    inv["domination_number"] = _domination_number(G) if "domination_number" in _need else 0
    inv["total_domination_number"] = _total_domination_number(G) if "total_domination_number" in _need else 0

    if "independence_number" in _need or "vertex_cover_number" in _need:
        alpha = _independence_number(G)
        inv["independence_number"] = alpha
        inv["vertex_cover_number"] = n - alpha
    else:
        inv["independence_number"] = 0
        inv["vertex_cover_number"] = 0

    inv["independent_domination_number"] = _independent_domination_number(G) if "independent_domination_number" in _need else 0
    inv["matching_number"] = len(nx.max_weight_matching(G, maxcardinality=True)) if "matching_number" in _need else 0
    inv["first_zagreb_index"] = sum(d ** 2 for d in degrees) if "first_zagreb_index" in _need else 0
    inv["second_zagreb_index"] = sum(G.degree(u) * G.degree(v) for u, v in G.edges()) if "second_zagreb_index" in _need else 0

    if "randic_index" in _need:
        inv["randic_index"] = sum(1.0 / math.sqrt(G.degree(u) * G.degree(v))
            for u, v in G.edges() if G.degree(u) > 0 and G.degree(v) > 0)
    else:
        inv["randic_index"] = 0

    if "harmonic_index" in _need:
        inv["harmonic_index"] = sum(2.0 / (G.degree(u) + G.degree(v))
            for u, v in G.edges() if G.degree(u) + G.degree(v) > 0)
    else:
        inv["harmonic_index"] = 0

    if "proximity" in _need or "remoteness" in _need:
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
    else:
        inv["proximity"] = 0.0
        inv["remoteness"] = 0.0

    _eigen_needed = {"largest_eigenvalue", "second_smallest_laplace_eigenvalue", "largest_distance_eigenvalue"}
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
    return ["n","m","minimum_degree","maximum_degree","average_degree","density",
            "diameter","radius","triangle_number","clique_number","domination_number",
            "total_domination_number","independence_number","vertex_cover_number",
            "independent_domination_number","matching_number","first_zagreb_index",
            "second_zagreb_index","randic_index","harmonic_index","proximity",
            "remoteness","largest_eigenvalue","second_smallest_laplace_eigenvalue",
            "largest_distance_eigenvalue"]


def _domination_number(G):
    dominated = set()
    dominating = set()
    for v in sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True):
        if v not in dominated:
            dominating.add(v)
            dominated.add(v)
            dominated.update(G.neighbors(v))
    return len(dominating)


def _total_domination_number(G):
    if G.number_of_nodes() == 0:
        return 0
    dominated = set()
    dominating = set()
    for v in sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True):
        if v not in dominated and G.degree(v) > 0:
            dominating.add(v)
            dominated.update(G.neighbors(v))
        if len(dominated) == G.number_of_nodes():
            break
    return max(len(dominating), 2) if G.number_of_nodes() >= 2 else len(dominating)


def _independence_number(G):
    try:
        ind_set = set()
        excluded = set()
        for v in sorted(G.nodes(), key=lambda v: G.degree(v)):
            if v not in excluded:
                ind_set.add(v)
                excluded.update(G.neighbors(v))
        return len(ind_set)
    except Exception:
        return 0


def _independent_domination_number(G):
    dominated = set()
    ind_dom = set()
    excluded = set()
    for v in sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True):
        if v not in excluded:
            ind_dom.add(v)
            excluded.add(v)
            excluded.update(G.neighbors(v))
            dominated.add(v)
            dominated.update(G.neighbors(v))
    return len(ind_dom)


def _distance_matrix(G):
    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    D = np.zeros((n, n))
    sp = dict(nx.all_pairs_shortest_path_length(G))
    for u in nodes:
        for v in nodes:
            D[node_idx[u]][node_idx[v]] = sp[u].get(v, 0)
    return D


def is_claw_free(G):
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) < 3:
            continue
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    a, b, c = neighbors[i], neighbors[j], neighbors[k]
                    if not G.has_edge(a, b) and not G.has_edge(b, c) and not G.has_edge(a, c):
                        return False
    return True


def is_tree(G):
    return nx.is_tree(G)


def satisfies_class(G, graph_classes):
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
