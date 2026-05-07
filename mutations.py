"""
mutations.py – Générateurs de graphes initiaux et mutations locales.
"""
import random
import networkx as nx
from invariants import satisfies_class, is_claw_free


# ─────────────────────────────────────────────
#  GÉNÉRATEURS INITIAUX
# ─────────────────────────────────────────────

def generate_initial_graphs(conjecture, n_graphs=8):
    """Génère une population initiale variée selon la classe."""
    classes = conjecture.graph_classes
    graphs = []
    generators = _get_generators(classes)
    for gen in generators:
        for _ in range(max(1, n_graphs // len(generators))):
            G = gen()
            if G is not None and satisfies_class(G, classes):
                graphs.append(G)
    # Compléter si besoin
    while len(graphs) < n_graphs:
        G = random.choice(generators)()
        if G is not None and satisfies_class(G, classes):
            graphs.append(G)
    return graphs[:n_graphs]


def _get_generators(classes):
    gens = []
    if "tree" in classes:
        gens += [_random_tree, _path_graph, _star_graph, _caterpillar, _random_tree_large]
    elif "claw_free" in classes:
        gens += [_cycle_graph, _complete_graph, _line_graph, _claw_free_random, _dense_claw_free]
    else:
        gens += [
            _random_connected, _cycle_graph, _complete_graph,
            _grid_graph, _random_connected_medium, _path_graph,
            _wheel_graph, _petersen_like,
        ]
    return gens if gens else [_random_connected]


def _random_tree(n=None):
    n = n or random.randint(5, 20)
    return nx.random_labeled_tree(n)


def _random_tree_large():
    return _random_tree(random.randint(15, 35))


def _path_graph():
    n = random.randint(4, 25)
    return nx.path_graph(n)


def _star_graph():
    n = random.randint(3, 15)
    return nx.star_graph(n)


def _caterpillar():
    """Arbre caterpillar: chemin central avec feuilles."""
    spine = random.randint(3, 10)
    G = nx.path_graph(spine)
    node_id = spine
    for v in range(spine):
        leaves = random.randint(0, 3)
        for _ in range(leaves):
            G.add_edge(v, node_id)
            node_id += 1
    return G


def _cycle_graph():
    n = random.randint(4, 20)
    return nx.cycle_graph(n)


def _complete_graph():
    n = random.randint(3, 10)
    return nx.complete_graph(n)


def _random_connected(n=None, p=None):
    n = n or random.randint(6, 18)
    p = p or random.uniform(0.2, 0.6)
    while True:
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            return G


def _random_connected_medium():
    return _random_connected(random.randint(10, 25), random.uniform(0.15, 0.5))


def _grid_graph():
    r = random.randint(2, 5)
    c = random.randint(2, 5)
    G = nx.grid_2d_graph(r, c)
    return nx.convert_node_labels_to_integers(G)


def _wheel_graph():
    n = random.randint(4, 12)
    return nx.wheel_graph(n)


def _petersen_like():
    G = nx.petersen_graph()
    return G


def _line_graph():
    """Graphe line (toujours sans griffe)."""
    n = random.randint(4, 12)
    H = nx.erdos_renyi_graph(n, random.uniform(0.3, 0.7))
    if H.number_of_edges() == 0:
        H = nx.path_graph(n)
    L = nx.line_graph(H)
    if L.number_of_nodes() == 0:
        return _cycle_graph()
    return nx.convert_node_labels_to_integers(L)


def _claw_free_random():
    """Construit un graphe connexe sans griffe par construction."""
    n = random.randint(6, 18)
    G = nx.cycle_graph(n)
    # Ajouter des arêtes qui ne créent pas de griffe
    edges_to_try = [(u, v) for u in range(n) for v in range(u+2, n)
                    if not G.has_edge(u, v) and abs(u - v) <= n // 2]
    random.shuffle(edges_to_try)
    for u, v in edges_to_try[:n]:
        G.add_edge(u, v)
        if not is_claw_free(G):
            G.remove_edge(u, v)
    return G


def _dense_claw_free():
    """Graphe claw-free dense = complement d'un graphe triangle-free."""
    n = random.randint(8, 16)
    # Graphe de Turán T(n,2) ~ biparti complet = son complément est union de cliques
    half = n // 2
    G = nx.complete_bipartite_graph(half, n - half)
    comp = nx.complement(G)
    if not nx.is_connected(comp):
        # Connecter
        comps = list(nx.connected_components(comp))
        for i in range(len(comps) - 1):
            u = list(comps[i])[0]
            v = list(comps[i+1])[0]
            comp.add_edge(u, v)
    return comp


# ─────────────────────────────────────────────
#  MUTATIONS
# ─────────────────────────────────────────────

def mutate(G: nx.Graph, graph_classes: list) -> nx.Graph:
    """Applique une mutation aléatoire à G."""
    H = G.copy()
    mutation_fns = _get_mutations(graph_classes)
    fn = random.choice(mutation_fns)
    try:
        H = fn(H)
    except Exception:
        pass
    return H


def _get_mutations(classes):
    """Sélectionne les mutations adaptées à la classe."""
    base = [
        _add_edge, _remove_edge, _add_vertex_with_edges,
        _remove_leaf, _add_leaf, _subdivide_edge,
    ]
    if "tree" in classes:
        return [_add_leaf, _remove_leaf, _add_leaf, _add_path, _remove_leaf]
    elif "claw_free" in classes:
        return [_add_edge, _add_edge, _remove_edge,
                _add_vertex_clique_neighbor, _add_triangle, _remove_leaf]
    return base + [_add_clique, _add_path, _densify_local]


def _add_edge(G):
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return G
    for _ in range(20):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            return G
    return G


def _remove_edge(G):
    edges = list(G.edges())
    if not edges:
        return G
    u, v = random.choice(edges)
    G.remove_edge(u, v)
    return G


def _add_vertex_with_edges(G):
    new_node = max(G.nodes(), default=-1) + 1
    G.add_node(new_node)
    nodes = list(G.nodes())
    nodes.remove(new_node)
    if nodes:
        k = random.randint(1, min(3, len(nodes)))
        for v in random.sample(nodes, k):
            G.add_edge(new_node, v)
    return G


def _remove_leaf(G):
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if leaves:
        G.remove_node(random.choice(leaves))
    return G


def _add_leaf(G):
    if G.number_of_nodes() == 0:
        return G
    v = random.choice(list(G.nodes()))
    new_node = max(G.nodes()) + 1
    G.add_edge(v, new_node)
    return G


def _subdivide_edge(G):
    edges = list(G.edges())
    if not edges:
        return G
    u, v = random.choice(edges)
    new_node = max(G.nodes()) + 1
    G.remove_edge(u, v)
    G.add_edge(u, new_node)
    G.add_edge(new_node, v)
    return G


def _add_path(G):
    if G.number_of_nodes() == 0:
        return G
    start = random.choice(list(G.nodes()))
    length = random.randint(1, 4)
    cur = start
    base = max(G.nodes()) + 1
    for i in range(length):
        G.add_edge(cur, base + i)
        cur = base + i
    return G


def _add_clique(G):
    if G.number_of_nodes() == 0:
        return G
    k = random.randint(2, 4)
    base = max(G.nodes()) + 1
    new_nodes = list(range(base, base + k))
    for u in new_nodes:
        for v in new_nodes:
            if u < v:
                G.add_edge(u, v)
    # Connecter à G
    attach = random.choice(list(G.nodes()))
    G.add_edge(attach, new_nodes[0])
    return G


def _densify_local(G):
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return G
    v = random.choice(nodes)
    neighbors = list(G.neighbors(v))
    if len(neighbors) >= 2:
        for u in neighbors:
            for w in neighbors:
                if u < w and not G.has_edge(u, w) and random.random() < 0.5:
                    G.add_edge(u, w)
    return G


def _add_vertex_clique_neighbor(G):
    """Ajoute un sommet connecté à tous les membres d'une clique (garde sans-griffe)."""
    nodes = list(G.nodes())
    if not nodes:
        return G
    # Chercher une clique de taille 2 ou 3
    new_v = max(G.nodes()) + 1
    # Connecter à 2 nœuds adjacents (triangle avec eux)
    edges = list(G.edges())
    if edges:
        u, v = random.choice(edges)
        G.add_edge(new_v, u)
        G.add_edge(new_v, v)
    return G


def _add_triangle(G):
    edges = list(G.edges())
    if not edges:
        return G
    u, v = random.choice(edges)
    new_v = max(G.nodes()) + 1
    G.add_edge(new_v, u)
    G.add_edge(new_v, v)
    return G


# ─────────────────────────────────────────────
#  RÉPARATION
# ─────────────────────────────────────────────

def repair(G: nx.Graph, graph_classes: list) -> nx.Graph:
    """Répare G pour qu'il satisfasse les contraintes de classe."""
    # Connexité
    if "connected" in graph_classes or "tree" in graph_classes:
        G = _repair_connectivity(G)

    # Arbre: supprimer les cycles
    if "tree" in graph_classes:
        G = _repair_tree(G)

    # Sans griffe: supprimer les griffes
    if "claw_free" in graph_classes:
        G = _repair_claw_free(G)

    # Biparti
    if "bipartite" in graph_classes:
        G = _repair_bipartite(G)

    return G


def _repair_connectivity(G):
    """Reconnecte les composantes connexes."""
    if G.number_of_nodes() < 2:
        return G
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return G
    for i in range(len(comps) - 1):
        u = random.choice(list(comps[i]))
        v = random.choice(list(comps[i + 1]))
        G.add_edge(u, v)
    return G


def _repair_tree(G):
    """Transforme G en arbre spanning."""
    if G.number_of_nodes() < 2:
        return G
    if not nx.is_connected(G):
        G = _repair_connectivity(G)
    T = nx.minimum_spanning_tree(G)
    return T


def _repair_claw_free(G):
    """Supprime les griffes induites en ajoutant des arêtes manquantes."""
    max_iter = 50
    for _ in range(max_iter):
        found_claw = False
        for v in list(G.nodes()):
            neighbors = list(G.neighbors(v))
            if len(neighbors) < 3:
                continue
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        a, b, c = neighbors[i], neighbors[j], neighbors[k]
                        missing = []
                        if not G.has_edge(a, b):
                            missing.append((a, b))
                        if not G.has_edge(b, c):
                            missing.append((b, c))
                        if not G.has_edge(a, c):
                            missing.append((a, c))
                        if len(missing) == 3:  # griffe trouvée
                            # Ajouter une arête aléatoire pour casser la griffe
                            G.add_edge(*random.choice(missing))
                            found_claw = True
                            break
                    if found_claw:
                        break
                if found_claw:
                    break
            if found_claw:
                break
        if not found_claw:
            break
    return G


def _repair_bipartite(G):
    """Retire les arêtes qui violent la bipartition."""
    if nx.is_bipartite(G):
        return G
    # Trouver une 2-coloration par BFS, supprimer les arêtes conflictuelles
    color = {}
    for start in G.nodes():
        if start not in color:
            queue = [start]
            color[start] = 0
            while queue:
                u = queue.pop(0)
                for v in G.neighbors(u):
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        G.remove_edge(u, v)
                        break
    return G
