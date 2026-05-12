"""
mutations.py – Générateurs de graphes et mutations optimisés par classe.
Stratégies agressives et diversifiées pour maximiser les chances de contre-exemples.
"""
import random
import networkx as nx
from invariants import satisfies_class, is_claw_free


# ─────────────────────────────────────────────
#  GÉNÉRATEURS INITIAUX
# ─────────────────────────────────────────────

def generate_initial_graphs(conjecture, n_graphs=12):
    """Génère une population initiale variée et ciblée selon la classe et les invariants."""
    classes = conjecture.graph_classes
    graphs = []
    generators = _get_generators(classes, conjecture)

    # Générer avec variété de tailles
    sizes = [5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 30, 35]
    for gen in generators:
        for sz in random.sample(sizes, min(3, len(sizes))):
            try:
                G = gen(sz)
                if G is not None and G.number_of_nodes() >= 2 and satisfies_class(G, classes):
                    graphs.append(G)
            except Exception:
                pass
        if len(graphs) >= n_graphs * 2:
            break

    # Compléter si besoin
    attempts = 0
    while len(graphs) < n_graphs and attempts < 100:
        attempts += 1
        gen = random.choice(generators)
        sz = random.choice(sizes)
        try:
            G = gen(sz)
            if G is not None and G.number_of_nodes() >= 2 and satisfies_class(G, classes):
                graphs.append(G)
        except Exception:
            pass

    return graphs[:n_graphs] if graphs else [nx.path_graph(6)]


def _get_generators(classes, conjecture=None):
    """Sélectionne les générateurs adaptés à la classe et aux invariants."""
    if "tree" in classes:
        return [
            _random_tree, _path_graph, _star_graph, _caterpillar,
            _double_star, _spider, _broom, _lobster, _balanced_tree
        ]
    elif "claw_free" in classes:
        return [
            _line_graph_random, _cycle_graph, _complete_graph_small,
            _power_of_cycle, _complement_triangle_free, _claw_free_dense,
            _line_graph_of_complete, _circulant_claw_free, _inflated_cycle
        ]
    else:
        # Connected graphs - very diverse
        return [
            _random_connected, _cycle_graph, _complete_graph_small,
            _grid_graph, _path_graph, _wheel_graph, _petersen_graph,
            _random_regular, _barabasi_albert, _complete_bipartite,
            _kneser_like, _friendship_graph, _book_graph
        ]


# ═══════════════════════════════════════════════
#  GÉNÉRATEURS POUR ARBRES
# ═══════════════════════════════════════════════

def _random_tree(n=None):
    n = n or random.randint(8, 25)
    return nx.random_labeled_tree(n)

def _path_graph(n=None):
    n = n or random.randint(5, 30)
    return nx.path_graph(n)

def _star_graph(n=None):
    n = n or random.randint(4, 20)
    return nx.star_graph(n - 1)

def _caterpillar(n=None):
    """Arbre caterpillar: chemin central + feuilles."""
    spine = random.randint(3, min(n or 15, 15))
    G = nx.path_graph(spine)
    node_id = spine
    target = (n or 15)
    for v in range(spine):
        leaves = random.randint(0, min(4, target - G.number_of_nodes()))
        for _ in range(leaves):
            G.add_edge(v, node_id)
            node_id += 1
            if G.number_of_nodes() >= target:
                return G
    return G

def _double_star(n=None):
    """Double star: deux sommets centraux avec feuilles."""
    n = n or random.randint(6, 20)
    k1 = random.randint(2, n - 3)
    k2 = n - k1 - 2
    G = nx.Graph()
    G.add_edge(0, 1)
    node_id = 2
    for _ in range(max(1, k1)):
        G.add_edge(0, node_id)
        node_id += 1
    for _ in range(max(1, k2)):
        G.add_edge(1, node_id)
        node_id += 1
    return G

def _spider(n=None):
    """Spider graph: centre + k chemins."""
    n = n or random.randint(7, 20)
    k = random.randint(3, min(6, n - 1))
    G = nx.Graph()
    center = 0
    node_id = 1
    legs_left = n - 1
    for i in range(k):
        leg_len = max(1, legs_left // (k - i))
        legs_left -= leg_len
        prev = center
        for _ in range(leg_len):
            G.add_edge(prev, node_id)
            prev = node_id
            node_id += 1
    return G

def _broom(n=None):
    """Broom: path + star at one end."""
    n = n or random.randint(6, 20)
    path_len = random.randint(2, n - 3)
    G = nx.path_graph(path_len)
    node_id = path_len
    leaves = n - path_len
    for _ in range(max(1, leaves)):
        G.add_edge(path_len - 1, node_id)
        node_id += 1
    return G

def _lobster(n=None):
    """Lobster: caterpillar avec feuilles de feuilles."""
    n = n or random.randint(10, 25)
    spine = random.randint(3, n // 3)
    G = nx.path_graph(spine)
    node_id = spine
    while G.number_of_nodes() < n:
        v = random.randint(0, G.number_of_nodes() - 1)
        if G.degree(v) <= 3:
            G.add_edge(v, node_id)
            node_id += 1
    return G

def _balanced_tree(n=None):
    """Arbre équilibré."""
    r = random.choice([2, 3])
    h = random.randint(2, 4)
    G = nx.balanced_tree(r, h)
    return G


# ═══════════════════════════════════════════════
#  GÉNÉRATEURS POUR GRAPHES CLAW-FREE
# ═══════════════════════════════════════════════

def _line_graph_random(n=None):
    """Line graph d'un graphe aléatoire (toujours claw-free)."""
    n = n or random.randint(6, 15)
    m_target = random.randint(n, min(n * 2, n * (n - 1) // 2))
    H = nx.gnm_random_graph(n, m_target)
    if H.number_of_edges() == 0:
        H = nx.path_graph(n)
    L = nx.line_graph(H)
    if L.number_of_nodes() < 3:
        return nx.cycle_graph(max(n, 5))
    L = nx.convert_node_labels_to_integers(L)
    if not nx.is_connected(L):
        L = L.subgraph(max(nx.connected_components(L), key=len)).copy()
        L = nx.convert_node_labels_to_integers(L)
    return L

def _line_graph_of_complete(n=None):
    """Line graph of K_n (very dense, claw-free)."""
    k = random.randint(4, min(8, n or 8))
    L = nx.line_graph(nx.complete_graph(k))
    return nx.convert_node_labels_to_integers(L)

def _power_of_cycle(n=None):
    """Puissance d'un cycle (claw-free)."""
    n = n or random.randint(6, 20)
    k = random.randint(2, max(2, n // 4))
    G = nx.cycle_graph(n)
    # Add edges for power
    nodes = list(range(n))
    for i in range(n):
        for j in range(2, k + 1):
            G.add_edge(i, (i + j) % n)
    return G

def _complement_triangle_free(n=None):
    """Complément d'un graphe triangle-free (toujours claw-free)."""
    n = n or random.randint(8, 16)
    # Triangle-free graph: biparti
    half = n // 2
    H = nx.complete_bipartite_graph(half, n - half)
    # Remove some edges to make it sparser
    edges = list(H.edges())
    remove_count = random.randint(0, len(edges) // 3)
    for e in random.sample(edges, min(remove_count, len(edges))):
        H.remove_edge(*e)
    comp = nx.complement(H)
    if not nx.is_connected(comp):
        comps = list(nx.connected_components(comp))
        for i in range(len(comps) - 1):
            u = list(comps[i])[0]
            v = list(comps[i + 1])[0]
            comp.add_edge(u, v)
    return comp

def _claw_free_dense(n=None):
    """Dense claw-free graph via iterated construction."""
    n = n or random.randint(8, 16)
    # Start from complete graph and keep it claw-free
    G = nx.complete_graph(min(n, 5))
    while G.number_of_nodes() < n:
        new_v = G.number_of_nodes()
        # Connect to a clique in G
        nodes = list(G.nodes())
        v = random.choice(nodes)
        nbrs = list(G.neighbors(v))
        connect_to = [v] + random.sample(nbrs, min(random.randint(1, 3), len(nbrs)))
        G.add_node(new_v)
        for u in connect_to:
            G.add_edge(new_v, u)
        if not is_claw_free(G):
            # Fix by connecting more
            for u in G.neighbors(new_v):
                for w in G.neighbors(new_v):
                    if u != w and not G.has_edge(u, w):
                        G.add_edge(u, w)
                        if is_claw_free(G):
                            break
    return G

def _circulant_claw_free(n=None):
    """Circulant graph with enough connections to be claw-free."""
    n = n or random.randint(8, 20)
    # C_n with connections to distance 1,2,...,k
    k = max(2, n // 4)
    G = nx.circulant_graph(n, list(range(1, k + 1)))
    return G

def _inflated_cycle(n=None):
    """Inflated cycle: replace each vertex by a clique."""
    cycle_len = random.randint(3, min(7, (n or 12) // 2))
    clique_size = random.randint(2, 4)
    G = nx.Graph()
    node_id = 0
    cliques = []
    for _ in range(cycle_len):
        clique = list(range(node_id, node_id + clique_size))
        for u in clique:
            for v in clique:
                if u < v:
                    G.add_edge(u, v)
        cliques.append(clique)
        node_id += clique_size
    # Connect consecutive cliques
    for i in range(cycle_len):
        j = (i + 1) % cycle_len
        # Connect all of clique i to all of clique j (or subset)
        for u in cliques[i]:
            for v in cliques[j]:
                G.add_edge(u, v)
    return G


# ═══════════════════════════════════════════════
#  GÉNÉRATEURS POUR GRAPHES CONNEXES
# ═══════════════════════════════════════════════

def _random_connected(n=None):
    n = n or random.randint(8, 20)
    p = random.uniform(0.15, 0.6)
    for _ in range(50):
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            return G
    return nx.path_graph(n)

def _cycle_graph(n=None):
    n = n or random.randint(5, 25)
    return nx.cycle_graph(n)

def _complete_graph_small(n=None):
    n = min(n or random.randint(4, 10), 12)
    return nx.complete_graph(n)

def _grid_graph(n=None):
    side = random.randint(2, max(2, int(math.sqrt(n or 16))))
    import math
    G = nx.grid_2d_graph(side, random.randint(2, side + 2))
    return nx.convert_node_labels_to_integers(G)

def _wheel_graph(n=None):
    n = n or random.randint(5, 15)
    return nx.wheel_graph(n)

def _petersen_graph(n=None):
    return nx.petersen_graph()

def _random_regular(n=None):
    n = n or random.randint(6, 20)
    d = random.randint(2, min(5, n - 1))
    if (n * d) % 2 != 0:
        n += 1
    try:
        return nx.random_regular_graph(d, n)
    except Exception:
        return nx.cycle_graph(n)

def _barabasi_albert(n=None):
    n = n or random.randint(8, 25)
    m_ba = random.randint(1, 4)
    return nx.barabasi_albert_graph(n, m_ba)

def _complete_bipartite(n=None):
    n = n or random.randint(6, 16)
    a = random.randint(2, n - 2)
    b = n - a
    return nx.complete_bipartite_graph(a, b)

def _kneser_like(n=None):
    """Graphe de Kneser-like ou Petersen."""
    try:
        return nx.petersen_graph()
    except Exception:
        return nx.cycle_graph(10)

def _friendship_graph(n=None):
    """Friendship graph: n triangles sharing a vertex."""
    k = random.randint(2, min(6, (n or 10) // 2))
    G = nx.Graph()
    center = 0
    node_id = 1
    for _ in range(k):
        G.add_edge(center, node_id)
        G.add_edge(center, node_id + 1)
        G.add_edge(node_id, node_id + 1)
        node_id += 2
    return G

def _book_graph(n=None):
    """Book graph: n triangles sharing an edge."""
    k = random.randint(2, min(8, n or 8))
    G = nx.Graph()
    G.add_edge(0, 1)
    for i in range(k):
        v = i + 2
        G.add_edge(0, v)
        G.add_edge(1, v)
    return G


# ═══════════════════════════════════════════════
#  MUTATIONS
# ═══════════════════════════════════════════════

def mutate(G: nx.Graph, graph_classes: list) -> nx.Graph:
    """Applique une ou plusieurs mutations aléatoires."""
    H = G.copy()
    mutation_fns = _get_mutations(graph_classes)
    # Appliquer 1-3 mutations
    n_muts = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    for _ in range(n_muts):
        fn = random.choice(mutation_fns)
        try:
            H = fn(H)
        except Exception:
            pass
    return H


def _get_mutations(classes):
    """Sélectionne les mutations adaptées à la classe."""
    if "tree" in classes:
        return [
            _add_leaf, _add_leaf, _remove_leaf, _add_path_tree,
            _move_leaf, _subdivide_edge, _contract_leaf_path,
            _add_pendant_star, _swap_subtree
        ]
    elif "claw_free" in classes:
        return [
            _add_edge, _add_edge, _add_triangle,
            _add_vertex_to_clique, _densify_neighborhood,
            _remove_edge_safe, _vertex_duplication,
            _add_twin, _complete_neighborhood
        ]
    else:
        return [
            _add_edge, _remove_edge, _add_vertex_with_edges,
            _remove_leaf, _add_leaf, _subdivide_edge,
            _add_clique_attached, _add_path_attached, _densify_local,
            _add_pendant_star, _swap_edges, _contract_edge,
            _vertex_duplication, _grow_random
        ]


# ─── Tree mutations ───────────────────────────

def _add_leaf(G):
    if G.number_of_nodes() == 0:
        return G
    v = random.choice(list(G.nodes()))
    new_node = max(G.nodes()) + 1
    G.add_edge(v, new_node)
    return G

def _remove_leaf(G):
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if leaves and G.number_of_nodes() > 3:
        G.remove_node(random.choice(leaves))
    return G

def _add_path_tree(G):
    """Add a path as a pendant."""
    if G.number_of_nodes() == 0:
        return G
    start = random.choice(list(G.nodes()))
    length = random.randint(1, 4)
    base = max(G.nodes()) + 1
    prev = start
    for i in range(length):
        G.add_edge(prev, base + i)
        prev = base + i
    return G

def _move_leaf(G):
    """Move a leaf from one vertex to another."""
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if not leaves or G.number_of_nodes() < 4:
        return G
    leaf = random.choice(leaves)
    parent = list(G.neighbors(leaf))[0]
    other_nodes = [v for v in G.nodes() if v != leaf and v != parent]
    if other_nodes:
        new_parent = random.choice(other_nodes)
        G.remove_edge(leaf, parent)
        G.add_edge(leaf, new_parent)
    return G

def _contract_leaf_path(G):
    """Remove a degree-2 vertex (contract path)."""
    deg2 = [v for v in G.nodes() if G.degree(v) == 2]
    if not deg2 or G.number_of_nodes() < 4:
        return G
    v = random.choice(deg2)
    nbrs = list(G.neighbors(v))
    G.remove_node(v)
    if not G.has_edge(nbrs[0], nbrs[1]):
        G.add_edge(nbrs[0], nbrs[1])
    return G

def _add_pendant_star(G):
    """Add a small star as pendant."""
    if G.number_of_nodes() == 0:
        return G
    attach = random.choice(list(G.nodes()))
    base = max(G.nodes()) + 1
    center = base
    G.add_edge(attach, center)
    leaves = random.randint(1, 3)
    for i in range(leaves):
        G.add_edge(center, base + 1 + i)
    return G

def _swap_subtree(G):
    """Swap two leaves between different parents."""
    leaves = [v for v in G.nodes() if G.degree(v) == 1]
    if len(leaves) < 2:
        return G
    l1, l2 = random.sample(leaves, 2)
    p1 = list(G.neighbors(l1))[0]
    p2 = list(G.neighbors(l2))[0]
    if p1 != p2:
        G.remove_edge(l1, p1)
        G.remove_edge(l2, p2)
        G.add_edge(l1, p2)
        G.add_edge(l2, p1)
    return G


# ─── Claw-free mutations ─────────────────────

def _add_edge(G):
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return G
    for _ in range(30):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            return G
    return G

def _remove_edge_safe(G):
    """Remove an edge, keeping connectivity."""
    edges = list(G.edges())
    if not edges:
        return G
    random.shuffle(edges)
    for u, v in edges[:10]:
        G.remove_edge(u, v)
        if nx.is_connected(G):
            return G
        G.add_edge(u, v)
    return G

def _add_triangle(G):
    """Add a triangle connected to existing edge."""
    edges = list(G.edges())
    if not edges:
        return G
    u, v = random.choice(edges)
    new_v = max(G.nodes()) + 1
    G.add_edge(new_v, u)
    G.add_edge(new_v, v)
    return G

def _add_vertex_to_clique(G):
    """Add vertex connected to an existing clique (safe for claw-free)."""
    nodes = list(G.nodes())
    if not nodes:
        return G
    new_v = max(G.nodes()) + 1
    # Find a clique (edge or triangle)
    edges = list(G.edges())
    if edges:
        u, v = random.choice(edges)
        G.add_node(new_v)
        G.add_edge(new_v, u)
        G.add_edge(new_v, v)
        # Also connect to common neighbors
        common = set(G.neighbors(u)) & set(G.neighbors(v))
        for w in common:
            if random.random() < 0.5:
                G.add_edge(new_v, w)
    return G

def _densify_neighborhood(G):
    """Make the neighborhood of a vertex more complete."""
    nodes = list(G.nodes())
    if len(nodes) < 3:
        return G
    v = random.choice(nodes)
    nbrs = list(G.neighbors(v))
    if len(nbrs) >= 2:
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if not G.has_edge(nbrs[i], nbrs[j]) and random.random() < 0.4:
                    G.add_edge(nbrs[i], nbrs[j])
    return G

def _vertex_duplication(G):
    """Duplicate a vertex (add twin with same neighbors)."""
    nodes = list(G.nodes())
    if not nodes:
        return G
    v = random.choice(nodes)
    new_v = max(G.nodes()) + 1
    G.add_node(new_v)
    for u in G.neighbors(v):
        G.add_edge(new_v, u)
    # Also connect to v (false twin -> true twin)
    if random.random() < 0.5:
        G.add_edge(new_v, v)
    return G

def _add_twin(G):
    """Add a true twin of a random vertex."""
    nodes = list(G.nodes())
    if not nodes:
        return G
    v = random.choice(nodes)
    new_v = max(G.nodes()) + 1
    G.add_node(new_v)
    G.add_edge(new_v, v)
    for u in G.neighbors(v):
        G.add_edge(new_v, u)
    return G

def _complete_neighborhood(G):
    """Make N(v) a clique for some v."""
    nodes = list(G.nodes())
    if not nodes:
        return G
    v = random.choice(nodes)
    nbrs = list(G.neighbors(v))
    for i in range(len(nbrs)):
        for j in range(i + 1, len(nbrs)):
            if not G.has_edge(nbrs[i], nbrs[j]):
                G.add_edge(nbrs[i], nbrs[j])
    return G


# ─── General connected mutations ─────────────

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
        k = random.randint(1, min(4, len(nodes)))
        for v in random.sample(nodes, k):
            G.add_edge(new_node, v)
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

def _add_clique_attached(G):
    if G.number_of_nodes() == 0:
        return G
    k = random.randint(2, 4)
    base = max(G.nodes()) + 1
    new_nodes = list(range(base, base + k))
    for u in new_nodes:
        for v in new_nodes:
            if u < v:
                G.add_edge(u, v)
    attach = random.choice(list(G.nodes()))
    G.add_edge(attach, new_nodes[0])
    return G

def _add_path_attached(G):
    if G.number_of_nodes() == 0:
        return G
    start = random.choice(list(G.nodes()))
    length = random.randint(2, 5)
    base = max(G.nodes()) + 1
    prev = start
    for i in range(length):
        G.add_edge(prev, base + i)
        prev = base + i
    return G

def _densify_local(G):
    nodes = list(G.nodes())
    if len(nodes) < 3:
        return G
    v = random.choice(nodes)
    neighbors = list(G.neighbors(v))
    if len(neighbors) >= 2:
        for u in neighbors:
            for w in neighbors:
                if u < w and not G.has_edge(u, w) and random.random() < 0.4:
                    G.add_edge(u, w)
    return G

def _swap_edges(G):
    """Swap two edges (2-opt style)."""
    edges = list(G.edges())
    if len(edges) < 2:
        return G
    for _ in range(10):
        e1, e2 = random.sample(edges, 2)
        a, b = e1
        c, d = e2
        if len({a, b, c, d}) == 4:
            G.remove_edge(a, b)
            G.remove_edge(c, d)
            if random.random() < 0.5:
                G.add_edge(a, c)
                G.add_edge(b, d)
            else:
                G.add_edge(a, d)
                G.add_edge(b, c)
            return G
    return G

def _contract_edge(G):
    """Contract a random edge."""
    edges = list(G.edges())
    if not edges or G.number_of_nodes() < 4:
        return G
    u, v = random.choice(edges)
    # Merge v into u
    for w in list(G.neighbors(v)):
        if w != u:
            G.add_edge(u, w)
    G.remove_node(v)
    return G

def _grow_random(G):
    """Add multiple vertices at once."""
    n_add = random.randint(2, 5)
    base = max(G.nodes()) + 1
    nodes = list(G.nodes())
    for i in range(n_add):
        new_v = base + i
        G.add_node(new_v)
        # Connect to 1-3 existing nodes
        targets = random.sample(nodes + list(range(base, base + i)),
                                min(random.randint(1, 3), len(nodes)))
        for t in targets:
            G.add_edge(new_v, t)
    return G


# ═══════════════════════════════════════════════
#  RÉPARATION
# ═══════════════════════════════════════════════

def repair(G: nx.Graph, graph_classes: list) -> nx.Graph:
    """Répare G pour qu'il satisfasse les contraintes de classe."""
    if G.number_of_nodes() < 2:
        return nx.path_graph(4)

    # Remove self-loops first
    G.remove_edges_from(nx.selfloop_edges(G))

    # Connexité d'abord
    if "connected" in graph_classes or "tree" in graph_classes:
        G = _repair_connectivity(G)

    # Arbre
    if "tree" in graph_classes:
        G = _repair_tree(G)

    # Sans griffe
    if "claw_free" in graph_classes:
        G = _repair_claw_free(G)

    # Biparti
    if "bipartite" in graph_classes:
        G = _repair_bipartite(G)

    # Planar
    if "planar" in graph_classes:
        G = _repair_planar(G)

    return G


def _repair_connectivity(G):
    """Reconnecte les composantes connexes."""
    if G.number_of_nodes() < 2:
        return G
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return G
    # Sort by size descending, connect smaller to larger
    comps.sort(key=len, reverse=True)
    for i in range(1, len(comps)):
        u = random.choice(list(comps[0]))
        v = random.choice(list(comps[i]))
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
    """Supprime les griffes en ajoutant des arêtes entre voisins indépendants."""
    max_iter = 100
    for _ in range(max_iter):
        claw = _find_claw(G)
        if claw is None:
            break
        v, a, b, c = claw
        # Add edge between two of the independent triple
        pairs = [(a, b), (b, c), (a, c)]
        random.shuffle(pairs)
        G.add_edge(*pairs[0])
    return G


def _find_claw(G):
    """Find a claw (K_{1,3}) in G. Returns (center, a, b, c) or None."""
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        if len(neighbors) < 3:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    a, b, c = neighbors[i], neighbors[j], neighbors[k]
                    if not G.has_edge(a, b) and not G.has_edge(b, c) and not G.has_edge(a, c):
                        return (v, a, b, c)
    return None


def _repair_bipartite(G):
    """Retire les arêtes qui violent la bipartition."""
    if nx.is_bipartite(G):
        return G
    color = {}
    edges_to_remove = []
    for start in G.nodes():
        if start in color:
            continue
        queue = [start]
        color[start] = 0
        while queue:
            u = queue.pop(0)
            for v in G.neighbors(u):
                if v not in color:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    edges_to_remove.append((u, v))
    for e in edges_to_remove:
        if G.has_edge(*e):
            G.remove_edge(*e)
    return G


def _repair_planar(G):
    """Remove edges to make G planar."""
    while not nx.check_planarity(G)[0]:
        edges = list(G.edges())
        if not edges:
            break
        G.remove_edge(*random.choice(edges))
    return G


# ─── Import math for grid_graph ───
import math
