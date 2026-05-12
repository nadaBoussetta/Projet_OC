"""
heuristic.py – Multi-strategy counter-example search.

Combines targeted Phase 0 (500+ structured graphs), exhaustive Phase 1,
and hill-climbing + SA Phase 2.  No external API calls – entirely local.
"""
import time
import random
import math
import networkx as nx

from invariants import compute_invariants, satisfies_class, get_needed_invariants, is_claw_free
from mutations import generate_initial_graphs, mutate, repair


# ──────────────────────────────────────────────────────────────
#  Counter-example container
# ──────────────────────────────────────────────────────────────

class Counterexample:
    def __init__(self, G, invariants, violation, conjecture_id):
        G.remove_edges_from(nx.selfloop_edges(G))
        self.G = G
        self.invariants = invariants
        self.violation = violation
        self.conjecture_id = conjecture_id
        self.graph6 = nx.to_graph6_bytes(G, header=False).decode().strip()

    def __repr__(self):
        return (f"Counterexample(conj={self.conjecture_id}, "
                f"violation={self.violation:.6f}, "
                f"n={self.G.number_of_nodes()}, "
                f"graph6={self.graph6})")


# ──────────────────────────────────────────────────────────────
#  Score function (local, no API)
# ──────────────────────────────────────────────────────────────

def heuristic_score(G, invariants, conjecture):
    """
    Invariant-aware score that guides the search toward violations.

    Proximity/remoteness (benchmark definition):
        proximity  = 1 / max_v(avg_dist_from_v)   in (0, 1]
        remoteness = 1 / min_v(avg_dist_from_v)   in (0, 1]
    Both values are small for spread-out graphs and close to 1 for compact ones.
    """
    viol  = conjecture.violation(invariants)
    score = 25.0 * viol       # primary signal

    x    = conjecture.x_name
    y    = conjecture.y_name
    sign = conjecture.sign    # '>=' or '<='
    n    = max(invariants.get("n", 1), 1)

    # ── proximity / remoteness ────────────────────────────────
    if "proximity" in (x, y) or "remoteness" in (x, y):
        prox = invariants.get("proximity", 0.0)   # 1/max_avg_dist  (0,1]
        rem  = invariants.get("remoteness", 0.0)  # 1/min_avg_dist  (0,1]
        max_avg = (1.0 / prox) if prox > 1e-9 else 0.0
        min_avg = (1.0 / rem)  if rem  > 1e-9 else 0.0

        if sign == ">=":
            # need y < f(x).
            if y == "proximity":
                score += max_avg * 2.0      # want max_avg large => prox small
            if y == "remoteness":
                score += min_avg * 2.0      # want min_avg large => rem small
            if x == "proximity":
                score += (1.0 - prox) * 3.0
            if x == "remoteness":
                score += (1.0 - rem) * 3.0
        else:
            if y == "proximity":
                score += prox * 5.0         # want prox large
            if y == "remoteness":
                score += rem  * 5.0
            if x == "proximity":
                score += prox * 2.0
            if x == "remoteness":
                score += rem  * 2.0

    # ── total_domination ─────────────────────────────────────
    elif "total_domination" in x or "total_domination" in y:
        gt = invariants.get("total_domination_number", 0)
        al = invariants.get("independence_number", 0)
        mu = invariants.get("matching_number", 0)
        vc = invariants.get("vertex_cover_number", 0)
        if sign == ">=":
            if y == "total_domination_number":
                score += (n - gt) * 0.3       # want small gamma_t
            else:
                score += gt * 0.5             # push x (gamma_t) up
        else:
            if y == "total_domination_number":
                score += gt * 0.5             # want large gamma_t
            else:
                score -= gt * 0.2

    # ── vertex_cover / independence ──────────────────────────
    elif "vertex_cover" in (x, y) or "independence" in (x, y):
        vc = invariants.get("vertex_cover_number", 0)
        al = invariants.get("independence_number", 0)
        if sign == ">=":
            if y == "vertex_cover_number":
                score += (n - vc) * 0.3
            if y == "independence_number":
                score += (n - al) * 0.3
        else:
            if y == "vertex_cover_number":
                score += vc * 0.4
            if y == "independence_number":
                score += al * 0.4

    # ── matching / independent_domination ────────────────────
    elif "matching" in (x, y) or "independent_domination" in (x, y):
        mu  = invariants.get("matching_number", 0)
        ido = invariants.get("independent_domination_number", 0)
        if sign == ">=":
            if y == "matching_number":
                score += (n // 2 - mu) * 0.4
            if y == "independent_domination_number":
                score += (n - ido) * 0.3
        else:
            if y == "matching_number":
                score += mu  * 0.5
            if y == "independent_domination_number":
                score += ido * 0.5

    # ── spectral invariants ───────────────────────────────────
    elif "eigenvalue" in x or "eigenvalue" in y:
        lde = invariants.get("largest_distance_eigenvalue", 0.0)
        lam = invariants.get("largest_eigenvalue", 0.0)
        al2 = invariants.get("second_smallest_laplace_eigenvalue", 0.0)
        score += lde * 0.15 + lam * 0.05 - al2 * 0.05

    # ── diameter / radius ─────────────────────────────────────
    elif "diameter" in (x, y) or "radius" in (x, y):
        d = invariants.get("diameter", 0)
        score += d * 0.5 - 0.02 * n

    # ── randic index ──────────────────────────────────────────
    elif "randic" in x or "randic" in y:
        Delta = invariants.get("maximum_degree", 0)
        delta = invariants.get("minimum_degree", 1) or 1
        score += (Delta - delta) * 0.3

    # ── density ───────────────────────────────────────────────
    elif "density" in (x, y):
        dens = invariants.get("density", 0.0)
        if sign == ">=":
            score += dens * 3.0
        else:
            score += (1.0 - dens) * 3.0

    else:
        Delta = invariants.get("maximum_degree", 0)
        score += 0.1 * Delta

    return score


# ──────────────────────────────────────────────────────────────
#  Main entry point
# ──────────────────────────────────────────────────────────────

def search_counterexample(conjecture, time_limit=60.0, score_fn=None, verbose=False):
    """
    Three-phase local search (no API calls):
      Phase 0 – targeted special-graph library  (~2 s)
      Phase 1 – exhaustive small graphs n=3..13
      Phase 2 – hill-climbing + SA with crossover
    """
    start_time = time.time()
    classes = conjecture.graph_classes
    needed  = get_needed_invariants(conjecture)

    if score_fn is None:
        score_fn = heuristic_score

    # ── Phase 0 ───────────────────────────────────────────────
    result = _try_special_graphs(conjecture, classes, needed)
    if result is not None:
        return result

    elapsed = time.time() - start_time

    # ── Phase 1 ───────────────────────────────────────────────
    budget1 = max(time_limit * 0.08, min(6.0, time_limit * 0.15))
    if elapsed < time_limit * 0.12:
        result = _exhaustive_small(conjecture, classes, needed,
                                   start_time, elapsed + budget1)
        if result is not None:
            return result

    # ── Phase 2 ───────────────────────────────────────────────
    pop_size   = 20
    max_no_imp = 100
    tabu       = set()

    population = generate_initial_graphs(conjecture, n_graphs=pop_size)
    scored = []
    _ctr = [0]  # tie-breaker counter to avoid comparing Graph objects
    def _entry(sc, G, inv):
        _ctr[0] += 1
        return (sc, _ctr[0], G, inv)

    for G in population:
        r = _eval(G, conjecture, classes, needed, score_fn)
        if r is None:
            continue
        if isinstance(r, Counterexample):
            return r
        sc, Gr, inv = r
        scored.append(_entry(sc, Gr, inv))

    if not scored:
        scored = [_entry(0.0, nx.path_graph(6), {})]

    scored.sort(reverse=True, key=lambda t: t[0])
    best_sc = scored[0][0]
    no_imp  = 0
    temp    = 6.0
    cool    = 0.993

    while time.time() - start_time < time_limit - 0.3:
        frac = (time.time() - start_time) / time_limit

        # Parent selection (tournament or random)
        if random.random() < 0.65:
            pool  = scored[:max(5, len(scored) // 2)]
            cands = random.sample(pool, min(3, len(pool)))
            _, _k, par, _ = max(cands, key=lambda t: t[0])
        else:
            _, _k, par, _ = scored[random.randint(0, min(len(scored) - 1, pop_size))]

        # Crossover with second parent occasionally
        if random.random() < 0.25 and len(scored) >= 2:
            _, _k2, par2, _ = scored[random.randint(0, min(len(scored) - 1, pop_size))]
            H = _crossover(par, par2, classes)
        else:
            n_mut = random.choices([1, 2, 3, 4], weights=[0.35, 0.35, 0.20, 0.10])[0]
            if no_imp > max_no_imp // 2:
                n_mut += 1
            H = par.copy()
            for _ in range(n_mut):
                H = mutate(H, classes)

        H = repair(H, classes)
        H.remove_edges_from(nx.selfloop_edges(H))
        if not satisfies_class(H, classes) or H.number_of_nodes() < 2:
            continue

        try:
            g6 = nx.to_graph6_bytes(H, header=False).decode().strip()
        except Exception:
            continue
        if g6 in tabu:
            continue
        tabu.add(g6)
        if len(tabu) > 6000:
            tabu = set(random.sample(list(tabu), 1200))

        try:
            inv_H = compute_invariants(H, needed)
        except Exception:
            continue

        v = conjecture.violation(inv_H)
        if v > 1e-9:
            return Counterexample(H, inv_H, v, conjecture.id)

        sc_H = score_fn(H, inv_H, conjecture)

        if sc_H > best_sc:
            best_sc = sc_H
            no_imp  = 0
        else:
            delta = sc_H - best_sc
            if temp > 0.01 and random.random() < math.exp(max(delta / temp, -30)):
                pass
            no_imp += 1

        temp *= cool
        scored.append(_entry(sc_H, H, inv_H))
        scored.sort(reverse=True, key=lambda t: t[0])
        scored = scored[:pop_size * 2]

        if no_imp >= max_no_imp:
            no_imp = 0
            temp   = max(2.5, temp * 3.0)

            new_pop = generate_initial_graphs(conjecture, n_graphs=pop_size)
            for Gn in new_pop:
                r = _eval(Gn, conjecture, classes, needed, score_fn)
                if r is None:
                    continue
                if isinstance(r, Counterexample):
                    return r
                sc, Gr, inv = r
                scored.append(_entry(sc, Gr, inv))

            for sz in _target_sizes(frac):
                try:
                    Gt = _generate_targeted(conjecture, classes, sz)
                    if Gt is None:
                        continue
                    r = _eval(Gt, conjecture, classes, needed, score_fn)
                    if r is None:
                        continue
                    if isinstance(r, Counterexample):
                        return r
                    sc, Gr, inv = r
                    scored.append(_entry(sc, Gr, inv))
                except Exception:
                    pass

            scored.sort(reverse=True, key=lambda t: t[0])
            scored = scored[:pop_size * 2]
            if scored:
                best_sc = scored[0][0]

    return None


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def _eval(G, conjecture, classes, needed, score_fn):
    if G is None or G.number_of_nodes() < 2:
        return None
    G.remove_edges_from(nx.selfloop_edges(G))
    if not satisfies_class(G, classes):
        return None
    try:
        inv = compute_invariants(G, needed)
        v = conjecture.violation(inv)
        if v > 1e-9:
            return Counterexample(G, inv, v, conjecture.id)
        sc = score_fn(G, inv, conjecture)
        return (sc, G, inv)   # 3-tuple; caller wraps in _entry() for the scored list
    except Exception:
        return None


def _crossover(G1, G2, classes):
    """Simple crossover: merge edges from two parents, reconnect."""
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()
    n  = (n1 + n2) // 2
    H  = nx.Graph()
    H.add_nodes_from(range(n))
    for u, v in G1.edges():
        if u < n and v < n:
            H.add_edge(u, v)
    for u, v in G2.edges():
        nu = u % n
        nv = v % n
        if nu != nv:
            H.add_edge(nu, nv)
    # ensure connectivity
    if not nx.is_connected(H):
        nodes = list(H.nodes())
        for comp in list(nx.connected_components(H))[1:]:
            src = list(comp)[0]
            dst = random.choice([x for x in nodes if x not in comp])
            H.add_edge(src, dst)
    return H


def _target_sizes(frac):
    if frac < 0.3:
        return [5, 7, 9, 11, 13, 15]
    elif frac < 0.6:
        return [10, 15, 18, 20, 25, 30]
    else:
        return [15, 20, 25, 30, 35, 40, 50]


# ──────────────────────────────────────────────────────────────
#  Phase 0 – targeted special-graph library
# ──────────────────────────────────────────────────────────────

def _try_special_graphs(conjecture, classes, needed):
    x    = conjecture.x_name
    y    = conjecture.y_name
    sign = conjecture.sign
    t0   = time.time()

    def _check(G):
        """Evaluate one graph; return Counterexample or None."""
        try:
            if G is None or G.number_of_nodes() < 2:
                return None
            G.remove_edges_from(nx.selfloop_edges(G))
            if not satisfies_class(G, classes):
                return None
            inv = compute_invariants(G, needed)
            v = conjecture.violation(inv)
            if v > 1e-9:
                return Counterexample(G, inv, v, conjecture.id)
        except Exception:
            pass
        return None

    # ── PRIORITY graphs – checked immediately, no time waste ──
    priority = []

    # Star-with-tail graphs for proximity/remoteness conjectures
    # K_{1,k} + tail of m: various distance properties
    if "proximity" in (x, y) or "remoteness" in (x, y) or "domination" in x or "domination" in y:
        for k in range(2, 18):
            for m in range(1, 16):
                priority.append(_star_with_tail(k, m))

    for G in priority:
        r = _check(G)
        if r is not None:
            return r

    # ── GENERAL graphs (with 4-second budget) ─────────────────
    graphs = []

    # universal basics (small n)
    if "tree" not in classes:
        for n in range(3, 22):
            graphs.append(nx.complete_graph(n))
            graphs.append(nx.cycle_graph(n))
            graphs.append(nx.path_graph(n))
            graphs.append(nx.star_graph(n))
            if n >= 4:
                graphs.append(nx.wheel_graph(n))

    # ── complete bipartite ────────────────────────────────────
    if "tree" not in classes:
        for a in range(1, 9):
            for b in range(a, 12):
                graphs.append(nx.complete_bipartite_graph(a, b))

    # ── Petersen + small named graphs ─────────────────────────
    if "tree" not in classes and "claw_free" not in classes:
        graphs.append(nx.petersen_graph())
        try:
            graphs.append(nx.dodecahedral_graph())
            graphs.append(nx.icosahedral_graph())
            graphs.append(nx.bull_graph())
            graphs.append(nx.diamond_graph())
        except Exception:
            pass
        for n in [6, 8, 10, 12, 14, 16]:
            for d in [3, 4]:
                if (n * d) % 2 == 0 and d < n:
                    try:
                        graphs.append(nx.random_regular_graph(d, n, seed=42 + n))
                    except Exception:
                        pass

    # ── PROXIMITY / REMOTENESS ────────────────────────────────
    if "proximity" in (x, y) or "remoteness" in (x, y):
        # K_k + pendant path (dense + outlier distance)
        for core in range(3, 16):
            for tail in range(1, 12):
                G = _clique_plus_path(core, tail)
                if satisfies_class(G, classes):
                    graphs.append(G)

        # Lollipop graphs
        for k in range(3, 14):
            for m in range(1, 12):
                try:
                    G = nx.lollipop_graph(k, m)
                    if satisfies_class(G, classes):
                        graphs.append(G)
                except Exception:
                    pass

        # Barbell graphs
        for k in range(3, 9):
            for bridge in range(1, 8):
                G = _barbell(k, bridge)
                if satisfies_class(G, classes):
                    graphs.append(G)

        # Paths and caterpillars (large max_avg_dist)
        for n in range(5, 40):
            graphs.append(nx.path_graph(n))
            G = _caterpillar_uniform(n, 1)
            if satisfies_class(G, classes):
                graphs.append(G)

        # Stars (large remoteness: center has min_avg_dist = 1)
        for n in range(3, 35):
            graphs.append(nx.star_graph(n))

        # Spider graphs
        for arms in range(3, 8):
            for length in range(2, 9):
                G = _spider_graph(arms, length)
                if satisfies_class(G, classes):
                    graphs.append(G)

        # Double stars
        for k1 in range(2, 12):
            for k2 in range(k1, 12):
                G = _double_star_graph(k1, k2)
                if satisfies_class(G, classes):
                    graphs.append(G)

        if "claw_free" in classes:
            for n in range(4, 20):
                L = nx.line_graph(nx.path_graph(n))
                L = nx.convert_node_labels_to_integers(L)
                if nx.is_connected(L):
                    graphs.append(L)
            for n in range(4, 18):
                L = nx.line_graph(nx.cycle_graph(n))
                L = nx.convert_node_labels_to_integers(L)
                graphs.append(L)
            for n in range(5, 20):
                try:
                    G = nx.power_graph(nx.path_graph(n), 2)
                    graphs.append(G)
                except Exception:
                    pass

    # ── TOTAL DOMINATION ─────────────────────────────────────
    if "total_domination" in x or "total_domination" in y:
        for n in range(4, 30):
            graphs.append(nx.cycle_graph(n))
        for n in range(6, 24):
            graphs.append(nx.circulant_graph(n, [1, 2]))
        for n in range(3, 22):
            graphs.append(nx.complete_graph(n))
            graphs.append(nx.star_graph(n))
        for base_n in range(4, 16):
            G = _corona_k1(base_n)
            if satisfies_class(G, classes):
                graphs.append(G)
        for arms in range(3, 10):
            G = _subdivided_star(arms)
            if satisfies_class(G, classes):
                graphs.append(G)
        # Caterpillars with many leaves (high gt, low mu/vc)
        for spine in range(3, 16):
            for leaves in range(1, 8):
                G = _caterpillar_uniform(spine, leaves)
                if satisfies_class(G, classes):
                    graphs.append(G)
        # Mixed-arm spiders (asymmetric domination structure)
        for n_arms in range(3, 7):
            for short in range(1, 4):
                for lng in range(short + 1, 6):
                    arms = [short] + [lng] * (n_arms - 1)
                    G = _spider_mixed(arms)
                    if satisfies_class(G, classes):
                        graphs.append(G)
        # Double stars (high gt for trees)
        for k1 in range(2, 14):
            for k2 in range(k1, 14):
                G = _double_star_graph(k1, k2)
                if satisfies_class(G, classes):
                    graphs.append(G)
        if "tree" in classes:
            for n in range(5, 40):
                graphs.append(nx.path_graph(n))
                graphs.append(nx.star_graph(n))
            for k in range(2, 12):
                G = _double_star_graph(k, k)
                if satisfies_class(G, classes):
                    graphs.append(G)
        if "claw_free" in classes:
            for k in range(3, 10):
                L = nx.line_graph(nx.complete_graph(k))
                graphs.append(nx.convert_node_labels_to_integers(L))
            for n in range(4, 18):
                L = nx.line_graph(nx.path_graph(n))
                graphs.append(nx.convert_node_labels_to_integers(L))
            # Cycle/clique with pendant paths (claw-free, high gt)
            for cl in range(3, 8):
                for pl in range(1, 5):
                    G = _cycle_with_pendant_paths(cl, pl)
                    if satisfies_class(G, classes):
                        graphs.append(G)
            for cl in range(3, 7):
                for pl in range(1, 4):
                    G = _clique_with_pendant_paths(cl, pl)
                    if satisfies_class(G, classes):
                        graphs.append(G)
        for n in range(8, 28):
            G = nx.cycle_graph(n).copy()
            if n >= 6:
                G.add_edge(0, n // 2)
            if satisfies_class(G, classes):
                graphs.append(G)

    # ── VERTEX COVER / INDEPENDENCE ───────────────────────────
    if "vertex_cover" in x or "vertex_cover" in y or \
       "independence" in x or "independence" in y:
        for a in range(2, 12):
            for b in range(a, 15):
                graphs.append(nx.complete_bipartite_graph(a, b))
        for n in range(3, 28):
            graphs.append(nx.star_graph(n))
            graphs.append(nx.path_graph(n))
            graphs.append(nx.cycle_graph(n))
        for n in range(3, 20):
            graphs.append(nx.complete_graph(n))
        for n in range(4, 18):
            graphs.append(nx.wheel_graph(n))
        for n in range(5, 32):
            graphs.append(nx.random_labeled_tree(n, seed=n))
        for k in range(2, 14):
            G = _double_star_graph(k, k)
            if satisfies_class(G, classes):
                graphs.append(G)
        for n in range(3, 10):
            for leaves in range(1, 8):
                G = _caterpillar_uniform(n, leaves)
                if satisfies_class(G, classes):
                    graphs.append(G)

    # ── MATCHING / INDEPENDENT DOMINATION ─────────────────────
    if "matching" in (x, y) or "independent_domination" in (x, y):
        for n in range(2, 16):
            graphs.append(nx.complete_bipartite_graph(n, n))
        for n in range(3, 28):
            graphs.append(nx.star_graph(n))
            graphs.append(nx.path_graph(n))
            graphs.append(nx.cycle_graph(n))
        for n in range(4, 22):
            graphs.append(nx.complete_graph(n))
            graphs.append(nx.wheel_graph(n))
        for n in range(6, 24):
            graphs.append(nx.circulant_graph(n, [1, 3]))
            graphs.append(nx.circulant_graph(n, [1, 2, 3]))
        # Double stars: low mu, high ido
        for k1 in range(2, 14):
            for k2 in range(k1, 14):
                G = _double_star_graph(k1, k2)
                if satisfies_class(G, classes):
                    graphs.append(G)
        # Caterpillars with many leaves
        for spine in range(3, 10):
            for leaves in range(1, 8):
                G = _caterpillar_uniform(spine, leaves)
                if satisfies_class(G, classes):
                    graphs.append(G)
        if "claw_free" in classes:
            for k in range(3, 10):
                L = nx.line_graph(nx.complete_graph(k))
                graphs.append(nx.convert_node_labels_to_integers(L))
            for n in range(4, 18):
                L = nx.line_graph(nx.path_graph(n))
                graphs.append(nx.convert_node_labels_to_integers(L))
            for n in range(4, 18):
                L = nx.line_graph(nx.cycle_graph(n))
                graphs.append(nx.convert_node_labels_to_integers(L))
            for cl in range(3, 8):
                for pl in range(1, 5):
                    G = _cycle_with_pendant_paths(cl, pl)
                    if satisfies_class(G, classes):
                        graphs.append(G)

    # ── TREE CLASS ─────────────────────────────────────────────
    if "tree" in classes:
        for n in range(4, 42):
            graphs.append(nx.path_graph(n))
            graphs.append(nx.star_graph(n - 1))
        for k in range(2, 16):
            G = _double_star_graph(k, k)
            if satisfies_class(G, classes):
                graphs.append(G)
        # Asymmetric double stars
        for k1 in range(2, 12):
            for k2 in range(k1 + 1, 14):
                G = _double_star_graph(k1, k2)
                if satisfies_class(G, classes):
                    graphs.append(G)
        for spine in range(3, 16):
            for leaves in range(1, 8):
                G = _caterpillar_uniform(spine, leaves)
                if satisfies_class(G, classes):
                    graphs.append(G)
        for arms in range(3, 8):
            for length in range(2, 9):
                G = _spider_graph(arms, length)
                if satisfies_class(G, classes):
                    graphs.append(G)
        # Mixed-arm spiders
        for n_arms in range(3, 7):
            for short in range(1, 4):
                for lng in range(short + 1, 6):
                    arms = [short] + [lng] * (n_arms - 1)
                    G = _spider_mixed(arms)
                    if satisfies_class(G, classes):
                        graphs.append(G)
        # Star-with-tail (various k, m)
        for k in range(2, 16):
            for m in range(1, 16):
                G = _star_with_tail(k, m)
                if satisfies_class(G, classes):
                    graphs.append(G)
        for r in [2, 3]:
            for h in range(2, 7):
                graphs.append(nx.balanced_tree(r, h))

    # ── CLAW-FREE ─────────────────────────────────────────────
    if "claw_free" in classes:
        for n in range(3, 24):
            graphs.append(nx.complete_graph(n))
            graphs.append(nx.cycle_graph(n))
        for n in range(5, 22):
            for k in range(2, max(2, n // 3) + 1):
                G = nx.circulant_graph(n, list(range(1, k + 1)))
                graphs.append(G)
        for k in range(3, 10):
            L = nx.line_graph(nx.complete_graph(k))
            graphs.append(nx.convert_node_labels_to_integers(L))
        for n in range(4, 20):
            L = nx.line_graph(nx.path_graph(n))
            graphs.append(nx.convert_node_labels_to_integers(L))
        for n in range(4, 18):
            L = nx.line_graph(nx.cycle_graph(n))
            graphs.append(nx.convert_node_labels_to_integers(L))
        for cycle_len in range(3, 8):
            for clique in range(2, 5):
                G = _inflated_cycle(cycle_len, clique)
                if satisfies_class(G, classes):
                    graphs.append(G)
        # Cycle/clique with pendant paths (claw-free, high gt)
        for cl in range(3, 10):
            for pl in range(1, 6):
                G = _cycle_with_pendant_paths(cl, pl)
                if satisfies_class(G, classes):
                    graphs.append(G)
        for cl in range(3, 8):
            for pl in range(1, 4):
                G = _clique_with_pendant_paths(cl, pl)
                if satisfies_class(G, classes):
                    graphs.append(G)

    # ── EIGENVALUES ────────────────────────────────────────────
    if "eigenvalue" in x or "eigenvalue" in y:
        for n in range(4, 24):
            graphs.append(nx.path_graph(n))
            graphs.append(nx.cycle_graph(n))
        for core in range(3, 12):
            for tail in range(1, 10):
                G = _clique_plus_path(core, tail)
                if satisfies_class(G, classes):
                    graphs.append(G)
        for n in range(3, 20):
            graphs.append(nx.star_graph(n))
        if "claw_free" in classes:
            for n in range(4, 20):
                L = nx.line_graph(nx.path_graph(n))
                graphs.append(nx.convert_node_labels_to_integers(L))

    # ── DENSITY ────────────────────────────────────────────────
    if "density" in (x, y):
        for n in range(3, 22):
            graphs.append(nx.complete_graph(n))
        for a in range(1, 10):
            for b in range(a, 10):
                graphs.append(nx.complete_bipartite_graph(a, b))
        for core in range(4, 15):
            for tail in range(1, 7):
                G = _clique_plus_path(core, tail)
                if satisfies_class(G, classes):
                    graphs.append(G)

    # ── RANDIC INDEX ───────────────────────────────────────────
    if "randic" in x or "randic" in y:
        for n in range(4, 24):
            graphs.append(nx.path_graph(n))
            graphs.append(nx.cycle_graph(n))
            graphs.append(nx.star_graph(n))
        for a in range(1, 9):
            for b in range(1, 9):
                graphs.append(nx.complete_bipartite_graph(a, b))
        for n in [6, 8, 10, 12, 14, 16, 18, 20]:
            for d in [2, 3, 4]:
                if (n * d) % 2 == 0 and d < n:
                    try:
                        graphs.append(nx.random_regular_graph(d, n, seed=n + d))
                    except Exception:
                        pass

    # ── Evaluate general candidates with time budget ──────────
    for G in graphs:
        if time.time() - t0 > 6.0:   # 6-second Phase 0 budget
            break
        try:
            if G.number_of_nodes() < 2:
                continue
            if not satisfies_class(G, classes):
                continue
            inv = compute_invariants(G, needed)
            if conjecture.violation(inv) > 1e-9:
                return Counterexample(G, inv, conjecture.violation(inv), conjecture.id)
        except Exception:
            continue

    return None


# ──────────────────────────────────────────────────────────────
#  Phase 1 – exhaustive small graphs
# ──────────────────────────────────────────────────────────────

def _exhaustive_small(conjecture, classes, needed, start_time, deadline):
    for n in range(3, 14):
        if time.time() > deadline:
            break
        attempts = 80 if n <= 8 else 40
        for _ in range(attempts):
            if time.time() > deadline:
                break
            try:
                if "tree" in classes:
                    G = nx.random_labeled_tree(n)
                elif "claw_free" in classes:
                    p = random.uniform(0.5, 0.92)
                    G = nx.erdos_renyi_graph(n, p)
                    if not nx.is_connected(G):
                        continue
                    if not is_claw_free(G):
                        continue
                else:
                    p = random.uniform(0.1, 0.85)
                    G = nx.erdos_renyi_graph(n, p)
                    if not nx.is_connected(G):
                        continue
                if not satisfies_class(G, classes):
                    continue
                inv = compute_invariants(G, needed)
                if conjecture.violation(inv) > 1e-9:
                    return Counterexample(G, inv, conjecture.violation(inv), conjecture.id)
            except Exception:
                continue
    return None


# ──────────────────────────────────────────────────────────────
#  Phase 2 targeted generator
# ──────────────────────────────────────────────────────────────

def _generate_targeted(conjecture, classes, n):
    x = conjecture.x_name
    y = conjecture.y_name

    if "tree" in classes:
        candidates = [
            nx.path_graph(n),
            nx.star_graph(n - 1),
            nx.random_labeled_tree(n),
            _double_star_graph(n // 3 + 1, max(1, n - n // 3 - 3)),
            _caterpillar_uniform(max(3, n // 2), 2),
            _spider_graph(3, max(2, n // 4)),
        ]
        for G in candidates:
            try:
                if G is not None and satisfies_class(G, classes):
                    return G
            except Exception:
                pass
        return nx.path_graph(n)

    if "claw_free" in classes:
        if n <= 12:
            return nx.complete_graph(n)
        k = int(math.sqrt(2 * n)) + 1
        L = nx.line_graph(nx.complete_graph(min(k, 10)))
        return nx.convert_node_labels_to_integers(L)

    # Connected graphs
    if "proximity" in (x, y) or "remoteness" in (x, y) or "density" in (x, y):
        core = max(3, n * 2 // 3)
        tail = max(1, n - core)
        G = _clique_plus_path(core, tail)
        if satisfies_class(G, classes):
            return G

    if "total_domination" in (x, y) or "matching" in (x, y):
        G = nx.cycle_graph(n)
        for _ in range(n // 3):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                G.add_edge(u, v)
        if nx.is_connected(G) and satisfies_class(G, classes):
            return G

    if "independence" in (x, y) or "vertex_cover" in (x, y):
        return nx.star_graph(n - 1)

    if "diameter" in (x, y) or "radius" in (x, y):
        return nx.path_graph(n)

    if "eigenvalue" in x or "eigenvalue" in y:
        return _clique_plus_path(max(3, n // 2), max(1, n - n // 2))

    for _ in range(20):
        p = random.uniform(0.15, 0.55)
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G) and satisfies_class(G, classes):
            return G
    return nx.path_graph(n)


# ──────────────────────────────────────────────────────────────
#  Graph construction helpers
# ──────────────────────────────────────────────────────────────

def _star_with_tail(k, m):
    """K_{1,k} (hub + k leaves) plus a path tail of m vertices: h-p1-p2-...-pm.
    Properties: n=1+k+m, vc=μ=1+floor(m/2),
    proximity = 2(k+m) / ((m+1)(m+2k))
    """
    G = nx.Graph()
    # hub = 0, leaves 1..k, tail k+1..k+m
    G.add_node(0)
    for i in range(1, k + 1):
        G.add_edge(0, i)
    prev = 0
    for j in range(1, m + 1):
        node = k + j
        G.add_edge(prev, node)
        prev = node
    return G


def _clique_plus_path(k, m):
    """K_k attached to a path of m extra vertices."""
    G = nx.complete_graph(k)
    prev = k - 1
    for i in range(m):
        G.add_edge(prev, k + i)
        prev = k + i
    return G


def _barbell(k, bridge):
    """Two K_k cliques joined by a path of length bridge."""
    G = nx.complete_graph(k)
    offset = k
    prev = k - 1
    for i in range(bridge):
        G.add_node(offset + i)
        G.add_edge(prev, offset + i)
        prev = offset + i
    c2_start = offset + bridge
    for u in range(k):
        for v in range(u + 1, k):
            G.add_node(c2_start + u)
            G.add_node(c2_start + v)
            G.add_edge(c2_start + u, c2_start + v)
    G.add_edge(prev, c2_start)
    return G


def _caterpillar_uniform(spine, leaves_per_node):
    G = nx.path_graph(max(spine, 2))
    nid = max(spine, 2)
    for v in range(max(spine, 2)):
        for _ in range(leaves_per_node):
            G.add_edge(v, nid)
            nid += 1
    return G


def _double_star_graph(k1, k2):
    k1 = max(1, k1)
    k2 = max(1, k2)
    G = nx.Graph()
    G.add_edge(0, 1)
    nid = 2
    for _ in range(k1):
        G.add_edge(0, nid)
        nid += 1
    for _ in range(k2):
        G.add_edge(1, nid)
        nid += 1
    return G


def _spider_graph(arms, arm_length):
    G = nx.Graph()
    G.add_node(0)
    nid = 1
    for _ in range(arms):
        prev = 0
        for _ in range(arm_length):
            G.add_edge(prev, nid)
            prev = nid
            nid += 1
    return G


def _corona_k1(n):
    """G ∘ K_1: K_n with a pendant vertex on every node."""
    G = nx.complete_graph(n)
    for v in range(n):
        G.add_edge(v, n + v)
    return G


def _subdivided_star(arms):
    """K_{1,arms} with each edge subdivided once."""
    G = nx.Graph()
    G.add_node(0)
    nid = 1
    for _ in range(arms):
        G.add_edge(0, nid)
        G.add_edge(nid, nid + 1)
        nid += 2
    return G


def _inflated_cycle(cycle_len, clique_size):
    G = nx.Graph()
    nid = 0
    cliques = []
    for _ in range(cycle_len):
        cl = list(range(nid, nid + clique_size))
        for u in cl:
            for v in cl:
                if u < v:
                    G.add_edge(u, v)
        cliques.append(cl)
        nid += clique_size
    for i in range(cycle_len):
        j = (i + 1) % cycle_len
        for u in cliques[i]:
            for v in cliques[j]:
                G.add_edge(u, v)
    return G


def _spider_mixed(arm_lengths):
    """Spider graph with arms of different lengths from center node 0."""
    G = nx.Graph()
    G.add_node(0)
    nid = 1
    for length in arm_lengths:
        prev = 0
        for _ in range(length):
            G.add_edge(prev, nid)
            prev = nid
            nid += 1
    return G


def _cycle_with_pendant_paths(cycle_len, path_len):
    """C_k with a pendant path of given length attached to each vertex.
    Claw-free when cycle_len >= 3 and path_len >= 1."""
    G = nx.cycle_graph(cycle_len)
    nid = cycle_len
    for v in range(cycle_len):
        prev = v
        for _ in range(path_len):
            G.add_edge(prev, nid)
            prev = nid
            nid += 1
    return G


def _clique_with_pendant_paths(clique_size, path_len):
    """K_k with a pendant path of given length on each vertex. Claw-free."""
    G = nx.complete_graph(clique_size)
    nid = clique_size
    for v in range(clique_size):
        prev = v
        for _ in range(path_len):
            G.add_edge(prev, nid)
            prev = nid
            nid += 1
    return G
