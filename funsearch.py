"""
funsearch.py – Partie 2 : Architecture inspirée de FunSearch.
"""
import time
import random
import requests

from invariants import compute_invariants, satisfies_class, get_needed_invariants
from mutations import generate_initial_graphs, mutate, repair


BASE_SCORE_FUNCTION = '''
def heuristic_score(G, invariants, conjecture):
    violation = conjecture.violation(invariants)
    n = invariants.get("n", 1) or 1
    m = invariants.get("m", 0)
    delta = invariants.get("minimum_degree", 0)
    Delta = invariants.get("maximum_degree", 0)
    diam = invariants.get("diameter", 0)
    gamma = invariants.get("domination_number", 0)
    alpha = invariants.get("independence_number", 0)
    tau = invariants.get("vertex_cover_number", 0)
    triangles = invariants.get("triangle_number", 0)
    density = 0
    if n > 1:
        density = 2 * m / (n * (n - 1))
    return (
        20.0 * violation
        + 0.5 * diam
        + 0.3 * Delta
        + 0.2 * triangles
        - 0.03 * n
        - 0.1 * abs(density - 0.4)
    )
'''

SYSTEM_PROMPT = """Tu es un expert en théorie des graphes et en optimisation combinatoire.
Tu dois proposer des fonctions Python qui guident la recherche vers des graphes violant des conjectures mathématiques.
Réponds UNIQUEMENT avec le code Python de la fonction, sans aucun commentaire ni balise markdown."""


def _build_evolution_prompt(best_functions, conjecture_info):
    funcs_str = "\n\n---\n\n".join([
        f"# Score moyen: {score:.4f}\n{code}"
        for score, code in best_functions[:3]
    ])
    return f"""Voici les meilleures fonctions de score pour guider la recherche de contre-exemples:

{funcs_str}

Contexte: {conjecture_info}

Propose une NOUVELLE fonction `heuristic_score(G, invariants, conjecture)` AMÉLIORÉE.
Règles:
1. Retourne un score numérique à MAXIMISER
2. `conjecture.violation(invariants)` > 0 = contre-exemple
3. Guide vers des graphes prometteurs AVANT la violation
4. Invariants disponibles: n, m, minimum_degree, maximum_degree, average_degree, density, diameter, radius, triangle_number, clique_number, domination_number, total_domination_number, independence_number, vertex_cover_number, independent_domination_number, matching_number, randic_index, harmonic_index, proximity, remoteness, largest_eigenvalue, second_smallest_laplace_eigenvalue
5. Adapte le bonus selon les invariants de la conjecture (x_name={conjecture_info})

Réponds UNIQUEMENT avec le code Python complet de heuristic_score, sans markdown."""


def _call_claude_api(prompt):
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30
        )
        data = response.json()
        if "content" in data and data["content"]:
            return data["content"][0].get("text", "").strip()
    except Exception as e:
        print(f"[LLM ERROR] {e}")
    return ""


def _safe_compile(code):
    try:
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        namespace = {}
        exec(code, namespace)
        fn = namespace.get("heuristic_score")
        if callable(fn):
            return fn, code
    except Exception as e:
        print(f"[COMPILE ERROR] {e}")
    return None, None


def _evaluate_score_function(score_fn, test_cases):
    if not test_cases:
        return 0.0
    total = 0.0
    count = 0
    for G, inv, conjecture in test_cases:
        try:
            sc = score_fn(G, inv, conjecture)
            viol = conjecture.violation(inv)
            total += sc + 10.0 * viol
            count += 1
        except Exception:
            pass
    return total / count if count > 0 else 0.0


class FunSearch:
    def __init__(self, n_iterations=5, population_size=4):
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.function_pool = []
        fn, code = _safe_compile(BASE_SCORE_FUNCTION)
        if fn:
            self.function_pool.append((0.0, code, fn))

    def evolve(self, test_cases, conjecture_info=""):
        print(f"\n[FunSearch] Démarrage ({self.n_iterations} itérations)")
        for i, (_, code, fn) in enumerate(self.function_pool):
            score = _evaluate_score_function(fn, test_cases)
            self.function_pool[i] = (score, code, fn)

        for iteration in range(self.n_iterations):
            print(f"  [FunSearch] Itération {iteration+1}/{self.n_iterations}")
            self.function_pool.sort(key=lambda x: x[0], reverse=True)
            best_for_llm = [(sc, code) for sc, code, _ in self.function_pool[:3]]
            prompt = _build_evolution_prompt(best_for_llm, conjecture_info)
            new_code = _call_claude_api(prompt)
            if new_code:
                fn_new, clean_code = _safe_compile(new_code)
                if fn_new:
                    score_new = _evaluate_score_function(fn_new, test_cases)
                    self.function_pool.append((score_new, clean_code, fn_new))
                    print(f"    Nouvelle fonction, score={score_new:.4f}")
            self.function_pool.sort(key=lambda x: x[0], reverse=True)
            self.function_pool = self.function_pool[:self.population_size]

        best = self.function_pool[0][0] if self.function_pool else 0
        print(f"[FunSearch] Terminé. Meilleur score: {best:.4f}")
        return self.function_pool[0][2] if self.function_pool else None

    def get_best_function(self):
        if self.function_pool:
            self.function_pool.sort(key=lambda x: x[0], reverse=True)
            return self.function_pool[0][2]
        return None

    def get_best_code(self):
        if self.function_pool:
            self.function_pool.sort(key=lambda x: x[0], reverse=True)
            return self.function_pool[0][1]
        return BASE_SCORE_FUNCTION


def search_counterexample_funsearch(conjecture, time_limit=60.0, score_fn=None, verbose=False):
    import networkx as nx
    from heuristic import Counterexample, heuristic_score

    if score_fn is None:
        score_fn = heuristic_score

    needed = get_needed_invariants(conjecture)
    start_time = time.time()
    classes = conjecture.graph_classes
    pop_size = 10
    max_no_improve = 60
    tabu = set()
    n_restarts = 0

    population = generate_initial_graphs(conjecture, n_graphs=pop_size)
    if not population:
        population = [nx.path_graph(random.randint(4, 12)) for _ in range(pop_size)]

    scored = []
    for G in population:
        try:
            inv = compute_invariants(G, needed)
            sc = score_fn(G, inv, conjecture)
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

    while time.time() - start_time < time_limit:
        pool = scored[:max(4, len(scored)//2)]
        candidates = random.sample(pool, min(3, len(pool)))
        _, parent_G, _ = max(candidates, key=lambda x: x[0])

        n_mutations = random.randint(1, 2 + min(3, n_restarts))
        H = parent_G.copy()
        for _ in range(n_mutations):
            H = mutate(H, classes)
        H = repair(H, classes)

        if not satisfies_class(H, classes) or H.number_of_nodes() < 2:
            continue

        try:
            g6 = nx.to_graph6_bytes(H, header=False).decode().strip()
        except Exception:
            continue
        if g6 in tabu:
            continue
        tabu.add(g6)
        if len(tabu) > 3000:
            tabu.clear()

        try:
            inv_H = compute_invariants(H, needed)
            sc_H = score_fn(H, inv_H, conjecture)
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

        if no_improve >= max_no_improve:
            n_restarts += 1
            new_graphs = generate_initial_graphs(conjecture, n_graphs=pop_size)
            for G_new in new_graphs:
                try:
                    inv_new = compute_invariants(G_new, needed)
                    sc_new = score_fn(G_new, inv_new, conjecture)
                    scored.append((sc_new, G_new, inv_new))
                    if conjecture.violation(inv_new) > 0:
                        return Counterexample(G_new, inv_new, conjecture.violation(inv_new), conjecture.id)
                except Exception:
                    pass
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:pop_size * 2]
            no_improve = 0

    return None
