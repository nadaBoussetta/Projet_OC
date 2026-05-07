"""
funsearch.py – Partie 2 : Architecture inspirée de FunSearch.
Le LLM propose, teste et améliore automatiquement des fonctions de score.
"""
import time
import random
import traceback
import requests
import json


# ─────────────────────────────────────────────
#  TEMPLATE DE FONCTION DE SCORE
# ─────────────────────────────────────────────

BASE_SCORE_FUNCTION = '''
def heuristic_score(G, invariants, conjecture):
    """
    G : graphe NetworkX
    invariants : dictionnaire des invariants calculés
    conjecture : objet décrivant la conjecture
    retourne un score numérique à maximiser
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
    density = 0
    if n > 1:
        density = 2 * m / (n * (n - 1))
    return (
        10.0 * violation
        + 0.3 * diam
        + 0.2 * Delta
        + 0.1 * triangles
        - 0.05 * n
        - 0.2 * abs(density - 0.5)
    )
'''

# ─────────────────────────────────────────────
#  PROMPTS POUR LE LLM
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un expert en théorie des graphes et en optimisation combinatoire.
Tu dois proposer des fonctions Python qui guident la recherche vers des graphes violant des conjectures mathématiques.
Réponds UNIQUEMENT avec le code Python de la fonction, sans aucun commentaire ni balise markdown."""

def _build_evolution_prompt(best_functions: list, conjecture_info: str) -> str:
    """Construit le prompt d'évolution pour le LLM."""
    funcs_str = "\n\n---\n\n".join([
        f"# Score moyen: {score:.4f}\n{code}"
        for score, code in best_functions[:3]
    ])

    return f"""Voici les meilleures fonctions de score trouvées pour guider la recherche de contre-exemples en théorie des graphes:

{funcs_str}

Contexte de la conjecture: {conjecture_info}

Ta mission: proposer une NOUVELLE fonction `heuristic_score(G, invariants, conjecture)` AMÉLIORÉE.

Règles:
1. La fonction doit retourner un score numérique à MAXIMISER
2. `conjecture.violation(invariants)` > 0 signifie contre-exemple trouvé
3. Le score doit guider vers des graphes PROMETTEURS même avant la violation
4. Utilise les invariants disponibles: n, m, minimum_degree, maximum_degree, average_degree, density, diameter, radius, triangle_number, clique_number, domination_number, total_domination_number, independence_number, vertex_cover_number, independent_domination_number, matching_number, randic_index, proximity, remoteness, largest_eigenvalue, second_smallest_laplace_eigenvalue
5. Pense aux relations entre invariants: ex. si la conjecture porte sur domination_number et independence_number, guide vers des graphes avec grand écart entre ces deux valeurs
6. Le score ne doit PAS coder en dur des contre-exemples connus

Réponds UNIQUEMENT avec le code Python complet de la fonction heuristic_score, sans markdown ni commentaire."""


def _call_claude_api(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Appelle l'API Claude pour générer une nouvelle fonction de score."""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": system,
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


def _safe_compile(code: str):
    """Compile et retourne la fonction, ou None si erreur."""
    try:
        # Nettoyer le code (enlever balises markdown si présentes)
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        namespace = {}
        exec(code, namespace)
        fn = namespace.get("heuristic_score")
        if callable(fn):
            return fn, code
    except Exception as e:
        print(f"[COMPILE ERROR] {e}")
    return None, None


def _evaluate_score_function(score_fn, test_cases: list) -> float:
    """Évalue une fonction de score sur des cas de test."""
    if not test_cases:
        return 0.0
    total = 0.0
    count = 0
    for G, inv, conjecture in test_cases:
        try:
            sc = score_fn(G, inv, conjecture)
            viol = conjecture.violation(inv)
            # Récompense: le score doit être corrélé avec la violation
            # et élevé pour les graphes proches de la violation
            total += sc + 5.0 * viol
            count += 1
        except Exception:
            pass
    return total / count if count > 0 else 0.0


# ─────────────────────────────────────────────
#  MOTEUR FUNSEARCH
# ─────────────────────────────────────────────

class FunSearch:
    """
    Implémentation de FunSearch: évolution de fonctions de score via LLM.
    """

    def __init__(self, n_iterations=5, population_size=4):
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.function_pool = []  # Liste de (score, code, fn)
        self._init_pool()

    def _init_pool(self):
        """Initialise le pool avec la fonction de base."""
        fn, code = _safe_compile(BASE_SCORE_FUNCTION)
        if fn:
            self.function_pool.append((0.0, code, fn))

    def evolve(self, test_cases: list, conjecture_info: str = ""):
        """
        Fait évoluer le pool de fonctions via le LLM.
        test_cases: liste de (G, invariants, conjecture)
        """
        print(f"\n[FunSearch] Démarrage de l'évolution ({self.n_iterations} itérations)")

        # Évaluer le pool initial
        for i, (_, code, fn) in enumerate(self.function_pool):
            score = _evaluate_score_function(fn, test_cases)
            self.function_pool[i] = (score, code, fn)

        for iteration in range(self.n_iterations):
            print(f"  [FunSearch] Itération {iteration + 1}/{self.n_iterations}")

            # Trier par score
            self.function_pool.sort(key=lambda x: x[0], reverse=True)

            # Préparer les meilleures fonctions pour le LLM
            best_for_llm = [(sc, code) for sc, code, _ in self.function_pool[:3]]

            # Appeler le LLM
            prompt = _build_evolution_prompt(best_for_llm, conjecture_info)
            new_code = _call_claude_api(prompt)

            if new_code:
                fn_new, clean_code = _safe_compile(new_code)
                if fn_new:
                    score_new = _evaluate_score_function(fn_new, test_cases)
                    self.function_pool.append((score_new, clean_code, fn_new))
                    print(f"    [FunSearch] Nouvelle fonction ajoutée, score={score_new:.4f}")

            # Garder les meilleures
            self.function_pool.sort(key=lambda x: x[0], reverse=True)
            self.function_pool = self.function_pool[:self.population_size]

        print(f"[FunSearch] Évolution terminée. Meilleur score: {self.function_pool[0][0]:.4f}")
        return self.function_pool[0][2]  # Retourne la meilleure fonction

    def get_best_function(self):
        """Retourne la meilleure fonction de score."""
        if self.function_pool:
            self.function_pool.sort(key=lambda x: x[0], reverse=True)
            return self.function_pool[0][2]
        return None

    def get_best_code(self):
        """Retourne le code de la meilleure fonction."""
        if self.function_pool:
            self.function_pool.sort(key=lambda x: x[0], reverse=True)
            return self.function_pool[0][1]
        return BASE_SCORE_FUNCTION


# ─────────────────────────────────────────────
#  HEURISTIQUE AVEC FUNSEARCH
# ─────────────────────────────────────────────

def search_counterexample_funsearch(conjecture, time_limit=60.0,
                                    score_fn=None, verbose=False):
    """
    Recherche de contre-exemple avec une fonction de score issue de FunSearch.
    Identique à heuristic.py mais avec score_fn injectable.
    """
    from invariants import compute_invariants, satisfies_class
    from mutations import generate_initial_graphs, mutate, repair
    import networkx as nx
    from heuristic import Counterexample

    if score_fn is None:
        from heuristic import heuristic_score
        score_fn = heuristic_score

    start_time = time.time()
    classes = conjecture.graph_classes
    pop_size = 8
    max_no_improve = 80
    tabu_size = 20
    best_score = float("-inf")
    best_result = None
    tabu = []

    population = generate_initial_graphs(conjecture, n_graphs=pop_size)
    if not population:
        population = [nx.path_graph(random.randint(4, 12)) for _ in range(pop_size)]

    scored = []
    for G in population:
        try:
            inv = compute_invariants(G)
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
        candidates = random.sample(scored, min(3, len(scored)))
        _, parent_G, _ = max(candidates, key=lambda x: x[0])

        n_mutations = random.randint(1, 3)
        H = parent_G.copy()
        for _ in range(n_mutations):
            H = mutate(H, classes)
        H = repair(H, classes)

        if not satisfies_class(H, classes) or H.number_of_nodes() < 2:
            continue

        try:
            inv_H = compute_invariants(H)
            sc_H = score_fn(H, inv_H, conjecture)
        except Exception:
            continue

        viol = conjecture.violation(inv_H)
        if viol > 0:
            elapsed = time.time() - start_time
            if verbose:
                print(f"  [FOUND] Contre-exemple en {elapsed:.2f}s | violation={viol:.4f}")
            return Counterexample(H, inv_H, viol, conjecture.id)

        g6 = nx.to_graph6_bytes(H, header=False).decode().strip()
        if g6 in tabu:
            continue
        tabu.append(g6)
        if len(tabu) > tabu_size:
            tabu.pop(0)

        if sc_H > best_score:
            best_score = sc_H
            no_improve = 0
        else:
            no_improve += 1

        scored.append((sc_H, H, inv_H))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:pop_size * 2]

        if no_improve >= max_no_improve:
            new_graphs = generate_initial_graphs(conjecture, n_graphs=pop_size)
            for G_new in new_graphs:
                try:
                    inv_new = compute_invariants(G_new)
                    sc_new = score_fn(G_new, inv_new, conjecture)
                    scored.append((sc_new, G_new, inv_new))
                    if conjecture.violation(inv_new) > 0:
                        return Counterexample(G_new, inv_new,
                                              conjecture.violation(inv_new), conjecture.id)
                except Exception:
                    pass
            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:pop_size * 2]
            no_improve = 0

    return None
