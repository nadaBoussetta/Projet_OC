"""
funsearch.py – Architecture FunSearch avec Groq LLM (Llama-3.3-70B).
Inspirée de "Mathematical discoveries from program search with large language models"
(Romera-Paredes et al., Nature 2024).

Pipeline:
  1. Seed  – bibliothèque de 10 fonctions de score manuelles
  2. Evolve – Groq génère de nouvelles fonctions Python via prompt
  3. Sandbox – exécution sécurisée du code généré (namespace restreint)
  4. Evaluate – score sur cas de test réels (vrais graphes + conjectures)
  5. Select  – top-k sélectionnés pour la génération suivante

Fallback offline automatique si GROQ_API_KEY absent ou API indisponible.
"""
import os
import re
import time
import random
import math
import inspect
import networkx as nx
from dotenv import load_dotenv

# ── Chargement de la clé API ────────────────────────────────────────────────
load_dotenv()
_GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
_GROQ_MODEL     = "llama-3.3-70b-versatile"
_GROQ_FALLBACK  = "llama-3.1-8b-instant"

try:
    from groq import Groq as _GroqClient
    _GROQ_AVAILABLE = bool(_GROQ_API_KEY)
except ImportError:
    _GROQ_AVAILABLE = False

from invariants import compute_invariants, satisfies_class, get_needed_invariants
from mutations import generate_initial_graphs


# ═══════════════════════════════════════════════════════════════════════════
#  SEED FUNCTIONS – Bibliothèque de démarrage (10 fonctions)
# ═══════════════════════════════════════════════════════════════════════════

def _score_v1(G, invariants, conjecture):
    """Pure violation score."""
    return conjecture.violation(invariants)


def _score_v2(G, invariants, conjecture):
    """Violation + diameter bonus."""
    v = conjecture.violation(invariants)
    return 15.0 * v + 0.5 * invariants.get("diameter", 0) - 0.02 * invariants.get("n", 1)


def _score_v3(G, invariants, conjecture):
    """Violation + degree heterogeneity."""
    v = conjecture.violation(invariants)
    Delta = invariants.get("maximum_degree", 0)
    delta = invariants.get("minimum_degree", 0)
    return 12.0 * v + 0.4 * (Delta - delta) + 0.3 * Delta


def _score_v4(G, invariants, conjecture):
    """Adaptive: boost by invariant type."""
    v = conjecture.violation(invariants)
    x, y = conjecture.x_name, conjecture.y_name
    score = 20.0 * v
    if "diameter" in (x, y):
        score += 0.6 * invariants.get("diameter", 0)
    elif "domination" in x or "domination" in y:
        score += 0.5 * invariants.get("domination_number", 0)
    elif "independence" in x or "independence" in y:
        score += 0.4 * invariants.get("independence_number", 0)
    elif "clique" in (x, y):
        score += 0.3 * invariants.get("clique_number", 0)
    elif "eigenvalue" in x or "eigenvalue" in y:
        score += 0.3 * invariants.get("largest_eigenvalue", 0)
    elif "matching" in (x, y):
        score += 0.3 * invariants.get("matching_number", 0)
    elif "zagreb" in x or "zagreb" in y:
        score += 0.2 * invariants.get("maximum_degree", 0) ** 2
    return score


def _score_v5(G, invariants, conjecture):
    """Non-linear: large jump near violation boundary."""
    v = conjecture.violation(invariants)
    if v > 0:
        return 1000.0 + v * 100
    elif v > -0.5:
        return 50.0 + v * 30
    elif v > -2:
        return 10.0 + v * 5
    return v + 0.1 * invariants.get("maximum_degree", 0)


def _score_v6(G, invariants, conjecture):
    """Density-aware: penalize extreme density."""
    v = conjecture.violation(invariants)
    n = invariants.get("n", 1)
    m = invariants.get("m", 0)
    density = (2 * m / (n * (n - 1))) if n > 1 else 0
    Delta = invariants.get("maximum_degree", 0)
    return (18.0 * v
            + 0.5 * math.log(n + 1) * abs(v + 0.1)
            + 0.2 * Delta
            - 0.3 * abs(density - 0.35))


def _score_v7(G, invariants, conjecture):
    """Extremality: prefer highly irregular graphs."""
    v = conjecture.violation(invariants)
    n = invariants.get("n", 1)
    Delta = invariants.get("maximum_degree", 0)
    delta = invariants.get("minimum_degree", 0)
    diam  = invariants.get("diameter", 0)
    return 15.0 * v + 0.1 * abs(Delta - n + 1) + 0.05 * diam + 0.1 * (Delta - delta)


def _score_v8(G, invariants, conjecture):
    """Stepped reward: bigger bonus closer to boundary."""
    v = conjecture.violation(invariants)
    score = 25.0 * v
    if v > -1:
        score += 3.0
    if v > -0.5:
        score += 5.0
    score += 0.1 * invariants.get("diameter", 0) + 0.05 * invariants.get("maximum_degree", 0)
    return score


def _score_v9(G, invariants, conjecture):
    """Multi-objective with size penalty."""
    v = conjecture.violation(invariants)
    n = invariants.get("n", 1)
    m = invariants.get("m", 0)
    diam    = invariants.get("diameter", 0)
    Delta   = invariants.get("maximum_degree", 0)
    density = (2 * m / (n * (n - 1))) if n > 1 else 0
    if v >= 0:
        return 500.0 + v * 50
    proximity = max(0.0, 10.0 + v * 5)
    return 20.0 * v + proximity + 0.2 * diam + 0.15 * Delta - 0.01 * n + 0.5 * (1.0 - abs(density - 0.4))


def _score_v10(G, invariants, conjecture):
    """Logarithmic emphasis on combined domination invariants."""
    v     = conjecture.violation(invariants)
    n     = invariants.get("n", 1)
    alpha = invariants.get("independence_number", 0)
    gamma = invariants.get("domination_number", 0)
    return 20.0 * v + 0.15 * (alpha + gamma) + math.log(max(1, n)) * max(0.0, v + 2)


SEED_FUNCTIONS = [
    _score_v1, _score_v2, _score_v3, _score_v4, _score_v5,
    _score_v6, _score_v7, _score_v8, _score_v9, _score_v10,
]


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPT ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are an expert in graph theory and combinatorial optimization.
Your task: write a Python score function to guide local search toward counter-examples
of graph theory conjectures.

SIGNATURE (mandatory):
def score(G, invariants, conjecture):
    \"\"\"One-line description.\"\"\"
    # implementation
    return float_value

Rules:
- Higher score = graph is a better candidate for violating the conjecture
- conjecture.violation(invariants) > 0 means the conjecture IS violated (goal achieved)
- Available invariants keys: n, m, minimum_degree, maximum_degree, average_degree, density,
  diameter, radius, triangle_number, clique_number, domination_number, total_domination_number,
  independence_number, vertex_cover_number, independent_domination_number, matching_number,
  first_zagreb_index, second_zagreb_index, randic_index, harmonic_index, proximity, remoteness,
  largest_eigenvalue, second_smallest_laplace_eigenvalue, largest_distance_eigenvalue
- conjecture.x_name, conjecture.y_name: invariant names; conjecture.sign: "<=" or ">="
- math module available; NO other imports
- Return ONLY the Python function, no explanations, no markdown"""

_USER_TEMPLATE = """Conjecture to refute: {y_name} {sign} f({x_name})
Graph classes: {classes}
Polynomial degree: {degree}

Best current functions (ranked by performance score):

{top_fns}

Write a NEW function called 'score' that outperforms the above.
Focus on: what structural properties make {y_name} large/small relative to f({x_name})?
Return ONLY the def score(...): block."""


# ═══════════════════════════════════════════════════════════════════════════
#  SANDBOXED CODE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

_BLOCKED = ["import", "__builtins__", "open(", "exec(", "eval(", "compile(",
            "subprocess", "socket", "shutil", "pickle", "os.", "sys."]

_SAFE_NS = {
    "__builtins__": {
        "abs": abs, "max": max, "min": min, "round": round,
        "int": int, "float": float, "bool": bool, "len": len,
        "sum": sum, "range": range, "enumerate": enumerate,
        "isinstance": isinstance, "hasattr": hasattr,
        "zip": zip, "list": list, "dict": dict, "tuple": tuple,
    },
    "math": math,
}


def _compile_fn(code: str):
    """
    Compile et retourne une fonction Python depuis du code généré par LLM.
    Retourne (callable, None) ou (None, str_error).
    """
    # Security screening
    for pat in _BLOCKED:
        if pat in code:
            return None, f"Blocked: '{pat}'"

    # Extract def block
    m = re.search(r"(def\s+\w+\s*\([^)]*\)\s*:(?:.|\n)*)", code)
    if not m:
        return None, "No def block found"

    fn_code = m.group(1)
    ns = dict(_SAFE_NS)
    try:
        exec(fn_code, ns)  # noqa: S102
    except SyntaxError as e:
        return None, f"SyntaxError: {e}"
    except Exception as e:
        return None, f"RuntimeError: {e}"

    fn = next((v for k, v in ns.items()
               if callable(v) and not k.startswith("_") and k not in _SAFE_NS), None)
    return (fn, None) if fn else (None, "No callable found")


# ═══════════════════════════════════════════════════════════════════════════
#  GROQ LLM INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

class _GroqSampler:
    """Wraps Groq API calls with retry and fallback logic."""

    def __init__(self):
        self._client = _GroqClient(api_key=_GROQ_API_KEY) if _GROQ_AVAILABLE else None
        self.model = _GROQ_MODEL
        self.calls = 0
        self.failures = 0

    def sample(self, conjecture, top_db: list) -> tuple:
        """
        Génère une nouvelle fonction de score.
        Retourne (callable, code_str) ou (None, error_msg).
        """
        if not self._client:
            return None, "Groq not available"

        # Build context: top-3 functions with source
        fn_blocks = []
        for rank, (sc, name, fn, code) in enumerate(top_db[:3], 1):
            src = code if code.startswith("def ") else _try_get_source(fn)
            fn_blocks.append(f"# Rank {rank} | score={sc:.4f}\n{src}")

        user_msg = _USER_TEMPLATE.format(
            y_name=conjecture.y_name,
            sign=conjecture.sign,
            x_name=conjecture.x_name,
            classes=", ".join(conjecture.graph_classes),
            degree=conjecture.degree,
            top_fns="\n\n".join(fn_blocks),
        )

        for attempt, model in enumerate([self.model, _GROQ_FALLBACK]):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.8 + attempt * 0.1,
                    max_tokens=700,
                    top_p=0.92,
                )
                self.calls += 1
                raw = resp.choices[0].message.content.strip()
                # Strip markdown
                raw = re.sub(r"```python\s*", "", raw)
                raw = re.sub(r"```\s*", "", raw)

                fn, err = _compile_fn(raw)
                if fn:
                    return fn, raw
                self.failures += 1
                # Retry with fallback model
            except Exception as e:
                self.failures += 1
                if attempt == 1:
                    return None, str(e)

        return None, "All attempts failed"

    @property
    def stats(self):
        return {"calls": self.calls, "failures": self.failures, "model": self.model}


def _try_get_source(fn) -> str:
    try:
        return inspect.getsource(fn)
    except Exception:
        return f"# {getattr(fn, '__name__', 'unknown')} (source unavailable)"


# ═══════════════════════════════════════════════════════════════════════════
#  FUNSEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class FunSearch:
    """
    Moteur FunSearch complet:
      - Programs database (seed + generated + combos)
      - Groq LLM sampler pour générer de nouvelles fonctions
      - Evaluateur sur vrais graphes
      - Combinaisons pondérées offline
      - Logging et export du meilleur code
    """

    def __init__(self,
                 n_iterations: int = 6,
                 population_size: int = 8,
                 use_groq: bool = True,
                 groq_budget: int = 8):
        self.n_iterations   = n_iterations
        self.population_size = population_size
        self.groq_budget    = groq_budget
        self.use_groq       = use_groq and _GROQ_AVAILABLE

        # DB entries: (eval_score, name, fn, code_str)
        self.db: list = [
            (0.0, fn.__name__, fn, _try_get_source(fn))
            for fn in SEED_FUNCTIONS
        ]

        self._sampler = _GroqSampler() if self.use_groq else None
        self._generated: list[str] = []  # generated code history

    # ── Public API ──────────────────────────────────────────────────────────

    def evolve(self, test_cases: list, conjecture=None):
        """Lance l'évolution et retourne la meilleure fonction callable."""
        if not test_cases:
            return _score_v4

        # 1. Evaluate seeds
        print(f"\n[FunSearch] Etape 1/3 – Seeds ({len(self.db)} fonctions)...")
        self._eval_all(test_cases)
        self._trim()
        self._print_top(3)

        # 2. Groq rounds
        if self.use_groq and conjecture is not None:
            rounds = min(self.groq_budget, self.n_iterations)
            print(f"\n[FunSearch] Etape 2/3 – Groq LLM ({rounds} appels, model={_GROQ_MODEL})...")
            for r in range(rounds):
                fn, code = self._sampler.sample(conjecture, self.db)
                if fn:
                    sc = self._eval_one(fn, test_cases)
                    name = f"groq_r{r+1}"
                    self.db.append((sc, name, fn, code))
                    self._generated.append(code)
                    self._trim()
                    print(f"  [r{r+1}] generated '{name}' score={sc:.4f} | best={self.db[0][0]:.4f}")
                else:
                    print(f"  [r{r+1}] Groq failed: {code[:70]}")
        elif not _GROQ_AVAILABLE:
            print("\n[FunSearch] Etape 2/3 – Groq non disponible (fallback offline)")
        else:
            print("\n[FunSearch] Etape 2/3 – Groq désactivé (pas de conjecture pivot)")

        # 3. Offline combinations
        offline_rounds = max(3, self.n_iterations)
        print(f"\n[FunSearch] Etape 3/3 – Combinaisons offline ({offline_rounds} rounds)...")
        for r in range(offline_rounds):
            combos = self._make_combos(test_cases)
            self.db.extend(combos)
            self._trim()
        self._print_top(5)

        best = self.db[0]
        print(f"\n[FunSearch] => Meilleure: '{best[1]}' (score={best[0]:.4f})")
        if self._sampler:
            s = self._sampler.stats
            print(f"[FunSearch] Groq: {s['calls']} appels, {s['failures']} echecs")
        return best[2]

    def get_best_function(self):
        return self.db[0][2] if self.db else _score_v4

    def get_best_code(self) -> str:
        if not self.db:
            return "# no function"
        sc, name, fn, code = self.db[0]
        header = f"# Best: {name}  (score={sc:.4f})\n"
        return header + (code if code.startswith("def ") else _try_get_source(fn))

    def get_all_generated_codes(self) -> list:
        return list(self._generated)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _eval_all(self, test_cases):
        self.db = [(self._eval_one(fn, test_cases), name, fn, code)
                   for _, name, fn, code in self.db]
        self.db.sort(key=lambda x: x[0], reverse=True)

    def _eval_one(self, fn, test_cases) -> float:
        total, count = 0.0, 0
        for G, inv, conj in test_cases:
            try:
                sc   = float(fn(G, inv, conj))
                viol = float(conj.violation(inv))
                total += sc + 10.0 * max(0.0, viol)
                count += 1
            except Exception:
                pass
        return (total / count) if count > 0 else -1e9

    def _trim(self):
        self.db.sort(key=lambda x: x[0], reverse=True)
        self.db = self.db[:self.population_size * 3]

    def _make_combos(self, test_cases) -> list:
        results = []
        pool = self.db[:min(6, len(self.db))]
        for _ in range(3):
            k   = random.randint(2, min(3, len(pool)))
            sel = random.sample(pool, k)
            w   = [random.uniform(0.2, 2.0) for _ in sel]
            s   = sum(w) or 1.0
            w   = [x / s for x in w]

            def _combo(G, inv, conj, _sel=sel, _w=w):
                t = 0.0
                for wi, (_, _, fi, _) in zip(_w, _sel):
                    try:
                        t += wi * fi(G, inv, conj)
                    except Exception:
                        pass
                return t

            sc   = self._eval_one(_combo, test_cases)
            name = f"combo_{random.randint(1000,9999)}"
            results.append((sc, name, _combo, "# weighted combo"))
        return results

    def _print_top(self, k):
        for rank, (sc, name, _, _) in enumerate(self.db[:k], 1):
            print(f"  #{rank}: '{name}'  score={sc:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_funsearch(conjectures: list,
                    n_sample: int = 8,
                    n_iterations: int = 6,
                    groq_budget: int = 8) -> FunSearch:
    """
    Initialise un FunSearch, génère des cas de test, lance l'évolution.
    Retourne le moteur FunSearch avec la meilleure fonction sélectionnée.
    """
    fs = FunSearch(n_iterations=n_iterations,
                   population_size=8,
                   use_groq=_GROQ_AVAILABLE,
                   groq_budget=groq_budget)

    sample = random.sample(conjectures, min(n_sample, len(conjectures)))
    test_cases = []
    for c in sample:
        for G in generate_initial_graphs(c, n_graphs=5):
            try:
                if satisfies_class(G, c.graph_classes):
                    inv = compute_invariants(G)
                    test_cases.append((G, inv, c))
            except Exception:
                pass

    # Pick the conjecture with the most graph-class constraints as Groq pivot
    pivot = max(sample, key=lambda c: len(c.graph_classes), default=sample[0])
    fs.evolve(test_cases, pivot)
    return fs


def search_counterexample_funsearch(conjecture,
                                     time_limit: float = 60.0,
                                     score_fn=None,
                                     verbose: bool = False):
    """Recherche de contre-exemple avec FunSearch score function."""
    from heuristic import search_counterexample
    return search_counterexample(conjecture,
                                  time_limit=time_limit,
                                  score_fn=score_fn,
                                  verbose=verbose)
