"""
validate.py – Valide les contre-exemples trouvés.
Usage: python src/validate.py results/results_final.json [--benchmark benchmark/benchmark.xlsx]
"""
import sys
import json
import os
import networkx as nx

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
os.chdir(_project_root)

from conjecture import load_benchmark
from invariants import compute_invariants, satisfies_class


def validate_counterexample(conj, graph6: str):
    """Valide un contre-exemple selon les règles du projet."""
    errors = []

    # 1. Décoder graph6
    try:
        G = nx.from_graph6_bytes(graph6.encode())
    except Exception as e:
        errors.append(f"Impossible de décoder graph6: {e}")
        return False, errors, None, None

    # 2. Vérifier la classe
    if not satisfies_class(G, conj.graph_classes):
        errors.append(f"Ne satisfait pas la classe {conj.graph_classes}")
        return False, errors, G, None

    # 3. Calculer les invariants
    try:
        inv = compute_invariants(G)
    except Exception as e:
        errors.append(f"Erreur calcul invariants: {e}")
        return False, errors, G, None

    # 4. Vérifier violation stricte
    viol = conj.violation(inv)
    if viol <= 0:
        errors.append(f"Pas de violation stricte: {viol:.6f}")
        return False, errors, G, inv

    return True, errors, G, inv


def validate_results(results_path: str, benchmark_path: str = "benchmark.xlsx"):
    """Valide tous les contre-exemples d'un fichier de résultats."""
    with open(results_path) as f:
        data = json.load(f)

    conjectures = {c.id: c for c in load_benchmark(benchmark_path)}
    results = data["results"]

    print(f"\n{'='*60}")
    print(f"VALIDATION DES CONTRE-EXEMPLES")
    print(f"Fichier: {results_path}")
    print(f"{'='*60}\n")

    valid_count = 0
    invalid_count = 0
    total_time = 0

    for r in results:
        if not r.get("found"):
            continue

        conj_id = r["conjecture_id"]
        graph6 = r.get("graph6")

        if conj_id not in conjectures:
            print(f"[WARN] Conjecture {conj_id} non trouvée")
            continue

        conj = conjectures[conj_id]
        ok, errors, G, inv = validate_counterexample(conj, graph6)

        if ok:
            valid_count += 1
            total_time += r.get("time", 0)
            print(f"  [OK] Conj {conj_id}: VALIDE (violation={conj.violation(inv):.4f}, t={r.get('time', 0):.2f}s)")
        else:
            invalid_count += 1
            print(f"  [FAIL] Conj {conj_id}: INVALIDE")
            for err in errors:
                print(f"      {err}")

    total_found = valid_count + invalid_count
    score = total_time + (len(results) - valid_count) * 120
    print(f"\n{'='*60}")
    print(f"Valides  : {valid_count}/{total_found}")
    print(f"Invalides: {invalid_count}/{total_found}")
    print(f"Score    : {score:.1f}")
    print(f"{'='*60}\n")

    return valid_count, invalid_count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate.py results/results_simple.json [--benchmark path]")
        sys.exit(1)

    results_path = sys.argv[1]
    benchmark_path = None
    if "--benchmark" in sys.argv:
        idx = sys.argv.index("--benchmark")
        if idx + 1 < len(sys.argv):
            benchmark_path = sys.argv[idx + 1]
    if benchmark_path is None:
        for p in ["benchmark/benchmark.xlsx", "benchmark.xlsx"]:
            if os.path.exists(p):
                benchmark_path = p
                break
        else:
            benchmark_path = "benchmark/benchmark.xlsx"

    validate_results(results_path, benchmark_path)
