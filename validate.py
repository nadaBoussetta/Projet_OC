"""
validate.py – Valide les contre-exemples trouvés et génère un rapport de validation.
Usage: python validate.py results/results_simple.json
"""
import sys
import json
import networkx as nx
from conjecture import load_benchmark
from invariants import compute_invariants, satisfies_class


def validate_counterexample(conj, graph6: str):
    """Valide un contre-exemple selon les règles du projet."""
    errors = []
    warnings = []

    # 1. Décoder le graphe depuis graph6
    try:
        G = nx.from_graph6_bytes(graph6.encode())
    except Exception as e:
        errors.append(f"Impossible de décoder graph6: {e}")
        return False, errors, warnings, None, None

    # 2. Vérifier la classe
    if not satisfies_class(G, conj.graph_classes):
        errors.append(f"Le graphe ne satisfait pas la classe {conj.graph_classes}")
        return False, errors, warnings, G, None

    # 3. Calculer les invariants
    try:
        inv = compute_invariants(G)
    except Exception as e:
        errors.append(f"Erreur de calcul des invariants: {e}")
        return False, errors, warnings, G, None

    # 4. Vérifier la violation stricte
    viol = conj.violation(inv)
    if viol <= 0:
        errors.append(f"Violation non strictement positive: {viol:.6f}")
        return False, errors, warnings, G, inv

    return True, errors, warnings, G, inv


def validate_results(results_path: str, benchmark_path: str):
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

    for r in results:
        if not r.get("found"):
            continue

        conj_id = r["conjecture_id"]
        graph6 = r.get("graph6")

        if conj_id not in conjectures:
            print(f"[WARN] Conjecture {conj_id} non trouvée dans le benchmark")
            continue

        conj = conjectures[conj_id]
        ok, errors, warnings, G, inv = validate_counterexample(conj, graph6)

        if ok:
            valid_count += 1
            print(f"  ✓ Conjecture {conj_id}: VALIDE (violation={conj.violation(inv):.4f})")
        else:
            invalid_count += 1
            print(f"  ✗ Conjecture {conj_id}: INVALIDE")
            for err in errors:
                print(f"      ERROR: {err}")

    print(f"\n{'='*60}")
    print(f"Valides:   {valid_count}")
    print(f"Invalides: {invalid_count}")
    print(f"{'='*60}\n")

    return valid_count, invalid_count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate.py results/results_simple.json")
        sys.exit(1)

    results_path = sys.argv[1]
    benchmark_path = "benchmark.xlsx"
    validate_results(results_path, benchmark_path)
