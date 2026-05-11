"""
main.py – Script principal : lance la recherche sur toutes les conjectures du benchmark.
Usage:
    python main.py [--mode simple|funsearch] [--time 60] [--output results/results.json]
"""
import sys
import os
import time
import json
import argparse
import random

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conjecture import load_benchmark
from heuristic import search_counterexample
from funsearch import FunSearch, search_counterexample_funsearch


def run_benchmark(benchmark_path: str, mode: str = "simple",
                  time_limit: float = 60.0, verbose: bool = True):
    """
    Lance la recherche sur toutes les conjectures du benchmark.
    Retourne les résultats et le score total.
    """
    print(f"\n{'='*60}")
    print(f"GraphBench Challenge – Mode: {mode.upper()}")
    print(f"Limite de temps: {time_limit}s par conjecture")
    print(f"{'='*60}\n")

    conjectures = load_benchmark(benchmark_path)
    print(f"[INFO] {len(conjectures)} conjectures chargées.\n")

    results = []
    total_score = 0
    n_found = 0

    # FunSearch: initialiser le moteur d'évolution si nécessaire
    best_score_fn = None
    if mode == "funsearch":
        print("[FunSearch] Initialisation du moteur LLM...\n")
        funsearch = FunSearch(n_iterations=4, population_size=4)

        # Générer des cas de test pour FunSearch avec les premières conjectures
        from mutations import generate_initial_graphs
        from invariants import compute_invariants, satisfies_class
        test_cases = []
        sample_conjectures = random.sample(conjectures, min(5, len(conjectures)))
        for c in sample_conjectures:
            graphs = generate_initial_graphs(c, n_graphs=3)
            for G in graphs:
                try:
                    if satisfies_class(G, c.graph_classes):
                        inv = compute_invariants(G)
                        test_cases.append((G, inv, c))
                except Exception:
                    pass

        conjecture_info = (
            f"Classes: {set(g for c in conjectures for g in c.graph_classes)}, "
            f"Invariants: {set(c.x_name for c in conjectures) | set(c.y_name for c in conjectures)}"
        )
        best_score_fn = funsearch.evolve(test_cases, conjecture_info)

        # Sauvegarder le code de la meilleure fonction
        best_code = funsearch.get_best_code()
        with open("results/best_score_function.py", "w") as f:
            f.write(best_code)
        print(f"\n[FunSearch] Meilleure fonction sauvegardée dans results/best_score_function.py\n")

    # Lancer la recherche sur chaque conjecture
    for i, conj in enumerate(conjectures):
        print(f"[{i+1:3d}/{len(conjectures)}] Conjecture {conj.id}: {conj.y_name} {conj.sign} f({conj.x_name})")
        print(f"        Classes: {conj.graph_classes}")

        t_start = time.time()

        if mode == "funsearch" and best_score_fn is not None:
            result = search_counterexample_funsearch(
                conj, time_limit=time_limit,
                score_fn=best_score_fn, verbose=verbose
            )
        else:
            result = search_counterexample(conj, time_limit=time_limit, verbose=verbose)

        elapsed = time.time() - t_start

        if result is not None:
            cost = elapsed
            n_found += 1
            print(f"        ✓ CONTRE-EXEMPLE TROUVÉ en {elapsed:.2f}s")
            print(f"          violation={result.violation:.4f}")
            print(f"          graph6={result.graph6}")
            print(f"          n={result.G.number_of_nodes()}, m={result.G.number_of_edges()}")
            results.append({
                "conjecture_id": conj.id,
                "found": True,
                "time": round(elapsed, 3),
                "cost": round(cost, 3),
                "violation": round(result.violation, 6),
                "graph6": result.graph6,
                "n": result.G.number_of_nodes(),
                "m": result.G.number_of_edges(),
                "invariants": {k: round(v, 6) if isinstance(v, float) else v
                               for k, v in result.invariants.items()},
            })
        else:
            cost = 120.0
            print(f"        ✗ Non trouvé (coût = 120)")
            results.append({
                "conjecture_id": conj.id,
                "found": False,
                "time": round(elapsed, 3),
                "cost": 120.0,
                "violation": None,
                "graph6": None,
            })

        total_score += cost
        print(f"        Score cumulé: {total_score:.1f}\n")

    # Résumé
    print("\n" + "="*60)
    print("RÉSULTATS FINAUX")
    print("="*60)
    print(f"Conjectures réfutées : {n_found}/{len(conjectures)}")
    print(f"Score total          : {total_score:.1f}")
    print(f"Temps moyen (trouvés): "
          f"{sum(r['time'] for r in results if r['found']) / max(n_found, 1):.2f}s")
    print("="*60)

    return results, total_score, n_found


def main():
    parser = argparse.ArgumentParser(description="GraphBench – Réfutation automatique de conjectures")
    parser.add_argument("--mode", choices=["simple", "funsearch"], default="simple",
                        help="Mode: 'simple' (Partie 1) ou 'funsearch' (Partie 2)")
    parser.add_argument("--time", type=float, default=60.0,
                        help="Limite de temps par conjecture (secondes)")
    # Cherche benchmark.xlsx dans plusieurs emplacements possibles
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.join(_script_dir, "benchmark.xlsx"),           # même dossier
        os.path.join(_script_dir, "../benchmark/benchmark.xlsx"),  # dossier parent/benchmark
        os.path.join(_script_dir, "benchmark/benchmark.xlsx"), # sous-dossier benchmark
        "benchmark.xlsx",                                       # répertoire courant
    ]
    _default_benchmark = next((p for p in _candidates if os.path.exists(p)), _candidates[0])
    parser.add_argument("--benchmark", type=str,
                        default=_default_benchmark,
                        help="Chemin vers le fichier benchmark xlsx")
    parser.add_argument("--output", type=str, default=None,
                        help="Fichier de sortie JSON pour les résultats")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    results, total_score, n_found = run_benchmark(
        benchmark_path=args.benchmark,
        mode=args.mode,
        time_limit=args.time,
        verbose=args.verbose,
    )

    # Sauvegarder les résultats
    output_path = args.output or f"results/results_{args.mode}.json"
    with open(output_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "total_score": round(total_score, 3),
            "n_found": n_found,
            "n_total": len(results),
            "results": results,
        }, f, indent=2)

    print(f"\n[INFO] Résultats sauvegardés dans {output_path}")
    return total_score


if __name__ == "__main__":
    main()
