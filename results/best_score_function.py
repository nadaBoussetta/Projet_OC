# Best: groq_r6  (score=54528.6628)
def score(G, invariants, conjecture):
    """Proximity-based scoring with emphasis on vertex degree spread, network structure, and diameter."""
    v = conjecture.violation(invariants)
    n = invariants.get("n", 1)
    m = invariants.get("m", 0)
    diam = invariants.get("diameter", 0)
    Delta = invariants.get("maximum_degree", 0)
    delta = invariants.get("minimum_degree", 0)
    density = (2 * m / (n * (n - 1))) if n > 1 else 0
    avg_degree = invariants.get("average_degree", 0)
    first_zagreb_index = invariants.get("first_zagreb_index", 0)
    second_zagreb_index = invariants.get("second_zagreb_index", 0)
    triangle_number = invariants.get("triangle_number", 0)
    clique_number = invariants.get("clique_number", 0)
    independent_domination_number = invariants.get("independent_domination_number", 0)
    if v > 0:
        return 3200.0 + v * 320 + 0.96 * diam + 0.48 * Delta - 0.022 * n + 0.18 * (Delta - delta) + 0.12 * (avg_degree - delta) + 0.08 * (first_zagreb_index / (n * Delta) if n * Delta > 0 else 0) + 0.06 * (second_zagreb_index / (n * Delta) if n * Delta > 0 else 0) + 0.12 * (triangle_number / (n * (n - 1) * (n - 2) / 6) if n > 2 else 0) + 0.08 * (clique_number / n if n > 0 else 0) + 0.1 * (independent_domination_number / n if n > 0 else 0)
    proximity = max(0.0, 55.0 + v * 28)
    return 200.0 * v + proximity + 0.72 * diam + 0.58 * Delta - 0.032 * n + 0.98 * (1.0 - abs(density - 0.4)) + 0.28 * (Delta - delta) + 0.22 * (avg_degree - delta) + 0.12 * (first_zagreb_index / (n * Delta) if n * Delta > 0 else 0) + 0.08 * (second_zagreb_index / (n * Delta) if n * Delta > 0 else 0) + 0.12 * (math.log(diam + 1) if diam > 0 else 0) + 0.12 * (triangle_number / (n * (n - 1) * (n - 2) / 6) if n > 2 else 0) + 0.08 * (clique_number / n if n > 0 else 0) + 0.1 * (independent_domination_number / n if n > 0 else 0)
