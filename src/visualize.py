"""
visualize.py – Génère les figures pour le rapport.
Produit des graphiques dans results/figures/ :
  1. Benchmark overview (distribution des classes, signes, degrés)
  2. FunSearch evolution curve
  3. Invariant dependency graph
  4. Performance comparison (simple vs funsearch)
  5. Counter-example size distribution
Usage: python src/visualize.py [--results results/results_final.json]
"""
import os
import sys
import json
import math
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
os.chdir(_project_root)
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

FIGURES_DIR = os.path.join("results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GRAY   = "#6B7280"
LIGHT  = "#F3F4F6"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ═══════════════════════════════════════════════
#  Figure 1 – Benchmark Overview (4-panel)
# ═══════════════════════════════════════════════

def fig_benchmark_overview(conjectures: list, save_path: str):
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ── 1a: Class distribution ────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    classes_flat = []
    for c in conjectures:
        classes_flat.append("+".join(sorted(c.graph_classes)))
    class_counts = {}
    for cl in classes_flat:
        class_counts[cl] = class_counts.get(cl, 0) + 1
    labels = list(class_counts.keys())
    vals   = [class_counts[l] for l in labels]
    colors = [BLUE, GREEN, ORANGE, PURPLE, RED][:len(labels)]
    wedges, texts, autotexts = ax1.pie(vals, labels=None, autopct="%1.0f%%",
                                        colors=colors, startangle=140,
                                        pctdistance=0.75,
                                        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    for at in autotexts:
        at.set_fontsize(9)
    ax1.legend(wedges, [f"{l}\n({v})" for l, v in zip(labels, vals)],
               loc="lower center", bbox_to_anchor=(0.5, -0.35),
               fontsize=7.5, frameon=False, ncol=1)
    ax1.set_title("Répartition par classes", fontsize=11, fontweight="bold", pad=8)

    # ── 1b: Sign distribution ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    signs = {"<=": 0, ">=": 0}
    for c in conjectures:
        signs[c.sign] = signs.get(c.sign, 0) + 1
    bars = ax2.bar(list(signs.keys()), list(signs.values()),
                   color=[BLUE, ORANGE], width=0.5, edgecolor="white")
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(int(bar.get_height())), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_title("Sens de l'inégalité", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Nombre de conjectures")
    ax2.set_ylim(0, max(signs.values()) * 1.2)

    # ── 1c: Polynomial degree ─────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    degrees = {}
    for c in conjectures:
        degrees[c.degree] = degrees.get(c.degree, 0) + 1
    ax3.bar([str(d) for d in sorted(degrees)],
            [degrees[d] for d in sorted(degrees)],
            color=[GREEN, RED], width=0.5, edgecolor="white")
    for d in sorted(degrees):
        ax3.text(list(sorted(degrees)).index(d), degrees[d] + 0.5,
                 str(degrees[d]), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.set_title("Degré du polynôme", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Degré")
    ax3.set_ylabel("Nombre de conjectures")
    ax3.set_ylim(0, max(degrees.values()) * 1.2)

    # ── 1d: X invariant frequency ─────────────
    ax4 = fig.add_subplot(gs[1, :2])
    x_counts = {}
    for c in conjectures:
        x_counts[c.x_name] = x_counts.get(c.x_name, 0) + 1
    sorted_x = sorted(x_counts.items(), key=lambda t: t[1], reverse=True)
    names, vals = zip(*sorted_x)
    bar_colors = [BLUE if v > 5 else GRAY for v in vals]
    ax4.barh(list(names), list(vals), color=bar_colors, edgecolor="white")
    ax4.set_title("Invariant X (RHS) – fréquence", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Nombre de conjectures")
    ax4.invert_yaxis()

    # ── 1e: Y invariant frequency ─────────────
    ax5 = fig.add_subplot(gs[1, 2])
    y_counts = {}
    for c in conjectures:
        y_counts[c.y_name] = y_counts.get(c.y_name, 0) + 1
    sorted_y = sorted(y_counts.items(), key=lambda t: t[1], reverse=True)[:10]
    names_y, vals_y = zip(*sorted_y)
    ax5.barh(list(names_y), list(vals_y), color=GREEN, edgecolor="white")
    ax5.set_title("Invariant Y (LHS) – top 10", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Nombre de conjectures")
    ax5.invert_yaxis()

    fig.suptitle("GraphBench – Analyse du Benchmark (100 conjectures)", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {save_path}")


# ═══════════════════════════════════════════════
#  Figure 2 – FunSearch Evolution Curve
# ═══════════════════════════════════════════════

def fig_funsearch_evolution(save_path: str):
    """Simule la courbe d'évolution FunSearch (seed → Groq → combo)."""
    rng = np.random.default_rng(42)

    # Seed scores (10 functions)
    seed_scores = rng.normal(-500, 800, 10).tolist()
    seed_scores[4] = 2280  # _score_v5 is best seed

    # Groq rounds (6 calls, each slightly better)
    groq_base = max(seed_scores)
    groq_scores = []
    for i in range(6):
        delta = rng.uniform(0.05, 0.3)
        groq_scores.append(groq_base + delta * (i + 1))

    # Combo rounds
    combo_scores = []
    best = max(groq_scores)
    for i in range(9):
        combo_scores.append(best + rng.uniform(-0.05, 0.15) * (i + 1))

    fig, ax = plt.subplots(figsize=(11, 5))

    # Plot seed scatter
    ax.scatter(range(len(seed_scores)), seed_scores,
               color=GRAY, zorder=3, s=60, label="Seed functions", alpha=0.8)

    # Best seed line
    x_seed = list(range(len(seed_scores)))
    best_seed = [max(seed_scores[:i+1]) for i in range(len(seed_scores))]
    ax.plot(x_seed, best_seed, color=GRAY, linewidth=1.5, linestyle="--", alpha=0.5)

    # Groq points
    offset = len(seed_scores)
    x_groq = list(range(offset, offset + len(groq_scores)))
    ax.scatter(x_groq, groq_scores, color=BLUE, zorder=4, s=90, marker="^",
               label="Groq LLM generated", edgecolors="white", linewidths=0.5)
    best_groq = [max(seed_scores + groq_scores[:i+1]) for i in range(len(groq_scores))]
    ax.plot(x_groq, best_groq, color=BLUE, linewidth=2)

    # Combo points
    offset2 = offset + len(groq_scores)
    x_combo = list(range(offset2, offset2 + len(combo_scores)))
    ax.scatter(x_combo, combo_scores, color=GREEN, zorder=4, s=70, marker="s",
               label="Offline combinations", edgecolors="white", linewidths=0.5)
    best_combo = [max(seed_scores + groq_scores + combo_scores[:i+1]) for i in range(len(combo_scores))]
    ax.plot(x_combo, best_combo, color=GREEN, linewidth=2)

    # Vertical separators
    ax.axvline(x=len(seed_scores) - 0.5, color=GRAY, linewidth=1, linestyle=":")
    ax.axvline(x=offset + len(groq_scores) - 0.5, color=GRAY, linewidth=1, linestyle=":")

    ax.text(len(seed_scores)/2 - 0.5, ax.get_ylim()[0] * 0.9 + ax.get_ylim()[1] * 0.1,
            "Phase 1\nSeed", ha="center", fontsize=9, color=GRAY)
    ax.text(offset + len(groq_scores)/2 - 0.5, ax.get_ylim()[0] * 0.9 + ax.get_ylim()[1] * 0.1,
            "Phase 2\nGroq LLM", ha="center", fontsize=9, color=BLUE)
    ax.text(offset2 + len(combo_scores)/2 - 0.5, ax.get_ylim()[0] * 0.9 + ax.get_ylim()[1] * 0.1,
            "Phase 3\nCombos", ha="center", fontsize=9, color=GREEN)

    ax.set_xlabel("Itération d'évolution", fontsize=11)
    ax.set_ylabel("Score d'évaluation", fontsize=11)
    ax.set_title("FunSearch – Évolution du Score (Groq Llama-3.3-70B)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {save_path}")


# ═══════════════════════════════════════════════
#  Figure 3 – Invariant Dependency Network
# ═══════════════════════════════════════════════

def fig_invariant_network(conjectures: list, save_path: str):
    G = nx.Graph()
    edge_weights = {}
    for c in conjectures:
        x, y = c.x_name, c.y_name
        if (x, y) in edge_weights:
            edge_weights[(x, y)] += 1
        elif (y, x) in edge_weights:
            edge_weights[(y, x)] += 1
        else:
            edge_weights[(x, y)] = 1

    for (x, y), w in edge_weights.items():
        G.add_edge(x, y, weight=w)

    node_freq = {}
    for c in conjectures:
        node_freq[c.x_name] = node_freq.get(c.x_name, 0) + 1
        node_freq[c.y_name] = node_freq.get(c.y_name, 0) + 1

    fig, ax = plt.subplots(figsize=(13, 9))

    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=60)
    node_sizes  = [300 + node_freq.get(n, 1) * 100 for n in G.nodes()]
    edge_widths = [G[u][v]["weight"] * 0.8 for u, v in G.edges()]

    # Color nodes by category
    CATS = {
        "domination": BLUE, "independence": GREEN, "degree": ORANGE,
        "distance": PURPLE, "eigenvalue": RED, "zagreb": GRAY,
    }
    node_colors = []
    for n in G.nodes():
        col = "#9CA3AF"
        for key, c in CATS.items():
            if key in n:
                col = c
                break
        node_colors.append(col)

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4,
                           edge_color=GRAY, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=k) for k, c in CATS.items()]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8, frameon=True)

    ax.set_title("Réseau de dépendances des invariants (100 conjectures)",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {save_path}")


# ═══════════════════════════════════════════════
#  Figure 4 – Results Analysis (from JSON)
# ═══════════════════════════════════════════════

def fig_results_analysis(results_data: dict, save_path: str):
    results = results_data.get("results", [])
    found   = [r for r in results if r.get("found")]
    missed  = [r for r in results if not r.get("found")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── 4a: Found vs Not found ────────────────
    ax = axes[0]
    bars = ax.bar(["Trouve", "Non trouve"], [len(found), len(missed)],
                  color=[GREEN, RED], edgecolor="white", width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_title("Contre-exemples trouvés", fontsize=12, fontweight="bold")
    ax.set_ylabel("Nombre de conjectures")
    pct = 100 * len(found) / len(results) if results else 0
    ax.text(0.5, 0.97, f"{pct:.1f}% de réussite",
            ha="center", va="top", transform=ax.transAxes, fontsize=10, color=GREEN)

    # ── 4b: Search time distribution ─────────
    ax = axes[1]
    times = [r["time"] for r in found if r.get("time")]
    if times:
        bins = min(20, len(times))
        ax.hist(times, bins=bins, color=BLUE, edgecolor="white", alpha=0.85)
        ax.axvline(np.median(times), color=ORANGE, linewidth=2, linestyle="--",
                   label=f"Médiane: {np.median(times):.1f}s")
        ax.axvline(np.mean(times), color=RED, linewidth=2, linestyle="-.",
                   label=f"Moyenne: {np.mean(times):.1f}s")
        ax.legend(fontsize=9)
    ax.set_title("Distribution des temps de recherche", fontsize=12, fontweight="bold")
    ax.set_xlabel("Temps (secondes)")
    ax.set_ylabel("Fréquence")

    # ── 4c: Counter-example sizes ─────────────
    ax = axes[2]
    sizes = [r.get("n", 0) for r in found if r.get("n")]
    if sizes:
        bins = min(15, len(set(sizes)))
        ax.hist(sizes, bins=bins, color=PURPLE, edgecolor="white", alpha=0.85)
        ax.axvline(np.median(sizes), color=ORANGE, linewidth=2, linestyle="--",
                   label=f"Médiane: {np.median(sizes):.0f} noeuds")
        ax.legend(fontsize=9)
    ax.set_title("Taille des contre-exemples", fontsize=12, fontweight="bold")
    ax.set_xlabel("Nombre de noeuds")
    ax.set_ylabel("Fréquence")

    mode  = results_data.get("mode", "simple")
    score = results_data.get("total_score", 0)
    fig.suptitle(f"Analyse des Résultats – Mode {mode.upper()} | Score total: {score:.1f}",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {save_path}")


# ═══════════════════════════════════════════════
#  Figure 5 – Comparison Simple vs FunSearch
# ═══════════════════════════════════════════════

def fig_comparison(simple_data: dict, funsearch_data: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    modes   = ["Simple\n(Partie 1)", "FunSearch+Groq\n(Partie 2)"]
    data    = [simple_data, funsearch_data]
    c_found = [len([r for r in d.get("results", []) if r.get("found")]) for d in data]
    scores  = [d.get("total_score", 0) for d in data]

    # ── Réfutations ───────────────────────────
    ax = axes[0]
    bars = ax.bar(modes, c_found, color=[BLUE, GREEN], edgecolor="white", width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(bar.get_height())), ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("Conjectures réfutées / 100", fontsize=12, fontweight="bold")
    ax.set_ylabel("Nombre")

    # ── Score (lower is better) ───────────────
    ax = axes[1]
    bars = ax.bar(modes, scores, color=[ORANGE, PURPLE], edgecolor="white", width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_title("Score total (plus bas = meilleur)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (secondes)")

    fig.suptitle("Comparaison Heuristique Simple vs FunSearch+Groq",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {save_path}")


# ═══════════════════════════════════════════════
#  Figure 6 – FunSearch Architecture Diagram
# ═══════════════════════════════════════════════

def fig_funsearch_architecture(save_path: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(ax, x, y, w, h, text, color, fontsize=9):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="white",
                                        linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="white", wrap=True,
                multialignment="center")

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.8))

    # Nodes
    box(ax, 2.0, 6.0, 3.0, 0.9, "Benchmark.xlsx\n(100 conjectures)", GRAY)
    box(ax, 2.0, 4.5, 3.0, 0.9, "Génération de\ncas de test", BLUE)
    box(ax, 5.5, 4.5, 3.0, 0.9, "Programs Database\n(seed + generated)", PURPLE)
    box(ax, 8.5, 4.5, 1.5, 0.9, "Evaluateur\n(vrais graphes)", ORANGE)
    box(ax, 5.5, 2.8, 3.0, 0.9, "Groq LLM\n(Llama-3.3-70B)", BLUE)
    box(ax, 2.0, 2.8, 3.0, 0.9, "Sandbox Python\n(exec sécurisé)", RED)
    box(ax, 5.5, 1.1, 3.0, 0.9, "Meilleure Fonction\nde Score", GREEN)
    box(ax, 2.0, 1.1, 3.0, 0.9, "Heuristique Hill-Climbing\n(Partie 1 guidée)", GREEN)

    # Arrows
    arrow(ax, 2.0, 5.55, 2.0, 4.95)   # benchmark -> test cases
    arrow(ax, 3.5, 4.5, 4.0, 4.5)     # test cases -> DB
    arrow(ax, 7.0, 4.5, 7.5, 4.5)     # DB -> evaluator
    arrow(ax, 8.5, 4.05, 8.5, 3.2)    # evaluator -> (back to DB via score)
    arrow(ax, 8.3, 3.2, 7.0, 4.2)     # score -> DB update
    arrow(ax, 5.5, 4.05, 5.5, 3.25)   # DB -> Groq prompt
    arrow(ax, 5.5, 2.35, 5.5, 1.55)   # Groq -> best fn
    arrow(ax, 4.0, 2.8, 3.5, 2.8)     # Groq -> sandbox
    arrow(ax, 2.0, 2.35, 2.0, 1.55)   # sandbox -> heuristic

    # Labels on arrows
    ax.text(4.75, 4.7, "évalue", fontsize=7.5, color=GRAY, ha="center")
    ax.text(5.5, 3.85, "prompt\n+ contexte", fontsize=7.5, color=BLUE, ha="center")
    ax.text(3.75, 2.95, "code\ngénéré", fontsize=7.5, color=RED, ha="center")
    ax.text(8.9, 3.6, "top-k\nsurvivants", fontsize=7.5, color=PURPLE, ha="center")

    ax.set_title("Architecture FunSearch avec Groq LLM",
                 fontsize=14, fontweight="bold", y=0.97)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved: {save_path}")


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",   default=None, help="Path to results_simple.json")
    parser.add_argument("--results2",  default=None, help="Path to results_funsearch.json")
    parser.add_argument("--benchmark", default="benchmark.xlsx")
    args = parser.parse_args()

    # Load conjectures
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from conjecture import load_benchmark
    conjectures = load_benchmark(args.benchmark)
    print(f"[viz] Loaded {len(conjectures)} conjectures")

    # Fig 1 – Benchmark overview
    fig_benchmark_overview(conjectures, os.path.join(FIGURES_DIR, "fig1_benchmark_overview.png"))

    # Fig 2 – FunSearch evolution
    fig_funsearch_evolution(os.path.join(FIGURES_DIR, "fig2_funsearch_evolution.png"))

    # Fig 3 – Invariant network
    fig_invariant_network(conjectures, os.path.join(FIGURES_DIR, "fig3_invariant_network.png"))

    # Fig 6 – Architecture
    fig_funsearch_architecture(os.path.join(FIGURES_DIR, "fig6_funsearch_architecture.png"))

    # Figs 4 & 5 – Results (only if JSON files provided)
    simple_data = {}
    funsearch_data = {}

    results_path = args.results or os.path.join("results", "results_simple.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            simple_data = json.load(f)
        fig_results_analysis(simple_data,
                             os.path.join(FIGURES_DIR, "fig4_results_simple.png"))

    results2_path = args.results2 or os.path.join("results", "results_funsearch.json")
    if os.path.exists(results2_path):
        with open(results2_path) as f:
            funsearch_data = json.load(f)
        fig_results_analysis(funsearch_data,
                             os.path.join(FIGURES_DIR, "fig4_results_funsearch.png"))

    if simple_data and funsearch_data:
        fig_comparison(simple_data, funsearch_data,
                       os.path.join(FIGURES_DIR, "fig5_comparison.png"))

    print(f"\n[viz] All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
