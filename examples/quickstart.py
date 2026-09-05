"""Generate every figure shown in the README from a synthetic cohort.

    python examples/quickstart.py

Nothing here needs data files. The synthetic cohort has a *known* structure
(see ``phenotopo.data.synthetic_cohort``), so each figure can be read against
ground truth: Neuro and Devel are blended, Bone is an island, Skin touches
Neuro, Renal is a tight satellite.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phenotopo import (
    group_connectivity,
    plot_connectivity_heatmap,
    knn_graph,
    plot_connectivity,
    plot_density,
    plot_small_multiples,
)
from phenotopo.data import synthetic_cohort, synthetic_hierarchy, synthetic_term_lists

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig_or_ax, name):
    fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


emb, labels, distance = synthetic_cohort(n=900, seed=0)

# 1. PAGA-style connectivity: which groups blend, which are separate
conn = group_connectivity(knn_graph(distance=distance, k=15), labels)
print("\nconnectivity (observed / expected k-NN edges):")
print(conn["ratio"].round(2).to_string())
save(plot_connectivity(conn, embedding=emb, labels=labels, min_ratio=0.5), "connectivity.png")
save(plot_connectivity_heatmap(conn), "connectivity_heatmap.png")

# 2. density contours: overlap made visible
save(plot_density(emb, labels, levels=(0.5, 0.85)), "density.png")

# 3. small multiples: one group per panel instead of one legend
save(plot_small_multiples(emb, labels, ncols=5), "small_multiples.png")

# 4. hyperbolic disk: specificity as a radial axis (needs gensim)
try:
    from phenotopo.hyperbolic import place_patients, plot_disk, poincare_terms, radial_specificity

    relations, leaves = synthetic_hierarchy(branches=5, depth=3, fanout=3, seed=0)
    term_lists = synthetic_term_lists(leaves, labels, seed=0)
    coords = poincare_terms(relations, epochs=60, seed=0)
    depth_weight = {t: 1.0 + t.count(".") for t in coords}      # deeper = more specific
    pts = place_patients(term_lists, coords, weights=depth_weight)
    landmarks = {f"branch {b}": coords[b] for b in leaves}
    ax = plot_disk(pts, labels, landmarks=landmarks)
    r = radial_specificity(pts)
    ax.set_title(f"Poincaré disk - radius = specificity (median {np.median(r):.2f})", fontsize=10)
    save(ax, "hyperbolic.png")
except ImportError as e:
    print("skipping hyperbolic figure:", e)

# 5. Mapper graph coloured by a per-patient outcome (needs kmapper)
try:
    from phenotopo.mapper import mapper_graph, node_values, plot_mapper

    rng = np.random.RandomState(0)
    yield_by_group = {"Neuro": 0.5, "Devel": 0.55, "Bone": 0.85, "Skin": 0.45, "Renal": 0.2}
    solved = np.array([rng.rand() < yield_by_group[g] for g in labels], dtype=float)
    _, g = mapper_graph(emb, n_cubes=12, perc_overlap=0.35)
    ax = plot_mapper(g, node_values(g, solved), cmap="viridis", label="fraction solved in node")
    ax.set_title("Mapper graph - shape of the cohort, coloured by diagnostic yield", fontsize=10)
    save(ax, "mapper.png")
except ImportError as e:
    print("skipping mapper figure:", e)
