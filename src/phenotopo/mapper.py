"""Topological (Mapper) graph of a cohort - the shape of the continuum without forcing clusters."""

from __future__ import annotations

import numpy as np


def mapper_graph(
    X: np.ndarray,
    lens: np.ndarray | None = None,
    n_cubes: int = 12,
    perc_overlap: float = 0.35,
    clusterer=None,
):
    """Build a Mapper graph with KeplerMapper (``tda`` extra).

    ``X`` are the points to cluster locally (an embedding or feature matrix);
    ``lens`` is the filter function projected to 1-2 dimensions (defaults to the
    first two columns of ``X``). Returns ``(graph_dict, networkx_graph)``; every
    node carries a ``members`` attribute with the row indices it covers.
    """
    try:
        import kmapper as km
    except ImportError as e:  # pragma: no cover
        raise ImportError("install the 'tda' extra: pip install phenotopo[tda]") from e
    import networkx as nx
    from sklearn.cluster import DBSCAN

    X = np.asarray(X)
    lens = X[:, :2] if lens is None else np.asarray(lens)
    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(
        lens, X,
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
        clusterer=clusterer or DBSCAN(eps=0.5, min_samples=3),
    )
    g = nx.Graph()
    for node, members in graph["nodes"].items():
        g.add_node(node, members=list(members))
    for a, bs in graph["links"].items():
        for b in bs:
            g.add_edge(a, b)
    return graph, g


def node_values(g, values, reducer=np.mean) -> dict:
    """Aggregate a per-point value (e.g. diagnostic yield) over each node's members."""
    values = np.asarray(values, dtype=float)
    return {n: float(reducer(values[g.nodes[n]["members"]])) for n in g.nodes}


def plot_mapper(g, node_color: dict | None = None, cmap: str = "viridis", label: str = "",
                seed: int = 0, ax=None, node_scale: float = 6.0):
    """Draw the Mapper graph; node size = members, colour = ``node_color``."""
    import matplotlib.pyplot as plt
    import networkx as nx

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8), dpi=200)
    pos = nx.spring_layout(g, seed=seed, k=0.6)
    sizes = [node_scale * len(g.nodes[n]["members"]) + 15 for n in g.nodes]
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.3, width=0.7, edge_color="#555555")
    if node_color is None:
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color="#2980b9", alpha=0.85)
    else:
        colours = [node_color[n] for n in g.nodes]
        nodes = nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color=colours,
                                       cmap=cmap, alpha=0.9, edgecolors="black", linewidths=0.4)
        cb = plt.colorbar(nodes, ax=ax, shrink=0.7)
        cb.set_label(label)
    ax.axis("off")
    return ax
