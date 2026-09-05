"""Overlap-honest plots: abstracted connectivity graph, density contours, small multiples."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .graph import group_centroids


def default_palette(groups) -> dict:
    """A stable colour for each group (tab20 + tab20b, then repeat)."""
    import matplotlib.pyplot as plt

    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)
    return {g: base[i % len(base)] for i, g in enumerate(groups)}


def _repel(texts, ax, xs=None, ys=None):
    """Push overlapping labels apart with adjustText if it is installed (no-op otherwise)."""
    try:
        from adjustText import adjust_text
    except ImportError:
        return
    if len(texts) < 2:
        return
    kw = dict(ax=ax, expand=(1.3, 1.6), arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5))
    if xs is not None:
        kw.update(x=list(xs), y=list(ys))
    adjust_text(texts, **kw)


def _new_axes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_connectivity(
    connectivity: dict,
    embedding: np.ndarray | None = None,
    labels=None,
    min_ratio: float = 1.0,
    palette: dict | None = None,
    node_scale: float = 6000.0,
    edge_scale: float = 3.5,
    ax=None,
    show_background: bool = True,
    fontsize: float = 8,
):
    """Draw the abstracted group graph.

    Nodes sit at the group's median position in ``embedding`` (or in a spring
    layout if no embedding is given); node area is proportional to group size;
    an edge is drawn where ``ratio >= min_ratio`` with width proportional to
    ``log(ratio)``. Reading it: thick edges are groups that phenotypically blend,
    isolated nodes are genuinely separate groups.
    """
    import networkx as nx

    ratio, sizes, groups = connectivity["ratio"], connectivity["sizes"], connectivity["labels"]
    palette = palette or default_palette(groups)
    ax = _new_axes(ax, (9, 8))

    if embedding is not None and labels is not None:
        pos_df = group_centroids(embedding, labels)
        pos = {g: pos_df.loc[g].to_numpy() for g in groups}
        if show_background:
            ax.scatter(np.asarray(embedding)[:, 0], np.asarray(embedding)[:, 1],
                       s=2, c="#dddddd", alpha=0.5, rasterized=True, zorder=0)
    else:
        g = nx.Graph()
        for a in groups:
            g.add_node(a)
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                r = float(ratio.loc[a, b])
                if r >= min_ratio and r > 0:
                    g.add_edge(a, b, weight=math.log(r + 1.0))
        # force-directed layout driven by the connectivity itself (as in PAGA);
        # k widens spacing so labels of many groups do not collide
        pos = nx.spring_layout(g, seed=0, weight="weight", k=2.2 / math.sqrt(max(len(groups), 2)), iterations=300)

    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            r = float(ratio.loc[a, b])
            if r >= min_ratio and r > 0:
                w = max(1.2, edge_scale * r)            # linear: ratio 1 -> 3.5 px, ratio 2 -> 7 px
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                        color="#333333", lw=w, alpha=0.85, zorder=1,
                        solid_capstyle="round")
                mid = (pos[a] + pos[b]) / 2.0
                ax.annotate(f"{r:.1f}", mid, fontsize=fontsize - 1, ha="center", va="center",
                            color="#333333", zorder=4,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#333333", lw=0.5, alpha=0.9))
    total = float(sizes.sum())
    texts = []
    for g in groups:
        area = node_scale * (sizes[g] / total) + 150
        ax.scatter(*pos[g], s=area, color=palette[g], edgecolors="black", linewidths=0.8,
                   alpha=0.95, zorder=2)
        radius_pt = math.sqrt(area) / 2.0
        texts.append(ax.annotate(f"{g}  (n={sizes[g]})", pos[g], fontsize=fontsize, fontweight="bold",
                    ha="center", va="bottom", xytext=(0, radius_pt + 3), textcoords="offset points",
                    zorder=3, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#888888",
                                        lw=0.4, alpha=0.9)))
    _repel(texts, ax, [pos[g][0] for g in groups], [pos[g][1] for g in groups])
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.set_xticks([]); ax.set_yticks([])
    return ax


def plot_density(
    embedding: np.ndarray,
    labels,
    groups=None,
    levels=(0.5, 0.8),
    palette: dict | None = None,
    min_size: int = 15,
    points: bool = True,
    ax=None,
):
    """Per-group kernel-density contours on a 2-D embedding.

    Each contour encloses the given fraction of that group's probability mass
    (``levels`` = highest-density regions). Overlapping contours are what a
    single scatter with twenty colours hides.
    """
    embedding = np.asarray(embedding); labels = np.asarray(labels)
    order = list(groups) if groups is not None else list(pd.unique(labels))
    palette = palette or default_palette(order)
    ax = _new_axes(ax, (9, 8))
    if points:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=2, c="#cccccc", alpha=0.5, rasterized=True, zorder=0)
    xs = np.linspace(embedding[:, 0].min(), embedding[:, 0].max(), 220)
    ys = np.linspace(embedding[:, 1].min(), embedding[:, 1].max(), 220)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.vstack([gx.ravel(), gy.ravel()])
    for g in order:
        pts = embedding[labels == g]
        if len(pts) < min_size:
            continue
        try:
            kde = gaussian_kde(pts.T)
        except np.linalg.LinAlgError:
            continue
        z = kde(grid).reshape(gx.shape)
        # thresholds enclosing the requested mass fractions
        sample = kde(pts.T)
        thresholds = [np.quantile(sample, 1 - lv) for lv in levels]
        ax.contour(gx, gy, z, levels=sorted(thresholds), colors=[palette[g]],
                   linewidths=[1.6, 0.9][: len(levels)], alpha=0.95, zorder=2)
        ax.plot([], [], color=palette[g], lw=1.6, label=f"{g} (n={len(pts)})")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, edgecolor="gray")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    return ax


def plot_small_multiples(
    embedding: np.ndarray,
    labels,
    groups=None,
    ncols: int = 5,
    palette: dict | None = None,
    panel: float = 2.6,
    point_size: float = 4,
):
    """One panel per group, that group highlighted over the grey cohort.

    The honest replacement for a legend with twenty indistinguishable colours.
    """
    import matplotlib.pyplot as plt

    embedding = np.asarray(embedding); labels = np.asarray(labels)
    order = list(groups) if groups is not None else [g for g, _ in
             sorted(pd.Series(labels).value_counts().items(), key=lambda kv: -kv[1])]
    palette = palette or default_palette(order)
    nrows = math.ceil(len(order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols, panel * nrows), dpi=200)
    axes = np.atleast_1d(axes).ravel()
    for ax, g in zip(axes, order):
        ax.scatter(embedding[:, 0], embedding[:, 1], s=1, c="#dddddd", rasterized=True)
        m = labels == g
        ax.scatter(embedding[m, 0], embedding[m, 1], s=point_size, c=[palette[g]], alpha=0.85, rasterized=True)
        ax.set_title(f"{g}  (n={int(m.sum())})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#bbbbbb")
    for ax in axes[len(order):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_connectivity_heatmap(
    connectivity: dict,
    groups=None,
    min_size: int = 0,
    vmax: float | None = None,
    ax=None,
    fontsize: float = 8,
):
    """Group × group connectivity ratio as a heatmap.

    The most readable summary once there are more than ~8 groups: every pair is
    shown, the diagonal (self-connectivity) tells how cohesive a group is, and
    values are printed in the cells. ``min_size`` drops small groups whose
    expected edge counts are tiny and whose ratios are therefore unstable.
    """
    import matplotlib.pyplot as plt

    ratio, sizes = connectivity["ratio"], connectivity["sizes"]
    order = [g for g in (groups or connectivity["labels"]) if sizes[g] >= min_size]
    m = ratio.loc[order, order].to_numpy(dtype=float)
    ax = _new_axes(ax, (0.55 * len(order) + 3, 0.5 * len(order) + 2.5))
    top = vmax if vmax is not None else max(2.0, float(np.nanpercentile(m[~np.eye(len(order), dtype=bool)], 95)))
    im = ax.imshow(np.clip(m, 0, top), cmap="magma_r", vmin=0, vmax=top, aspect="auto")
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    labs = [f"{g} (n={sizes[g]})" for g in order]
    ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=fontsize)
    ax.set_yticklabels(labs, fontsize=fontsize)
    for i in range(len(order)):
        for j in range(len(order)):
            v = m[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=fontsize - 1.5,
                    color="white" if v > 0.6 * top else "black")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    cb = plt.colorbar(im, ax=ax, shrink=0.7); cb.set_label("observed / expected k-NN edges")
    return ax
