"""Multi-configuration robustness protocol for group connectivity.

A single connectivity ratio depends on choices nobody should trust blindly:
the distance, the neighbourhood size and the sample. This module recomputes
every ratio under a grid of distances x k, attaches a permutation q-value and a
bootstrap CI to each, and reports a relationship only if it holds in *every*
configuration. Small groups are excluded up front because their expected edge
counts are tiny and their ratios explode.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import knn_graph
from .stats import bootstrap_ratio, permutation_test


def connectivity_robustness(
    distances: dict,
    labels,
    ks=(10, 15, 30),
    min_size: int = 100,
    n_perm: int = 1000,
    n_boot: int = 200,
    blend: float = 1.5,
    separation: float = 0.5,
    alpha: float = 0.05,
    random_state: int = 0,
    primary: tuple | None = None,
) -> dict:
    """Run the protocol.

    Parameters
    ----------
    distances
        ``{"name": square distance matrix}`` - e.g. cosine and SimGIC.
    labels
        Group label per row of the matrices.
    ks
        Neighbourhood sizes to sweep.
    min_size
        Groups smaller than this are dropped before anything is computed.
    blend, separation
        Effect-size thresholds. *blend* / *cohesive* needs ratio > ``blend`` with
        the bootstrap CI above 1 and q < ``alpha``; *separation* needs
        ratio < ``separation`` and q < ``alpha`` - in every configuration.
    primary
        ``(distance_name, k)`` reported as the headline estimate; defaults to the
        first distance and the middle k.

    Returns
    -------
    dict with ``table`` (one row per pair x configuration), ``summary`` (one row
    per pair with the verdict), ``groups``, ``n_patients``, ``configs``, ``primary``.
    """
    labels = np.asarray(labels)
    sizes = pd.Series(labels).value_counts()
    groups = [g for g in sizes.index if sizes[g] >= min_size]
    if len(groups) < 2:
        raise ValueError(f"fewer than two groups with n >= {min_size}")
    keep = np.isin(labels, groups)
    lab = labels[keep]
    names = list(distances)
    primary = primary or (names[0], list(ks)[len(ks) // 2])

    rows = []
    for name in names:
        d = np.asarray(distances[name])[np.ix_(keep, keep)]
        for k in ks:
            adj = knn_graph(distance=d, k=k)
            sig = permutation_test(adj, lab, n_perm=n_perm, random_state=random_state, alpha=alpha)
            boot = bootstrap_ratio(adj, lab, n_boot=n_boot, random_state=random_state)
            for _, r in sig["pairs"].iterrows():
                a, b = r.group_a, r.group_b
                rows.append({
                    "distance": name, "k": k, "group_a": a, "group_b": b, "self": a == b,
                    "ratio": r.ratio,
                    "ci_lo": float(boot["ci_lo95"].loc[a, b]), "ci_hi": float(boot["ci_hi95"].loc[a, b]),
                    "null_lo": r.null_lo95, "null_hi": r.null_hi95, "z": r.z, "q": r.q_BH,
                })
    table = pd.DataFrame(rows)
    table["pair"] = [f"{a} (self)" if a == b else f"{a} – {b}" for a, b in zip(table.group_a, table.group_b)]

    def verdict(g):
        strong = bool(((g.ratio > blend) & (g.ci_lo > 1.0) & (g.q < alpha)).all())
        sep = bool(((g.ratio < separation) & (g.q < alpha)).all())
        if bool(g["self"].iloc[0]):
            if strong:
                return "cohesive"
            return "diffuse" if bool((g.ratio < blend).all()) else "not robust"
        return "blend" if strong else "separation" if sep else "not robust"

    prim = table[(table.distance == primary[0]) & (table.k == primary[1])].set_index("pair")
    gb = table.groupby("pair")
    summary = pd.DataFrame({
        "self": gb["self"].first(),
        "n_a": gb["group_a"].first().map(sizes), "n_b": gb["group_b"].first().map(sizes),
        "ratio": prim["ratio"], "ci_lo": prim["ci_lo"], "ci_hi": prim["ci_hi"],
        "ratio_min": gb["ratio"].min(), "ratio_max": gb["ratio"].max(),
        "q_max": gb["q"].max(), "n_configs_sig": gb["q"].apply(lambda s: int((s < alpha).sum())),
        "verdict": gb.apply(verdict),
    }).sort_values(["self", "ratio"], ascending=[True, False])
    return {"table": table, "summary": summary, "groups": groups, "n_patients": int(keep.sum()),
            "configs": [(n, k) for n in names for k in ks], "primary": primary,
            "thresholds": {"blend": blend, "separation": separation, "alpha": alpha, "min_size": min_size}}


def plot_forest(result: dict, short_names: dict | None = None, ax=None, colours=None):
    """Forest plot of every pair under every configuration, with verdicts.

    One row per pair (between-system rows above the rule, within-system below);
    for each row one point + 95 % bootstrap CI per configuration - colour =
    distance, size = k. Points that agree across configurations are the visual
    proof of robustness; the ✓ verdicts are the protocol's conclusion.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    table, summary = result["table"], result["summary"]
    names = list(dict.fromkeys(table.distance)); ks = sorted(table.k.unique())
    palette = colours or dict(zip(names, ["#1f5fa8", "#d9731a", "#2e8b57", "#8e44ad"]))
    sn = short_names or {}
    label_of = lambda pair: pair if not sn else (lambda a, b: f"{sn.get(a, a)} – {sn.get(b, b)}" if b else f"{sn.get(a, a)} (self)")(
        *((pair.split(" – ") + [None])[:2] if " – " in pair else (pair.replace(" (self)", ""), None)))
    off_rows = summary[~summary["self"]].index.tolist(); self_rows = summary[summary["self"]].index.tolist()
    order = off_rows + self_rows
    th = result["thresholds"]
    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, 0.42 * len(order) + 2.2), dpi=300)
    offsets = {k: o for k, o in zip(ks, np.linspace(-0.22, 0.22, len(ks)))}
    msize = {k: s for k, s in zip(ks, np.linspace(4, 8, len(ks)))}
    for yi, pr in enumerate(order):
        y = len(order) - 1 - yi
        for _, r in table[table.pair == pr].iterrows():
            yy = y + offsets[r.k]
            ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=palette[r.distance], lw=1.0, alpha=0.45, solid_capstyle="round")
            ax.plot(r.ratio, yy, "o", ms=msize[r.k], color=palette[r.distance], mec="white", mew=0.4)
        v = summary.loc[pr, "verdict"]
        if v != "not robust":
            colour = {"blend": "#1a7f37", "cohesive": "#1a7f37", "separation": "#8a2f2f", "diffuse": "#6c6c6c"}[v]
            ax.text(0.031, y, "✓ " + v, fontsize=7.5, fontweight="bold", va="center", ha="left", color=colour)
    for x, ls in ((1.0, "-"), (th["blend"], "--"), (th["separation"], "--")):
        ax.axvline(x, color="gray", lw=0.9 if x == 1 else 0.7, ls=ls, zorder=0)
    if self_rows and off_rows:
        ax.axhline(len(self_rows) - 0.5, color="black", lw=0.6)
        ax.text(0.5, len(order) - 0.5 + 0.55, "between groups", fontsize=8, style="italic", color="gray", va="center")
        ax.text(0.5, len(self_rows) - 0.5 - 0.45, "within group (cohesion)", fontsize=8, style="italic", color="gray", va="top")
    ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels([label_of(p) for p in order], fontsize=8.5)
    ax.set_xscale("log")
    lo = max(0.02, float(np.nanmin(table.ci_lo[table.ci_lo > 0])) * 0.6) if (table.ci_lo > 0).any() else 0.03
    ax.set_xlim(min(0.03, lo), float(table.ci_hi.max()) * 1.8)
    ticks = [t for t in [0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 5, 10, 20, 50] if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
    ax.set_xlabel("connectivity ratio = observed / expected k-NN edges  (log scale; 95 % bootstrap CI)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=palette[n], marker="o", ls="", label=n) for n in names]
    handles += [Line2D([], [], color="gray", marker="o", ms=msize[k], ls="", label=f"k = {k}") for k in ks]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", frameon=True, edgecolor="gray", ncol=2)
    return ax
