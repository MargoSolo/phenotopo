"""Hyperbolic (Poincaré-disk) views of a term hierarchy and of the patients on it.

Why hyperbolic: a hierarchy grows exponentially with depth, which a Euclidean
plane cannot fit without distortion but the Poincaré disk can - the root sits
near the centre and specific leaf terms crowd the rim. Placing patients in the
same disk turns *phenotyping specificity* into a visible radial axis.
"""

from __future__ import annotations

import numpy as np


def poincare_terms(relations, size: int = 2, epochs: int = 50, seed: int = 0, **kwargs) -> dict:
    """Embed a hierarchy in the Poincaré disk with ``gensim``.

    ``relations`` is an iterable of ``(child, parent)`` pairs. Returns a
    ``{term: ndarray(size)}`` dictionary. Requires the ``hyperbolic`` extra.
    """
    try:
        from gensim.models.poincare import PoincareModel
    except ImportError as e:  # pragma: no cover
        raise ImportError("install the 'hyperbolic' extra: pip install phenotopo[hyperbolic]") from e
    relations = [tuple(r) for r in relations]
    n_nodes = len({x for r in relations for x in r})
    # gensim samples `negative` distractor nodes per update; a tiny hierarchy
    # cannot supply the default 10, so cap it instead of failing.
    kwargs.setdefault("negative", 10)
    kwargs["negative"] = max(1, min(kwargs["negative"], n_nodes - 2))
    model = PoincareModel(relations, size=size, seed=seed, **kwargs)
    model.train(epochs=epochs, print_every=10**9)
    return {t: np.asarray(model.kv[t], dtype=float) for t in model.kv.index_to_key}


def _to_klein(x: np.ndarray) -> np.ndarray:
    n2 = np.sum(x * x, axis=-1, keepdims=True)
    return 2.0 * x / (1.0 + n2)


def _to_poincare(k: np.ndarray) -> np.ndarray:
    n2 = np.clip(np.sum(k * k, axis=-1, keepdims=True), 0.0, 1.0 - 1e-9)
    return k / (1.0 + np.sqrt(1.0 - n2))


def einstein_midpoint(points: np.ndarray, weights=None) -> np.ndarray:
    """Weighted hyperbolic centroid of Poincaré-disk points (Einstein midpoint).

    Computed in the Klein model, where the midpoint is the Lorentz-factor-weighted
    mean, then mapped back to the disk. This is the proper hyperbolic average, not
    a Euclidean mean that would drift toward the origin.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    k = _to_klein(p)
    gamma = 1.0 / np.sqrt(np.clip(1.0 - np.sum(k * k, axis=-1), 1e-9, None))
    coef = (gamma * w)[:, None]
    m = (coef * k).sum(axis=0) / coef.sum()
    return _to_poincare(m[None, :])[0]


def place_patients(term_lists, term_coords: dict, weights: dict | None = None) -> np.ndarray:
    """Put every patient in the disk at the Einstein midpoint of its terms.

    ``weights`` (e.g. information content) up-weight specific terms so a patient
    with one vague and one precise finding sits nearer the precise one. Terms
    absent from ``term_coords`` are ignored; a patient with none lands at the
    origin.
    """
    out = np.zeros((len(term_lists), 2))
    for i, terms in enumerate(term_lists):
        pts, ws = [], []
        for t in terms:
            c = term_coords.get(t)
            if c is not None:
                pts.append(c)
                ws.append(1.0 if weights is None else float(weights.get(t, 1.0)))
        if pts:
            out[i] = einstein_midpoint(np.array(pts), np.array(ws))
    return out


def radial_specificity(points: np.ndarray) -> np.ndarray:
    """Distance from the disk centre, 0 (generic) to 1 (maximally specific)."""
    return np.linalg.norm(np.asarray(points), axis=1)


def plot_disk(
    points: np.ndarray,
    labels=None,
    landmarks: dict | None = None,
    palette: dict | None = None,
    point_size: float = 8,
    ax=None,
):
    """Scatter points inside the unit disk, optionally with labelled landmark terms."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from .layout import default_palette

    points = np.asarray(points)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, lw=1.2, color="black"))
    for r in (0.5, 0.8):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.5, ls=":", color="#999999"))
    if labels is None:
        ax.scatter(points[:, 0], points[:, 1], s=point_size, c="#2980b9", alpha=0.7, rasterized=True)
    else:
        labels = np.asarray(labels)
        order = list(pd.unique(labels))
        palette = palette or default_palette(order)
        for g in order:
            m = labels == g
            ax.scatter(points[m, 0], points[m, 1], s=point_size, c=[palette[g]], alpha=0.75,
                       edgecolors="none", rasterized=True, label=f"{g} (n={int(m.sum())})")
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, edgecolor="gray")
    if landmarks:
        from .layout import _repel

        texts = []
        for name, xy in landmarks.items():
            ax.scatter(xy[0], xy[1], marker="*", s=90, c="black", zorder=5, edgecolors="white", linewidths=0.5)
            texts.append(ax.annotate(name, xy, fontsize=7, fontweight="bold", xytext=(4, 4),
                                     textcoords="offset points", zorder=6,
                                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)))
        _repel(texts, ax, [xy[0] for xy in landmarks.values()], [xy[1] for xy in landmarks.values()])
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
    ax.axis("off")
    return ax
