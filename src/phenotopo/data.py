"""Synthetic cohorts with designed structure, for examples and tests."""

from __future__ import annotations

import numpy as np


def synthetic_cohort(n: int = 900, seed: int = 0):
    """A cohort whose structure is known in advance, so plots can be read.

    Five groups in a 2-D latent space:

    * ``Neuro`` — large, diffuse core;
    * ``Devel`` — sits on top of the Neuro core (strongly blended);
    * ``Bone`` — a well-separated island;
    * ``Skin`` — a smaller island touching Neuro on one side;
    * ``Renal`` — a tight satellite.

    Returns ``(embedding, labels, distance)``.
    """
    rng = np.random.RandomState(seed)
    spec = {
        "Neuro": (0.45, (0.0, 0.0), 1.10),
        "Devel": (0.20, (0.4, 0.4), 0.80),
        "Bone": (0.15, (5.0, 0.5), 0.55),
        "Skin": (0.12, (-2.6, 2.4), 0.55),
        "Renal": (0.08, (1.8, -3.2), 0.30),
    }
    pts, labs = [], []
    for name, (frac, mu, sd) in spec.items():
        m = int(round(n * frac))
        pts.append(rng.normal(mu, sd, size=(m, 2)))
        labs += [name] * m
    emb = np.vstack(pts)
    labels = np.array(labs)
    d = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1))
    return emb, labels, d


def synthetic_hierarchy(branches: int = 4, depth: int = 3, fanout: int = 3, seed: int = 0):
    """A small tree ``(relations, leaves_by_branch)`` shaped like an ontology.

    ``relations`` are ``(child, parent)`` pairs rooted at ``"root"``; each branch
    has ``depth`` levels with ``fanout`` children per node.
    """
    relations, leaves = [], {}
    for b in range(branches):
        top = f"B{b}"
        relations.append((top, "root"))
        frontier = [top]
        for level in range(1, depth + 1):
            nxt = []
            for parent in frontier:
                for c in range(fanout):
                    child = f"{parent}.{c}"
                    relations.append((child, parent))
                    nxt.append(child)
            frontier = nxt
        leaves[top] = frontier
    return relations, leaves


def synthetic_term_lists(leaves: dict, labels, seed: int = 0, per_patient=(2, 5)):
    """Give each synthetic patient terms drawn from the branch matching its label.

    Labels are mapped onto branches in order; ``Devel`` patients draw from two
    branches to mimic phenotypic overlap. A few patients get only one shallow
    term so the radial specificity axis is visible.
    """
    rng = np.random.RandomState(seed)
    branches = list(leaves)
    groups = list(dict.fromkeys(labels))
    branch_of = {g: branches[i % len(branches)] for i, g in enumerate(groups)}
    out = []
    for lab in labels:
        pool = list(leaves[branch_of[lab]])
        if lab == "Devel":
            pool += list(leaves[branches[0]])
        k = rng.randint(per_patient[0], per_patient[1] + 1)
        terms = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        if rng.rand() < 0.08:                 # sparse, shallow phenotyping
            terms = [branch_of[lab]]
        out.append(terms)
    return out
