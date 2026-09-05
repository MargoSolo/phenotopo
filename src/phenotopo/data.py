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


def synthetic_hpo_cohort(n: int = 400, seed: int = 0, thin_site: bool = True):
    """A small annotated cohort with *designed* faults, for QC and comparison demos.

    Two diagnoses drawn from different branches of a toy ontology, plus three
    things that are planted on purpose so a quality report has something true to
    find:

    * patients from ``Site B`` are phenotyped thinly (one or two terms against four
      to seven elsewhere) - the annotation-depth confound;
    * a handful of patients carry the *other* diagnosis's phenotype - discordant
      cases;
    * some patients carry both a term and its parent - redundant annotation.

    Returns a :class:`~phenotopo.cohort.Cohort` with metadata columns
    ``diagnosis``, ``site`` and ``solved``.
    """
    import pandas as pd

    from .cohort import Cohort, Ontology

    branches = {
        "GENE_A": ["Status epilepticus", "Focal seizure", "Intellectual disability",
                   "Global developmental delay", "Hypotonia", "Ataxia", "Microcephaly"],
        "GENE_B": ["Short stature", "Kyphosis", "Joint hypermobility", "Arachnodactyly",
                   "Osteopenia", "Pes planus", "Facial dysmorphism"],
    }
    parents = {"Seizure": "Nervous system", "Status epilepticus": "Seizure",
               "Focal seizure": "Seizure", "Intellectual disability": "Nervous system",
               "Global developmental delay": "Nervous system", "Hypotonia": "Nervous system",
               "Ataxia": "Nervous system", "Microcephaly": "Head",
               "Short stature": "Skeletal system", "Scoliosis": "Skeletal system",
               "Kyphosis": "Scoliosis", "Joint hypermobility": "Skeletal system",
               "Arachnodactyly": "Skeletal system", "Osteopenia": "Skeletal system",
               "Pes planus": "Skeletal system", "Facial dysmorphism": "Head",
               "Nervous system": "root", "Skeletal system": "root", "Head": "root"}
    onto = Ontology(list(parents.items()), {t: t for t in list(parents) + ["root"]})

    rng = np.random.RandomState(seed)
    ids, present, meta = [], [], []
    for i in range(n):
        dx = "GENE_A" if i < n // 2 else "GENE_B"
        site = "B" if thin_site and rng.rand() < 0.25 else "A"
        pool = branches[dx]
        if rng.rand() < 0.04:                                   # discordant case
            pool = branches["GENE_B" if dx == "GENE_A" else "GENE_A"]
        if site == "B":                                          # thinly phenotyped site
            terms = set(rng.choice(pool, size=rng.randint(1, 3), replace=False))
        else:
            terms = set(rng.choice(pool, size=rng.randint(4, 8), replace=False))
        for child, parent in (("Status epilepticus", "Seizure"), ("Kyphosis", "Scoliosis")):
            if child in terms and rng.rand() < 0.25:
                terms.add(parent)                                 # redundant ancestor
        ids.append(f"P{i:03d}")
        present.append(terms)
        meta.append({"diagnosis": dx, "site": site, "solved": bool(rng.rand() < 0.55)})
    return Cohort(ids=ids, present=present, metadata=pd.DataFrame(meta), ontology=onto)
