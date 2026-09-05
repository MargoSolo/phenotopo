"""Patients who do not sit where their label says they should.

Two different things get called an outlier, and mixing them produces nonsense, so
they are reported separately:

* **isolation** - the patient is far from everybody (sparse or unusual phenotype);
* **discordance** - the patient has plenty of close neighbours, and they belong to
  a different group than the one the patient is assigned to.

Neither is a claim that a diagnosis is wrong. The honest reading is *phenotypically
discordant with the assigned group* - a candidate for review, in a diagnostic
cohort, a genotype-phenotype study or a reclassification project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def patient_outliers(distance, labels=None, ids=None, k: int = 15, n_neighbors_shown: int = 3,
                     isolation_pct: float = 95.0, discordance: float = 0.8) -> pd.DataFrame:
    """Rank patients by isolation and by disagreement with their neighbourhood.

    Parameters
    ----------
    distance
        Square distance matrix in the phenotype space (not an embedding - a 2-D
        layout distorts exactly the distances this uses).
    labels
        Assigned group per patient (diagnosis, gene, system). Optional: without
        labels only isolation is computed.
    k
        Neighbourhood size.
    isolation_pct
        Percentile of mean k-NN distance above which a patient is called isolated.
    discordance
        Fraction of the neighbourhood belonging to other groups above which a
        patient is called discordant.

    Returns
    -------
    DataFrame sorted by ``discordance`` then ``isolation_pct``, with the nearest
    neighbours and the majority label around each patient.
    """
    d = np.asarray(distance, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("`distance` must be a square matrix")
    n = d.shape[0]
    k = int(min(k, n - 1))
    if k < 1:
        raise ValueError("need at least two patients")
    ids = list(ids) if ids is not None else list(range(n))
    dd = d.copy()
    np.fill_diagonal(dd, np.inf)
    order = np.argsort(dd, axis=1)[:, :k]
    mean_d = np.take_along_axis(dd, order, axis=1).mean(axis=1)
    iso_pct = pd.Series(mean_d).rank(pct=True).to_numpy() * 100.0

    rows = []
    labels = np.asarray(labels) if labels is not None else None
    for i in range(n):
        nb = order[i]
        row = {"id": ids[i], "mean_knn_distance": float(mean_d[i]), "isolation_pct": float(iso_pct[i])}
        if labels is not None:
            nb_labels = labels[nb]
            same = float(np.mean(nb_labels == labels[i]))
            counts = pd.Series(nb_labels).value_counts()
            row.update({
                "label": labels[i],
                "same_label_fraction": same,
                "discordance": 1.0 - same,
                "neighbourhood_majority": counts.index[0],
                "majority_fraction": float(counts.iloc[0] / k),
            })
        row["nearest"] = ", ".join(
            f"{ids[j]}" + (f" [{labels[j]}]" if labels is not None else "") + f" ({dd[i, j]:.2f})"
            for j in nb[:n_neighbors_shown])
        rows.append(row)

    out = pd.DataFrame(rows)
    isolated = out["isolation_pct"] >= isolation_pct
    if labels is not None:
        disc = (out["discordance"] >= discordance) & (out["neighbourhood_majority"] != out["label"])
        out["flag"] = np.where(isolated & disc, "isolated + discordant",
                      np.where(disc, "phenotypically discordant with assigned group",
                      np.where(isolated, "isolated", "")))
        out = out.sort_values(["discordance", "isolation_pct"], ascending=False)
    else:
        out["flag"] = np.where(isolated, "isolated", "")
        out = out.sort_values("isolation_pct", ascending=False)
    return out.reset_index(drop=True)


def explain_outlier(cohort, index: int, distance=None, labels=None, k: int = 15, top: int = 5) -> dict:
    """Why is this patient unusual? The terms that separate it from its neighbours.

    Reports the patient's most specific terms, the terms it has that its
    neighbourhood mostly lacks, and the terms its neighbourhood has that it lacks -
    the latter being the most useful column in practice, because a missing common
    phenotype is often a phenotyping gap rather than biology.
    """
    ic = cohort.information_content()
    prop = cohort.propagated()
    name = cohort.ontology.name if cohort.ontology is not None else (lambda t: t)
    out = {"id": cohort.ids[index], "n_terms": len(cohort.present[index]),
           "most_specific_terms": [(t, name(t), round(float(ic.get(t, 0.0)), 2))
                                   for t in sorted(cohort.present[index],
                                                   key=lambda t: -float(ic.get(t, 0.0)))[:top]]}
    if distance is None:
        return out
    d = np.asarray(distance, dtype=float).copy()
    np.fill_diagonal(d, np.inf)
    nb = np.argsort(d[index])[:k]
    freq_nb = pd.Series([t for j in nb for t in prop[j]]).value_counts() / len(nb)
    mine = prop[index]
    unique = sorted(((t, float(freq_nb.get(t, 0.0))) for t in mine if float(ic.get(t, 0.0)) > 0),
                    key=lambda x: (x[1], -float(ic.get(x[0], 0.0))))[:top]
    missing = sorted(((t, float(f)) for t, f in freq_nb.items()
                      if t not in mine and f >= 0.5 and float(ic.get(t, 0.0)) > 0),
                     key=lambda x: -x[1])[:top]
    out.update({
        "neighbours": [cohort.ids[j] for j in nb[:top]],
        "unusual_for_neighbourhood": [(t, name(t), round(f, 2)) for t, f in unique],
        "expected_but_absent": [(t, name(t), round(f, 2)) for t, f in missing],
    })
    if labels is not None:
        labels = np.asarray(labels)
        counts = pd.Series(labels[nb]).value_counts()
        out["assigned_label"] = labels[index]
        out["neighbourhood_majority"] = counts.index[0]
        out["majority_fraction"] = float(counts.iloc[0] / len(nb))
    return out
