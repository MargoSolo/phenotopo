"""Phenotyping quality control - is the structure you see real, or annotation depth?

The most common way to be fooled by a phenotype cohort is not a bad embedding: it
is that some patients were phenotyped in four vague terms and others in twenty
specific ones, and that the difference tracks a site, a clinician or a diagnostic
era. Then any map shows structure, and the structure is bookkeeping.

:func:`phenotype_qc` scores every patient's annotation - how many terms, how
specific, how deep, how redundant, whether absence and onset were recorded at all -
and flags the under-phenotyped. :func:`annotation_bias` asks the follow-up question
directly: how much of the apparent group separation can be recovered from
annotation depth *alone*?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal


def phenotype_qc(cohort, min_terms: int = 3, specificity_pct: float = 10.0) -> dict:
    """Per-patient phenotyping quality, plus a cohort-level summary.

    Parameters
    ----------
    cohort
        A :class:`phenotopo.cohort.Cohort`.
    min_terms
        A patient with at most this many asserted terms is flagged
        ``LOW_ANNOTATION_DEPTH``.
    specificity_pct
        A patient below this percentile of mean information content is flagged
        ``LOW_SPECIFICITY_RELATIVE_TO_COHORT`` - the terms are there, but they are
        vague relative to the rest of *this* cohort.

    Flags are deliberately **relative and descriptive**. Six well-chosen terms can
    describe a skeletal dysplasia completely while fifteen vague ones describe a
    neurodevelopmental case badly, so a flag never says a patient is
    "under-phenotyped" in absolute terms - it says the annotation stands out against
    this cohort and that ``review`` is recommended.

    Returns
    -------
    dict with ``patients`` (one row per patient, **in cohort order**, so it lines up
    with labels and distance matrices), ``summary`` (cohort figures) and ``flagged``
    (the subset carrying any flag, worst first).

    Columns of ``patients``
    -----------------------
    ``n_terms`` asserted terms · ``n_propagated`` after the true-path rule ·
    ``n_excluded`` explicitly absent phenotypes · ``n_onset`` observations with an
    age · ``mean_ic`` / ``total_ic`` specificity (information content) ·
    ``mean_depth`` mean ontology depth · ``n_redundant`` asserted terms that are
    ancestors of another asserted term (bookkeeping noise, not information) ·
    ``specificity_pct`` percentile of ``mean_ic`` within the cohort · ``flags`` ·
    ``review``.
    """
    ic = cohort.information_content()
    onto = cohort.ontology
    prop = cohort.propagated()
    rows = []
    for i, pid in enumerate(cohort.ids):
        terms = sorted(cohort.present[i])
        ics = [float(ic.get(t, 0.0)) for t in terms]
        depths = [onto.depth(t) for t in terms] if onto is not None else []
        redundant = 0
        if onto is not None and len(terms) > 1:
            strict = set().union(*[onto.ancestors(t, include_self=False) for t in terms])
            redundant = sum(t in strict for t in terms)
        rows.append({
            "id": pid,
            "n_terms": len(terms),
            "n_propagated": len(prop[i]),
            "n_excluded": len(cohort.excluded[i]),
            "n_onset": len(cohort.onset[i]),
            "mean_ic": float(np.mean(ics)) if ics else 0.0,
            "total_ic": float(np.sum(ics)),
            "mean_depth": float(np.mean(depths)) if depths else np.nan,
            "n_redundant": int(redundant),
        })
    pat = pd.DataFrame(rows)
    pat["specificity_pct"] = pat["mean_ic"].rank(pct=True) * 100.0

    flags = []
    for _, r in pat.iterrows():
        f = []
        if r["n_terms"] == 0:
            f.append("NO_PHENOTYPE_RECORDED")
        elif r["n_terms"] <= min_terms:
            f.append("LOW_ANNOTATION_DEPTH")
        if r["n_terms"] > 0 and r["specificity_pct"] < specificity_pct:
            f.append("LOW_SPECIFICITY_RELATIVE_TO_COHORT")
        if r["n_redundant"] > 0:
            f.append("HIGH_REDUNDANCY")
        flags.append("; ".join(f))
    pat["flags"] = flags   # kept in cohort order, so it aligns with labels and distances
    pat["review"] = np.where(pat["flags"] != "", "review recommended", "")

    n = max(len(pat), 1)
    summary = {
        "n_patients": len(pat),
        "median_terms": float(pat["n_terms"].median()),
        "iqr_terms": [float(pat["n_terms"].quantile(0.25)), float(pat["n_terms"].quantile(0.75))],
        "median_mean_ic": float(pat["mean_ic"].median()),
        "pct_low_annotation_depth": 100.0 * (pat["n_terms"] <= min_terms).sum() / n,
        "pct_low_specificity": 100.0 * (pat["specificity_pct"] < specificity_pct).sum() / n,
        "pct_with_redundant_terms": 100.0 * (pat["n_redundant"] > 0).sum() / n,
        "pct_with_excluded_phenotypes": 100.0 * (pat["n_excluded"] > 0).sum() / n,
        "pct_with_onset": 100.0 * (pat["n_onset"] > 0).sum() / n,
        "pct_review_recommended": 100.0 * (pat["flags"] != "").sum() / n,
    }
    flagged = pat[pat["flags"] != ""].sort_values(["n_terms", "mean_ic"]).reset_index(drop=True)
    return {"patients": pat, "summary": summary, "flagged": flagged,
            "thresholds": {"min_terms": min_terms, "specificity_pct": specificity_pct}}


def annotation_bias(qc, labels, distance: np.ndarray | None = None, k: int = 15,
                    n_splits: int = 5, random_state: int = 0,
                    features=("n_terms", "n_propagated")) -> dict:
    """Is the difference between groups phenotype, or how thoroughly they were annotated?

    Two answers, both plain:

    1. **Do the groups differ in annotation depth at all?** Kruskal-Wallis on the
       number of terms and on mean information content, with epsilon-squared as
       the effect size (0.01 small, 0.06 moderate, 0.14 large).
    2. **Could the group signal be annotation depth?** If a ``distance`` matrix is
       given, group labels are predicted by cross-validated k-NN twice: from the
       phenotype distance, and from ``features`` alone. The result is reported as
       ``confound_risk`` (LOW / MODERATE / HIGH), **not** as a causal verdict:
       annotation counts predicting the group does not establish that the group
       separation is caused by them. A group that genuinely differs in phenotype
       severity is often also annotated more thoroughly, and the association runs
       group -> severity -> annotation depth. High risk means the analysis has to
       address the confound (matching, stratification, or a depth-controlled
       comparison), not that the finding is artifactual.

    ``features`` defaults to pure counts (``n_terms``, ``n_propagated``) - *how much*
    was written down. Specificity measures (``mean_ic``, ``mean_depth``) are
    deliberately **not** included: they depend on *which* terms a patient has, so
    two groups with genuinely different phenotypes differ in them, and adding them
    would report real biology as annotation bias.

    ``qc`` is the result of :func:`phenotype_qc` (or its ``patients`` frame).
    """
    pat = qc["patients"] if isinstance(qc, dict) else pd.DataFrame(qc)
    labels = np.asarray(labels)
    if len(labels) != len(pat):
        raise ValueError("labels and QC table must have the same length")
    groups = list(pd.unique(labels))

    by_group = pd.DataFrame({
        "n": [int((labels == g).sum()) for g in groups],
        "median_terms": [float(pat.loc[labels == g, "n_terms"].median()) for g in groups],
        "median_mean_ic": [float(pat.loc[labels == g, "mean_ic"].median()) for g in groups],
        "pct_flagged": [100.0 * (pat.loc[labels == g, "flags"] != "").mean() for g in groups],
    }, index=pd.Index(groups, name="group")).sort_values("n", ascending=False)

    tests = {}
    for col in ("n_terms", "mean_ic"):
        samples = [pat.loc[labels == g, col].to_numpy() for g in groups if (labels == g).sum() > 1]
        pooled = np.concatenate(samples) if samples else np.array([])
        if len(samples) > 1 and len(np.unique(pooled)) > 1:      # kruskal is undefined for all-tied data
            h, p = kruskal(*samples)
            n_tot = sum(len(s) for s in samples)
            eps2 = float((h - len(samples) + 1) / (n_tot - len(samples))) if n_tot > len(samples) else np.nan
            tests[col] = {"H": float(h), "p": float(p), "epsilon_sq": eps2,
                          "effect": "large" if eps2 >= 0.14 else "moderate" if eps2 >= 0.06
                          else "small" if eps2 >= 0.01 else "negligible"}

    out = {"by_group": by_group, "kruskal": tests}

    if distance is not None:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        keep = pd.Series(labels).map(pd.Series(labels).value_counts()) >= n_splits
        keep = keep.to_numpy()
        y = labels[keep]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        d = np.asarray(distance)[np.ix_(keep, keep)]
        acc_phen = float(np.mean(cross_val_score(
            KNeighborsClassifier(n_neighbors=k, metric="precomputed"), d, y, cv=cv)))
        depth = pat.loc[keep, list(features)].to_numpy()
        acc_depth = float(np.mean(cross_val_score(
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), depth, y, cv=cv)))
        base = float(pd.Series(y).value_counts(normalize=True).max())
        lift_phen, lift_depth = acc_phen - base, acc_depth - base
        share = float(np.clip(lift_depth / lift_phen, 0, 1)) if lift_phen > 1e-9 else np.nan
        risk = ("HIGH" if share == share and share > 0.5
                else "MODERATE" if share == share and share > 0.25 else "LOW")
        out["recovery"] = {
            "phenotype": acc_phen, "annotation_only": acc_depth, "majority_baseline": base,
            "annotation_share_of_lift": share, "k": k, "n_used": int(keep.sum()),
            "features": list(features),
            "confound_risk": risk,
            "note": ("Annotation counts predict the group about as well as phenotype does. "
                     "This flags a risk, it does not establish that the biological signal is "
                     "artifactual: a group that genuinely differs in phenotype severity may also "
                     "be annotated more thoroughly." if risk != "LOW" else
                     "Annotation counts add little beyond the baseline, so the group signal is "
                     "unlikely to be driven by how much was written down."),
        }
    return out


def plot_qc(qc, labels=None, ax=None, min_terms: int | None = None, colours=("#2980b9", "#c0392b")):
    """How thoroughly was the cohort phenotyped, and is that even across groups?

    Left: distribution of asserted terms per patient, with the under-phenotyped
    tail shaded. Right (only with ``labels``): terms per group, so a site or a
    diagnosis that was annotated differently from the rest is visible at a glance.

    Returns the :class:`~matplotlib.figure.Figure` (this view has two panels, so a
    single axis would not describe it).
    """
    import matplotlib.pyplot as plt

    pat = qc["patients"] if isinstance(qc, dict) else pd.DataFrame(qc)
    if min_terms is None:
        min_terms = qc["thresholds"]["min_terms"] if isinstance(qc, dict) else 3
    if ax is None:
        _, ax = plt.subplots(1, 2 if labels is not None else 1,
                             figsize=(10 if labels is not None else 5.5, 3.4), dpi=200)
    axes = np.atleast_1d(ax)
    fig = axes[0].figure

    n = pat["n_terms"].to_numpy()
    bins = np.arange(0, max(n.max(), min_terms + 1) + 2) - 0.5
    axes[0].hist(n, bins=bins, color=colours[0], alpha=0.85)
    axes[0].axvspan(bins[0], min_terms + 0.5, color=colours[1], alpha=0.12)
    axes[0].axvline(min_terms + 0.5, color=colours[1], lw=1, ls="--")
    share = 100.0 * (n <= min_terms).mean()
    axes[0].set_xlabel("HPO terms per patient")
    axes[0].set_ylabel("patients")
    axes[0].set_title(f"{share:.0f} % at or below {min_terms} terms (review recommended)", fontsize=9)

    if labels is not None:
        labels = np.asarray(labels)
        order = pd.Series(labels).value_counts().index.tolist()
        data = [pat.loc[labels == g, "n_terms"].to_numpy() for g in order]
        axes[1].boxplot(data, widths=0.6, showfliers=False, medianprops=dict(color=colours[1]))
        axes[1].set_xticks(np.arange(1, len(order) + 1))
        axes[1].set_xticklabels([f"{g}\n(n={len(d)})" for g, d in zip(order, data)], fontsize=8)
        axes[1].set_ylabel("HPO terms per patient")
        axes[1].set_title("annotation depth by group", fontsize=9)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    return fig
