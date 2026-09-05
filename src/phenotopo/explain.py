"""Why do two groups differ? Ontology-aware phenotype comparison.

A connectivity ratio says two groups are separated; the next question is always
*which phenotypes separate them*. Running one test per HPO term and sorting by
p-value answers it badly, for two reasons that are specific to ontologies:

* **The terms are not independent.** A patient annotated *Status epilepticus* is,
  after propagation, also annotated *Seizure*, *Abnormal nervous system
  physiology* and so on up to the root. Benjamini-Hochberg assumes far more
  independence than that, and a whole ancestor chain lights up for one finding.
  :func:`explain_groups` therefore offers a **Westfall-Young max-T permutation**
  procedure: the group labels are shuffled and the *largest* prevalence
  difference over all terms is recorded each time, so the null already contains
  the ontology's correlation structure and the adjusted p-values control the
  family-wise error rate without assuming independence.
* **Significance is not the answer.** With a few thousand patients, a 2-point
  prevalence difference is significant and clinically empty. Terms are reported
  only above an explicit **effect-size threshold** (percentage points), and the
  redundant ancestors of an already-reported term are pruned, so what comes back
  is a short list of distinct phenotypes rather than a chain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def _wilson_diff_ci(x1, n1, x2, n2, z: float = 1.959963985):
    """Newcombe's score interval for a difference of two proportions."""
    def wilson(x, n):
        if n == 0:
            return 0.0, 1.0
        p = x / n
        d = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / d
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return max(0.0, centre - half), min(1.0, centre + half)
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    p1, p2 = (x1 / n1 if n1 else 0.0), (x2 / n2 if n2 else 0.0)
    return (p1 - p2) - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), \
           (p1 - p2) + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def explain_groups(
    cohort,
    labels,
    group_a,
    group_b=None,
    min_prevalence: float = 0.05,
    min_effect: float = 0.10,
    alpha: float = 0.05,
    n_perm: int = 1000,
    prune_redundant: bool = True,
    redundancy_tol: float = 0.02,
    random_state: int = 0,
) -> dict:
    """Phenotypes that distinguish ``group_a`` from ``group_b`` (default: everyone else).

    Parameters
    ----------
    min_prevalence
        A term is tested only if at least this fraction of one group carries it -
        the rest are noise with no power.
    min_effect
        Minimum absolute prevalence difference (0.10 = 10 percentage points) for a
        term to be reported. Significance alone never qualifies a term.
    n_perm
        Westfall-Young permutations. ``0`` falls back to Fisher + Benjamini-Hochberg,
        which is faster but assumes an independence the ontology does not have.
    prune_redundant
        Drop a term when a more specific reported term has essentially the same
        prevalence in both groups (within ``redundancy_tol``) - i.e. it is the same
        finding seen further up the ontology. Ties are resolved in favour of the
        deeper term, so *Seizure* is dropped for *Status epilepticus* and never the
        other way round.

    Returns
    -------
    dict with ``table`` (all tested terms), ``top`` (those passing effect **and**
    significance, most distinctive first), ``n_tested``, ``groups``, ``method``.

    Columns: ``term``, ``name``, ``ic``, ``prevalence_a``/``_b``, ``effect_pp``
    (percentage points, positive = enriched in ``group_a``), ``ci_lo_pp``/``ci_hi_pp``,
    ``p``, ``p_adjusted``, ``n_a``/``n_b``, ``excluded_a``/``_b`` (fraction in whom the
    phenotype was **looked for and ruled out**, reported but never mixed into the
    test), ``redundant_with``.
    """
    labels = np.asarray(labels)
    if len(labels) != len(cohort):
        raise ValueError("labels must have one entry per patient")
    mask_a = labels == group_a
    mask_b = (labels == group_b) if group_b is not None else ~mask_a
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    if n_a < 2 or n_b < 2:
        raise ValueError(f"both groups need >= 2 patients (got {n_a} and {n_b})")

    prop = cohort.propagated()
    ic = cohort.information_content()
    onto = cohort.ontology
    name = onto.name if onto is not None else (lambda t: t)

    idx = np.where(mask_a | mask_b)[0]
    in_a = mask_a[idx].astype(float)
    terms = sorted(set().union(*[prop[i] for i in idx]) if len(idx) else set())
    if not terms:
        raise ValueError("no annotated terms in the two groups")
    tpos = {t: j for j, t in enumerate(terms)}
    X = np.zeros((len(idx), len(terms)), dtype=np.float32)
    for r, i in enumerate(idx):
        for t in prop[i]:
            X[r, tpos[t]] = 1.0

    count_a = in_a @ X
    count_b = (1.0 - in_a) @ X
    prev_a, prev_b = count_a / n_a, count_b / n_b
    keep = np.where((prev_a >= min_prevalence) | (prev_b >= min_prevalence))[0]
    if len(keep) == 0:
        raise ValueError(f"no term reaches min_prevalence={min_prevalence} in either group")
    X, terms = X[:, keep], [terms[j] for j in keep]
    count_a, count_b = count_a[keep], count_b[keep]
    prev_a, prev_b = prev_a[keep], prev_b[keep]
    effect = prev_a - prev_b

    p = np.array([fisher_exact([[int(a), n_a - int(a)], [int(b), n_b - int(b)]])[1]
                  for a, b in zip(count_a, count_b)])
    if n_perm and n_perm > 0:
        rng = np.random.RandomState(random_state)
        g = in_a.copy()
        max_null = np.empty(n_perm)
        for s in range(n_perm):
            rng.shuffle(g)
            e = (g @ X) / n_a - ((1.0 - g) @ X) / n_b
            max_null[s] = np.abs(e).max()
        p_adj = np.array([((max_null >= abs(e)).sum() + 1) / (n_perm + 1) for e in effect])
        method = f"Westfall-Young max-T permutation (n={n_perm}), FWER-controlled"
    else:
        from .stats import benjamini_hochberg
        p_adj = benjamini_hochberg(p)
        method = "Fisher exact + Benjamini-Hochberg (assumes independence the ontology lacks)"

    # Explicitly excluded phenotypes are reported alongside, never mixed into the
    # test: "not recorded" and "looked for and absent" are different observations,
    # and only the second is evidence of absence.
    excl = cohort.propagated_excluded()
    if any(excl):
        E = np.zeros((len(idx), len(terms)), dtype=np.float32)
        pos_of = {t: j for j, t in enumerate(terms)}
        for r, i in enumerate(idx):
            for t in excl[i]:
                j = pos_of.get(t)
                if j is not None:
                    E[r, j] = 1.0
        excluded_a = (in_a @ E) / n_a
        excluded_b = ((1.0 - in_a) @ E) / n_b
    else:
        excluded_a = excluded_b = np.zeros(len(terms))

    ci = [_wilson_diff_ci(int(a), n_a, int(b), n_b) for a, b in zip(count_a, count_b)]
    table = pd.DataFrame({
        "term": terms,
        "name": [name(t) for t in terms],
        "ic": [round(float(ic.get(t, 0.0)), 2) for t in terms],
        "prevalence_a": prev_a, "prevalence_b": prev_b,
        "effect_pp": 100.0 * effect,
        "ci_lo_pp": [100.0 * c[0] for c in ci], "ci_hi_pp": [100.0 * c[1] for c in ci],
        "p": p, "p_adjusted": p_adj, "n_a": n_a, "n_b": n_b,
        "excluded_a": excluded_a, "excluded_b": excluded_b,
    })
    table["redundant_with"] = ""

    passing = table[(table["p_adjusted"] < alpha) & (table["effect_pp"].abs() >= 100.0 * min_effect)].copy()
    # most distinctive first; on ties the *more specific* term wins, so that an
    # ancestor carrying the same signal is the one pruned below, never the child.
    passing["_abs"] = passing["effect_pp"].abs()
    passing["_depth"] = [onto.depth(t) if onto is not None else 0 for t in passing["term"]]
    passing = passing.sort_values(["_abs", "_depth", "ic"], ascending=False).drop(columns=["_abs", "_depth"])

    if prune_redundant and onto is not None and len(passing):
        kept, prev = [], {r.term: (r.prevalence_a, r.prevalence_b) for r in passing.itertuples()}
        for r in passing.itertuples():
            more_specific = [k for k in kept
                             if r.term in onto.ancestors(k, include_self=False)
                             and abs(prev[k][0] - r.prevalence_a) <= redundancy_tol
                             and abs(prev[k][1] - r.prevalence_b) <= redundancy_tol]
            if more_specific:
                table.loc[table["term"] == r.term, "redundant_with"] = more_specific[0]
            else:
                kept.append(r.term)
        passing = passing[passing["term"].isin(kept)]
        table.loc[table["term"].isin(kept), "redundant_with"] = ""

    order = table["effect_pp"].abs().sort_values(ascending=False).index
    return {
        "table": table.reindex(order).reset_index(drop=True),
        "top": passing.reset_index(drop=True),
        "n_tested": len(terms), "method": method,
        "groups": (group_a, group_b if group_b is not None else "rest"),
        "sizes": {"a": n_a, "b": n_b},
        "thresholds": {"min_prevalence": min_prevalence, "min_effect": min_effect, "alpha": alpha},
    }


def plot_explain(result: dict, ax=None, max_terms: int = 15, colours=("#c0392b", "#2980b9")):
    """Horizontal effect plot of the reported terms, with confidence intervals."""
    import matplotlib.pyplot as plt

    top = result["top"].head(max_terms).iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.42 * max(len(top), 3) + 1.4), dpi=200)
    if not len(top):
        ax.text(0.5, 0.5, "no term passes the effect-size and significance thresholds",
                ha="center", va="center", fontsize=9, color="0.35", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    y = np.arange(len(top))
    colour = [colours[0] if e > 0 else colours[1] for e in top["effect_pp"]]
    ax.barh(y, top["effect_pp"], color=colour, height=0.6, alpha=0.85)
    ax.errorbar(top["effect_pp"], y,
                xerr=[top["effect_pp"] - top["ci_lo_pp"], top["ci_hi_pp"] - top["effect_pp"]],
                fmt="none", ecolor="0.3", elinewidth=1, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.name[:42]}  ({r.prevalence_a:.0%} vs {r.prevalence_b:.0%})"
                        for r in top.itertuples()], fontsize=8)
    ax.axvline(0, color="0.2", lw=1)
    a, b = result["groups"]
    ax.set_xlabel(f"prevalence difference, percentage points   (right = {a}, left = {b})", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return ax
