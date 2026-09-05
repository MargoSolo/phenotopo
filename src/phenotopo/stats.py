"""Inference for group connectivity: permutation significance and bootstrap CIs.

``group_connectivity`` compares observed cross-group k-NN edges with an
analytical expectation. That says *how far* from chance a ratio is, not whether
it could have arisen by chance given the graph's actual structure and the group
sizes. Two complementary answers:

* :func:`permutation_test` - shuffle the group labels over the *fixed* graph
  many times; the ratios obtained form the null distribution of each pair.
  Gives p-values (with Benjamini-Hochberg correction across all pairs),
  z-scores and a 95 % null interval.
* :func:`bootstrap_ratio` - re-estimate the ratio on random 80 % subsamples of
  patients; the percentile interval is a precision interval for the estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import group_connectivity


def _ratio_from_members(rows, cols, data, member, m) -> np.ndarray:
    observed = np.zeros((m, m), dtype=float)
    np.add.at(observed, (member[rows], member[cols]), data)
    observed = (observed + observed.T) / 2.0
    degree = observed.sum(axis=1)
    total = degree.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.outer(degree, degree) / total
        return np.where(expected > 0, observed / expected, 0.0)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values for a 1-D array of p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def permutation_test(
    adjacency,
    labels,
    n_perm: int = 1000,
    random_state: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Permutation null for every group-pair connectivity ratio.

    Labels are permuted across patients (group sizes preserved, graph fixed) and
    the ratio recomputed each time. For each pair the result reports the
    observed ratio, the null mean and 95 % interval, one-sided p-values for
    enrichment (``ratio > null``) and depletion, a two-sided p-value, a z-score
    and Benjamini-Hochberg q-values across all unique pairs (diagonal included).

    ``pairs`` is a long-format table sorted by q-value - the thing to report.
    """
    labels = np.asarray(labels)
    groups = list(pd.unique(labels))
    idx = {g: i for i, g in enumerate(groups)}
    member = np.array([idx[g] for g in labels])
    m = len(groups)
    a = adjacency.tocoo()
    rows, cols, data = a.row, a.col, a.data

    observed = _ratio_from_members(rows, cols, data, member, m)
    rng = np.random.RandomState(random_state)
    null = np.empty((n_perm, m, m), dtype=float)
    for b in range(n_perm):
        null[b] = _ratio_from_members(rows, cols, data, rng.permutation(member), m)

    null_mean = null.mean(axis=0)
    null_sd = null.std(axis=0)
    lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
    p_enrich = ((null >= observed).sum(axis=0) + 1) / (n_perm + 1)
    p_deplete = ((null <= observed).sum(axis=0) + 1) / (n_perm + 1)
    p_two = np.minimum(1.0, 2.0 * np.minimum(p_enrich, p_deplete))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(null_sd > 0, (observed - null_mean) / null_sd, 0.0)

    iu = np.triu_indices(m)
    q_flat = benjamini_hochberg(p_two[iu])
    q = np.zeros((m, m)); q[iu] = q_flat; q = np.maximum(q, q.T)

    def df(x):
        return pd.DataFrame(x, index=groups, columns=groups)

    rows_out = []
    for k, (i, j) in enumerate(zip(*iu)):
        rows_out.append({
            "group_a": groups[i], "group_b": groups[j], "ratio": float(observed[i, j]),
            "null_mean": float(null_mean[i, j]), "null_lo95": float(lo[i, j]), "null_hi95": float(hi[i, j]),
            "z": float(z[i, j]), "p_enrich": float(p_enrich[i, j]), "p_deplete": float(p_deplete[i, j]),
            "p_two_sided": float(p_two[i, j]), "q_BH": float(q_flat[k]),
            "direction": "enriched" if observed[i, j] > null_mean[i, j] else "depleted",
            "significant": bool(q_flat[k] < alpha),
        })
    pairs = pd.DataFrame(rows_out).sort_values("q_BH").reset_index(drop=True)
    return {
        "ratio": df(observed), "null_mean": df(null_mean), "null_lo95": df(lo), "null_hi95": df(hi),
        "z": df(z), "p_two_sided": df(p_two), "q_BH": df(q), "pairs": pairs,
        "n_perm": n_perm, "alpha": alpha, "labels": groups,
    }


def bootstrap_ratio(
    adjacency,
    labels,
    n_boot: int = 300,
    subsample: float = 0.8,
    random_state: int = 0,
) -> dict:
    """Percentile interval for each ratio from repeated patient subsamples.

    Answers "how precisely is this ratio estimated?" (as opposed to the
    permutation test's "could it be chance?"). Groups that vanish from a
    subsample contribute nothing for that draw.
    """
    labels = np.asarray(labels)
    groups = list(pd.unique(labels))
    n = adjacency.shape[0]
    size = int(round(subsample * n))
    rng = np.random.RandomState(random_state)
    adjacency = adjacency.tocsr()
    draws = np.full((n_boot, len(groups), len(groups)), np.nan)
    gi = {g: i for i, g in enumerate(groups)}
    for b in range(n_boot):
        take = np.sort(rng.choice(n, size, replace=False))
        sub = group_connectivity(adjacency[take][:, take], labels[take])
        r = sub["ratio"]
        for a in r.index:
            for c in r.columns:
                draws[b, gi[a], gi[c]] = r.loc[a, c]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    df = lambda x: pd.DataFrame(x, index=groups, columns=groups)
    return {"ci_lo95": df(lo), "ci_hi95": df(hi), "median": df(np.nanmedian(draws, axis=0)),
            "n_boot": n_boot, "subsample": subsample, "labels": groups}
