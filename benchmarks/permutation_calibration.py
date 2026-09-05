"""Does the permutation adjustment in ``explain_groups`` actually control the error rate?

The claim in :mod:`phenotopo.explain` is that a single-step max-statistic label
permutation controls the family-wise error rate over a set of HPO terms that are
*not* independent, because propagation makes every term correlated with its
ancestors. A claim like that is worth simulating rather than asserting.

    python benchmarks/permutation_calibration.py

Two experiments, both on synthetic cohorts with a propagated ontology:

**A - calibration under the null.** Labels are assigned at random, so every reported
term is a false positive. A procedure controlling the FWER at 0.05 should raise at
least one term in at most ~5 % of simulated cohorts. Uncorrected testing and
Benjamini-Hochberg are shown alongside - BH controls the false discovery rate, not
the family-wise rate, so it is expected to exceed 5 % here; that is the point of the
comparison, not a defect of BH.

**B - power.** One branch of the ontology is given a real prevalence difference. The
question is how much of the null-case protection is paid for in missed true effects.

Runtime is a few minutes; results are written to ``permutation_calibration.csv``.
"""

import os
import time

import numpy as np
import pandas as pd

from phenotopo import Cohort, Ontology, explain_groups

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05
N_SIM = 200
N_PERM = 500
N_PATIENTS = 200
SEED = 0


def toy_ontology(branches=4, per_branch=4):
    """A shallow ontology: root -> branch -> leaf, so leaves share ancestors."""
    relations, leaves = [], {}
    for b in range(branches):
        top = f"B{b}"
        relations.append((top, "root"))
        leaves[top] = []
        for i in range(per_branch):
            leaf = f"{top}.{i}"
            relations.append((leaf, top))
            leaves[top].append(leaf)
            for j in range(2):                       # a second level, to deepen the chains
                relations.append((f"{leaf}.{j}", leaf))
                leaves[top].append(f"{leaf}.{j}")
    names = {t: t for t, _ in relations}
    names["root"] = "root"
    return Ontology(relations, names), leaves


SIGNAL_TERM = "B0.0"          # one specific leaf carries the planted difference


def simulate(rng, onto, leaves, effect=0.0):
    """A cohort of two groups.

    ``effect`` is the extra prevalence of one specific term (``SIGNAL_TERM``) in
    group A - a difference of that many percentage points, not a diffuse shift
    spread over a whole branch, which would be undetectable by construction.
    """
    pool = [t for terms in leaves.values() for t in terms]
    present, labels = [], []
    for i in range(N_PATIENTS):
        group = "A" if i < N_PATIENTS // 2 else "B"
        terms = set(rng.choice(pool, size=rng.randint(3, 7), replace=False))
        if effect and group == "A" and rng.rand() < effect:
            terms.add(SIGNAL_TERM)
        labels.append(group)
        present.append(terms)
    cohort = Cohort(ids=[f"P{i}" for i in range(N_PATIENTS)], present=present, ontology=onto)
    return cohort, np.array(labels)


def run(effect, method, n_sim=N_SIM, seed=SEED):
    """Fraction of simulated cohorts reporting at least one term, and how many."""
    rng = np.random.RandomState(seed)
    onto, leaves = toy_ontology()
    hits, counts, tested, signal = 0, [], [], []
    for _ in range(n_sim):
        cohort, labels = simulate(rng, onto, leaves, effect=effect)
        kwargs = dict(min_prevalence=0.05, min_effect=0.0, alpha=ALPHA, prune_redundant=False)
        if method == "permutation":
            res = explain_groups(cohort, labels, "A", "B", n_perm=N_PERM, random_state=int(rng.randint(1e6)), **kwargs)
        elif method == "benjamini-hochberg":
            res = explain_groups(cohort, labels, "A", "B", n_perm=0, **kwargs)
        else:                                          # uncorrected
            res = explain_groups(cohort, labels, "A", "B", n_perm=0, **kwargs)
            res = {"top": res["table"][res["table"]["p"] < ALPHA], "n_tested": res["n_tested"]}
        n = len(res["top"])
        hits += int(n > 0)
        counts.append(n)
        tested.append(res["n_tested"])
        if effect:
            signal.append(SIGNAL_TERM in set(res["top"]["term"]))
    return {"method": method, "effect": effect, "any_reported": hits / n_sim,
            "mean_terms_reported": float(np.mean(counts)), "mean_terms_tested": float(np.mean(tested)),
            "signal_detected": float(np.mean(signal)) if signal else float("nan"), "n_sim": n_sim}


if __name__ == "__main__":
    rows, t0 = [], time.time()
    print(f"A · calibration under the null ({N_SIM} cohorts of {N_PATIENTS}, alpha = {ALPHA})")
    for method in ("uncorrected", "benjamini-hochberg", "permutation"):
        r = run(0.0, method)
        rows.append({**r, "experiment": "null"})
        print(f"  {method:<20} at least one false term in {r['any_reported']:6.1%} of cohorts "
              f"({r['mean_terms_reported']:.2f} terms on average, {r['mean_terms_tested']:.0f} tested)")
    print(f"\nB · power: one term given a real prevalence difference in group A")
    for effect in (0.1, 0.2, 0.4):
        for method in ("benjamini-hochberg", "permutation"):
            r = run(effect, method, n_sim=max(60, N_SIM // 3))
            rows.append({**r, "experiment": "power"})
            print(f"  effect {effect:>3.0%}  {method:<20} planted term recovered in {r['signal_detected']:6.1%} "
                  f"of cohorts ({r['mean_terms_reported']:.2f} terms reported)")
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "permutation_calibration.csv"), index=False)
    print(f"\nwrote permutation_calibration.csv in {time.time() - t0:.0f}s")
