"""Four case studies on real, published rare-disease cases (GA4GH Phenopacket Store).

    phenotopo ontology install hp          # once
    python examples/phenopacket_store/run_case_studies.py

The corpus is 10,377 phenopackets curated from the literature by the Monarch
Initiative (Danis et al., 2025), downloaded on demand into the local cache and never
committed here. Case-study numbers and figures are written next to this script.

1  Does phenotype structure recover known disease structure?
2  Can annotation depth manufacture apparent separation?
3  Does discordance find atypical published patients?
4  Which HPO terms actually distinguish two related disorders?
5  Do explicitly excluded phenotypes carry information? (87 % of this corpus has them)
"""

import json
import os
import sys
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from phenotopo import (
    Ontology,
    annotation_bias,
    connectivity_robustness,
    explain_groups,
    explain_outlier,
    from_phenopackets,
    patient_outliers,
    phenotype_qc,
    plot_explain,
    plot_forest,
    plot_qc,
)
from phenotopo.cli import cache_dir, cached_ontology

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RELEASE = "0.1.27"
URL = f"https://github.com/monarch-initiative/phenopacket-store/releases/download/{RELEASE}/all_phenopackets.zip"
MIN_N = 80          # diseases smaller than this give unstable connectivity ratios
K = 15
os.makedirs(FIG, exist_ok=True)
results = {}


def corpus_path() -> str:
    root = os.path.join(cache_dir(), "phenopacket-store")   # cached outside the repository
    packets = os.path.join(root, "packets")
    if not os.path.isdir(packets):
        os.makedirs(root, exist_ok=True)
        archive = os.path.join(root, "all_phenopackets.zip")
        if not os.path.exists(archive):
            print(f"downloading {URL}")
            urllib.request.urlretrieve(URL, archive)          # noqa: S310 - explicit, printed
        with zipfile.ZipFile(archive) as z:
            z.extractall(packets)
    return packets


def save(fig_or_ax, name):
    fig = fig_or_ax.figure if hasattr(fig_or_ax, "figure") else fig_or_ax
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", name)


def knn_accuracy(distance, labels, k=K, splits=5, seed=0):
    counts = pd.Series(labels).value_counts()
    keep = pd.Series(labels).map(counts).to_numpy() >= splits
    y, d = np.asarray(labels)[keep], np.asarray(distance)[np.ix_(keep, keep)]
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    acc = float(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=k, metric="precomputed"), d, y, cv=cv)))
    return acc, float(pd.Series(y).value_counts(normalize=True).max())


if cached_ontology() is None:
    sys.exit("run `phenotopo ontology install hp` first")
print("reading the corpus …")
full = from_phenopackets(corpus_path(), ontology=Ontology.from_obo(cached_ontology()))
counts = full.metadata["disease"].value_counts()
diseases = [d for d in counts.index if counts[d] >= MIN_N]
mask = full.metadata["disease"].isin(diseases).to_numpy() & np.array([len(p) > 0 for p in full.present])
idx = np.where(mask)[0]
cohort = type(full)(ids=[full.ids[i] for i in idx], present=[full.present[i] for i in idx],
                    excluded=[full.excluded[i] for i in idx], onset=[full.onset[i] for i in idx],
                    metadata=full.metadata.iloc[idx].reset_index(drop=True), ontology=full.ontology)
labels = cohort.labels("disease")
short = {d: (d.split(",")[0][:34]) for d in diseases}
print(f"corpus {len(full)} patients / {counts.size} diseases  →  analysis set {len(cohort)} patients, "
      f"{len(diseases)} diseases with n ≥ {MIN_N}")
results["corpus"] = {"release": RELEASE, "n_total": len(full), "n_diseases_total": int(counts.size),
                     "n_analysed": len(cohort), "diseases": {d: int(counts[d]) for d in diseases},
                     "min_n": MIN_N}

D = cohort.distance()                      # SimGIC
qc = phenotype_qc(cohort)
results["qc"] = qc["summary"]
print("QC:", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in qc["summary"].items()})
save(plot_qc(qc, labels=None), "qc_terms_per_patient.png")

# ── 1 · does phenotype structure recover known disease structure? ─────────────
print("\n1 · recovering known disease structure")
acc, base = knn_accuracy(D, labels)
print(f"  k-NN recovers the reported diagnosis in {acc:.1%} of cases (majority baseline {base:.1%})")
rob = connectivity_robustness(cohort.distances(), labels, ks=(10, 15, 30), min_size=MIN_N,
                              n_perm=1000, n_boot=200, random_state=0)
summary = rob["summary"]
summary.to_csv(os.path.join(HERE, "case1_connectivity.csv"))
save(plot_forest(rob, short_names=short, top_between=12), "case1_connectivity_forest.png")
#   16 diseases make 136 rows; the figure keeps the most connected pairs and the
#   16 within-disease rows, and case1_connectivity.csv carries all of them.
verdicts = summary["verdict"].value_counts().to_dict()
print("  verdicts:", verdicts)
for kind in ("cohesive", "blend", "separation"):
    rows = summary[summary["verdict"] == kind]
    if len(rows):
        print(f"  {kind}: " + "; ".join(f"{i} {r.ratio:.1f}" for i, r in list(rows.iterrows())[:6]))
results["case1"] = {"knn_disease_recovery": acc, "majority_baseline": base,
                    "verdicts": {k: int(v) for k, v in verdicts.items()},
                    "cohesive": summary[summary.verdict == "cohesive"]["ratio"].round(2).to_dict(),
                    "blends": summary[summary.verdict == "blend"]["ratio"].round(2).to_dict(),
                    "separations": summary[summary.verdict == "separation"]["ratio"].round(2).to_dict(),
                    "diffuse": summary[summary.verdict == "diffuse"]["ratio"].round(2).to_dict()}

# ── 2 · can annotation depth manufacture apparent separation? ─────────────────
print("\n2 · annotation depth as a confounder")
bias = annotation_bias(qc, labels, distance=D, k=K)
by_group = bias["by_group"].sort_values("median_terms")
by_group.to_csv(os.path.join(HERE, "case2_annotation_depth.csv"))
r = bias["recovery"]
print(f"  all {len(diseases)} diseases: phenotype {r['phenotype']:.1%} vs counts alone "
      f"{r['annotation_only']:.1%} (permutation baseline {r['permutation_baseline']:.1%}) → risk {r['confound_risk']}")
thin, thick = by_group.index[0], by_group.index[-1]
pair = np.isin(labels, [thin, thick])
sub_qc = {"patients": qc["patients"].loc[pair].reset_index(drop=True),
          "thresholds": qc["thresholds"], "summary": qc["summary"]}
pair_bias = annotation_bias(sub_qc, labels[pair], distance=D[np.ix_(pair, pair)], k=K)
pr = pair_bias["recovery"]
print(f"  most unequal pair — {short[thin]} (median {by_group.loc[thin,'median_terms']:.0f} terms) vs "
      f"{short[thick]} (median {by_group.loc[thick,'median_terms']:.0f}): phenotype {pr['phenotype']:.1%}, "
      f"counts alone {pr['annotation_only']:.1%} (baseline {pr['permutation_baseline']:.1%}) → risk {pr['confound_risk']}")
results["case2"] = {"all_diseases": {k: v for k, v in r.items() if k != "note"},
                    "most_unequal_pair": {"thin": thin, "thick": thick,
                                          "median_terms": [float(by_group.loc[thin, "median_terms"]),
                                                           float(by_group.loc[thick, "median_terms"])],
                                          **{k: v for k, v in pr.items() if k != "note"}},
                    "median_terms_by_disease": by_group["median_terms"].to_dict()}
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=170)
order = by_group.index
ax.barh(range(len(order)), by_group["median_terms"], color="#2980b9", alpha=0.85)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{short[d]} (n={int(by_group.loc[d,'n'])})" for d in order], fontsize=8)
ax.set_xlabel("median HPO terms per published case", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
save(ax, "case2_annotation_depth.png")

# ── 3 · does discordance find atypical published patients? ────────────────────
print("\n3 · phenotypically discordant published cases")
out = patient_outliers(D, labels, ids=cohort.ids, k=K)
disc = out[out["flag"].str.contains("discordant")]
disc.to_csv(os.path.join(HERE, "case3_discordant.csv"), index=False)   # the flagged rows; the full table is one call away
print(f"  {len(disc)} of {len(out)} cases ({len(disc)/len(out):.1%}) sit in a neighbourhood dominated "
      f"by a different reported diagnosis")
per = (disc.groupby("label").size() / pd.Series({d: int((labels == d).sum()) for d in diseases})).dropna()
print("  highest rate:", "; ".join(f"{short[d]} {v:.0%}" for d, v in per.sort_values(ascending=False).head(4).items()))
example = disc.iloc[0]
ex = explain_outlier(cohort, cohort.ids.index(example["id"]), D, labels, k=K)
print(f"  example {ex['id']}: reported {short.get(ex['assigned_label'], ex['assigned_label'])}, "
      f"neighbourhood {short.get(ex['neighbourhood_majority'], ex['neighbourhood_majority'])} "
      f"({ex['majority_fraction']:.0%}), {ex['n_terms']} terms")
results["case3"] = {"n_discordant": int(len(disc)), "n_total": int(len(out)),
                    "rate": float(len(disc) / len(out)),
                    "rate_by_disease": {d: float(v) for d, v in per.sort_values(ascending=False).items()},
                    "example": {k: v for k, v in ex.items() if k in
                                ("id", "assigned_label", "neighbourhood_majority", "majority_fraction", "n_terms")},
                    "example_unusual_terms": [t[1] for t in ex["unusual_for_neighbourhood"]],
                    "example_expected_but_absent": [t[1] for t in ex["expected_but_absent"]]}

# ── 4 · which terms distinguish two related disorders? ────────────────────────
print("\n4 · what separates two related neurodevelopmental syndromes")
pair_a, pair_b = "KBG syndrome", "Glass syndrome"
diff = explain_groups(cohort, labels, pair_a, pair_b, min_prevalence=0.05, min_effect=0.15,
                      n_perm=1000, random_state=0)
diff["top"].to_csv(os.path.join(HERE, "case4_kbg_vs_glass.csv"), index=False)
print(f"  {pair_a} (n={diff['sizes']['a']}) vs {pair_b} (n={diff['sizes']['b']}): "
      f"{len(diff['top'])} distinguishing terms of {diff['n_tested']} tested")
for row in diff["top"].head(6).itertuples():
    side = pair_a if row.effect_pp > 0 else pair_b
    print(f"    {row.name[:44]:<44} {abs(row.effect_pp):5.1f} pp → {side}")
save(plot_explain(diff, max_terms=14), "case4_kbg_vs_glass.png")
results["case4"] = {"groups": [pair_a, pair_b], "sizes": diff["sizes"], "n_tested": diff["n_tested"],
                    "method": diff["method"],
                    "top": diff["top"].head(12)[["name", "prevalence_a", "prevalence_b", "effect_pp",
                                                 "ci_lo_pp", "ci_hi_pp", "p_adjusted"]].round(3).to_dict("records")}

# ── 5 · do explicitly excluded phenotypes carry information? ──────────────────
print("\n5 · explicitly excluded phenotypes")
share = float(np.mean([bool(e) for e in cohort.excluded]))
print(f"  {share:.0%} of these cases record at least one phenotype as looked-for-and-absent")
acc_neg, _ = knn_accuracy(cohort.distance("simgic", negatives="use"), labels)
print(f"  disease recovery: {acc:.1%} without negatives → {acc_neg:.1%} with them")
results["case5"] = {"share_with_negatives": share, "knn_without": acc, "knn_with": acc_neg,
                    "delta_pp": 100 * (acc_neg - acc)}

with open(os.path.join(HERE, "case_studies_results.json"), "w") as fh:
    json.dump(results, fh, indent=1, default=float)
print("\nwrote case_studies_results.json")
