# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [0.6.0] — 2026-09-06

### Added
- **Benchmarks on real, published cases** (`benchmarks/phenopacket_store/`): five case
  studies over 2,691 cases from Phenopacket Store 0.1.27 — diagnosis recovery 93.8 %
  against a 17.2 % baseline, 16/16 diseases robustly cohesive with a single blended
  pair, the annotation-confound check LOW across all diseases and HIGH for the most
  unequally annotated pair, macrodontia recovered as the top KBG-vs-Glass
  discriminator, and a 2.2-point gain from using explicitly excluded phenotypes.
  Corpora are cached outside the repository; only derived tables and figures are kept.
- **`benchmarks/permutation_calibration.py`** — the FWER and power simulation that
  `explain_groups` refers to: the max-statistic permutation lands on 5.0 % at a nominal
  5 % (uncorrected: 75 %) and is more powerful than Benjamini–Hochberg on planted
  effects (87.9 % vs 81.8 %).

### Changed
- `annotation_bias` reports balanced accuracy against a label-permutation baseline
  rather than plain accuracy against the majority class, guards `k` on small folds, and
  takes configurable risk `bands`.
- `phenotype_qc` separates absolute completeness heuristics from cohort-relative
  anomalies and warns at cohort level when the whole cohort is thinly phenotyped, so a
  uniformly under-annotated cohort is no longer invisible to a percentile rule.
- `explain_groups` describes its adjustment precisely as a single-step max-statistic
  label permutation.
- `from_phenopackets` searches directory trees; real corpora nest one folder per gene.

## [0.5.1] — 2026-09-05

### Fixed
- Package metadata: the summary shown on PyPI was still the 0.4.0 one; the license is
  now the SPDX expression `MIT` rather than the full licence text pasted into the
  metadata field; added Documentation and Changelog project URLs.

### Added
- `.zenodo.json`, so an archived release carries proper software metadata.

## [0.5.0] — 2026-09-05

### Changed
- **`Cohort.distance()` now defaults to SimGIC**, the IC-weighted ontology-aware
  measure, with cosine kept as a sensitivity analysis; `Cohort.distances()` returns
  both, ready for the robustness protocol. (Behaviour change: an explicit
  `distance("cosine")` reproduces the old default.)
- QC flags are relative and descriptive — `LOW_ANNOTATION_DEPTH`,
  `LOW_SPECIFICITY_RELATIVE_TO_COHORT`, `HIGH_REDUNDANCY`, resolving to
  `review recommended` — replacing the absolute `under-phenotyped` / `non-specific`.
  A short, well-chosen phenotype list is not a poor one. Summary keys renamed
  accordingly (`pct_low_annotation_depth`, `pct_low_specificity`,
  `pct_review_recommended`).
- `annotation_bias` reports `confound_risk` (LOW / MODERATE / HIGH) with an explicit
  note that annotation counts predicting a group does **not** establish that the
  separation is caused by them, replacing the causal verdict string.

### Added
- `distance(..., negatives="use")`: excluded phenotypes (propagated downwards, since
  excluding a parent excludes its children) enter the similarity, so a shared
  ruled-out phenotype counts. Refuses to run where absence was never recorded;
  `onset=` is declared and raises rather than silently ignoring.
- `Ontology.descendants`, `Cohort.propagated_excluded`, `Cohort.has_negatives`,
  `Cohort.excluded_terms`, `Cohort.distances`.
- `explain_groups` reports `excluded_a` / `excluded_b` — the fraction in whom a
  phenotype was looked for and ruled out — alongside the test, never inside it.
- Command line (`phenotopo`): `demo` (synthetic cohort → report in a browser),
  `report` (Phenopackets or a CSV/XLSX table → the same HTML, no Python needed),
  `ontology install` / `path` (explicit local ontology cache; the only place the
  package touches the network).

## [0.4.0] — 2026-09-05

### Added
- `cohort.Cohort` / `cohort.Ontology` with readers `from_hpo_table` and
  `from_phenopackets` (GA4GH Phenopackets v2: excluded phenotypes, onset, disease and
  gene), ancestor propagation, information content, and cosine / SimGIC distances —
  so an analysis no longer starts with the user building a distance matrix.
- `qc.phenotype_qc`: per-patient annotation quality (terms, specificity, ontology
  depth, redundant ancestors, recorded absence and onset) with flags, plus
  `qc.plot_qc`.
- `qc.annotation_bias`: Kruskal–Wallis on annotation depth across groups and a
  cross-validated k-NN comparison of group recovery from phenotype versus from
  annotation counts alone — how much of the apparent structure is bookkeeping.
- `outliers.patient_outliers` (isolation and neighbourhood discordance reported
  separately) and `outliers.explain_outlier` (terms that make a patient unusual, and
  the terms its neighbourhood has that it lacks).
- `explain.explain_groups`: ontology-aware group comparison with a Westfall–Young
  max-T permutation (FWER control that respects parent–child dependence), Newcombe
  confidence intervals, an effect-size threshold and pruning of redundant ancestor
  terms; `explain.plot_explain`.
- `report.cohort_report`: one self-contained local HTML file with QC, annotation
  bias, outliers, robustness verdicts and group comparisons, figures embedded.
- `data.synthetic_hpo_cohort`: an annotated cohort with designed faults (a thinly
  phenotyped site, discordant cases, redundant ancestor terms).

### Changed
- `phenotype_qc` returns its patient table in cohort order, so it lines up with
  labels and distance matrices; the flagged subset is the sorted view.

## [0.3.1] — 2026-09-03

### Fixed
- Robustness tests assumed group sizes that the synthetic cohort does not have; simplified
  pair-label formatting in `plot_forest`.

## [0.3.0] — 2026-09-03

### Added
- `robustness.connectivity_robustness`: the multi-configuration protocol (distances × k,
  minimum group size, effect-size thresholds, all-configurations verdicts) and
  `robustness.plot_forest`.
- Continuous integration (pytest on push / pull request), `CITATION.cff`.

## [0.2.0] — 2026-09-03

### Added
- `stats.permutation_test`: label-permutation null for every connectivity ratio
  (p-values, BH q-values, z-scores, 95 % null interval), `stats.bootstrap_ratio`
  (subsampling percentile CI), significance overlay on the connectivity heatmap.
- `plot_connectivity_heatmap`; optional adjustText label repulsion.

## [0.1.0] — 2026-09-03

### Added
- `knn_graph`, `group_connectivity`: PAGA-style connectivity between labelled groups
  (observed k-NN edges relative to a configuration-model expectation).
- `plot_connectivity`: abstracted group graph drawn at group centroids.
- `plot_density`, `plot_small_multiples`: overlap-honest alternatives to a single
  many-colour scatter.
- `hyperbolic`: Poincaré embedding of a term hierarchy, Einstein-midpoint placement
  of patients in the disk, radial specificity axis.
- `mapper`: KeplerMapper wrapper producing a `networkx` graph coloured by any
  per-patient value.
- `data.synthetic_cohort` / `data.synthetic_hierarchy` for examples and tests.
