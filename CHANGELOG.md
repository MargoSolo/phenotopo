# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

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
