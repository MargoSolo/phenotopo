# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

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
