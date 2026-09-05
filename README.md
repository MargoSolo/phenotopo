# phenotopo

**Continuum-honest maps of phenotype cohorts.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A cohort of rare-disease patients described by HPO terms is, more often than not,
a **continuum with local structure** — not a set of well-separated islands. The
standard picture (one UMAP, twenty colours, clusters read by eye) hides exactly
that. `phenotopo` provides views that were designed for the continuum case and
that borrow, deliberately, from fields that solved the same problem earlier:

| View | Borrowed from | What it answers |
|---|---|---|
| **Group-connectivity graph** | PAGA, single-cell genomics | Which labelled groups genuinely blend into each other, and which are separate? |
| **Density contours** | spatial statistics | Where do groups overlap — shown instead of hidden? |
| **Small multiples** | data-visualisation practice | How does *each* group sit in the cohort, without a 20-colour legend? |
| **Poincaré disk** | hyperbolic geometry / hierarchical embeddings | How *specific* is each patient's phenotyping? (radius = specificity) |
| **Mapper graph** | topological data analysis | What is the *shape* of the cohort — branches, bridges, flares — with no forced clustering? |

Inputs are minimal: a distance or feature matrix, group labels, and optionally any
2-D embedding you already trust. `phenotopo` never computes the embedding for you
unless asked — it is a lens on top of your existing analysis, not a replacement.

---

## Installation

```bash
pip install -e .                 # core: connectivity, density, small multiples
pip install -e ".[hyperbolic]"   # + Poincaré disk (gensim)
pip install -e ".[tda]"          # + Mapper graph (kmapper)
pip install -e ".[all]"          # everything, incl. umap-learn
```

---

## The views, on a cohort whose structure is known

Every figure below is produced by [`examples/quickstart.py`](examples/quickstart.py) from a
synthetic cohort with *designed* structure, so each can be read against ground truth:
**Neuro** is a large diffuse core, **Devel** sits on top of it (blended), **Bone** is a
separate island, **Skin** touches Neuro on one side, **Renal** is a tight satellite.

### 1 · Group-connectivity graph (PAGA-style)

```python
from phenotopo import knn_graph, group_connectivity, plot_connectivity

conn = group_connectivity(knn_graph(distance=D, k=15), labels)
plot_connectivity(conn, embedding=emb, labels=labels)
```

<p align="center"><img src="examples/figures/connectivity.png" width="640"></p>

Nodes sit at each group's median position; node area is group size; an edge is
drawn where groups share more k-NN edges than a degree-preserving null model
expects (`ratio ≥ 1`), with width ∝ log ratio. **Thick edge = the groups blend;
no edge = genuinely separate.** Here Neuro–Devel is thick, Bone is isolated, Skin
hangs off Neuro — exactly as designed. This is the single most honest summary of
"local coherence without discrete clusters".

### 2 · Density contours

```python
from phenotopo import plot_density
plot_density(emb, labels, levels=(0.5, 0.85))
```

<p align="center"><img src="examples/figures/density.png" width="640"></p>

Each contour encloses 50 % / 85 % of that group's probability mass (highest-density
regions). Overlap between groups is drawn, not obscured by point-cloud overplotting.

### 3 · Small multiples

```python
from phenotopo import plot_small_multiples
plot_small_multiples(emb, labels, ncols=5)
```

<p align="center"><img src="examples/figures/small_multiples.png" width="760"></p>

One panel per group, highlighted over the grey cohort. The standard cure for a
legend nobody can read.

### 4 · Poincaré disk — specificity as a radial axis

```python
from phenotopo.hyperbolic import poincare_terms, place_patients, plot_disk

coords = poincare_terms(relations)                     # (child, parent) pairs, e.g. HPO is_a
pts    = place_patients(term_lists, coords, weights=ic)  # Einstein midpoint of each patient's terms
plot_disk(pts, labels, landmarks={"Neurology": coords["HP:0000707"]})
```

<p align="center"><img src="examples/figures/hyperbolic.png" width="560"></p>

A hierarchy grows exponentially with depth; the Poincaré disk fits it in two
dimensions with the root near the centre and specific leaves at the rim. Patients
are placed at the **Einstein midpoint** (the proper hyperbolic centroid) of their
terms, weighted by information content, so *how specifically a patient was
phenotyped* becomes visible as distance from the centre — the annotation-depth
confound, made geometric instead of hidden.

### 5 · Mapper graph — the shape of the cohort

```python
from phenotopo.mapper import mapper_graph, node_values, plot_mapper

_, g = mapper_graph(emb, n_cubes=12, perc_overlap=0.35)
plot_mapper(g, node_values(g, solved), label="fraction solved in node")
```

<p align="center"><img src="examples/figures/mapper.png" width="640"></p>

Mapper covers the space with overlapping bins, clusters locally inside each and
links bins that share points. The result shows branches, bridges and flares
**without forcing every patient into a cluster**. Colouring nodes by any
per-patient value (here a synthetic diagnostic yield) reveals *where* in phenotype
space an outcome concentrates.

---

## Reading the connectivity ratio

| `ratio` | Meaning |
|---|---|
| ≫ 1 | groups blend — many more cross edges than chance |
| ≈ 1 | as connected as random mixing |
| ≪ 1 | separated — a real boundary in phenotype space |

The null model preserves every group's total degree (the configuration model
behind modularity), so large groups are not rewarded merely for being large.

---

## API

| Module | Contents |
|---|---|
| `phenotopo.graph` | `knn_graph`, `group_connectivity`, `group_centroids` |
| `phenotopo.layout` | `plot_connectivity`, `plot_density`, `plot_small_multiples`, `default_palette` |
| `phenotopo.hyperbolic` | `poincare_terms`, `einstein_midpoint`, `place_patients`, `radial_specificity`, `plot_disk` |
| `phenotopo.mapper` | `mapper_graph`, `node_values`, `plot_mapper` |
| `phenotopo.data` | `synthetic_cohort`, `synthetic_hierarchy`, `synthetic_term_lists` |

## Tests

```bash
pytest -q
```

Runs on synthetic data only. **No patient data belongs in this repository** —
`.gitignore` blocks ontology dumps, cohort files and analysis output.

## References

- Wolf F.A. et al. *PAGA: graph abstraction reconciles clustering with trajectory inference through a topology preserving map of single cells.* Genome Biology, 2019.
- Nickel M., Kiela D. *Poincaré embeddings for learning hierarchical representations.* NeurIPS, 2017.
- Ungar A.A. *Analytic hyperbolic geometry.* World Scientific, 2005 — Einstein midpoint.
- Singh G., Mémoli F., Carlsson G. *Topological methods for the analysis of high dimensional data sets.* Eurographics, 2007 — Mapper.
- van Veen H.J. et al. *Kepler Mapper.* Journal of Open Source Software, 2019.

## License

MIT — see [LICENSE](LICENSE).
