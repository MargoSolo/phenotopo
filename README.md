# phenotopo

**Phenotype QC, outliers and honest cohort structure for HPO / Phenopacket rare-disease cohorts.**

[![tests](https://github.com/MargoSolo/phenotopo/actions/workflows/tests.yml/badge.svg)](https://github.com/MargoSolo/phenotopo/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![version](https://img.shields.io/badge/version-0.4.0-informational.svg)](CHANGELOG.md)

Two things go wrong when a rare-disease cohort is analysed by HPO terms.

The first is **bookkeeping mistaken for biology**: some patients were phenotyped in
two vague terms and others in twenty specific ones, the difference tracks a site or
a clinician, and every map then shows structure that is really annotation depth.
The second is **clusters that are not there**: a cohort described by phenotype is
usually a continuum with local structure, and one UMAP with twenty colours invites
everyone to read islands into it.

`phenotopo` is built around those two failures. It scores how well each patient was
phenotyped and tests whether the group signal survives that; it finds the patients
who sit away from their own group; it says which phenotypes actually separate two
groups, tested in a way that respects the ontology; and it reports cohort structure
only where the answer survives a change of distance, of *k* and of resampling.

Everything runs locally. No patient data leaves the machine, and the package makes
no network calls.

---

## Installation

```bash
pip install -e .                 # core: cohorts, QC, outliers, comparison, connectivity
pip install -e ".[hyperbolic]"   # + Poincaré disk (gensim)
pip install -e ".[tda]"          # + Mapper graph (kmapper)
pip install -e ".[all]"          # everything, incl. umap-learn
```

If `adjustText` is installed, overlapping labels on the connectivity graph and the
Poincaré disk are pushed apart automatically; without it the plots still work.

## Quick start

```python
import phenotopo as pt

cohort = pt.from_phenopackets("patients/", ontology="hp.obo")   # or from_hpo_table(df, ...)
qc     = pt.phenotype_qc(cohort)                                # who was phenotyped thinly?
bias   = pt.annotation_bias(qc, cohort.labels("site"),          # is the group signal bookkeeping?
                            distance=cohort.distance("cosine"))
out    = pt.patient_outliers(cohort.distance("cosine"),         # who sits away from their group?
                             cohort.labels("diagnosis"), ids=cohort.ids)
diff   = pt.explain_groups(cohort, cohort.labels("gene"),       # which phenotypes separate them?
                           "GENE_A", "GENE_B")
pt.cohort_report(cohort, labels="diagnosis",                    # one local HTML file
                 distance=cohort.distance("cosine"),
                 comparisons=diff, path="cohort_report.html")
```

A `Cohort` is optional throughout: every analysis function also takes a plain
distance matrix and an array of labels, so `phenotopo` sits on top of a pipeline
you already have rather than replacing it.

`from_phenopackets` reads GA4GH Phenopackets v2 and keeps what a bare list of term
IDs throws away — **excluded** phenotypes (looked for and absent), **onset** per
observation, and the disease and gene from `interpretations`, so
`explain_groups(..., "gene")` works straight after reading.

---

The figures below come from [`examples/quickstart.py`](examples/quickstart.py), run on
synthetic cohorts with *designed* faults and structure, so each can be checked against
ground truth.

## 1 · Phenotyping quality

```python
qc = pt.phenotype_qc(cohort)
qc["summary"]        # median terms, % under-phenotyped, % non-specific, % redundant, ...
qc["flagged"]        # worst-annotated patients first, with reasons
pt.plot_qc(qc, cohort.labels("site"))
```

<p align="center"><img src="examples/figures/qc.png" width="760"></p>

Per patient: terms asserted and after propagation, explicitly excluded phenotypes,
observations with an onset, mean and total information content, mean ontology depth,
**redundant ancestor terms** (a term recorded next to its own child — bookkeeping, not
information), the specificity percentile, and a flag: *under-phenotyped*,
*non-specific*, *redundant ancestors*.

Then the question that decides whether any of the rest can be believed:

```python
bias = pt.annotation_bias(qc, labels, distance=D)
bias["kruskal"]      # do the groups differ in annotation depth? H, p, epsilon-squared
bias["recovery"]     # {'phenotype': 0.92, 'annotation_depth_only': 1.00, ...}
```

Group labels are predicted twice by cross-validated k-NN — from the phenotype
distance, and from **how much was written down alone** (`n_terms`, `n_propagated`).
When the second is as good as the first, the map is bookkeeping. Specificity measures
are deliberately excluded from that second model: they depend on *which* terms a
patient has, so including them would report real biology as bias.

## 2 · Outliers — who sits away from their own group

```python
out = pt.patient_outliers(D, labels, ids=cohort.ids, k=15)
pt.explain_outlier(cohort, index, D, labels)
```

Two things get called an outlier, and mixing them produces nonsense, so they are
reported separately: **isolation** (far from everybody — a sparse or unusual
phenotype) and **discordance** (plenty of close neighbours, and they belong to a
different group). `explain_outlier` names the terms that make a patient unusual for
its neighbourhood, and — often more useful — the terms its neighbourhood has that it
lacks, which is as likely to be a phenotyping gap as biology.

The output is never a claim that a diagnosis is wrong. It is *phenotypically
discordant with the assigned group*: a candidate for review in a diagnostic cohort,
a genotype–phenotype study or a reclassification project.

## 3 · What separates two groups

```python
diff = pt.explain_groups(cohort, labels, "GENE_A", "GENE_B", min_effect=0.15)
diff["top"]     # term, name, prevalence in each group, effect in pp with CI, adjusted p
pt.plot_explain(diff)
```

<p align="center"><img src="examples/figures/explain.png" width="700"></p>

One test per HPO term, sorted by p-value, answers this badly, for two reasons
specific to ontologies:

- **The terms are not independent.** After propagation a patient annotated *Status
  epilepticus* is also annotated *Seizure* and everything up to the root, so one
  finding lights up a whole ancestor chain and Benjamini–Hochberg assumes far more
  independence than exists. `explain_groups` therefore uses a **Westfall–Young max-T
  permutation**: labels are shuffled and the *largest* prevalence difference over all
  terms is recorded each time, so the null already contains the ontology's
  correlation structure and the adjusted p-values control the family-wise error rate.
- **Significance is not the answer.** With a few thousand patients a 2-point
  difference is significant and clinically empty. Terms are reported only above an
  explicit **effect-size threshold** in percentage points, with a Newcombe confidence
  interval, and redundant ancestors of an already-reported term are pruned — ties go
  to the deeper term, so *Seizure* is dropped for *Status epilepticus* and never the
  other way round.

## 4 · Cohort structure, only where it is robust

```python
res = pt.connectivity_robustness({"cosine": D_cos, "simgic": D_gic}, labels,
                                 ks=(10, 15, 30), min_size=100)
res["summary"]        # per pair: ratio, CI, range across configurations, verdict
pt.plot_forest(res)
```

The connectivity ratio compares the k-NN edges crossing between two groups with the
number expected under a degree-preserving null (the PAGA abstraction, Wolf et al.
2019): `1` is random mixing, `> 1` blending, `< 1` separation, and the diagonal is
within-group cohesion. A ratio from one distance, one *k* and one sample is a single
draw, so the protocol (i) drops groups below `min_size`, whose expected counts are
tiny and whose ratios explode; (ii) recomputes every ratio for every distance × *k*
with a permutation *q* and a bootstrap CI; (iii) applies an **effect-size threshold**,
not just significance; and (iv) issues a verdict only if the criterion holds in
**every** configuration. What survives is a sentence you can write down.

The underlying views, when you want to look rather than test:

```python
conn = pt.group_connectivity(pt.knn_graph(distance=D, k=15), labels)
pt.plot_connectivity(conn, embedding=emb, labels=labels)          # node-link, few groups
pt.plot_connectivity_heatmap(conn, min_size=30,                   # every pair, many groups
                             significance=pt.permutation_test(pt.knn_graph(distance=D), labels))
```

<p align="center">
<img src="examples/figures/connectivity.png" width="420">
<img src="examples/figures/connectivity_heatmap.png" width="380">
</p>

Nodes sit at each group's median position, node area is group size, edge width ∝ log
ratio. Past about eight groups a node–link drawing stops being readable and the
heatmap is the honest form: every pair, the separated ones (`ratio < 1`) included,
`*` where the permutation q survives correction.

## 5 · Looking at the continuum

```python
pt.plot_density(emb, labels, levels=(0.5, 0.85))    # where groups overlap, drawn not hidden
pt.plot_small_multiples(emb, labels, ncols=5)       # one panel per group, no 20-colour legend
```

<p align="center">
<img src="examples/figures/density.png" width="420">
<img src="examples/figures/small_multiples.png" width="420">
</p>

Two further views, for hierarchy and shape:

```python
from phenotopo.hyperbolic import poincare_terms, place_patients, plot_disk
from phenotopo.mapper import mapper_graph, node_values, plot_mapper
```

<p align="center">
<img src="examples/figures/hyperbolic.png" width="380">
<img src="examples/figures/mapper.png" width="440">
</p>

The **Poincaré disk** fits a hierarchy into two dimensions with the root at the centre
and specific leaves at the rim; patients sit at the Einstein midpoint of their terms,
so phenotyping specificity becomes a radial axis — the annotation-depth confound made
geometric instead of hidden. The **Mapper graph** covers the space with overlapping
bins and clusters locally inside each, showing branches, bridges and flares without
forcing every patient into a cluster; colour it by any per-patient outcome (here a
synthetic diagnostic yield) to see where that outcome concentrates.

## 6 · One file to hand over

```python
pt.cohort_report(cohort, labels="diagnosis", distance=D,
                 robustness=res, comparisons=[diff], path="cohort_report.html")
```

Cohort overview, QC and the annotation-bias verdict, the outlier list, the robustness
table with its forest plot and every group comparison — in a single self-contained
HTML file, figures embedded, no server and no network. It opens in a browser and
survives being emailed.

---

## Reading the connectivity ratio

| `ratio` | Meaning |
|---|---|
| ≫ 1 | groups blend — many more cross edges than chance |
| ≈ 1 | as connected as random mixing |
| ≪ 1 | separated — a real boundary in phenotype space |

The null preserves every group's total degree (the configuration model behind
modularity), so large groups are not rewarded merely for being large.

## API

| Module | Contents |
|---|---|
| `phenotopo.cohort` | `Cohort`, `Ontology`, `from_hpo_table`, `from_phenopackets` |
| `phenotopo.qc` | `phenotype_qc`, `annotation_bias`, `plot_qc` |
| `phenotopo.outliers` | `patient_outliers`, `explain_outlier` |
| `phenotopo.explain` | `explain_groups`, `plot_explain` |
| `phenotopo.report` | `cohort_report` |
| `phenotopo.graph` | `knn_graph`, `group_connectivity`, `group_centroids` |
| `phenotopo.stats` | `permutation_test`, `bootstrap_ratio`, `benjamini_hochberg` |
| `phenotopo.robustness` | `connectivity_robustness`, `plot_forest` |
| `phenotopo.layout` | `plot_connectivity`, `plot_connectivity_heatmap`, `plot_density`, `plot_small_multiples`, `default_palette` |
| `phenotopo.hyperbolic` | `poincare_terms`, `einstein_midpoint`, `place_patients`, `radial_specificity`, `plot_disk` |
| `phenotopo.mapper` | `mapper_graph`, `node_values`, `plot_mapper` |
| `phenotopo.data` | `synthetic_cohort`, `synthetic_hpo_cohort`, `synthetic_hierarchy`, `synthetic_term_lists` |

## Scope

`phenotopo` deliberately does **not** prioritise genes or diagnoses for a single
patient — Exomiser and LIRICAL do that — and does not capture phenotypes, pedigrees
or variants, which is PhenoTips' job. It sits after capture and beside
prioritisation: quality control, cohort structure, outliers and group comparison,
reading Phenopackets so it fits between the two.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Tests

```bash
pytest -q
```

Runs on synthetic data only. **No patient data belongs in this repository** —
`.gitignore` blocks ontology dumps, cohort files and analysis output.

## References

- Wolf F.A. et al. *PAGA: graph abstraction reconciles clustering with trajectory inference through a topology preserving map of single cells.* Genome Biology, 2019.
- Westfall P.H., Young S.S. *Resampling-based multiple testing.* Wiley, 1993 — max-T procedure.
- Newcombe R.G. *Interval estimation for the difference between independent proportions.* Statistics in Medicine, 1998.
- Pesquita C. et al. *Metrics for GO based protein semantic similarity: a systematic evaluation.* BMC Bioinformatics, 2008 — SimGIC.
- Jacobsen J.O.B. et al. *The GA4GH Phenopacket schema defines a computable representation of clinical data.* Nature Biotechnology, 2022.
- Nickel M., Kiela D. *Poincaré embeddings for learning hierarchical representations.* NeurIPS, 2017.
- Ungar A.A. *Analytic hyperbolic geometry.* World Scientific, 2005 — Einstein midpoint.
- Singh G., Mémoli F., Carlsson G. *Topological methods for the analysis of high dimensional data sets.* Eurographics, 2007 — Mapper.
- van Veen H.J. et al. *Kepler Mapper.* Journal of Open Source Software, 2019.

## License

MIT — see [LICENSE](LICENSE).
