"""phenotopo - a QC and explainability toolkit for phenotype cohorts.

A cohort of rare-disease patients is usually a continuum with local structure,
not a set of islands. Standard practice - one scatter plot, twenty colours,
clusters read by eye - hides that. ``phenotopo`` offers views built for the
continuum case:

* :func:`group_connectivity` / :func:`plot_connectivity` - a PAGA-style abstracted
  graph: which labelled groups genuinely blend into each other, and which are
  separate (Wolf et al., *Genome Biology* 2019, generalised beyond single cells);
* :func:`plot_density` - per-group highest-density contours, so overlap is visible;
* :func:`plot_small_multiples` - one panel per group instead of a 20-colour legend;
* :mod:`phenotopo.hyperbolic` - a Poincaré-disk view in which phenotyping
  *specificity* becomes a radial axis;
* :mod:`phenotopo.mapper` - a topological (Mapper) graph of the cohort's shape;
* :mod:`phenotopo.stats` - permutation significance and bootstrap intervals for the
  connectivity ratios, so a "2.6" comes with a p-value and a CI;
* :mod:`phenotopo.robustness` - the multi-configuration protocol (distances x k,
  effect-size thresholds, all-configurations verdicts) and its forest plot - the
  version of the connectivity result that belongs in a manuscript.

Before any of that, the questions asked first in a clinic:

* :func:`phenotype_qc` - who was phenotyped thinly, vaguely or redundantly, and
  :func:`annotation_bias` - how much of the apparent group structure is annotation
  depth rather than biology;
* :func:`patient_outliers` / :func:`explain_outlier` - who sits away from their own
  group, and which terms put them there;
* :func:`explain_groups` - which phenotypes actually separate two groups, tested in
  a way that respects the ontology's parent-child dependence and an effect-size
  threshold;
* :func:`cohort_report` - all of it in one local HTML file.

Inputs are deliberately minimal: a distance or feature matrix, group labels and
(optionally) any 2-D embedding you already trust - or a
:class:`~phenotopo.cohort.Cohort` read from an HPO table or GA4GH Phenopackets,
which computes those for you.
"""

from .cohort import Cohort, Ontology, from_hpo_table, from_phenopackets
from .explain import explain_groups, plot_explain
from .graph import group_centroids, group_connectivity, knn_graph
from .layout import default_palette, plot_connectivity, plot_connectivity_heatmap, plot_density, plot_small_multiples
from .outliers import explain_outlier, patient_outliers
from .qc import annotation_bias, phenotype_qc, plot_qc
from .report import cohort_report
from . import cli, cohort, data, explain, hyperbolic, mapper, outliers, qc, report, robustness, stats
from .robustness import connectivity_robustness, plot_forest
from .stats import benjamini_hochberg, bootstrap_ratio, permutation_test

__version__ = "0.6.0"
__all__ = [
    "Cohort", "Ontology", "from_hpo_table", "from_phenopackets",
    "phenotype_qc", "annotation_bias", "plot_qc", "patient_outliers", "explain_outlier",
    "explain_groups", "plot_explain", "cohort_report",
    "knn_graph", "group_connectivity", "group_centroids",
    "plot_connectivity", "plot_connectivity_heatmap", "plot_density", "plot_small_multiples", "default_palette",
    "cli", "cohort", "data", "explain", "hyperbolic", "mapper", "outliers", "qc", "report", "stats",
    "permutation_test", "bootstrap_ratio", "benjamini_hochberg",
    "robustness", "connectivity_robustness", "plot_forest", "__version__",
]
