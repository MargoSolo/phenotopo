"""phenotopo - continuum-honest maps of phenotype cohorts.

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

Inputs are deliberately minimal: a distance or feature matrix, group labels and
(optionally) any 2-D embedding you already trust.
"""

from .graph import group_centroids, group_connectivity, knn_graph
from .layout import default_palette, plot_connectivity, plot_connectivity_heatmap, plot_density, plot_small_multiples
from . import data, hyperbolic, mapper, robustness, stats
from .robustness import connectivity_robustness, plot_forest
from .stats import benjamini_hochberg, bootstrap_ratio, permutation_test

__version__ = "0.3.0"
__all__ = [
    "knn_graph", "group_connectivity", "group_centroids",
    "plot_connectivity", "plot_connectivity_heatmap", "plot_density", "plot_small_multiples", "default_palette",
    "data", "hyperbolic", "mapper", "stats",
    "permutation_test", "bootstrap_ratio", "benjamini_hochberg",
    "robustness", "connectivity_robustness", "plot_forest", "__version__",
]
