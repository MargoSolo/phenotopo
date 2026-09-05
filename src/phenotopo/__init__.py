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
* :mod:`phenotopo.mapper` - a topological (Mapper) graph of the cohort's shape.

Inputs are deliberately minimal: a distance or feature matrix, group labels and
(optionally) any 2-D embedding you already trust.
"""

from .graph import group_centroids, group_connectivity, knn_graph
from .layout import default_palette, plot_connectivity, plot_density, plot_small_multiples
from . import data, hyperbolic, mapper

__version__ = "0.1.0"
__all__ = [
    "knn_graph", "group_connectivity", "group_centroids",
    "plot_connectivity", "plot_density", "plot_small_multiples", "default_palette",
    "data", "hyperbolic", "mapper", "__version__",
]
