"""Command line: see a report in thirty seconds, or produce one without writing code.

    phenotopo demo                          # synthetic cohort -> report, opened in a browser
    phenotopo report patients/ --labels diagnosis -o report.html
    phenotopo ontology install hp           # download the HPO release into the local cache
    phenotopo ontology path                 # where the cached ontology lives

The library itself never touches the network. ``ontology install`` is the single
exception, it is explicit, it prints the URL it fetches, and it exists so that
nothing downloads behind your back while an analysis runs.
"""

from __future__ import annotations

import argparse
import os
import sys

HPO_URL = "https://purl.obolibrary.org/obo/hp.obo"


def cache_dir() -> str:
    """Where cached ontologies live (override with ``PHENOTOPO_CACHE``)."""
    base = os.environ.get("PHENOTOPO_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "phenotopo")
    os.makedirs(base, exist_ok=True)
    return base


def cached_ontology(name: str = "hp") -> str | None:
    """Path to the cached ontology file, or ``None`` if it has not been installed."""
    path = os.path.join(cache_dir(), f"{name}.obo")
    return path if os.path.exists(path) else None


def install_ontology(name: str = "hp", url: str | None = None, source: str | None = None) -> str:
    """Put an ontology in the cache, from ``source`` on disk or by download."""
    target = os.path.join(cache_dir(), f"{name}.obo")
    if source:
        import shutil
        shutil.copyfile(source, target)
        print(f"copied {source} -> {target}")
        return target
    url = url or HPO_URL
    from urllib.request import urlopen
    print(f"downloading {url}")
    with urlopen(url) as response, open(target, "wb") as fh:      # noqa: S310 - explicit user action
        fh.write(response.read())
    print(f"saved {target} ({os.path.getsize(target) / 1e6:.1f} MB)")
    return target


def _resolve_ontology(explicit: str | None):
    from .cohort import Ontology

    path = explicit or cached_ontology()
    if path is None:
        sys.exit("no ontology available: pass --ontology PATH, or run `phenotopo ontology install hp`")
    return Ontology.from_obo(path)


def _open(path: str, no_open: bool) -> None:
    print(path)
    if no_open:
        return
    import webbrowser
    webbrowser.open(f"file://{path}")


def cmd_demo(args) -> None:
    from .data import synthetic_hpo_cohort
    from .explain import explain_groups
    from .report import cohort_report

    cohort = synthetic_hpo_cohort(n=400, seed=0)
    distance = cohort.distance()
    diff = explain_groups(cohort, cohort.labels("diagnosis"), "GENE_A", "GENE_B",
                          n_perm=300, min_effect=0.15)
    path = cohort_report(cohort, labels="diagnosis", distance=distance, comparisons=diff,
                         path=args.out, title="phenotopo demo (synthetic cohort)")
    print("Synthetic cohort with designed faults: a thinly phenotyped site, discordant "
          "cases and redundant ancestor terms - so every section has something true to find.")
    _open(path, args.no_open)


def cmd_report(args) -> None:
    from .cohort import from_hpo_table, from_phenopackets
    from .explain import explain_groups
    from .report import cohort_report

    ontology = _resolve_ontology(args.ontology)
    src = args.input
    if os.path.isdir(src) or src.endswith(".json"):
        cohort = from_phenopackets(src, ontology=ontology)
    else:
        cohort = from_hpo_table(src, id_col=args.id_col, terms_col=args.terms_col,
                                sep=args.sep, ontology=ontology)
    print(f"{len(cohort)} patients, {len(cohort.terms())} propagated terms")
    distance = cohort.distance(args.metric)
    comparisons = None
    if args.compare:
        a, b = (args.compare + [None])[:2] if isinstance(args.compare, list) else (args.compare, None)
        comparisons = explain_groups(cohort, cohort.labels(args.labels), a, b,
                                     n_perm=args.n_perm, min_effect=args.min_effect)
    path = cohort_report(cohort, labels=args.labels, distance=distance,
                         comparisons=comparisons, path=args.out, title=args.title)
    _open(path, args.no_open)


def cmd_ontology(args) -> None:
    if args.action == "install":
        install_ontology(args.name, url=args.url, source=args.source)
    else:
        print(cached_ontology(args.name) or f"{args.name} not installed (cache: {cache_dir()})")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="phenotopo", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="build a report from a synthetic cohort and open it")
    demo.add_argument("-o", "--out", default="phenotopo_demo.html")
    demo.add_argument("--no-open", action="store_true")
    demo.set_defaults(func=cmd_demo)

    rep = sub.add_parser("report", help="QC + outliers + comparison report for your own cohort")
    rep.add_argument("input", help="directory of Phenopackets, a .json packet, or a CSV/XLSX table")
    rep.add_argument("-o", "--out", default="cohort_report.html")
    rep.add_argument("--labels", default=None, help="metadata column to group by")
    rep.add_argument("--compare", nargs="+", default=None, metavar="GROUP",
                     help="one or two groups to compare within --labels")
    rep.add_argument("--metric", default="simgic", choices=["simgic", "cosine"])
    rep.add_argument("--ontology", default=None, help="path to hp.obo (default: the cached one)")
    rep.add_argument("--id-col", default="id")
    rep.add_argument("--terms-col", default="hpo")
    rep.add_argument("--sep", default=",")
    rep.add_argument("--n-perm", type=int, default=1000)
    rep.add_argument("--min-effect", type=float, default=0.10)
    rep.add_argument("--title", default=None)
    rep.add_argument("--no-open", action="store_true")
    rep.set_defaults(func=cmd_report)

    onto = sub.add_parser("ontology", help="manage the local ontology cache")
    onto.add_argument("action", choices=["install", "path"])
    onto.add_argument("name", nargs="?", default="hp")
    onto.add_argument("--url", default=None, help="download from here instead of the HPO release")
    onto.add_argument("--source", default=None, help="copy an existing local file instead of downloading")
    onto.set_defaults(func=cmd_ontology)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
