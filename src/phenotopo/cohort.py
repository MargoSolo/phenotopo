"""The cohort: patients, their HPO terms and the ontology that relates them.

Everything else in the package can be driven by a bare distance matrix, and that
stays true. This module is the convenience layer above it: it reads the formats a
clinical cohort actually arrives in - a table of HPO terms, or GA4GH Phenopackets -
propagates annotations along the ontology, weights them by information content and
hands back the matrices the analysis functions expect.

Three things are kept that a plain list of term IDs throws away, because losing them
silently is worse than not using them yet:

* **excluded** phenotypes (explicitly looked for and absent) - clinically not the same
  as "not mentioned";
* **onset** per observation;
* arbitrary per-patient **metadata** (diagnosis, gene, solved/unsolved, site).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, lil_matrix


class Ontology:
    """A directed acyclic graph of ``is_a`` relations, with term names.

    Built from ``(child, parent)`` pairs or from an ``hp.obo`` release. Only the
    two operations the analysis needs are exposed: ancestors of a term (the
    true-path rule) and its depth below the root.
    """

    def __init__(self, relations, names: dict | None = None):
        self.parents: dict[str, set] = defaultdict(set)
        self.children: dict[str, set] = defaultdict(set)
        for child, parent in relations:
            self.parents[child].add(parent)
            self.children[parent].add(child)
        self.names = dict(names or {})
        self._ancestors: dict[str, frozenset] = {}
        self._descendants: dict[str, frozenset] = {}
        self._depth: dict[str, int] = {}

    @classmethod
    def from_obo(cls, path: str) -> "Ontology":
        """Parse the ``is_a`` skeleton of an OBO file (e.g. the HPO release).

        Obsolete terms are skipped. Deliberately minimal - no external OBO
        library is required to use the package.
        """
        relations, names, term, name, obsolete = [], {}, None, None, False
        with open(path, encoding="utf-8") as fh:
            in_term = False
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("["):
                    if in_term and term and not obsolete and name:
                        names[term] = name
                    in_term, term, name, obsolete = line.strip() == "[Term]", None, None, False
                elif in_term:
                    if line.startswith("id: "):
                        term = line[4:].strip()
                    elif line.startswith("name: "):
                        name = line[6:].strip()
                    elif line.startswith("is_obsolete: true"):
                        obsolete = True
                    elif line.startswith("is_a: ") and term:
                        relations.append((term, line[6:].split("!")[0].strip()))
        if in_term and term and not obsolete and name:
            names[term] = name
        relations = [(c, p) for c, p in relations if c in names and p in names]
        return cls(relations, names)

    def ancestors(self, term: str, include_self: bool = True) -> frozenset:
        """All ancestors of ``term`` along every branch (HPO is a DAG, not a tree)."""
        if term not in self._ancestors:
            seen, stack = set(), [term]
            while stack:
                t = stack.pop()
                for p in self.parents.get(t, ()):
                    if p not in seen:
                        seen.add(p)
                        stack.append(p)
            self._ancestors[term] = frozenset(seen)
        return frozenset({term}) | self._ancestors[term] if include_self else self._ancestors[term]

    def descendants(self, term: str, include_self: bool = True) -> frozenset:
        """All descendants of ``term``.

        Absence propagates the other way from presence: if *Seizure* was looked for
        and excluded, every kind of seizure is excluded too.
        """
        if term not in self._descendants:
            seen, stack = set(), [term]
            while stack:
                t = stack.pop()
                for c in self.children.get(t, ()):
                    if c not in seen:
                        seen.add(c)
                        stack.append(c)
            self._descendants[term] = frozenset(seen)
        return frozenset({term}) | self._descendants[term] if include_self else self._descendants[term]

    def depth(self, term: str) -> int:
        """Shortest number of ``is_a`` steps from ``term`` up to a root (root = 0)."""
        if term not in self._depth:
            frontier, d, seen = {term}, 0, {term}
            while frontier:
                parents = {p for t in frontier for p in self.parents.get(t, ())} - seen
                if not parents:
                    break
                seen |= parents
                frontier = parents
                d += 1
            self._depth[term] = d
        return self._depth[term]

    def name(self, term: str) -> str:
        return self.names.get(term, term)

    def __len__(self) -> int:
        return len(set(self.parents) | set(self.children))


@dataclass
class Cohort:
    """Patients with HPO annotations, optionally an ontology and metadata.

    ``present`` and ``excluded`` are lists of term sets, one per patient; ``onset``
    maps a term to an ISO-8601 age string where it was recorded. Nothing here is
    required to be complete - missing pieces are simply reported as missing by
    :func:`phenotopo.qc.phenotype_qc`.
    """

    ids: list
    present: list
    excluded: list = field(default_factory=list)
    onset: list = field(default_factory=list)
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    ontology: Ontology | None = None

    def __post_init__(self):
        n = len(self.ids)
        self.present = [set(t) for t in self.present]
        self.excluded = [set(t) for t in (self.excluded or [set()] * n)]
        self.onset = list(self.onset or [{}] * n)
        if not (len(self.present) == len(self.excluded) == len(self.onset) == n):
            raise ValueError("ids, present, excluded and onset must have the same length")
        if self.metadata is None or len(self.metadata) == 0:
            self.metadata = pd.DataFrame(index=pd.Index(self.ids, name="id"))
        else:
            self.metadata = pd.DataFrame(self.metadata).reset_index(drop=True)
            self.metadata.index = pd.Index(self.ids, name="id")
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.ids)

    # ---- representation -------------------------------------------------
    def propagated(self) -> list:
        """Present terms plus all their ancestors (true-path rule)."""
        if "prop" not in self._cache:
            if self.ontology is None:
                self._cache["prop"] = [set(t) for t in self.present]
            else:
                anc = self.ontology.ancestors
                self._cache["prop"] = [set().union(*[anc(t) for t in ts]) if ts else set()
                                       for ts in self.present]
        return self._cache["prop"]

    def propagated_excluded(self) -> list:
        """Explicitly excluded terms plus all their descendants (absence propagates down)."""
        if "prop_excl" not in self._cache:
            if self.ontology is None:
                self._cache["prop_excl"] = [set(t) for t in self.excluded]
            else:
                desc = self.ontology.descendants
                self._cache["prop_excl"] = [set().union(*[desc(t) for t in ts]) if ts else set()
                                            for ts in self.excluded]
        return self._cache["prop_excl"]

    def has_negatives(self) -> bool:
        """True if any patient has an explicitly excluded phenotype recorded."""
        return any(self.excluded)

    def terms(self) -> list:
        """Sorted vocabulary of propagated terms present anywhere in the cohort."""
        if "terms" not in self._cache:
            self._cache["terms"] = sorted(set().union(*self.propagated()) if len(self) else set())
        return self._cache["terms"]

    def information_content(self) -> pd.Series:
        """IC(t) = -log2 f(t), the cohort frequency of ``t`` after propagation.

        Terms annotating everyone (the root) get IC = 0.
        """
        if "ic" not in self._cache:
            terms, prop, n = self.terms(), self.propagated(), max(len(self), 1)
            count = pd.Series(0, index=terms, dtype=float)
            for s in prop:
                if s:
                    count[list(s)] += 1
            freq = count / n
            ic = -np.log2(freq.where(freq > 0, 1.0))
            self._cache["ic"] = ic.where(count < n, 0.0)
        return self._cache["ic"]

    def feature_matrix(self, weight: str = "ic", polarity: str = "present") -> csr_matrix:
        """Sparse patients x terms matrix.

        ``weight`` is ``"ic"`` or ``"binary"``; ``polarity`` is ``"present"`` (the
        propagated phenotypes) or ``"excluded"`` (phenotypes recorded as looked for
        and absent, propagated downwards).
        """
        key = f"X_{weight}_{polarity}"
        if key not in self._cache:
            sets = self.propagated() if polarity == "present" else self.propagated_excluded()
            terms = self.terms() if polarity == "present" else self.excluded_terms()
            idx = {t: i for i, t in enumerate(terms)}
            ic = self.information_content()
            m = lil_matrix((len(self), len(terms)), dtype=np.float32)
            for i, s in enumerate(sets):
                for t in s:
                    if t in idx:
                        m[i, idx[t]] = float(ic.get(t, 1.0)) if weight == "ic" else 1.0
            self._cache[key] = m.tocsr()
        return self._cache[key]

    def excluded_terms(self) -> list:
        """Vocabulary of propagated excluded terms."""
        if "excl_terms" not in self._cache:
            sets = self.propagated_excluded()
            self._cache["excl_terms"] = sorted(set().union(*sets) if sets else set())
        return self._cache["excl_terms"]

    def distance(self, metric: str = "simgic", negatives: str = "ignore",
                 onset: str = "ignore") -> np.ndarray:
        """Square distance matrix between patients.

        Parameters
        ----------
        metric
            ``"simgic"`` (default) - the IC-weighted Jaccard of the propagated term
            sets (Pesquita et al. 2008). It is the ontology-aware measure the HPO
            literature uses, and it is the recommended default here.
            ``"cosine"`` - cosine distance between IC-weighted propagated vectors;
            a good sensitivity analysis rather than a primary choice.
        negatives
            ``"ignore"`` (default) uses only recorded phenotypes. ``"use"`` also
            uses **excluded** phenotypes: two patients in whom the same phenotype was
            looked for and ruled out are more similar for it, while presence in one
            and exclusion in the other counts only against them. Requires excluded
            phenotypes in the data (Phenopackets carry them; a plain term list does
            not) - the distinction *not recorded* vs *looked for and absent* is only
            available when it was recorded.
        onset
            Reserved; only ``"ignore"`` is implemented. Onset is read and reported by
            :func:`phenotopo.qc.phenotype_qc` but does not yet enter any similarity.
        """
        if onset != "ignore":
            raise ValueError("onset-aware similarity is not implemented yet (onset='ignore')")
        if negatives not in ("ignore", "use"):
            raise ValueError("negatives must be 'ignore' or 'use'")
        if negatives == "use" and not self.has_negatives():
            raise ValueError("no excluded phenotypes recorded in this cohort; use negatives='ignore'")
        key = f"D_{metric}_{negatives}"
        if key in self._cache:
            return self._cache[key]
        ic = self.information_content()

        if metric == "cosine":
            blocks = [self.feature_matrix("ic", "present")]
            if negatives == "use":
                blocks.append(self.feature_matrix("ic", "excluded"))
            w = hstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]
            norm = np.sqrt(np.asarray(w.multiply(w).sum(1)).ravel()) + 1e-12
            d = 1.0 - (w @ w.T).toarray() / np.outer(norm, norm)
        elif metric == "simgic":
            shared, total = self._simgic_parts("present", ic)
            if negatives == "use":
                s2, t2 = self._simgic_parts("excluded", ic)
                shared, total = shared + s2, total + t2
            union = total[:, None] + total[None, :] - shared
            with np.errstate(divide="ignore", invalid="ignore"):
                d = 1.0 - np.where(union > 0, shared / np.where(union > 0, union, 1.0), 0.0)
        else:
            raise ValueError("metric must be 'simgic' or 'cosine'")
        np.fill_diagonal(d, 0.0)
        d = np.clip((d + d.T) / 2.0, 0.0, None)
        self._cache[key] = d
        return d

    def _simgic_parts(self, polarity: str, ic):
        b = self.feature_matrix("binary", polarity)
        w = self.feature_matrix("ic", polarity)
        shared = (w @ b.T).toarray().astype(float)
        terms = self.terms() if polarity == "present" else self.excluded_terms()
        total = np.asarray(b @ np.array([float(ic.get(t, 1.0)) for t in terms])).ravel()
        return shared, total

    def distances(self, metrics=("simgic", "cosine"), **kwargs) -> dict:
        """``{name: matrix}`` for several metrics - ready for the robustness protocol."""
        return {m: self.distance(m, **kwargs) for m in metrics}

    def labels(self, column: str) -> np.ndarray:
        """Group labels from a metadata column, for connectivity / QC / comparison."""
        if column not in self.metadata.columns:
            raise KeyError(f"no metadata column {column!r}; have {list(self.metadata.columns)}")
        return self.metadata[column].to_numpy()


# ---- readers ------------------------------------------------------------
def from_hpo_table(
    table,
    id_col: str = "id",
    terms_col: str = "hpo",
    sep: str = ",",
    excluded_col: str | None = None,
    ontology: Ontology | str | None = None,
    metadata_cols=None,
) -> Cohort:
    """Cohort from a table (DataFrame, CSV or Excel path) of HPO term lists.

    ``terms_col`` holds terms separated by ``sep`` (``"HP:0001250, HP:0002133"``);
    anything that is not an ``HP:`` identifier is ignored. ``metadata_cols``
    (default: every other column) is carried through for grouping.
    """
    if isinstance(table, str):
        table = pd.read_excel(table) if table.lower().endswith((".xlsx", ".xls")) else pd.read_csv(table)
    df = pd.DataFrame(table)
    if isinstance(ontology, str):
        ontology = Ontology.from_obo(ontology)

    def split(v):
        if not isinstance(v, str):
            return set()
        return {t.strip() for t in v.split(sep) if t.strip().upper().startswith("HP:")}

    present = [split(v) for v in df[terms_col]]
    excluded = [split(v) for v in df[excluded_col]] if excluded_col else [set()] * len(df)
    cols = list(metadata_cols) if metadata_cols is not None else [
        c for c in df.columns if c not in {id_col, terms_col, excluded_col}]
    return Cohort(ids=[str(i) for i in df[id_col]], present=present, excluded=excluded,
                  metadata=df[cols] if cols else pd.DataFrame(index=df.index), ontology=ontology)


def from_phenopackets(path: str, ontology: Ontology | str | None = None) -> Cohort:
    """Cohort from GA4GH Phenopackets (v2 JSON): a directory tree of files, or one file.

    Directories are searched recursively, because real corpora nest their packets one
    folder per gene or disease.

    Reads ``phenotypicFeatures`` (id, ``excluded``, ``onset``), the subject id and
    sex, and, where present, the disease and gene from ``interpretations`` /
    ``diseases`` - so ``compare("gene")`` works straight after reading. Snake_case
    keys are accepted alongside camelCase.
    """
    if isinstance(ontology, str):
        ontology = Ontology.from_obo(ontology)
    if os.path.isdir(path):
        files = sorted(str(f) for f in Path(path).rglob("*.json"))     # corpora nest by gene/disease
    else:
        files = [path]
    if not files:
        raise ValueError(f"no .json phenopackets found in {path!r}")

    def get(d, *names, default=None):
        for n in names:
            if isinstance(d, dict) and n in d:
                return d[n]
        return default

    ids, present, excluded, onsets, meta = [], [], [], [], []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            packets = json.load(fh)
        for pkt in (packets if isinstance(packets, list) else [packets]):
            subject = get(pkt, "subject", default={}) or {}
            pid = str(get(subject, "id", default=None) or get(pkt, "id", default=os.path.basename(f)))
            pres, excl, ons = set(), set(), {}
            for feat in get(pkt, "phenotypicFeatures", "phenotypic_features", default=[]) or []:
                term = get(get(feat, "type", default={}) or {}, "id")
                if not term:
                    continue
                (excl if get(feat, "excluded", default=False) else pres).add(term)
                age = get(get(get(feat, "onset", default={}) or {}, "age", default={}) or {},
                          "iso8601duration", "iso8601Duration")
                if age:
                    ons[term] = age
            row = {"sex": get(subject, "sex", default=None)}
            diseases = get(pkt, "diseases", default=[]) or []
            if diseases:
                term = get(diseases[0], "term", default={}) or {}
                row["disease"] = get(term, "label") or get(term, "id")
            for interp in get(pkt, "interpretations", default=[]) or []:
                diag = get(interp, "diagnosis", default={}) or {}
                dt = get(diag, "disease", default={}) or {}
                row.setdefault("disease", get(dt, "label") or get(dt, "id"))
                for gi in get(diag, "genomicInterpretations", "genomic_interpretations", default=[]) or []:
                    vi = get(gi, "variantInterpretation", "variant_interpretation", default={}) or {}
                    vd = get(vi, "variationDescriptor", "variation_descriptor", default={}) or {}
                    gene = get(vd, "geneContext", "gene_context", default={}) or {}
                    symbol = get(gene, "symbol") or get(gene, "valueId", "value_id")
                    if symbol:
                        row.setdefault("gene", symbol)
            row["solved"] = bool(row.get("gene"))
            ids.append(pid); present.append(pres); excluded.append(excl); onsets.append(ons); meta.append(row)
    return Cohort(ids=ids, present=present, excluded=excluded, onset=onsets,
                  metadata=pd.DataFrame(meta), ontology=ontology)
