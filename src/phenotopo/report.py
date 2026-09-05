"""One self-contained HTML file with everything a cohort review needs.

Clinicians and co-authors do not run Python, and a folder of PNGs is not a
result. :func:`cohort_report` assembles the QC table, the annotation-bias check,
the outlier list, the robustness verdicts and any group comparisons into a single
file that opens in a browser, travels by email and needs no server. Figures are
embedded, so nothing breaks when the file is moved.

Nothing leaves the machine: the report is written to disk, and no network call is
made anywhere in this package.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import os

import pandas as pd

_CSS = """
:root { color-scheme: light; }
body { font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 1080px; padding: 32px 24px 64px; color: #1a1a1a; background: #fff; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 36px 0 8px; border-bottom: 1px solid #e3e3e3; padding-bottom: 4px; }
.sub { color: #666; margin: 0 0 24px; font-size: 13px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0 4px; }
.card { border: 1px solid #e3e3e3; border-radius: 6px; padding: 10px 14px; min-width: 130px; }
.card .v { font-size: 20px; font-weight: 600; } .card .k { color: #666; font-size: 12px; }
table { border-collapse: collapse; font-size: 12.5px; margin: 10px 0; width: 100%; }
th, td { border: 1px solid #e3e3e3; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f2f6fa; font-weight: 600; } tr:nth-child(even) td { background: #fafafa; }
.note { color: #555; font-size: 13px; margin: 6px 0 0; } .warn { color: #a33; font-weight: 600; }
.ok { color: #2b7a3d; font-weight: 600; } img { max-width: 100%; height: auto; display: block; margin: 10px 0; }
.wrap { overflow-x: auto; } footer { margin-top: 48px; color: #888; font-size: 12px; }
"""


def _fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _table(df: pd.DataFrame, max_rows: int = 40, float_fmt: str = "{:.3g}") -> str:
    if df is None or not len(df):
        return '<p class="note">nothing to show.</p>'
    shown = df.head(max_rows)
    out = f'<div class="wrap">{shown.to_html(index=False, float_format=lambda v: float_fmt.format(v), border=0)}</div>'
    if len(df) > max_rows:
        out += f'<p class="note">{len(df) - max_rows} further rows omitted.</p>'
    return out


def _cards(pairs) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="v">{html.escape(str(v))}</div><div class="k">{html.escape(str(k))}</div></div>'
        for k, v in pairs) + "</div>"


def cohort_report(
    cohort,
    labels=None,
    distance=None,
    path: str = "cohort_report.html",
    label_name: str = "group",
    robustness: dict | None = None,
    comparisons=None,
    figures=None,
    title: str | None = None,
    k: int = 15,
) -> str:
    """Write the report and return its path.

    Parameters
    ----------
    cohort, labels, distance
        The cohort, an optional grouping (a metadata column name or an array) and
        an optional phenotype distance matrix. QC always runs; the annotation-bias
        check needs ``labels``, the outlier section needs ``distance``.
    robustness
        Result of :func:`phenotopo.connectivity_robustness`; its verdict table and
        forest plot are included.
    comparisons
        Results of :func:`phenotopo.explain_groups` (one, or a list).
    figures
        Extra ``(caption, matplotlib figure)`` pairs to embed.
    """
    from .qc import annotation_bias, phenotype_qc, plot_qc
    from .outliers import patient_outliers

    if isinstance(labels, str):
        label_name, labels = labels, cohort.labels(labels)
    qc = phenotype_qc(cohort)
    s = qc["summary"]
    parts = [f"<h1>{html.escape(title or 'Phenotype cohort report')}</h1>",
             f'<p class="sub">{len(cohort)} patients · generated {_dt.date.today().isoformat()} · '
             f'phenotopo — runs locally, no data leaves this machine</p>']

    parts.append("<h2>Cohort overview</h2>")
    parts.append(_cards([
        ("patients", len(cohort)),
        ("median HPO terms", f"{s['median_terms']:.0f}"),
        ("low annotation depth", f"{s['pct_low_annotation_depth']:.0f} %"),
        ("low specificity", f"{s['pct_low_specificity']:.0f} %"),
        ("with excluded phenotypes", f"{s['pct_with_excluded_phenotypes']:.0f} %"),
        ("with onset recorded", f"{s['pct_with_onset']:.0f} %"),
    ]))

    parts.append("<h2>Phenotyping quality</h2>")
    parts.append(f'<p class="note">Review recommended for <b>{s["pct_review_recommended"]:.0f} %</b> of patients — '
                 f'{s["pct_low_annotation_depth"]:.0f} % with ≤ {qc["thresholds"]["min_terms"]} terms, '
                 f'{s["pct_low_specificity"]:.0f} % below the {qc["thresholds"]["specificity_pct"]:.0f}th percentile of '
                 f'information content within this cohort, {s["pct_with_redundant_terms"]:.0f} % carrying redundant '
                 f'ancestor terms. Flags are relative to this cohort and descriptive: a short, well-chosen '
                 f'phenotype list is not necessarily a poor one.</p>')
    try:
        parts.append(_fig_to_img(plot_qc(qc, labels)))
    except Exception:
        pass
    parts.append(_table(qc["flagged"][["id", "n_terms", "mean_ic", "mean_depth", "n_redundant",
                                       "specificity_pct", "flags"]], max_rows=25))

    if labels is not None:
        bias = annotation_bias(qc, labels, distance=distance, k=k)
        parts.append(f"<h2>Annotation bias across {html.escape(label_name)}s</h2>")
        parts.append(_table(bias["by_group"].reset_index(), max_rows=25))
        for col, t in bias["kruskal"].items():
            parts.append(f'<p class="note">{col}: Kruskal–Wallis H = {t["H"]:.1f}, p = {t["p"]:.2g}, '
                         f'ε² = {t["epsilon_sq"]:.3f} ({t["effect"]} effect).</p>')
        r = bias.get("recovery")
        if r:
            cls = "warn" if r["confound_risk"] != "LOW" else "ok"
            parts.append(f'<p class="note">Group recovery by cross-validated k-NN: phenotype '
                         f'<b>{r["phenotype"]:.1%}</b>, annotation counts alone <b>{r["annotation_only"]:.1%}</b>, '
                         f'majority baseline {r["majority_baseline"]:.1%} → annotation-confound risk '
                         f'<span class="{cls}">{html.escape(r["confound_risk"])}</span>. '
                         f'{html.escape(r["note"])}</p>')

    if distance is not None:
        out = patient_outliers(distance, labels, ids=cohort.ids, k=k)
        flagged = out[out["flag"] != ""]
        parts.append("<h2>Outliers</h2>")
        parts.append(f'<p class="note">{len(flagged)} of {len(out)} patients flagged. '
                     f'“Discordant” means the phenotype neighbourhood belongs to a different '
                     f'{html.escape(label_name)} — a candidate for review, not a statement that the assignment '
                     f'is wrong.</p>')
        cols = [c for c in ["id", "label", "discordance", "neighbourhood_majority", "isolation_pct",
                            "flag", "nearest"] if c in flagged.columns]
        parts.append(_table(flagged[cols], max_rows=30))

    if robustness is not None:
        parts.append("<h2>Group connectivity — robustness protocol</h2>")
        cfgs = ", ".join(f"{d} k={k_}" for d, k_ in robustness["configs"])
        parts.append(f'<p class="note">{robustness["n_patients"]} patients, '
                     f'{len(robustness["groups"])} groups ≥ {robustness["thresholds"]["min_size"]}, '
                     f'{len(robustness["configs"])} configurations ({html.escape(cfgs)}). A verdict is issued only '
                     f'where the criterion holds in every configuration.</p>')
        summary = robustness["summary"].reset_index()
        keep = [c for c in ["pair", "ratio", "ci_lo", "ci_hi", "ratio_min", "ratio_max",
                            "n_configs_sig", "verdict"] if c in summary.columns]
        parts.append(_table(summary[keep], max_rows=40))
        try:
            from .robustness import plot_forest
            parts.append(_fig_to_img(plot_forest(robustness).figure))
        except Exception:
            pass

    if comparisons is not None:
        comps = comparisons if isinstance(comparisons, (list, tuple)) else [comparisons]
        for res in comps:
            a, b = res["groups"]
            parts.append(f"<h2>{html.escape(str(a))} vs {html.escape(str(b))}</h2>")
            parts.append(f'<p class="note">n = {res["sizes"]["a"]} vs {res["sizes"]["b"]}; '
                         f'{res["n_tested"]} terms tested; {html.escape(res["method"])}; reported above '
                         f'{100 * res["thresholds"]["min_effect"]:.0f} percentage points.</p>')
            cols = ["name", "prevalence_a", "prevalence_b", "effect_pp", "ci_lo_pp", "ci_hi_pp", "p_adjusted"]
            parts.append(_table(res["top"][cols], max_rows=25))
            try:
                from .explain import plot_explain
                parts.append(_fig_to_img(plot_explain(res).figure))
            except Exception:
                pass

    for caption, fig in (figures or []):
        parts.append(f"<h2>{html.escape(str(caption))}</h2>{_fig_to_img(fig)}")

    parts.append('<footer>Generated by phenotopo. Verdicts are descriptive: they say a phenotype pattern is or is '
                 'not robust, never that a diagnosis is right or wrong.</footer>')
    doc = ("<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>{html.escape(title or 'Phenotype cohort report')}</title><style>{_CSS}</style></head>"
           f"<body>{''.join(parts)}</body></html>")
    path = os.path.abspath(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
