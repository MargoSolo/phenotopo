"""Generate the repository preview image (1280x640) from real package output.

    python examples/social_preview.py

Every number and mark below is produced by phenotopo itself, on the synthetic
cohort with designed faults - nothing is drawn by hand.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phenotopo import annotation_bias, connectivity_robustness, patient_outliers, phenotype_qc
from phenotopo.data import synthetic_hpo_cohort

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "social_preview.png")
BLUE, RED, INK, MUTED, GRID = "#2980b9", "#c0392b", "#1a1a1a", "#6b6b6b", "#e3e3e3"
GOOD, WARN = "#2b7a3d", "#b3541e"

cohort = synthetic_hpo_cohort(n=600, seed=0)
site = cohort.labels("site")
D = cohort.distance()
qc = phenotype_qc(cohort)
bias = annotation_bias(qc, site, distance=D)
rob = connectivity_robustness(cohort.distances(), cohort.labels("diagnosis"),
                              ks=(10, 15, 30), min_size=50, n_perm=300, n_boot=100)

# classical MDS (PCoA) of the phenotype distances - a plain map of the cohort
J = np.eye(len(D)) - 1.0 / len(D)
vals, vecs = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
emb = vecs[:, -2:][:, ::-1] * np.sqrt(np.maximum(vals[-2:][::-1], 0))

fig = plt.figure(figsize=(8, 4), dpi=160)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 3, left=0.06, right=0.975, top=0.60, bottom=0.185, wspace=0.38)

fig.text(0.055, 0.945, "phenotopo", fontsize=24, fontweight="bold", color=INK, va="top")
fig.text(0.055, 0.815, "Phenotype structure is easy to see, and easy to imagine.",
         fontsize=11, color=INK, va="top")
fig.text(0.055, 0.735, "This tells the two apart.", fontsize=11, color=MUTED, va="top")
fig.text(0.975, 0.945, "pip install phenotopo", fontsize=9.5, color=MUTED, ha="right", va="top",
         family="DejaVu Sans Mono")

def panel_title(ax, step, text):
    ax.set_title(f"{step} · {text}", fontsize=9, color=INK, loc="left", pad=6, fontweight="bold")

def chip(ax, text, colour, x=0.04, y=0.93):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, color=colour, fontweight="bold",
            va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=colour, edgecolor="none", alpha=0.13))

# 1 - how the cohort was annotated, before anything is believed
ax = fig.add_subplot(gs[0])
depth = [qc["patients"].loc[site == g, "n_terms"].to_numpy() for g in ("A", "B")]
bp = ax.boxplot(depth, widths=0.5, showfliers=False, patch_artist=True)
for patch, colour in zip(bp["boxes"], (BLUE, RED)):
    patch.set_facecolor(colour); patch.set_alpha(0.28); patch.set_edgecolor(colour)
for key in ("whiskers", "caps"):
    for artist in bp[key]:
        artist.set_color(MUTED)
for median in bp["medians"]:
    median.set_color(INK)
panel_title(ax, "1", "How was it annotated?")
ax.set_xticks([1, 2]); ax.set_xticklabels(["Site A", "Site B"], fontsize=8.5, color=INK)
ax.set_ylabel("HPO terms per patient", fontsize=8.5, color=MUTED)
ax.tick_params(axis="y", labelsize=7.5, colors=MUTED)
ax.set_ylim(0, max(max(d) for d in depth) * 1.62)
r = bias["recovery"]
chip(ax, f"annotation-confound risk: {r['confound_risk']}", WARN)
ax.text(0.045, 0.845, f"site predicted from term counts alone: {r['annotation_only']:.0%}",
        transform=ax.transAxes, fontsize=7, color=MUTED, va="center")
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.spines[["top", "right"]].set_visible(False)

# 2 - patients whose neighbours belong to the other group
ax = fig.add_subplot(gs[1])
diagnosis = cohort.labels("diagnosis")
out = patient_outliers(D, diagnosis, ids=cohort.ids, k=15).set_index("id")
flagged = np.array([bool("discordant" in out.loc[i, "flag"]) for i in cohort.ids])
for label, colour in (("GENE_A", BLUE), ("GENE_B", RED)):
    m = diagnosis == label
    ax.scatter(emb[m, 0], emb[m, 1], s=6, c=colour, alpha=0.55, edgecolors="none", label=label)
ax.scatter(emb[flagged, 0], emb[flagged, 1], s=42, facecolors="none", edgecolors=INK,
           linewidths=0.9, zorder=4, label=f"discordant ({int(flagged.sum())})")
panel_title(ax, "2", "Who sits in the wrong place?")
ax.legend(fontsize=7, frameon=True, framealpha=0.92, edgecolor=GRID, loc="upper left",
          markerscale=1.3, handletextpad=0.3, borderpad=0.4, labelspacing=0.3)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_color(GRID)

# 3 - what is reported: only verdicts that survive every configuration
ax = fig.add_subplot(gs[2])
summary = rob["summary"]
rows = list(summary.index)[::-1]
for i, pair in enumerate(rows):
    row = summary.loc[pair]
    verdict = row["verdict"]
    colour = GOOD if verdict in ("cohesive", "blend") else RED if verdict == "separation" else MUTED
    ax.errorbar(row["ratio"], i, xerr=[[max(row["ratio"] - row["ci_lo"], 0)],
                                       [max(row["ci_hi"] - row["ratio"], 0)]],
                fmt="o", ms=6, color=colour, ecolor=colour, elinewidth=1.6, capsize=2.5)
    mark = "" if verdict == "not robust" else "\u2713 "
    ax.text(0.97, i + 0.32, f"{mark}{verdict}", transform=ax.get_yaxis_transform(), fontsize=7.5,
            color=colour, va="center", ha="right", fontweight="bold" if mark else "normal")
ax.axvline(1, color=INK, lw=1)
ax.set_xscale("log")
ax.set_xlim(0.09, 40)
ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels([str(p).replace(" – ", "–").replace("GENE_", "") for p in rows], fontsize=7.5, color=INK)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel("connectivity ratio (log, 95 % CI)", fontsize=7.5, color=MUTED, labelpad=2)
ax.tick_params(axis="x", labelsize=7.5, colors=MUTED)
panel_title(ax, "3", "What is actually robust?")
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.spines[["top", "right"]].set_visible(False)

fig.text(0.5, 0.05, "HPO / GA4GH Phenopackets   ·   QC   ·   outliers   ·   group comparison   ·   robust structure   ·   MIT",
         fontsize=8.5, color=MUTED, ha="center")
fig.savefig(OUT, dpi=160, facecolor="white")
print("saved", OUT, "-", plt.imread(OUT).shape)
