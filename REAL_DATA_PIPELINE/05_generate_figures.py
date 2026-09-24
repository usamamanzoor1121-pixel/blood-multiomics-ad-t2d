#!/usr/bin/env python3
"""
05_generate_figures.py
========================
Publication-style figures for the real-data AD x T2D blood multi-omics
reanalysis, using the same validated colorblind-safe palette as the T2D
lncRNA project (references/palette.md in the dataviz skill).

New figures (existing ROC/SHAP/QC figures from the ML/preprocessing scripts
are reused as-is, not regenerated):
  FigA_AD_volcano.png         AD DEG volcano plot (real signal, 4,743 sig genes)
  FigB_T2D_volcano.png        T2D DEG volcano plot (real null result, 0 sig genes)
  FigC_BRS_T_distribution.png BRS-T (transcriptomic) score by diagnosis
  FigD_BIRTHS_distribution.png BIRTHS score by diagnosis
  FigE_ML_performance.png     AD vs T2D test AUC across all 5 models
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TAB = Path("data/tables")
FIG = Path("data/figures/analysis")
FIG.mkdir(parents=True, exist_ok=True)

# ── Validated palette (dataviz skill references/palette.md) ────────────────
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
SURFACE = "#fcfcfb"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.axisbelow": True, "savefig.dpi": 200,
})


def volcano(deg_path, title, outname, fc_thresh=0.3, p_thresh=0.05):
    df = pd.read_csv(deg_path)
    df["neglog10p"] = -np.log10(df["pvalue"].clip(lower=1e-300))
    sig = (df["padj"] < p_thresh) & (df["log2FC"].abs() > fc_thresh)
    n_sig = sig.sum()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(df.loc[~sig, "log2FC"], df.loc[~sig, "neglog10p"], s=8, alpha=0.3,
               color=INK_MUTED, linewidths=0, zorder=2, label=f"Not significant (n={(~sig).sum():,})")
    ax.scatter(df.loc[sig, "log2FC"], df.loc[sig, "neglog10p"], s=10, alpha=0.5,
               color=RED if n_sig > 0 else AQUA, linewidths=0, zorder=3,
               label=f"padj<{p_thresh} & |log2FC|>{fc_thresh} (n={n_sig:,})")

    if n_sig > 0:
        top = df[sig].sort_values("padj").head(8)
        for gene, row in top.iterrows():
            ax.annotate(df.loc[gene, "gene"], (row["log2FC"], row["neglog10p"]),
                        xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK_PRIMARY)

    ax.set_xlabel("log₂ fold-change (disease vs Control)")
    ax.set_ylabel("−log₁₀(p-value)")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9, markerscale=2)
    plt.tight_layout()
    plt.savefig(FIG / outname)
    plt.close()
    log.info(f"Saved {outname} ({n_sig:,} significant genes)")


def score_distribution(scores_path, score_col, group_order, colors, title, outname):
    df = pd.read_csv(scores_path, index_col=0)
    df = df[df["diagnosis"].isin(group_order)]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    positions = range(len(group_order))
    data = [df.loc[df["diagnosis"] == g, score_col].dropna().values for g in group_order]

    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                     showfliers=False, medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
                     whiskerprops=dict(color=INK_MUTED), capprops=dict(color=INK_MUTED))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=10, alpha=0.4,
                   color=color, linewidths=0, zorder=3)

    for i, (g, vals) in enumerate(zip(group_order, data)):
        ax.text(i, ax.get_ylim()[1], f"n={len(vals)}", ha="center", va="bottom",
                fontsize=8.5, color=INK_SECONDARY)

    ax.set_xticks(positions)
    ax.set_xticklabels(group_order, fontsize=10)
    ax.set_ylabel(score_col.replace("_", "-"))
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", pad=16, wrap=True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / outname)
    plt.close()
    log.info(f"Saved {outname}")


def ml_performance():
    ad = pd.read_csv(TAB / "ad_ml_results.csv")
    t2d = pd.read_csv(TAB / "t2d_ml_results.csv")
    models = ad["model"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(models))
    w = 0.36
    ax.bar(x - w / 2, t2d["test_auc"], width=w, color=BLUE, label="T2D (3-class, n=189)")
    ax.bar(x + w / 2, ad["test_auc"], width=w, color=ORANGE, label="AD (3-class, n=706)")
    ax.set_xlim(-0.9, len(models) - 0.3)
    ax.axhline(1 / 3, color=INK_MUTED, linewidth=1, linestyle="-", zorder=1)
    ax.text(-0.85, 1 / 3 + 0.02, "chance (1/3)", fontsize=8, color=INK_MUTED, ha="left")

    for i, (t2d_v, ad_v) in enumerate(zip(t2d["test_auc"], ad["test_auc"])):
        ax.text(i - w / 2, t2d_v + 0.01, f"{t2d_v:.2f}", ha="center", fontsize=8, color=INK_SECONDARY)
        ax.text(i + w / 2, ad_v + 0.01, f"{ad_v:.2f}", ha="center", fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0, 1.0)
    ax.set_title("Real-data ML classification: AD outperforms T2D\n(contrary to the original simulated-data narrative)",
                  fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "FigE_ML_performance.png")
    plt.close()
    log.info("Saved FigE_ML_performance.png")


if __name__ == "__main__":
    volcano(TAB / "ad_deg_results.csv", "AD blood DEG volcano plot (real GSE63060+61 data)",
            "FigA_AD_volcano.png")
    volcano(TAB / "t2d_deg_results.csv", "T2D blood DEG volcano plot (real GSE221521 data)",
            "FigB_T2D_volcano.png")
    score_distribution(TAB / "brs_t_scores_real.csv", "BRS_T", ["Control", "MCI", "AD"],
                        [BLUE, AQUA, ORANGE],
                        "BRS-T (transcriptomic-only) by diagnosis — no meaningful separation",
                        "FigC_BRS_T_distribution.png")
    score_distribution(TAB / "births_scores_real.csv", "BIRTHS", ["Control", "PreDM", "T2D"],
                        [BLUE, AQUA, ORANGE],
                        "BIRTHS score by diagnosis — real, modest, significant\n(Kruskal-Wallis p=0.008)",
                        "FigD_BIRTHS_distribution.png")
    ml_performance()
    log.info("All figures generated in data/figures/analysis/")
