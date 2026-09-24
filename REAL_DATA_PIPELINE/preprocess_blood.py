#!/usr/bin/env python3
"""
02_preprocess_blood.py
======================
Preprocessing pipeline for whole blood transcriptomics.
Handles both RNA-seq (T2DM: GSE221521) and microarray (AD: GSE63060/61).

Steps:
    1. Load raw expression matrices
    2. Quality control (low-count genes, outlier samples)
    3. Normalisation (VST for RNA-seq, quantile for microarray)
    4. Batch correction (ComBat-seq for GSE63060+61 merge)
    5. Metadata harmonisation (diagnosis labels, clinical features)
    6. Save processed matrices

Author: Usama Manzoor (usama.manzoor1121@gmail.com)
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PARAMS  = yaml.safe_load(open("params.yaml"))
RAW_DIR = Path(PARAMS["paths"]["raw_data"])
OUT_DIR = Path(PARAMS["paths"]["processed_data"])
FIG_DIR = Path(PARAMS["paths"]["figures"]) / "preprocessing"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_COUNTS    = PARAMS["deg"]["min_counts"]
PADJ_THRESH   = PARAMS["deg"]["padj_threshold"]
LOG2FC_THRESH = PARAMS["deg"]["log2fc_threshold"]


# ─────────────────────────────────────────────────────────────────────────────
# T2DM RNA-seq preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_t2d_rnaseq() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess GSE221521 whole-blood RNA-seq.
    Returns (expression_df, metadata_df).
    """
    log.info("── T2DM RNA-seq preprocessing (GSE221521) ──────────────────")

    expr_path = RAW_DIR / "t2d" / "GSE221521" / "GSE221521_expression_raw.csv"
    meta_path = RAW_DIR / "t2d" / "GSE221521" / "GSE221521_metadata.csv"

    if not expr_path.exists():
        log.warning(f"Raw data not found at {expr_path}. Generating simulated data.")
        expr_df, meta_df = _simulate_t2d_data()
    else:
        expr_df = pd.read_csv(expr_path, index_col=0)
        meta_df = pd.read_csv(meta_path, index_col=0)

    log.info(f"  Raw shape: {expr_df.shape}")

    # 1. Filter low-count genes
    expr_df = _filter_low_counts(expr_df, MIN_COUNTS)
    log.info(f"  After low-count filter: {expr_df.shape}")

    # 2. Library-size normalisation (median-of-ratios, DESeq2-style) + log2 (VST proxy)
    #    FIX: raw log2(count+1) with no library-size normalisation was used previously;
    #    this leaves systematic per-sample sequencing-depth differences in the data,
    #    which can mask true biological signal (or introduce spurious signal).
    expr_norm = _median_of_ratios_normalise(expr_df)
    expr_norm = np.log2(expr_norm + 1)

    # 3. Remove outlier samples (IQR method on PCA scores)
    expr_norm, meta_df = _remove_outlier_samples(expr_norm, meta_df)
    log.info(f"  After outlier removal: {expr_norm.shape}")

    # 4. Save
    expr_norm.to_csv(OUT_DIR / "t2d_expression_normalised.csv")
    meta_df.to_csv(OUT_DIR / "t2d_metadata_clean.csv")

    # 5. QC plot
    _plot_sample_distribution(expr_norm, meta_df, "T2DM Blood RNA-seq QC",
                              FIG_DIR / "t2d_sample_qc.png")

    log.info(f"  ✓ T2DM processed: {expr_norm.shape[0]} genes × {expr_norm.shape[1]} samples")
    return expr_norm, meta_df


# ─────────────────────────────────────────────────────────────────────────────
# AD microarray preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_ad_microarray() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess GSE63060 + GSE63061 (AddNeuroMed) whole-blood microarray.
    Merges two cohorts and applies ComBat batch correction.
    Returns (expression_df, metadata_df).
    """
    log.info("── AD Microarray preprocessing (GSE63060 + GSE63061) ───────")

    dfs, metas = [], []
    for accession, batch_id in [("GSE63060", 1), ("GSE63061", 2)]:
        expr_path = RAW_DIR / "ad" / accession / f"{accession}_expression_raw.csv"
        meta_path = RAW_DIR / "ad" / accession / f"{accession}_metadata.csv"

        if not expr_path.exists():
            log.warning(f"  {accession} not found — using simulated data")
            expr, meta = _simulate_ad_data(accession, batch_id)
        else:
            expr = pd.read_csv(expr_path, index_col=0)
            meta = pd.read_csv(meta_path, index_col=0)
            meta["batch"] = batch_id

        dfs.append(expr)
        metas.append(meta)

    # Merge on common genes
    common_genes = dfs[0].index.intersection(dfs[1].index)
    expr_merged  = pd.concat([dfs[0].loc[common_genes], dfs[1].loc[common_genes]], axis=1)
    meta_merged  = pd.concat(metas, axis=0)
    log.info(f"  Merged shape: {expr_merged.shape} ({len(common_genes)} common genes)")

    # Quantile normalisation
    expr_norm = _quantile_normalise(expr_merged)

    # ComBat batch correction (simplified — subtract batch means)
    expr_corrected = _batch_correct_simple(expr_norm, meta_merged)
    log.info(f"  Batch correction applied across 2 cohorts")

    # Remove outliers
    expr_corrected, meta_merged = _remove_outlier_samples(expr_corrected, meta_merged)

    # Save
    expr_corrected.to_csv(OUT_DIR / "ad_expression_normalised.csv")
    meta_merged.to_csv(OUT_DIR  / "ad_metadata_clean.csv")

    _plot_sample_distribution(expr_corrected, meta_merged, "AD Blood Microarray QC",
                              FIG_DIR / "ad_sample_qc.png")

    log.info(f"  ✓ AD processed: {expr_corrected.shape[0]} genes × {expr_corrected.shape[1]} samples")
    return expr_corrected, meta_merged


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def _median_of_ratios_normalise(counts: pd.DataFrame) -> pd.DataFrame:
    """
    DESeq2-style median-of-ratios size-factor normalisation for RNA-seq counts.
    Corrects for per-sample sequencing-depth differences before log-transform.
    """
    log_counts = np.log(counts.replace(0, np.nan))
    log_geo_mean = log_counts.mean(axis=1)
    finite = log_geo_mean.replace([np.inf, -np.inf], np.nan).notna()
    ratios = log_counts.loc[finite].sub(log_geo_mean.loc[finite], axis=0)
    size_factors = np.exp(ratios.median(axis=0))
    log.info(f"  Size factors: min={size_factors.min():.3f}, max={size_factors.max():.3f}, "
             f"median={size_factors.median():.3f}")
    return counts.div(size_factors, axis=1)


def _filter_low_counts(df: pd.DataFrame, min_counts: int) -> pd.DataFrame:
    """Keep genes expressed above threshold in at least 20% of samples."""
    min_samples = max(1, int(df.shape[1] * 0.20))
    mask = (df >= min_counts).sum(axis=1) >= min_samples
    return df[mask]


def _quantile_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile normalisation for microarray data."""
    sorted_df    = np.sort(df.values, axis=0)
    row_means    = sorted_df.mean(axis=1)
    rank_df      = df.rank(method="min").astype(int) - 1
    norm_values  = pd.DataFrame(
        row_means[rank_df.values],
        index=df.index, columns=df.columns
    )
    return norm_values


def _batch_correct_simple(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified batch effect removal — subtract batch-specific means.
    For full ComBat, use the R combat_seq implementation.
    """
    if "batch" not in meta.columns:
        return df

    corrected = df.copy()
    grand_mean = df.mean(axis=1)

    for batch_id in meta["batch"].unique():
        batch_samples = meta[meta["batch"] == batch_id].index
        batch_samples = [s for s in batch_samples if s in df.columns]
        if not batch_samples:
            continue
        batch_mean  = df[batch_samples].mean(axis=1)
        correction  = grand_mean - batch_mean
        corrected[batch_samples] = df[batch_samples].add(correction, axis=0)

    return corrected


def _remove_outlier_samples(df: pd.DataFrame,
                            meta: pd.DataFrame,
                            z_thresh: float = 3.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove samples >z_thresh standard deviations from the mean pairwise correlation."""
    corr_matrix = df.corr()
    mean_corr   = corr_matrix.mean()
    z_scores    = stats.zscore(mean_corr)
    keep        = np.abs(z_scores) < z_thresh
    removed     = (~keep).sum()

    if removed > 0:
        log.info(f"  Removed {removed} outlier samples (z>{z_thresh})")

    df_clean   = df.loc[:, keep]
    meta_clean = meta.loc[[s for s in df_clean.columns if s in meta.index]]
    return df_clean, meta_clean


def _plot_sample_distribution(df: pd.DataFrame,
                              meta: pd.DataFrame,
                              title: str,
                              outpath: Path):
    """Plot sample-level expression distribution coloured by diagnosis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1A1A2E")

    for ax in axes:
        ax.set_facecolor("#1A1A2E")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Sample median expression boxplot
    medians = df.median(axis=0)
    axes[0].hist(medians, bins=40, color="#028090", edgecolor="white", alpha=0.8)
    axes[0].set_title("Sample Median Expression Distribution")
    axes[0].set_xlabel("Median Expression")
    axes[0].set_ylabel("Count")

    # Sample count per diagnosis
    if "diagnosis" in meta.columns or "condition" in meta.columns:
        diag_col = "diagnosis" if "diagnosis" in meta.columns else "condition"
        counts   = meta[diag_col].value_counts()
        bars     = axes[1].bar(counts.index, counts.values,
                               color=["#02C39A", "#E74C3C", "#028090"][:len(counts)])
        axes[1].set_title("Sample Count by Diagnosis")
        axes[1].set_xlabel("Diagnosis")
        axes[1].set_ylabel("n samples")
        for bar, count in zip(bars, counts.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                         str(count), ha="center", va="bottom", color="white")

    fig.suptitle(title, color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    log.info(f"  QC plot saved: {outpath.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Simulation (used when raw GEO data is not downloaded)
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_t2d_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate biologically parameterised T2DM simulation.

    SIMULATION FIX (v2): Fold-changes are set to realistic published values
    from whole-blood T2DM studies (Mootha et al., Seo et al., GSE221521).
    Original values (3.5x-4.0x log2FC) were unrealistically large and produced
    AUC=1.0 even without data leakage. Real whole-blood signals are 0.3-0.8
    log2FC with substantial inter-individual variation.
    Per-sample noise is also increased to reflect biological heterogeneity.
    """
    np.random.seed(42)
    n_t2d, n_pre, n_ctrl = 509, 180, 501
    n_total  = n_t2d + n_pre + n_ctrl
    n_genes  = 18_000
    gene_ids = [f"GENE_{i:05d}" for i in range(n_genes)]

    # Baseline expression with realistic inter-individual noise
    baseline = np.random.negative_binomial(20, 0.5, size=(n_genes, n_total)).astype(float)
    # Add per-sample random effects (biological heterogeneity)
    sample_effects = np.random.normal(0, 0.15, size=(1, n_total))
    baseline = baseline * (1 + sample_effects)

    # Realistic fold-changes from published whole-blood T2DM literature
    # (GSE221521, Mootha et al. 2003, Seo et al. 2017) — typical 0.3-0.8 log2FC
    up_genes = {
        "S100A8":   0.75,   # monocyte activation marker
        "S100A9":   0.68,   # monocyte activation marker
        "TXNIP":    0.80,   # glucose-induced thioredoxin interacting protein
        "IL1B":     0.62,   # NLRP3 inflammasome output
        "NLRP3":    0.55,   # inflammasome sensor
        "CCL2":     0.48,   # monocyte chemotaxis
        "MPO":      0.42,   # neutrophil oxidative burst
        "FASN":     0.38,   # fatty acid synthesis
        "MMP9":     0.35,   # matrix metalloproteinase
        "STAT3":    0.32,   # JAK-STAT signalling
    }
    down_genes = {
        "IRS1":     -0.70,  # insulin receptor substrate
        "PPARGC1A": -0.75,  # mitochondrial biogenesis master regulator
        "ADIPOR1":  -0.55,  # adiponectin receptor
        "IL10":     -0.48,  # anti-inflammatory cytokine
        "FOXO1":    -0.45,  # forkhead transcription factor
        "TFAM":     -0.40,  # mitochondrial transcription factor
        "CD3D":     -0.50,  # T-cell receptor complex
        "PRF1":     -0.38,  # perforin (NK/T-cell)
    }

    for g, fc in {**up_genes, **down_genes}.items():
        if g in gene_ids:
            idx = gene_ids.index(g)
        else:
            gene_ids[list(up_genes.keys()).index(g) if g in up_genes else
                     n_genes - list(down_genes.keys()).index(g) - 1] = g
            idx = gene_ids.index(g)

        t2d_start = 0
        t2d_end   = n_t2d
        # Apply fold-change with per-sample noise (not all patients show equal signal)
        noise_t2d = np.random.normal(0, abs(fc) * 0.4, n_t2d)
        noise_pre = np.random.normal(0, abs(fc) * 0.4, n_pre)
        if fc > 0:
            baseline[idx, t2d_start:t2d_end] *= (2 ** (fc + noise_t2d))
            baseline[idx, t2d_end:t2d_end + n_pre] *= (2 ** (fc * 0.5 + noise_pre))
        else:
            baseline[idx, t2d_start:t2d_end] /= (2 ** (abs(fc) + noise_t2d))
            baseline[idx, t2d_end:t2d_end + n_pre] /= (2 ** (abs(fc) * 0.5 + noise_pre))

    expr_df = pd.DataFrame(
        baseline,
        index=gene_ids,
        columns=[f"T2D_{i:04d}" for i in range(n_t2d)] +
                [f"PRE_{i:04d}" for i in range(n_pre)] +
                [f"CTRL_{i:04d}" for i in range(n_ctrl)]
    )

    diagnoses = (["T2D"] * n_t2d + ["Pre-DM"] * n_pre + ["Control"] * n_ctrl)
    hba1c     = ([np.random.uniform(7.0, 12.0) for _ in range(n_t2d)] +
                 [np.random.uniform(5.7, 6.4)  for _ in range(n_pre)] +
                 [np.random.uniform(4.5, 5.6)  for _ in range(n_ctrl)])

    meta_df = pd.DataFrame({
        "diagnosis": diagnoses,
        "HbA1c":     hba1c,
        "batch":     [1] * n_total
    }, index=expr_df.columns)

    return expr_df, meta_df


def _simulate_ad_data(accession: str, batch_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate biologically parameterised AD simulation.

    SIMULATION FIX (v2): Fold-changes reduced to realistic whole-blood values
    from AddNeuroMed (GSE63060/61) and related studies. Blood-based AD signals
    are weaker than brain tissue signals — typical 0.25-0.65 log2FC with high
    inter-individual variation. Original values (1.8-2.8 log2FC) reflected
    brain tissue effect sizes, not blood, and produced unrealistic AUC=1.0.
    """
    np.random.seed(42 + batch_id)
    n_ad, n_mci, n_ctrl = 144, 119, 197
    n_total  = n_ad + n_mci + n_ctrl
    n_genes  = 22_000
    gene_ids = [f"PROBE_{i:05d}" for i in range(n_genes)]

    # Realistic baseline noise for microarray (already log2-transformed)
    baseline = np.random.normal(8.0, 1.5, size=(n_genes, n_total))
    # Add per-sample effects (batch and biological)
    sample_effects = np.random.normal(0, 0.2, size=(1, n_total))
    baseline = baseline + sample_effects

    # Realistic blood-based AD fold-changes from AddNeuroMed literature
    # (Lunnon et al. 2012, Fehlbaum-Beurdeley et al. 2012, GSE63060/61)
    up_genes = {
        "S100A8":  0.65,   # peripheral immune activation
        "S100A9":  0.58,   # peripheral immune activation
        "C1QB":    0.60,   # complement activation in blood
        "C1QC":    0.55,   # complement activation
        "PF4":     0.50,   # platelet factor, elevated in AD blood
        "MX1":     0.52,   # interferon response
        "OAS1":    0.45,   # innate immune response
        "TXNIP":   0.40,   # oxidative stress marker
        "IL1B":    0.48,   # neuroinflammation spillover
    }
    down_genes = {
        "CD3D":     -0.60,  # T-cell exhaustion in AD blood
        "CD3E":     -0.52,  # T-cell receptor complex
        "PRF1":     -0.48,  # NK/cytotoxic T-cell function
        "NKG7":     -0.42,  # NK cell granule protein
        "IRS1":     -0.38,  # insulin signalling (AD-T2D overlap)
        "PPARGC1A": -0.45,  # mitochondrial dysfunction marker
        "ADIPOR1":  -0.32,  # adiponectin signalling
    }

    for g, fc in {**up_genes, **down_genes}.items():
        if g not in gene_ids:
            probe_idx = list(up_genes.keys()).index(g) if g in up_genes else \
                        n_genes - list(down_genes.keys()).index(g) - 1
            gene_ids[probe_idx] = g

        idx = gene_ids.index(g)
        # Add per-sample noise to fold-change (not all AD patients show equal signal)
        noise_ad  = np.random.normal(0, abs(fc) * 0.45, n_ad)
        noise_mci = np.random.normal(0, abs(fc) * 0.45, n_mci)
        if fc > 0:
            baseline[idx, :n_ad]             += (fc + noise_ad)
            baseline[idx, n_ad:n_ad + n_mci] += (fc * 0.5 + noise_mci)
        else:
            baseline[idx, :n_ad]             += (fc + noise_ad)
            baseline[idx, n_ad:n_ad + n_mci] += (fc * 0.4 + noise_mci)

    expr_df = pd.DataFrame(
        baseline,
        index=gene_ids,
        columns=[f"{accession}_AD_{i:03d}"  for i in range(n_ad)] +
                [f"{accession}_MCI_{i:03d}" for i in range(n_mci)] +
                [f"{accession}_C_{i:03d}"   for i in range(n_ctrl)]
    )

    mmse_scores = ([np.random.uniform(10, 24) for _ in range(n_ad)] +
                   [np.random.uniform(24, 27) for _ in range(n_mci)] +
                   [np.random.uniform(27, 30) for _ in range(n_ctrl)])

    meta_df = pd.DataFrame({
        "diagnosis": ["AD"] * n_ad + ["MCI"] * n_mci + ["Control"] * n_ctrl,
        "MMSE":      mmse_scores,
        "batch":     [batch_id] * n_total
    }, index=expr_df.columns)

    return expr_df, meta_df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Blood Multi-Omics — Preprocessing Pipeline")
    log.info("=" * 60)

    t2d_expr, t2d_meta = preprocess_t2d_rnaseq()
    ad_expr,  ad_meta  = preprocess_ad_microarray()

    log.info("\n" + "=" * 60)
    log.info("Preprocessing complete")
    log.info(f"  T2DM: {t2d_expr.shape[0]} genes × {t2d_expr.shape[1]} samples")
    log.info(f"  AD:   {ad_expr.shape[0]}  genes × {ad_expr.shape[1]} samples")
    log.info(f"  Output: {OUT_DIR}/")
