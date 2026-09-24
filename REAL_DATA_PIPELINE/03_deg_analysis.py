#!/usr/bin/env python3
"""
03_deg_analysis.py
===================
Differential expression on the REAL preprocessed blood data, per disease.
T2D: T2D vs Control (Welch t-test on log2 VST-proxy values, BH-FDR).
AD:  AD vs Control (Welch t-test on log2 quantile-normalised/batch-corrected
     microarray values, BH-FDR).

Outputs (feed compute_real_crossdisease_concordance() in 01_biomarker_fixes.py):
    data/tables/t2d_deg_results.csv   (gene, log2FC, pvalue, padj)
    data/tables/ad_deg_results.csv    (gene, log2FC, pvalue, padj)
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROC = Path("data/processed")
OUT = Path("data/tables")
OUT.mkdir(parents=True, exist_ok=True)


def run_deg(expr_path, meta_path, case_label, ctrl_label, out_path, disease_name):
    log.info("=" * 60)
    log.info(f"DEG analysis — {disease_name}: {case_label} vs {ctrl_label}")
    log.info("=" * 60)

    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path, index_col=0)
    common = expr.columns.intersection(meta.index)
    expr = expr[common]
    meta = meta.loc[common]

    case_idx = meta[meta["diagnosis"] == case_label].index
    ctrl_idx = meta[meta["diagnosis"] == ctrl_label].index
    log.info(f"  {case_label} n={len(case_idx)}, {ctrl_label} n={len(ctrl_idx)}")

    results = []
    case_vals_all = expr[case_idx].values
    ctrl_vals_all = expr[ctrl_idx].values
    for i, gene in enumerate(expr.index):
        case_vals = case_vals_all[i]
        ctrl_vals = ctrl_vals_all[i]
        if case_vals.std() + ctrl_vals.std() < 1e-9:
            continue
        t, p = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False)
        lfc = case_vals.mean() - ctrl_vals.mean()  # data already log2-scale
        results.append({"gene": gene, "log2FC": lfc, "pvalue": p})

    deg = pd.DataFrame(results)
    _, padj, _, _ = multipletests(deg["pvalue"], method="fdr_bh")
    deg["padj"] = padj
    deg = deg.sort_values("padj")
    deg.to_csv(out_path, index=False)

    n_sig = (deg["padj"] < 0.05).sum()
    n_sig_lfc = ((deg["padj"] < 0.05) & (deg["log2FC"].abs() > 0.3)).sum()
    log.info(f"  Tested {len(deg):,} genes")
    log.info(f"  Significant (padj<0.05): {n_sig:,}")
    log.info(f"  Significant + |log2FC|>0.3: {n_sig_lfc:,}")
    log.info(f"  Saved: {out_path}")
    return deg


if __name__ == "__main__":
    run_deg(
        PROC / "t2d_expression_normalised.csv",
        PROC / "t2d_metadata_clean.csv",
        "T2D", "Control",
        OUT / "t2d_deg_results.csv", "T2DM"
    )
    run_deg(
        PROC / "ad_expression_normalised.csv",
        PROC / "ad_metadata_clean.csv",
        "AD", "Control",
        OUT / "ad_deg_results.csv", "AD"
    )
    log.info("\nDEG analysis complete for both diseases.")
