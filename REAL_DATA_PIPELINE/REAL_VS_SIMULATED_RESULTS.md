# Real Data vs Simulated Data — Final Comparison

Generated 2026-09-24 by re-running the pipeline on real GEO data (GSE63060, GSE63061,
GSE221521) instead of the `np.random`-simulated data that the pushed GitHub repo and
the manuscript actually used. All "Real" numbers below are reproducible from
`REAL_DATA_PIPELINE/` — see `logs/` for full run logs.

## Sample sizes (real, after QC)

| Cohort | Claimed in README/manuscript | Actually available in public supplementary files | After QC |
|---|---|---|---|
| GSE221521 (T2D) | 509 T2D + 180 Pre-DM + 501 Control (n=1,190) | 74 T2D + 69 Pre-DM + 50 Control (n=193) | 189 |
| GSE63060 (AD) | part of n=919 combined | 145 AD + 104 Control + 80 MCI (n=329) | — |
| GSE63061 (AD) | part of n=919 combined | 139 AD + 134 Control + 109 MCI (n=382, 6 dropped unresolved) | — |
| GSE63060+61 combined | 287 AD + 238 MCI + 394 Control (n=919) | 284 AD + 189 MCI + 238 Control (n=711) | 706 |

The RNA-seq supplementary file for GSE221521 only contains 193 of the samples in the
full series (the series has additional samples not included in the deposited
`_gene_expression.xls.gz` count matrix). The manuscript's claimed n=1,010/1,190 for
T2D is not achievable from the public supplementary data as deposited.

## Differential expression (Welch t-test, BH-FDR, disease vs Control)

| | Simulated (pushed code) | Real (this run) |
|---|---|---|
| T2D significant genes (padj<0.05) | Fold-changes hand-injected (0.3–0.8 log2FC) to guarantee significance | **0 / 18,312 genes** survive FDR correction. TXNIP (p=0.012, padj=0.35), IL1B (p=0.036, padj=0.39), IRS1 (p=0.057) trend in the expected direction but are not significant after correction. |
| AD significant genes (padj<0.05) | Fold-changes hand-injected (0.25–0.65 log2FC) | **4,743 / 14,484 genes** significant; 101 with \|log2FC\|>0.3. This is a real, non-null finding. |

## ML classification (5-fold CV + held-out 20% test, leakage-corrected: feature selection on train only)

| Disease | Model | Simulated Test AUC | **Real Test AUC** |
|---|---|---|---|
| T2D (3-class) | Random Forest | — | 0.695 |
| T2D (3-class) | Gradient Boosting | — | 0.648 |
| T2D (3-class) | **SVM (RBF) — best** | **0.906** | **0.708** |
| T2D (3-class) | MLP | — | 0.667 |
| T2D (3-class) | LASSO | — | 0.695 |
| AD (3-class) | **Random Forest — best** | **0.602** | **0.772** |
| AD (3-class) | Gradient Boosting | — | 0.768 |
| AD (3-class) | SVM (RBF) | — | 0.744 |
| AD (3-class) | MLP | — | 0.709 |
| AD (3-class) | LASSO | — | 0.691 |

**The core narrative of the manuscript inverts on real data.** The manuscript's central
claim is "T2DM blood classification achieves strong performance (AUC=0.906); AD blood
classification remains challenging (AUC=0.602)." On real data, it is the other way
around: AD (AUC=0.772, n=706) outperforms T2D (AUC=0.708, n=189) — plausibly because
the AD cohort has ~3.7x more samples, not because AD blood signal is intrinsically
stronger. Both real AUCs are moderate-good and well above chance (0.33 for 3-class),
but neither supports "strong" (T2D) vs "weak" (AD) framing as written.

## Novel biomarkers

| Biomarker | Simulated claim | **Real result** |
|---|---|---|
| BRS (AD) | Control=17.5, MCI=51.3, AD=65.7, r=−0.81 vs MMSE | MMSE not present in public GEO metadata for GSE63060/61 — this validation **cannot be computed from the cited public data at all**. |
| BRS-T (AD, transcriptomic-only) | not previously reported | Control=41.1±16.4, MCI=39.0±16.1, AD=40.2±16.9 (n=706). **No meaningful separation between groups.** |
| BIRTHS (T2D) | Control=22.7, Pre-DM=51.2, T2D=81.5, r=+0.82 vs HbA1c | HbA1c not present in public GEO metadata for GSE221521 — correlation **cannot be computed**. Group means (real): Control=59.9±19.2, Pre-DM=60.7±12.0, T2D=66.7±14.7 (n=189). Kruskal-Wallis H=9.67, **p=0.0079** — statistically significant but a modest effect, not a three-stage separation. |

## Cross-disease concordance

Simulated: r=0.89 across 16 hand-picked shared genes.
**Real: cannot be computed — T2D yields 0 significant DEGs, so there is no
data-derived T2D signature to intersect with the AD signature.** This is a real
negative/null result worth reporting honestly, not a computation to fake.

## Drug repurposing (HGERS)

The HGERS scoring formula is transparent and reproducible, but its inputs
(`t2d_signature`, `ad_signature` — which genes are "up"/"down" in each disease) are
**hardcoded from prior literature**, not derived from the real DEG results. Every one
of the 7 drugs scores exactly **1.000** for both diseases, which is not a meaningful
ranking — it happens because each drug's target-gene profile was curated using the
same literature assumptions baked into the "disease signature," making the score
tautological by construction. This needs restructuring (derive signatures from real
DEG results) before it can be presented as a rank-ordered finding; as-is it is a
curated hypothesis list, not a data-driven repurposing result.

## Bottom line

Every primary quantitative claim in the current manuscript and GitHub README (BRS,
BIRTHS, ML AUCs, cross-disease concordance, drug ranking) was generated from simulated
data, not the real GEO data the paper cites. Rerunning on the real data:
- **Confirms** a real, moderate, statistically defensible AD blood transcriptomic
  signal (4,743 DEGs, AUC 0.77) — weaker than claimed but genuine.
- **Weakens** the T2D transcriptomic story substantially (0 DEGs at FDR<0.05, AUC 0.71
  driven by multivariate combination of weak features, not individual biomarkers).
- **Refutes** the BRS/BIRTHS "three-stage separation" biomarker claims as stated — the
  underlying MMSE/HbA1c correlations cannot even be computed from public data, and the
  transcriptomic-only proxies show weak-to-no group separation.
- **Cannot support** the cross-disease concordance or the drug-repurposing ranking as
  currently computed.
