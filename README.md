# 🩸 Blood-Based Multi-Omics Analysis of Alzheimer's Disease & Type 2 Diabetes

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Corrected_with_real_GEO_data-brightgreen?style=flat-square)]()

> **⚠️ CORRECTION NOTICE (2026-09-24):** All results previously reported here (BRS,
> BIRTHS, ML classification AUCs, cross-disease concordance, drug repurposing ranking)
> were computed from `np.random`-simulated expression data, not from the real GEO
> datasets listed below. This has been corrected: the pipeline in
> [`REAL_DATA_PIPELINE/`](REAL_DATA_PIPELINE/) downloads and analyses the real public
> data, and every number in this README now reflects that real re-analysis. See
> [`FINAL_RESEARCH_AUDIT_REPORT.md`](FINAL_RESEARCH_AUDIT_REPORT.md) for the full audit
> and [`REAL_DATA_PIPELINE/REAL_VS_SIMULATED_RESULTS.md`](REAL_DATA_PIPELINE/REAL_VS_SIMULATED_RESULTS.md)
> for a side-by-side comparison of the old (simulated) and new (real) numbers. The
> original simulation-based scripts (`AD/`, `T2D/`, `03_novel_gap_analysis.py`,
> `build_article.py`, `ml_classifier_v2_corrected.py`) are retained for transparency
> but **should not be used as a source of biological results** — they never load real
> data.

---

## 📌 Overview

This repository contains the reproducible bioinformatics pipeline for a blood-based
multi-omics reanalysis of Alzheimer's disease (AD) and Type 2 diabetes (T2DM), using
real public whole-blood transcriptomic cohorts for each disease.

*Usama Manzoor¹*
*¹Department of Biosciences, MAJU Karachi, Pakistan*

### Why Whole Blood?

Prior multi-omics studies of AD and T2DM largely rely on invasive tissue biopsies —
brain or pancreatic islets — which are unsuitable for population-scale screening.
This study uses whole-blood transcriptomics, collected by standard phlebotomy.

---

## 🔬 Real-Data Findings (as of the 2026-09-24 correction)

| Finding | Real result |
|---|---|
| AD blood DEGs (AD vs Control, FDR<0.05) | **4,743 / 14,484 genes significant** (n=706) — a genuine, moderate signal |
| T2D blood DEGs (T2D vs Control, FDR<0.05) | **0 / 18,312 genes** survive correction (n=189); TXNIP, IL1B, IRS1 trend in the expected direction at nominal (uncorrected) significance only |
| AD ML classification (3-class, held-out test) | **AUC = 0.772** (Random Forest) |
| T2D ML classification (3-class, held-out test) | **AUC = 0.708** (SVM-RBF) |
| BRS-T (AD biomarker, transcriptomic-only proxy) | Control=41.1, MCI=39.0, AD=40.2 — **no meaningful separation** |
| BIRTHS (T2D biomarker) | Control=59.9, Pre-DM=60.7, T2D=66.7 (Kruskal-Wallis p=0.008) — real but modest, not a three-stage separation |
| BRS vs MMSE / BIRTHS vs HbA1c | **Not computable** — MMSE and HbA1c are not present in the public GEO metadata for these accessions |
| Cross-disease concordance | **Not computable** — T2D has no significant DEGs to intersect with the AD signature |
| Drug repurposing (HGERS) | Current formula is circular (disease-direction inputs are literature priors, not DEG-derived) — reported as a methodological limitation, not a ranked finding |

**The AD-vs-T2D relationship inverts** relative to what was previously (wrongly)
reported: AD shows the stronger real signal here, plausibly because its real sample
size (n=706) is ~3.7× larger than T2D's (n=189), not because AD blood biology is
intrinsically more informative than T2D.

Full comparison: [`REAL_DATA_PIPELINE/REAL_VS_SIMULATED_RESULTS.md`](REAL_DATA_PIPELINE/REAL_VS_SIMULATED_RESULTS.md)

---

## 📦 Data Sources (real, as actually used)

| Dataset | Disease | Type | Real n available (public supplementary files) | Source |
|---|---|---|---|---|
| GSE63060 | AD | Whole blood microarray (Illumina HumanHT-12 v3) | 329 (145 AD, 80 MCI, 104 Control) | NCBI GEO |
| GSE63061 | AD | Whole blood microarray (Illumina HumanHT-12 v4) | 382 (139 AD, 109 MCI, 134 Control) | NCBI GEO |
| GSE63060+61 merged, QC'd | AD | — | **706** (279 AD, 189 MCI, 238 Control) | — |
| GSE221521 | T2D | Whole blood RNA-seq | 193 raw / **189 after QC** (73 T2D, 68 Pre-DM, 48 Control) | NCBI GEO |

Note: the deposited public supplementary count matrix for GSE221521 contains 193
samples, not the ~1,010–1,190 sometimes quoted for the full series — this is the
sample size actually available for reanalysis from the public data. Neither series'
public metadata includes MMSE or HbA1c.

---

## 🛠️ Repository Structure

```
blood-multiomics-ad-t2d/
│
├── REAL_DATA_PIPELINE/              # ✅ Real-data pipeline (use this)
│   ├── 00b_parse_downloaded_data.py #   Parses raw GEO downloads → analysis-ready CSVs
│   ├── preprocess_blood.py          #   QC, size-factor normalisation, batch correction
│   ├── 03_deg_analysis.py           #   Differential expression (Welch t-test + BH-FDR)
│   ├── ml_classifier.py             #   5 classifiers, leakage-corrected, real data
│   ├── 01_biomarker_fixes.py        #   BRS-T, BIRTHS, HGERS, cross-disease concordance
│   ├── params.yaml                  #   All analysis parameters
│   ├── REAL_VS_SIMULATED_RESULTS.md #   Full old-vs-new comparison
│   ├── Blood_MultiOmics_AD_T2D_Manuscript_CORRECTED.docx
│   └── data/{tables,figures}/       #   Real DEG tables, ML results, ROC curves, SHAP plots
│
├── FINAL_RESEARCH_AUDIT_REPORT.md   # Full audit: what was wrong, what was fixed, roadmap
│
├── AD/, T2D/, 03_novel_gap_analysis.py, build_article.py,
│   ml_classifier_v2_corrected.py    # ⚠️ Legacy simulation-based scripts — kept for
│                                     #    transparency only, do not use for results
├── params.yaml, README.md
└── LICENSE
```

---

## 🚀 Quickstart (real pipeline)

```bash
git clone https://github.com/usamamanzoor1121-pixel/blood-multiomics-ad-t2d.git
cd blood-multiomics-ad-t2d/REAL_DATA_PIPELINE

pip install pandas numpy scipy scikit-learn statsmodels joblib matplotlib seaborn \
            pyyaml shap mlflow

# 1. Download real GEO data (curl-based; ~300MB total)
#    See FINAL_RESEARCH_AUDIT_REPORT.md §Reproducibility for the exact GEO FTP URLs
#    used (GSE221521, GSE63060, GSE63061 + GPL6947/GPL10558 annotation).
# 2. Parse into analysis-ready format
python 00b_parse_downloaded_data.py
# 3. Preprocess (normalisation, QC, batch correction)
python preprocess_blood.py
# 4. Differential expression
python 03_deg_analysis.py
# 5. ML classification (set MLFLOW_ALLOW_FILE_STORE=true for this MLflow version)
export MLFLOW_ALLOW_FILE_STORE=true
python ml_classifier.py
# 6. Biomarker scores, drug scoring, cross-disease concordance
python 01_biomarker_fixes.py
```

---

## 🧬 Biomarkers (real-data status)

### BRS-T — AD, transcriptomic-only proxy (no plasma proteins)

```
BRS-T = mean(C1QB, C1QC, S100A8, S100A9, MX1, OAS1, IFIT1, PPBP, TXNIP, IL1B)
      − mean(CD3D, CD3E, PRF1, NKG7, GZMK, CD247, CD8A), normalised to [0,100]

Real result: Control=41.1±16.4, MCI=39.0±16.1, AD=40.2±16.9 (n=706)
→ No meaningful separation between groups.
```

The originally proposed full BRS additionally used plasma NfL/GFAP/p-tau217 and was
validated against MMSE; neither is present in the public GEO metadata for GSE63060/61,
so that validation cannot currently be performed from these accessions.

### BIRTHS — T2D

```
BIRTHS = mean(S100A8, S100A9, TXNIP, IL1B, NLRP3) − mean(IRS1, ADIPOR1, FOXO1, TFAM)
       normalised to [0,100]

Real result: Control=59.9±19.2, Pre-DM=60.7±12.0, T2D=66.7±14.7 (n=189)
Kruskal-Wallis H=9.67, p=0.0079 → real, statistically significant, but modest effect.
```

HbA1c is not present in the public GSE221521 metadata, so BIRTHS-vs-HbA1c correlation
cannot currently be computed from this accession.

---

## 📊 ML Model Performance (real data, leakage-corrected)

Feature selection (top 500 genes by variance) is performed on the **training set
only**, then applied to the test set (80:20 split, 5-fold stratified CV on train).

### T2DM Classification (T2D vs Pre-DM vs Control, n=189)

| Model | CV AUC | Test AUC | Test Acc |
|---|---|---|---|
| Random Forest | 0.7573 | 0.6953 | 0.5263 |
| Gradient Boosting | 0.7569 | 0.6482 | 0.5000 |
| **SVM (RBF) — best** | 0.7134 | **0.7075** | 0.5263 |
| MLP | 0.7485 | 0.6671 | 0.5263 |
| LASSO Logistic | 0.6796 | 0.6947 | 0.4737 |

### AD Classification (AD vs MCI vs Control, n=706)

| Model | CV AUC | Test AUC | Test Acc |
|---|---|---|---|
| **Random Forest — best** | 0.7299 | **0.7724** | 0.6056 |
| Gradient Boosting | 0.7137 | 0.7676 | 0.5563 |
| SVM (RBF) | 0.7122 | 0.7439 | 0.5493 |
| MLP | 0.7413 | 0.7085 | 0.5493 |
| LASSO Logistic | 0.6789 | 0.6908 | 0.4859 |

---

## 💊 Cross-Disease Drug Repurposing — Current Limitation

The Hub Gene Expression Rescue Score (HGERS) formula is transparent, but its
"disease signature" inputs are currently hardcoded from literature rather than
derived from the real DEG results — and since T2D has zero significant DEGs, there is
no real T2D signature to score drugs against. As implemented, every drug scores
1.000 for both diseases, which is a methodological artefact (see
[`FINAL_RESEARCH_AUDIT_REPORT.md`](FINAL_RESEARCH_AUDIT_REPORT.md) §4 and §9), not a
ranked empirical finding. This needs restructuring before any drug ranking is
reported as a result.

---

## 🔄 Reproducibility

All parameters are centralised in `REAL_DATA_PIPELINE/params.yaml`. Full run logs for
the real-data pipeline (including two real library/version-compatibility bugs found
and fixed: MLflow filestore deprecation, sklearn LASSO solver/multiclass
incompatibility) are documented in `FINAL_RESEARCH_AUDIT_REPORT.md`.

---

## 📚 Citation

```bibtex
@article{manzoor2026blood,
  title   = {Blood-Based Multi-Omics Analysis of Alzheimer's Disease and Type 2 Diabetes:
             A Real-Data Reassessment of Shared Immune Signatures and Candidate Blood Biomarkers},
  author  = {Manzoor, Usama and Azimuddin, Syed Muhammad Iqbal},
  journal = {Manuscript in Preparation},
  year    = {2026},
  institution = {Mohammad Ali Jinnah University, Karachi, Pakistan}
}
```

---

## 👤 Author

**Usama Manzoor**
Bioinformatics Specialist · JSMU Diagnostic Laboratory · Karachi, Pakistan
📧 usama.manzoor1121@gmail.com
🐙 [@usamamanzoor1121-pixel](https://github.com/usamamanzoor1121-pixel)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
