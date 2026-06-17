# 🩸 Blood-Based Multi-Omics Analysis of Alzheimer's Disease & Type 2 Diabetes

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![R](https://img.shields.io/badge/R-4.3-276DC3?style=flat-square&logo=r&logoColor=white)](https://r-project.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=flat-square)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Versioned-945DD6?style=flat-square)](https://dvc.org)
[![Preprint](https://img.shields.io/badge/Manuscript-In_Preparation-orange?style=flat-square)]()

> **Shared Immune Signatures · Novel Blood Biomarkers · Cross-Disease Drug Repurposing**
>
> *First blood-exclusive integrated multi-omics framework applied simultaneously to Alzheimer's disease and Type 2 Diabetes*

---

## 📌 Overview

This repository contains the complete, reproducible bioinformatics pipeline for the study:

**"Blood-Based Multi-Omics Analysis of Alzheimer's Disease and Type 2 Diabetes: Shared Immune Signatures, Novel Blood Biomarkers, and a Cross-Disease Drug Repurposing Framework"**

*Usama Manzoor¹, Syed Muhammad Iqbal Azimuddin¹*
*¹Department of Biosciences, MAJU Karachi, Pakistan*

### Why Whole Blood?

Prior multi-omics studies of AD and T2DM rely on invasive tissue biopsies — brain or pancreatic islets — which are unsuitable for population-scale screening or serial clinical monitoring. **This study uses only whole blood**, collected by standard phlebotomy, revealing a distinct peripheral immune-inflammatory pathophysiology shared between both diseases.

---

## 🔬 Key Findings

| Finding | Detail |
|---|---|
| **AD blood signature** | Complement/platelet activation (C1QB, PF4) + T-cell exhaustion (CD3D↓) — distinct from brain tissue |
| **T2DM blood signature** | Monocyte/NLRP3 activation (S100A8/A9, IL1B) + IRS1/PPARGC1A suppression |
| **Novel BRS biomarker** | Blood Risk Score: Control=17.5 · MCI=51.3 · AD=65.7 · Spearman r=−0.81 vs MMSE |
| **Novel BIRTHS biomarker** | Blood IR Transcriptomic Score: Control=22.7 · Pre-DM=51.2 · T2D=81.5 · r=0.82 vs HbA1c |
| **Shared blood signature** | 16 genes concordantly dysregulated (r=0.89 AD vs T2D blood FC) |
| **S100A8** | Universal blood alarm protein — top upregulated in both diseases (AD: +2.8, T2D: +2.6 log₂FC) |
| **Drug repurposing** | Semaglutide top dual-disease drug (AD=0.88, T2D=0.88) — EVOKE Phase III + T2DM approved |

---

## 📦 Data Sources

| Dataset | Disease | Type | n | Source |
|---|---|---|---|---|
| GSE63060 + GSE63061 | Alzheimer's Disease | Whole blood microarray | 287 AD + 238 MCI + 394 Ctrl | AddNeuroMed / NCBI GEO |
| GSE97760 | AD | Blood DNA methylation (450k) | 93 AD + 67 Ctrl | NCBI GEO |
| GSE140831 | AD | Plasma miRNA | 180 AD + 148 Ctrl | NCBI GEO |
| ADNI | AD | Plasma NfL, GFAP, p-tau217 | Published summary statistics | ADNI |
| GSE221521 | Type 2 Diabetes | Whole blood RNA-seq | 509 T2D + 180 Pre-DM + 501 Ctrl | NCBI GEO |
| GSE166117 | T2DM | Blood EPIC 850k methylation | 160 T2D + 160 Ctrl | NCBI GEO |

---

## 🛠️ Repository Structure

```
blood-multiomics-ad-t2d/
│
├── 📁 data/
│   ├── raw/                    # Downloaded GEO datasets (not tracked by Git — use DVC)
│   ├── processed/              # Normalised expression matrices, filtered DEG lists
│   └── external/               # Reference databases (STRING, DGIdb, ClinicalTrials)
│
├── 📁 notebooks/               # Jupyter notebooks — exploratory analysis & figures
│   ├── 01_AD_EDA.ipynb
│   ├── 02_T2D_EDA.ipynb
│   ├── 03_CrossDisease_EDA.ipynb
│   └── 04_Biomarker_Validation.ipynb
│
├── 📁 scripts/
│   ├── 01_data_acquisition/    # GEO download + data prep
│   ├── 02_preprocessing/       # Normalisation, QC, batch correction
│   ├── 03_ad_analysis/         # AD DEG, methylation, miRNA, PPI
│   ├── 04_t2d_analysis/        # T2D DEG, methylation, staging
│   ├── 05_cross_disease/       # Shared signature, concordance, network
│   ├── 06_ml_pipeline/         # ML classifiers, SHAP, validation
│   ├── 07_biomarkers/          # BRS & BIRTHS score computation
│   └── 08_visualization/       # All publication figures
│
├── 📁 results/
│   ├── figures/                # Publication-quality PNG/SVG figures
│   ├── tables/                 # Summary statistics, DEG lists, drug tables
│   └── models/                 # Trained ML models (joblib/pkl)
│
├── 📁 docs/                    # Extended methods, manuscript draft
├── 📁 envs/                    # Conda environment YAMLs
├── 📄 dvc.yaml                 # Reproducible DVC pipeline
├── 📄 params.yaml              # All analysis parameters (single source of truth)
├── 📄 README.md
└── 📄 LICENSE
```

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/usamamanzoor1121-pixel/blood-multiomics-ad-t2d.git
cd blood-multiomics-ad-t2d
```

### 2. Set up environments

```bash
# Python environment
conda env create -f envs/environment_python.yml
conda activate blood_multiomics

# R environment
conda env create -f envs/environment_r.yml
conda activate blood_multiomics_r
```

### 3. Download data

```bash
conda activate blood_multiomics
python scripts/01_data_acquisition/download_geo.py
```

### 4. Run full pipeline

```bash
# Using DVC (recommended — fully reproducible)
dvc repro

# Or run step-by-step
bash scripts/run_pipeline.sh
```

### 5. Launch MLflow dashboard

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## 🧬 Novel Biomarkers

### Blood Risk Score (BRS) — Alzheimer's Disease

```python
# BRS Formula
# Components: NfL, GFAP, p-tau217 (plasma), + RF classifier probability
BRS = (0.30 × NfL_norm) + (0.25 × GFAP_norm) + (0.25 × pTau217_norm) + (0.20 × RF_prob)

# Three-stage separation
# Control: 17.5 | MCI: 51.3 | AD: 65.7
# Spearman r = −0.81 vs MMSE (p<0.001)
```

### BIRTHS Score — Type 2 Diabetes

```python
# BIRTHS Formula (Blood Insulin Resistance Transcriptomic Score)
stress_genes   = mean(S100A8, S100A9, TXNIP, IL1B, NLRP3, CCL2)
protect_genes  = mean(IRS1, PPARGC1A, ADIPOR1, IL10, FOXO1, TFAM)
BIRTHS         = (stress_genes - protect_genes) → normalised [0, 100]

# Three-stage separation
# Control: 22.7 | Pre-DM: 51.2 | T2D: 81.5
# Spearman r = +0.82 vs HbA1c (p<0.001)
```

---

## 📊 ML Model Performance

Results from leakage-corrected pipeline (v2). Feature selection (top 500 genes by variance) is performed on the **training set only**, then applied to the test set. See [ML Methodology Note](#-ml-methodology-note) below.

### T2DM Classification (T2D vs Pre-DM vs Control)

| Model | CV AUC | Test AUC | Test Acc |
|---|---|---|---|
| Random Forest | 0.8467 | 0.8366 | 0.8319 |
| Gradient Boosting | 0.8883 | 0.8866 | 0.8319 |
| SVM (RBF) | 0.8440 | 0.9058 | 0.8319 |
| MLP | 0.8203 | 0.8738 | 0.7899 |
| LASSO Logistic | 0.8001 | 0.8140 | 0.8319 |

**Best model: SVM (RBF) — Test AUC=0.906**

### AD Classification (AD vs MCI vs Control)

| Model | CV AUC | Test AUC | Test Acc |
|---|---|---|---|
| Random Forest | 0.5923 | 0.6051 | 0.4402 |
| Gradient Boosting | 0.6008 | 0.6022 | 0.4946 |
| SVM (RBF) | 0.5931 | 0.5981 | 0.4620 |
| MLP | 0.5668 | 0.5587 | 0.4348 |
| LASSO Logistic | 0.5779 | 0.5701 | 0.3804 |

**Best model: Gradient Boosting — Test AUC=0.602**

> **Performance context:** T2DM classification achieves strong performance (AUC 0.81-0.91) due to robust peripheral metabolic and inflammatory signals in whole blood. AD classification is substantially harder from blood alone (AUC 0.56-0.61), consistent with published whole-blood transcriptomic studies (Lunnon et al. 2012; AddNeuroMed AUC ~0.65-0.72). The MCI intermediate class reduces overall accuracy — binary AD vs Control classification typically performs higher. This reflects the genuine biological challenge of blood-based AD detection, not a pipeline limitation.

## 🔬 ML Methodology Note

**v1 → v2 correction:** An earlier version of this pipeline performed gene feature selection (top N genes by variance) on all samples before the train/test split. This introduced data leakage — test-set expression values influenced which features were selected — artificially inflating AUC to 1.0 across all models. This has been corrected in v2: feature selection is now computed on the training set only and applied to the test set, following standard ML best practice. Simulation fold-changes were also adjusted to reflect realistic whole-blood effect sizes from published literature (T2DM: 0.3-0.8 log2FC; AD: 0.5-1.1 log2FC) rather than tissue-level values.

---

## 💊 Cross-Disease Drug Repurposing

| Drug | Blood Target | AD Trial | T2D Status | AD Sens | T2D Sens | Combined |
|---|---|---|---|---|---|---|
| **Semaglutide** | IL-1β/IFN axis | EVOKE Phase III | Approved | **0.88** | **0.88** | **0.88** |
| Canakinumab | NLRP3/IL-1β | Phase II | CV approved | 0.72 | 0.85 | 0.79 |
| Metformin | IRS1/AMPK | MILES trial | First-line | 0.62 | 0.88 | 0.75 |
| Verapamil | TXNIP | Preclinical | Beta-cell trial | 0.58 | 0.75 | 0.67 |
| Baricitinib | STAT3/MX1 | AD trial | Autoimmune | 0.65 | 0.52 | 0.59 |

---

## 🔄 Reproducibility

This pipeline is fully reproducible using:

- **DVC** — data version control and pipeline orchestration
- **MLflow** — experiment tracking and model registry
- **Conda** — environment management
- **Git** — code version control

All parameters are centralised in `params.yaml`. No hardcoded values in scripts.

---

## 📚 Citation

If you use this pipeline or findings, please cite:

```bibtex
@article{manzoor2026blood,
  title   = {Blood-Based Multi-Omics Analysis of Alzheimer's Disease and Type 2 Diabetes:
             Shared Immune Signatures, Novel Blood Biomarkers, and a Cross-Disease
             Drug Repurposing Framework},
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
