# Final Research Audit Report — Blood-Based Multi-Omics AD × T2D
**Repository:** github.com/usamamanzoor1121-pixel/blood-multiomics-ad-t2d
**Local project:** D:\Cancer research\AD & T2D
**Audit date:** 2026-09-24
**Auditor role:** senior computational genomics / bioinformatics review, with hands-on
re-execution of the pipeline on real data (not a document-only review)

---

## Executive Summary

The pushed GitHub repository, the local project files, and the manuscript
(`Blood_MultiOmics_AD_T2D_Final Manuscript.docx`) all present a complete results
narrative — biomarker scores, ML classification AUCs, differential expression,
cross-disease concordance, drug repurposing — that was **computed entirely from
`np.random`-simulated data**, not from the real public GEO datasets (GSE63060,
GSE63061, GSE221521, etc.) the paper cites as its data source. A "v2" fix already
merged into the repo corrected a real ML data-leakage bug but did not switch to real
data. The manuscript additionally contains a false, garbled claim inserted in §2.5
("All results reported here use real GEO data...") that contradicts the code, plus
several internally self-contradictory numbers (the abstract states two different AUC
pairs for the same models; §3.4 states two different BIRTHS score sets in the same
paragraph). Local zip archives show that real-data replacement scripts had already
been drafted by a previous session but never run or merged.

**Action taken in this engagement:** the real data was downloaded, parsed,
preprocessed (fixing an additional real bug — RNA-seq counts had no library-size
normalisation), and the full analysis (DEG, ML classification, biomarker scores,
cross-disease concordance, drug scoring) was re-run end-to-end on real data. See
`REAL_DATA_PIPELINE/` for all code, logs, tables and figures, and
`REAL_DATA_PIPELINE/REAL_VS_SIMULATED_RESULTS.md` for the full old-vs-new comparison.

**Headline result:** on real data, AD shows a genuine, moderate, statistically robust
blood transcriptomic signal (4,743/14,484 genes significant at FDR<0.05; ML test
AUC=0.772, n=706) — weaker than claimed but real. T2D shows **no** genes surviving
FDR correction (0/18,312) despite moderate ML performance (test AUC=0.708, n=189),
consistent with a real but underpowered/distributed signal rather than the "strong,
robust" story claimed. The proposed BRS and BIRTHS biomarkers do not show their
claimed separation in real data, and their headline clinical validations (vs MMSE,
vs HbA1c) are **not computable at all** from public GEO metadata — a data-availability
problem that should have been caught before those correlations were reported. This
reverses the manuscript's central claim ("T2D strong, AD weak") and substantially
narrows what the study can currently claim.

This is not a project that needs "polishing before publication" — it needed, and has
now received, a full re-grounding in real data. What remains is finishing the
manuscript/figure updates (started here), addressing the still-open methodological
gaps below, and an honest reassessment of what is publishable.

---

## 1. Project Inventory

| Location | Contents | Status |
|---|---|---|
| `D:\Cancer research\AD & T2D\` (local, top level) | 8 PNG figures, 1 PDF article, 1 manuscript .docx, 2 zip archives, `AD_T2D_MultiOmics_Pipeline/` | Original — untouched |
| `AD_T2D_MultiOmics_Pipeline/` | `01_data_collection.py` (real GEO accession guide — never actually called by the other scripts), `AD/02_AD_analysis.py`, `T2D/02_T2D_analysis.py`, `03_novel_gap_analysis.py`, `build_article.py` | All simulate data via `np.random`; none load real GEO files |
| `blood_multiomics_all_fixes.zip` | `00_download_real_data.py`, `preprocess_blood.py`, `ml_classifier.py` (v2, leakage-corrected), `01_biomarker_fixes.py`, `params.yaml`, `RESOLUTION_GUIDE.md` | Pre-drafted real-data pipeline — never run before this audit |
| `blood_multiomics_fixes_v2.zip` | Earlier/partial version of the same fix set | Superseded by `all_fixes` |
| GitHub repo (cloned for inspection) | Top-level scripts (duplicated from `AD_T2D_MultiOmics_Pipeline/`), `ml_classifier_v2_corrected.py`, `params.yaml`, README, figures, PDF | `ml_classifier_v2_corrected.py` still calls `simulate_ad_v3()`/`simulate_t2d()` — leakage-fixed but not real-data |
| `REAL_DATA_PIPELINE/` (**new, this audit**) | Real downloaded GEO data, parsing script, patched preprocessing/ML/biomarker scripts, DEG script, logs, real figures/tables, corrected manuscript draft | New — see below |

No original files were deleted or overwritten. All new work is in
`REAL_DATA_PIPELINE/` and this report.

---

## 2. Data Summary (real, as actually reanalysed)

| Dataset | Platform | Claimed n | **Real n available / after QC** |
|---|---|---|---|
| GSE221521 (T2D) | Illumina NovaSeq RNA-seq, whole blood | 509 T2D+180 PreDM+501 Ctrl (n=1,190) | 74+69+50 = 193 / **189** |
| GSE63060 (AD) | Illumina HumanHT-12 v3 (GPL6947) | part of n=919 | 145 AD+80 MCI+104 Ctrl = 329 |
| GSE63061 (AD) | Illumina HumanHT-12 v4 (GPL10558) | part of n=919 | 139 AD+109 MCI+134 Ctrl = 382 (388−6 unresolved) |
| GSE63060+61 merged | — | 287 AD+238 MCI+394 Ctrl (n=919) | 284 AD+189 MCI+238 Ctrl = 711 / **706** |

**Critical, previously unflagged limitation:** neither series' public GEO
`!Sample_characteristics_ch1` metadata contains the clinical severity variables the
manuscript correlates against (MMSE for AD; HbA1c for T2D). GSE221521 metadata has
only `tissue` and `cell type`; GSE63060/61 metadata has `status, age, ethnicity,
gender, case-control flag`. The BRS-vs-MMSE and BIRTHS-vs-HbA1c validations central to
the manuscript's abstract **cannot be computed from the cited public accessions at
all**, real or simulated methodology aside.

---

## 3. Biological Question (as reconstructible from the code)

**Stated question:** Do AD and T2DM share a peripheral blood immune-inflammatory
signature detectable by whole-blood multi-omics, usable for (a) staging/diagnosis via
composite biomarkers (BRS, BIRTHS), (b) ML classification, and (c) cross-disease drug
repurposing?

**Actual analytical unit:** per-sample (donor-level), which is appropriate — this is
bulk blood transcriptomics, not single-cell, so there is no pseudoreplication concern
of the scRNA-seq kind. The real concern here was different: unverified data
provenance, not unit-of-analysis.

---

## 4. Statistical Audit

| Issue | Severity | Status |
|---|---|---|
| Primary results computed from simulated, not real, data | **CRITICAL** | **Fixed this session** — real pipeline run end-to-end |
| RNA-seq counts log-transformed with no library-size normalisation | **CRITICAL** (methodological) | **Fixed this session** — median-of-ratios size-factor normalisation added (size factors ranged 0.235–2.13×, confirming this mattered) |
| ML feature selection before train/test split (v1) | HIGH | Already fixed in v2 (confirmed correct: selection uses training-set variance only) |
| No multiple-testing-aware effect size reporting for T2D (0 genes survive FDR, but literature-typical effect sizes were reported instead) | **CRITICAL** | **Fixed this session** — real DEG results reported honestly (0/18,312 at FDR<0.05) |
| HGERS drug score circular (disease-direction inputs are literature priors, not DEG-derived) | HIGH | **Identified this session, not yet fixed** — needs restructuring (see roadmap) |
| Cross-disease concordance requires real DEG in both diseases; T2D has none | HIGH | **Identified this session** — correctly returns "cannot compute" rather than a fabricated r |
| AD batch correction (two cohorts) uses simplified batch-mean subtraction, not full ComBat | MEDIUM | Documented as a limitation; acceptable for a first pass, worth revisiting |
| Outlier removal by z>3.5 on mean pairwise correlation | LOW | Reasonable simple approach; adequate here |
| Small T2D sample size (n=189) relative to features (500 selected genes) | MEDIUM | Inherent to the public data available; flagged in manuscript limitations |

---

## 5. Dr. Yulia Medvedeva — Research Alignment

Her MBZUAI profile page returned HTTP 403 to automated fetching (repeatedly, across
multiple attempts), so this section is built from cross-referenced search-engine
summaries of MBZUAI's faculty listing and third-party academic profile aggregators,
**not a single directly-read source** — treat specifics as reasonably well-corroborated
but not independently verified against her own page text.

**Her stated research direction:** single-cell RNA-seq and ATAC-seq to study dynamic
gene activity and regulatory landscapes; long non-coding RNAs as regulators of gene
expression, splicing, and epigenetic memory; integrating high-resolution multi-omics
with genetic/phenotypic data for precision medicine, with **type 2 diabetes named
explicitly** as a disease of interest, plus an emphasis on ancestry and population
diversity in human health. A recent (2026) paper in her general research area —
"Scalable single-cell total RNA sequencing unifies coding and noncoding
transcriptomics," *Nature Biotechnology* — covers non-coding transcriptomics in
immunity and brain development; I could not verify her authorship on this specific
paper due to repeated access blocks on PMC/bioRxiv, so it is noted as topically
relevant context only, not attributed to her without verification.

**Overlap map:**

| Her focus | This project | Overlap |
|---|---|---|
| T2D multi-omics, precision medicine | T2D is one of two diseases studied | **Direct, strong** — topic-level overlap |
| Single-cell RNA-seq / ATAC-seq | Bulk whole-blood RNA-seq / microarray, no single-cell data | **Weak methodologically** — this project doesn't yet use single-cell resolution |
| lncRNA regulatory genomics | Not analysed at all currently (only protein-coding genes: S100A8, TXNIP, IRS1, etc.) | **Absent, but a natural extension** — see roadmap |
| Ancestry / population diversity in health | Single-country/cohort-limited public data (AddNeuroMed = European; GSE221521 origin not stated in metadata used) | **Not addressed** |

**Honest assessment:** the overlap is real at the disease-topic level (T2D,
multi-omics, precision medicine) but this project, as currently built, does not use
her methodological specialty (single-cell, ATAC-seq, lncRNA-centric analysis). If this
work is intended to connect to her lab, the most credible bridge is not "polish the
current bulk-blood ML pipeline" but **add an lncRNA-specific reanalysis layer** to the
real blood transcriptomic data already now in hand (see Roadmap §9), since lncRNA
probes/transcripts are present in RNA-seq data (GSE221521) and can be extracted from
the Ensembl gene_type column already downloaded but currently discarded.

---

## 6. Novelty Assessment

| Claim | Real-data support |
|---|---|
| AD blood transcriptomic signature exists and is ML-classifiable | **Supported** (4,743 DEGs, AUC 0.77) — but AUC and DEG magnitude are far more modest than the original claim, and are broadly consistent with prior AddNeuroMed literature (Lunnon et al. reported AUC ~0.65–0.72), so this is confirmatory, not strikingly novel |
| T2D blood transcriptomic signature is "robust" | **Not supported** — 0 DEGs at FDR<0.05; moderate ML AUC likely reflects a distributed weak signal or sample-size limitation, not a robust signature |
| BRS discriminates Control/MCI/AD | **Not supported** in its transcriptomic-only form (near-identical means); original plasma-protein form cannot be validated from public data |
| BIRTHS discriminates Control/Pre-DM/T2D | **Weakly supported** — real, statistically significant (p=0.008) but modest effect, not three-stage separation |
| Shared 16-gene cross-disease signature | **Not supported** — cannot be computed (T2D has no real DEGs) |
| Semaglutide as top dual-disease drug | **Not supported as computed** — HGERS score is circular/non-discriminating (all drugs score 1.0); this is a curated hypothesis, not a data-driven finding |

**What would make this genuinely novel:** a real, adequately-powered T2D whole-blood
cohort (current n=189 is likely underpowered for single-gene FDR significance given
realistic effect sizes), matched clinical metadata for biomarker validation, and an
lncRNA-specific layer that current literature on AD/T2D blood multi-omics has not
emphasized — this is the strongest genuine gap.

---

## 7. Reproducibility Audit

| Item | Before this audit | After |
|---|---|---|
| Real data ever downloaded/used | No | Yes — `REAL_DATA_PIPELINE/00b_parse_downloaded_data.py` |
| Dependencies pinned | No `requirements.txt`/env file with versions in repo | Still needed (see roadmap) |
| Random seeds | Set for simulation only (`np.random.seed(42)`) — irrelevant once simulation is removed | ML pipeline seeds are set (`random_state=42`) — good |
| Hard-coded paths | `wget` calls assume a Unix environment; failed in this Windows/git-bash environment | Replaced with a portable `curl`-based downloader + direct parser |
| MLflow / library version compatibility | Untested — the repo's own `ml_classifier_v2_corrected.py` was never actually run against real data by its authors | Found and fixed two real version-compatibility bugs (MLflow filestore deprecation; skops untrusted-type model logging; sklearn LASSO solver/multiclass incompatibility) — see `REAL_DATA_PIPELINE/logs/` |
| GPL/probe-to-symbol mapping | Anticipated in code comments but never implemented/run | Implemented and run (`load_gpl_annotation`) |

---

## 8. Red-Team Review — "Why should this paper not be published?"

1. **The primary results are not reproducible from the data the paper cites**, because
   they were never computed from that data. *(This audit's core finding — now
   addressed by rerunning on real data; the manuscript in its pre-audit form should
   not be submitted anywhere.)*
2. **A manuscript sentence falsely claims real-data provenance** ("All results
   reported here use real GEO data...") while the code demonstrably still simulates.
   *Resolved by the rewritten manuscript in `REAL_DATA_PIPELINE/`, which is now
   consistent with the actually-executed real pipeline.*
3. **Internally contradictory numbers within the same document** (two different AUC
   pairs in the abstract; two different BIRTHS triplets in §3.4). *Resolved in the
   rewrite.*
4. **The two headline biomarkers' clinical validations are not computable from the
   cited public data** (no MMSE, no HbA1c in GEO metadata) — a reviewer with GEO
   familiarity would catch this in minutes. *Now disclosed explicitly.*
5. **T2D shows no genes surviving multiple-testing correction** — a reviewer would
   ask how "strong performance" and "robust" biological claims can be reconciled with
   a null DEG result. *Now reported honestly as a null/weak result.*
6. **Drug-repurposing scores are tautological** (disease signature and drug profiles
   share the same literature source, guaranteeing high scores). *Flagged, not yet
   restructured — remains an open item for any resubmission.*
7. **Sample size for T2D (n=189) is far below what the abstract implies** (n=1,010 or
   1,190 is quoted elsewhere) — a reviewer checking the GEO accession would find the
   discrepancy. *Now stated accurately.*

---

## 9. Recommended Roadmap (prioritized)

### CRITICAL (done this session)
- Replace simulated data with real GEO data end-to-end. **Done.**
- Fix RNA-seq library-size normalisation. **Done.**
- Rewrite manuscript to remove false claims and internal contradictions. **Done** (`REAL_DATA_PIPELINE/Blood_MultiOmics_AD_T2D_Manuscript_CORRECTED.docx`).
- Update GitHub repo to reflect real pipeline, not simulated. **Done this session** (see commit).

### HIGH PRIORITY (not yet done — next steps)
- Restructure HGERS drug scoring to derive disease-direction inputs from real DEG
  results (or explicitly relax to nominal p<0.05 with clear caveats for T2D, since no
  gene survives FDR) instead of literature priors, or reframe it explicitly as a
  curated hypothesis table rather than a ranked empirical result.
- Investigate T2D null DEG result further: check for unmodeled confounders (age, sex,
  batch — not currently in the T2D metadata at all, unlike AD), and consider whether
  a larger or differently-composed T2D cohort (e.g., a second public accession) could
  be added to gain power.
- Add full ComBat (not batch-mean subtraction) for the AD cohort merge; consider
  `pyComBat` or R's `sva::ComBat`.
- Pin dependency versions (`requirements.txt` / conda env yaml) — three real
  version-compatibility bugs were hit and fixed in this session alone.

### MEDIUM PRIORITY
- Extract and analyse lncRNAs specifically from GSE221521 (the raw file's
  `gene_type` column, discarded during parsing, would let this be added without a new
  download) — this is the most credible bridge to Dr. Medvedeva's stated research
  interests and the biggest unexploited asset already in hand.
- Attempt to obtain matched clinical metadata (HbA1c, MMSE) for GSE221521/GSE63060/61
  via direct contact with data depositors or a different accession with richer public
  metadata, to make BRS/BIRTHS clinical validation possible at all.
- Re-run DNA methylation (GSE97760, GSE166117) and plasma miRNA (GSE140831) analyses
  referenced in the original concept but not reanalysed in this pass.

### OPTIONAL
- Single-cell deconvolution of the bulk blood signal (e.g., CIBERSORTx) to determine
  which real immune cell subtype(s) drive the AD signal — would also create a
  concrete bridge toward single-cell-resolution follow-up work.
- Cross-cohort external validation using an independent AD or T2D blood series if one
  with compatible metadata can be identified.

---

## 10. Decision Tree Outcome

This project falls into **Path C** (important methodological/data problems existed
and have now been corrected, with conclusions reassessed) transitioning toward
**Path B** (the corrected foundation is now sound; targeted extensions — lncRNA layer,
larger T2D cohort, HGERS restructuring — would substantially strengthen it before any
submission). It is **not** ready for Path A (polish and submit) in either its
pre-audit or current form — the current form is scientifically honest but the T2D
half of the story is now weak enough that it may need to be reframed (e.g., "AD blood
transcriptomics are classifiable; T2D blood transcriptomics show a real but weak and
underpowered signal requiring larger cohorts") rather than presented as a symmetric
dual-disease success story.

---

## 11. What I would do next (concrete sequence)

1. Get a second, larger T2D whole-blood RNA-seq public cohort (search GEO beyond
   GSE221521) to address the power problem directly — this is the single highest-value
   next step.
2. Extract lncRNAs from the already-downloaded GSE221521 raw file and rerun DEG/ML
   restricted to lncRNAs vs protein-coding genes separately — directly extends toward
   Dr. Medvedeva's stated interests and could become the paper's genuine novel angle.
3. Restructure HGERS to be data-derived, or drop the drug-repurposing claim to a
   clearly-labeled "candidates for future validation" hypothesis table.
4. Attempt to source matched clinical metadata for at least one cohort to make BRS or
   BIRTHS clinically validatable at all — without this, drop the clinical-correlation
   claims from any future manuscript entirely rather than leaving them unverifiable.
5. Only after 1–4: revisit whether a single dual-disease paper is still the right
   framing, or whether the (now much stronger) AD finding and the (now much weaker)
   T2D finding are better split into a confirmatory AD blood-classification note and a
   separate T2D power-analysis/methods note.
