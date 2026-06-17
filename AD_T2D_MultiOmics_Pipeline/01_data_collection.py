"""
=============================================================================
STEP 1 — DATA COLLECTION GUIDE
Projects: Alzheimer's Disease (AD) + Type 2 Diabetes (T2D)
=============================================================================
This script documents every database, download command, and file format
needed to reproduce these pipelines on real data.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# ALZHEIMER'S DISEASE — DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────

AD_DATABASES = {

    "GEO — RNA-Seq (Primary)": {
        "accession": "GSE53697",
        "description": "Human brain RNA-seq: AD vs Control (prefrontal cortex)",
        "samples": "329 AD + 188 Control",
        "data_types": ["Gene expression (FPKM)", "Clinical metadata"],
        "download": [
            "# R:",
            "library(GEOquery)",
            "gse <- getGEO('GSE53697', GSEMatrix=TRUE)",
            "# Python (via GEOparse):",
            "pip install GEOparse",
            "import GEOparse",
            "gse = GEOparse.get_GEO('GSE53697', destdir='data/raw/')",
        ],
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53697"
    },

    "GEO — RNA-Seq (Secondary bulk)": {
        "accession": "GSE132903",
        "description": "AD bulk RNA-seq — entorhinal cortex + temporal gyrus",
        "samples": "474 samples (AD stages I-VI)",
        "data_types": ["Raw counts matrix", "Braak staging"],
        "download": ["# Same GEOparse approach as above"],
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132903"
    },

    "AMP-AD / Synapse (Best dataset)": {
        "accession": "syn5550404",
        "description": "ROSMAP — Religious Orders Study/Memory and Aging Project",
        "samples": "1,210 samples | RNA-seq + Methylation + Proteomics",
        "data_types": ["RNA-seq counts", "DNA methylation (450k)",
                       "Proteomics (TMT-MS)", "Clinical: MMSE, CDR, Braak"],
        "download": [
            "# 1. Register at synapse.org (free)",
            "pip install synapseclient",
            "import synapseclient",
            "syn = synapseclient.Synapse()",
            "syn.login('your_email', 'your_password')",
            "# RNA-seq",
            "syn.get('syn8691899', downloadLocation='data/raw/rnaseq/')",
            "# Methylation",
            "syn.get('syn4586376', downloadLocation='data/raw/methylation/')",
            "# Proteomics",
            "syn.get('syn17015360', downloadLocation='data/raw/proteomics/')",
        ],
        "url": "https://www.synapse.org/#!Synapse:syn5550404"
    },

    "ADNI — Neuroimaging + Biomarkers": {
        "accession": "ADNI",
        "description": "Alzheimer's Disease Neuroimaging Initiative",
        "samples": "1,800+ participants | MRI + PET + CSF + Blood biomarkers",
        "data_types": ["MRI volumes", "Amyloid PET", "CSF Abeta42/tau/p-tau",
                       "Cognitive scores (ADAS-Cog, MMSE)"],
        "download": [
            "# 1. Apply at adni.loni.usc.edu (1-2 days approval)",
            "# 2. Download ADNIMERGE R package",
            "install.packages('ADNIMERGE')",
            "library(ADNIMERGE)",
            "data(adnimerge)",
        ],
        "url": "https://adni.loni.usc.edu/"
    },

    "GWAS Catalog — AD risk variants": {
        "accession": "GCST90027158",
        "description": "Largest AD GWAS meta-analysis (Bellenguez et al. 2022)",
        "samples": "111,326 cases + 677,663 controls",
        "download": [
            "wget https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/",
            "# Filter for genome-wide significant SNPs (p < 5e-8)",
        ],
        "url": "https://www.ebi.ac.uk/gwas/studies/GCST90027158"
    },

    "Allen Brain Atlas — Cell-type expression": {
        "description": "Single-cell RNA-seq of human brain regions",
        "download": ["# API: http://api.brain-map.org/"],
        "url": "https://portal.brain-map.org/"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# TYPE 2 DIABETES — DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────

T2D_DATABASES = {

    "GEO — Pancreatic Islet RNA-Seq": {
        "accession": "GSE50397",
        "description": "Human pancreatic islets: T2D vs Non-diabetic",
        "samples": "77 T2D + 80 Non-diabetic",
        "data_types": ["Gene expression", "Insulin secretion phenotypes"],
        "download": ["import GEOparse; gse = GEOparse.get_GEO('GSE50397')"],
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50397"
    },

    "GEO — Multi-tissue RNA-Seq": {
        "accession": "GSE41762",
        "description": "T2D across pancreas, liver, adipose, skeletal muscle",
        "samples": "148 samples across 4 tissues",
        "download": ["import GEOparse; gse = GEOparse.get_GEO('GSE41762')"],
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41762"
    },

    "GTEx — Multi-tissue expression (Best)": {
        "description": "Genotype-Tissue Expression: 54 tissues, 17,382 samples",
        "data_types": ["RNA-seq TPM", "eQTL data", "Tissue-specific expression"],
        "download": [
            "# Direct download:",
            "wget https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/",
            "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz",
            "# Filter for: Pancreas, Liver, Adipose, Skeletal Muscle",
        ],
        "url": "https://gtexportal.org/home/datasets"
    },

    "DIAGRAM GWAS — T2D genetic variants": {
        "accession": "GCST90132315",
        "description": "T2D multi-ancestry GWAS (Mahajan et al. 2022)",
        "samples": "2.5 million participants",
        "download": [
            "wget https://ftp.ebi.ac.uk/pub/databases/gwas/",
            "summary_statistics/GCST90132315/",
        ],
        "url": "https://www.ebi.ac.uk/gwas/studies/GCST90132315"
    },

    "HMDB — Metabolomics": {
        "description": "Human Metabolome Database — T2D metabolite signatures",
        "data_types": ["Plasma metabolite concentrations", "Metabolic pathway maps"],
        "download": ["wget https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip"],
        "url": "https://hmdb.ca/"
    },

    "UK Biobank (Application required)": {
        "description": "500k participants: genetics + metabolomics + clinical",
        "data_types": ["NMR metabolomics (249 measures)", "Genotyping array",
                       "HbA1c, glucose, insulin", "Lifestyle data"],
        "note": "Apply at ukbiobank.ac.uk — 4-6 week approval",
        "url": "https://www.ukbiobank.ac.uk/"
    },

    "GEO — DNA Methylation (Pancreas)": {
        "accession": "GSE76352",
        "description": "Pancreatic islet methylation: T2D vs control",
        "samples": "25 T2D + 25 Control",
        "download": ["import GEOparse; gse = GEOparse.get_GEO('GSE76352')"],
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76352"
    }
}

print("=" * 65)
print("  DATA COLLECTION GUIDE READY")
print("=" * 65)
print("\n📊 ALZHEIMER'S DISEASE — Data Sources:")
for name, info in AD_DATABASES.items():
    print(f"\n  ► {name}")
    print(f"    Samples: {info.get('samples', 'Various')}")
    print(f"    URL: {info['url']}")

print("\n\n📊 TYPE 2 DIABETES — Data Sources:")
for name, info in T2D_DATABASES.items():
    print(f"\n  ► {name}")
    print(f"    Samples: {info.get('samples', 'Various')}")
    print(f"    URL: {info['url']}")
