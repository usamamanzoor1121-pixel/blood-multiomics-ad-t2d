#!/usr/bin/env python3
"""
00b_parse_downloaded_data.py
=============================
Parses the ALREADY-DOWNLOADED raw GEO files (via curl, see logs/) into the
exact CSV format preprocess_blood.py expects. This replaces the wget-based
downloader in 00_download_real_data.py (wget unavailable in this environment)
but reuses its metadata-parsing logic.

Inputs (already on disk):
    data/raw/t2d/GSE221521/GSE221521_gene_expression.xls.gz
    data/raw/t2d/GSE221521/GSE221521_series_matrix.txt.gz
    data/raw/ad/GSE63060/GSE63060_normalized.txt.gz
    data/raw/ad/GSE63060/GSE63060_series_matrix.txt.gz
    data/raw/ad/GSE63061/GSE63061_normalized.txt.gz
    data/raw/ad/GSE63061/GSE63061_series_matrix.txt.gz
    data/raw/annot/GPL6947.annot.gz   (GSE63060 probe->symbol)
    data/raw/annot/GPL10558.annot.gz  (GSE63061 probe->symbol)

Outputs:
    data/raw/t2d/GSE221521/GSE221521_expression_raw.csv  (genes x samples, counts)
    data/raw/t2d/GSE221521/GSE221521_metadata.csv         (diagnosis)
    data/raw/ad/GSE63060/GSE63060_expression_raw.csv      (gene symbols x samples)
    data/raw/ad/GSE63060/GSE63060_metadata.csv            (diagnosis, age, gender)
    data/raw/ad/GSE63061/GSE63061_expression_raw.csv
    data/raw/ad/GSE63061/GSE63061_metadata.csv
"""
import gzip
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW = Path("data/raw")


# ── T2D: GSE221521 ──────────────────────────────────────────────────────────
def parse_t2d():
    log.info("=" * 60)
    log.info("Parsing GSE221521 (T2DM whole blood RNA-seq)")
    log.info("=" * 60)

    count_file = RAW / "t2d/GSE221521/GSE221521_gene_expression.xls.gz"
    series_file = RAW / "t2d/GSE221521/GSE221521_series_matrix.txt.gz"
    out_expr = RAW / "t2d/GSE221521/GSE221521_expression_raw.csv"
    out_meta = RAW / "t2d/GSE221521/GSE221521_metadata.csv"

    full_df = pd.read_csv(count_file, sep="\t", compression="gzip", low_memory=False)
    log.info(f"  Raw file shape: {full_df.shape}")

    full_df = full_df.set_index("gene_name")
    count_cols = [c for c in full_df.columns if c.endswith("_count")]
    log.info(f"  Found {len(count_cols)} count columns")
    expr_df = full_df[count_cols].copy()
    expr_df.columns = [c.replace("_count", "") for c in expr_df.columns]
    expr_df = expr_df[~expr_df.index.duplicated(keep="first")]
    expr_df = expr_df[expr_df.index.notna()]
    log.info(f"  Expression matrix: {expr_df.shape[0]:,} genes x {expr_df.shape[1]:,} samples")

    # Metadata: parse diagnosis from !Sample_title ("leukocytes, DM group RNA1")
    with gzip.open(series_file, "rt") as f:
        lines = f.readlines()
    titles = []
    for line in lines:
        if line.startswith("!Sample_title"):
            titles = [t.strip().strip('"') for t in line.rstrip("\n").split("\t")[1:]]
            break

    sample_to_condition = {}
    for title in titles:
        sid = title.split()[-1]  # e.g. "RNA1"
        t_low = title.lower()
        if ", dm group" in t_low:
            cond = "T2D"
        elif ", dr group" in t_low:
            cond = "PreDM"
        elif "control" in t_low:
            cond = "Control"
        else:
            cond = "Unknown"
        sample_to_condition[sid] = cond

    conditions = [sample_to_condition.get(s, "Unknown") for s in expr_df.columns]
    meta_df = pd.DataFrame({"diagnosis": conditions}, index=expr_df.columns)
    log.info(f"  Conditions: {meta_df['diagnosis'].value_counts().to_dict()}")

    unknown = (meta_df["diagnosis"] == "Unknown").sum()
    if unknown:
        log.warning(f"  {unknown} samples with unresolved diagnosis (dropping)")
        keep = meta_df["diagnosis"] != "Unknown"
        expr_df = expr_df.loc[:, keep.values]
        meta_df = meta_df.loc[keep]

    expr_df.to_csv(out_expr)
    meta_df.to_csv(out_meta)
    log.info(f"  Saved: {out_expr} + {out_meta}")


# ── AD: GSE63060 / GSE63061 ─────────────────────────────────────────────────
def load_gpl_annotation(gpl_path: Path) -> dict:
    log.info(f"  Loading probe->symbol map from {gpl_path.name}")
    with gzip.open(gpl_path, "rt", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("!platform_table_begin"))
    end = next(i for i, l in enumerate(lines) if l.startswith("!platform_table_end"))
    header = lines[start + 1].rstrip("\n").split("\t")
    id_idx = header.index("ID")
    sym_idx = header.index("Gene symbol")
    mapping = {}
    for line in lines[start + 2:end]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) > max(id_idx, sym_idx) and parts[sym_idx]:
            mapping[parts[id_idx]] = parts[sym_idx]
    log.info(f"  Loaded {len(mapping):,} probe->symbol mappings")
    return mapping


def parse_ad_series_matrix(series_file: Path) -> pd.DataFrame:
    with gzip.open(series_file, "rt") as f:
        lines = f.readlines()

    def get_row(tag):
        for line in lines:
            if line.startswith(tag):
                return [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
        return None

    titles = get_row("!Sample_title")
    char_rows = [l for l in lines if l.startswith("!Sample_characteristics_ch1")]

    def parse_char_row(line):
        vals = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
        key = vals[0].split(":")[0].strip().lower() if vals else ""
        return key, [v.split(":", 1)[-1].strip() if ":" in v else v for v in vals]

    status, age, gender = None, None, None
    for line in char_rows:
        key, vals = parse_char_row(line)
        if key == "status":
            status = vals
        elif key == "age":
            age = vals
        elif key == "gender":
            gender = vals

    n = len(titles)
    diag_map = {"AD": "AD", "MCI": "MCI", "CTL": "Control", "CONTROL": "Control"}
    diagnosis = [diag_map.get((status[i] if status else "").upper(), "Unknown") for i in range(n)]

    meta_df = pd.DataFrame({
        "diagnosis": diagnosis,
        "age": [float(a) if a and a.replace(".", "", 1).isdigit() else None for a in (age or [None] * n)],
        "gender": gender if gender else [None] * n,
    }, index=titles)
    return meta_df


def parse_ad(accession: str, gpl_annot_file: str):
    log.info("=" * 60)
    log.info(f"Parsing {accession} (AddNeuroMed blood microarray)")
    log.info("=" * 60)

    outdir = RAW / f"ad/{accession}"
    expr_file = outdir / f"{accession}_normalized.txt.gz"
    series_file = outdir / f"{accession}_series_matrix.txt.gz"
    out_expr = outdir / f"{accession}_expression_raw.csv"
    out_meta = outdir / f"{accession}_metadata.csv"

    expr_df = pd.read_csv(expr_file, sep="\t", compression="gzip", index_col=0, low_memory=False)
    log.info(f"  Raw probe matrix: {expr_df.shape}")

    probe_to_symbol = load_gpl_annotation(RAW / "annot" / gpl_annot_file)
    expr_df.index = [probe_to_symbol.get(p, p) for p in expr_df.index]
    expr_df = expr_df[~expr_df.index.str.startswith("ILMN_")]
    expr_df = expr_df[expr_df.index != ""]
    expr_df = expr_df.groupby(expr_df.index).mean()  # collapse multi-probe genes
    log.info(f"  After probe->symbol mapping: {expr_df.shape[0]:,} unique genes x {expr_df.shape[1]:,} samples")

    meta_df = parse_ad_series_matrix(series_file)
    common = expr_df.columns.intersection(meta_df.index)
    log.info(f"  Samples matched between expression and metadata: {len(common)}/{expr_df.shape[1]}")
    expr_df = expr_df[common]
    meta_df = meta_df.loc[common]

    unknown = (meta_df["diagnosis"] == "Unknown").sum()
    if unknown:
        log.warning(f"  {unknown} samples with unresolved diagnosis (dropping)")
        keep = meta_df["diagnosis"] != "Unknown"
        expr_df = expr_df.loc[:, keep.values] if hasattr(keep, "values") else expr_df.loc[:, keep]
        meta_df = meta_df.loc[keep]

    log.info(f"  Conditions: {meta_df['diagnosis'].value_counts().to_dict()}")

    expr_df.to_csv(out_expr)
    meta_df.to_csv(out_meta)
    log.info(f"  Saved: {out_expr} + {out_meta}")


if __name__ == "__main__":
    parse_t2d()
    parse_ad("GSE63060", "GPL6947.annot.gz")
    parse_ad("GSE63061", "GPL10558.annot.gz")
    log.info("\nAll real GEO data parsed successfully.")
