import scanpy as sc
import pandas as pd
from anndata import AnnData
import numpy as np

def read_qc(dirpath: str) -> AnnData:
    """Reads data from a given filepath into an AnnData object and performs QC gene annotation (original version)."""
    adata = sc.read_10x_mtx(dirpath, var_names='gene_symbols', cache=True)
    adata.var["mt"] = adata.var_names.str.startswith("Mt-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    return adata

def read_qc_corrected(dirpath: str) -> AnnData:
    """Reads data and performs QC gene annotation and cell filtering as in preprocessing.ipynb."""
    adata = sc.read_10x_mtx(dirpath, var_names='gene_symbols', cache=True)
    # Annotate mitochondrial genes (human: 'MT-')
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # Filter cells: keep cells with n_genes_by_counts >= 200, pct_counts_mt < 10
    adata = adata[adata.obs['n_genes_by_counts'] >= 200, :]
    adata = adata[adata.obs['pct_counts_mt'] < 10, :]
    return adata

def main():
    heart_dir = "droplet/Heart_and_Aorta-10X_P7_4"
    thymus_dir = "droplet/Thymus-10X_P7_11"

    # Heart & Aorta
    adata_orig_heart = read_qc(heart_dir)
    adata_corr_heart = read_qc_corrected(heart_dir)
    print("--- Heart & Aorta ---")
    print(f"Original QC: {adata_orig_heart.n_obs} cells")
    print(f"Corrected QC: {adata_corr_heart.n_obs} cells")
    print(f"Cells removed by corrected QC: {adata_orig_heart.n_obs - adata_corr_heart.n_obs}\n")

    # Thymus
    adata_orig_thymus = read_qc(thymus_dir)
    adata_corr_thymus = read_qc_corrected(thymus_dir)
    print("--- Thymus ---")
    print(f"Original QC: {adata_orig_thymus.n_obs} cells")
    print(f"Corrected QC: {adata_corr_thymus.n_obs} cells")
    print(f"Cells removed by corrected QC: {adata_orig_thymus.n_obs - adata_corr_thymus.n_obs}")

if __name__ == "__main__":
    main()
