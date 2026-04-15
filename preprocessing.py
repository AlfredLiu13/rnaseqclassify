from __future__ import annotations
import scanpy as sc 
import pandas as pd
from anndata import AnnData
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from sklearn.svm import SVC
import os 

def read_qc(dirpath: str) -> AnnData: 

    """Reads data from a given filepath into an AnnData object and performs QC filtering of doublets"""

    # Load the data
    adata = sc.read_10x_mtx(dirpath, var_names='gene_symbols', cache=True)

    # QC 
    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("Mt-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    return adata


def normalize(adata: AnnData) -> None: 

    """Normalizes all cells by sequencing depth and log transformation"""

    # save raw counts
    adata.layers['raw_counts'] = adata.X

    # normalize by sequencing depth such that cells have same count 
    sc.pp.normalize_total(adata) 

    # log transform data
    sc.pp.log1p(adata) 

def reduce(adata: AnnData, num_genes: int) -> AnnData: 

    """selects highly variable genes and performs PCA"""

    print(f'Total Cells: {adata.n_obs}')
    print(f'Total Genes: {adata.n_vars}')

    # annotate highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=num_genes)
    sc.pl.highly_variable_genes(adata)

    # subset to highly variable genes
    if 'highly_variable' not in adata.var.columns:
        adata.var['highly_variable'] = False
    adata = adata[:, adata.var['highly_variable']]
    
    print(f'After subsetting to HVGs: {adata.shape}')
    
    # scale and PCA
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pl.pca(adata, size=10)
    
    return adata

def annotate(adata: AnnData, annotation_filepath: str, organ_type: str = "heart") :
    """
    Maps cells to cluster annotation. Outputs features dataset and labels dataset. 

        1. Maps each unique cluster/cell type to wa number 
        2. Maps each sample to the label
        3. Splits data into 70/30 train/test datasets 
    """
    
    # read annotation file
    annotations = pd.read_csv(annotation_filepath)
    
    # filter for the correct batch
    if organ_type == "heart" : 
        batch = "10X_P7_4"
    elif organ_type == "thymus": 
        batch = "10X_P7_11"

    annotations = annotations[annotations['cell'].str.contains(batch)]
    
    # extract barcode and format to match AnnData
    annotations = annotations.copy()
    annotations['barcode'] = annotations['cell'].str.split('_').str[-1] + '-1'
    annotations.set_index('barcode', inplace=True)
    
    # keep only cell_ontology_class and cluster.ids columns from annotations file
    annotations = annotations[['cell_ontology_class', 'cluster.ids']]
    
    # merge columns with adata.obs
    for col in annotations.columns:
        adata.obs[col] = annotations[col]
    
    return adata 


def get_train_test(adata: AnnData, label_col: str = "cluster.ids", test_size: float = 0.3, random_state: int = 42) -> tuple:
    """
    Extracts PCA features and labels from annotated AnnData and splits into train/test sets for SVM.
    
    Returns numpy arrays compatible with scikit-learn's SVM.Classifier:
    - X_train, X_test: 2D arrays (n_samples, n_features) - PCA coordinates
    - y_train, y_test: 1D arrays (n_samples,) - class labels
    """
    
    # get feature and labels matrix 
    X = adata.obsm['X_pca']  # PCA coordinates (2D array)
    y = np.asarray(adata.obs[label_col].values)  # Convert to 1D numpy array
    
    # filter out cells with NaN labels (unmatched annotations)
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"Using {len(y)} annotated cells (removed {np.sum(~valid_idx)} unmatched cells)")
    
    # split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y.astype(int)
    )
    
    return X_train, X_test, y_train, y_test


def plot_svm_accuracy(model, X_test, y_test, save_path: str = "svm_confusion_matrix.png"):
    """
    Plot confusion matrix and accuracy metrics, then save to file.
    
    Parameters:
    - model: trained SVM classifier
    - X_test: test features (2D array)
    - y_test: test labels (1D array)
    - save_path: path to save the plot (default: svm_confusion_matrix.png)
    """
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # plot confusion matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=True)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # plot accuracy, precision, recall, f1 
    ax2.text(0.5, 0.92, f'Accuracy: {accuracy:.4f}', ha='center', fontsize=14, weight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.55, classification_report(y_test, y_pred), ha='center', va='top', fontsize=10, family='monospace', transform=ax2.transAxes)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close() 

if __name__ == "__main__" : 

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    dir_path = os.path.join(script_dir, "droplet/Thymus-10X_P7_11")
    annotation_fpath = os.path.join(script_dir, "droplet/annotations_droplet.csv")
    num_var_genes = 2000

    # preprocess the file     
    adata = read_qc(dir_path) 
    normalize(adata) 
    adata = reduce(adata, num_var_genes)
    adata = annotate(adata, annotation_fpath, "heart")

    adata_preprocess = adata.copy()
    adata_preprocess.write_h5ad('heart_aorta_qc_normalized_pca.h5ad')

    # Export first 50 PCA coordinates for thymus dataset
    pca_df = pd.DataFrame(
        adata.obsm['X_pca'][:, :50],  # First 50 PCs
        columns=[f'PC_{i+1}' for i in range(50)],
        index=adata.obs_names
    )
    pca_df.to_csv('thymus_pca_coordinates.csv')


    # # split into training and testing datasets
    # X_train, X_test, y_train, y_test = get_train_test(adata) 
    
    # # run SVM and test accuracy 
    # svm = SVC(kernel='rbf')
    # svm.fit(X_train, y_train)
    # accuracy = svm.score(X_test, y_test)

    # # plot 
    # save_path = os.path.join(script_dir, "svm_accuracy.png")
    # plot_svm_accuracy(svm, X_test, y_test, save_path=save_path)