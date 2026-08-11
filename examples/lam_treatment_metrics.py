
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings

import torch
from anndata import AnnData
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
sys.path.append("../")
import scgpt as scg

from scgpt.tokenizer.gene_tokenizer import GeneVocab


from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

adata = sc.read_h5ad("../data/finetune_lam_treatment_20260428.h5ad")

path_dir = "../../../LAM_Treatment/Outputs/seurat_embeddings/"
file_embeds = os.listdir(path_dir)


for i in file_embeds:
    # Load coordinates
    umap_df = pd.read_csv(os.path.join(path_dir,i), index_col=0)

    # Reindex to match the scanpy object exactly
    umap_df = umap_df.reindex(adata.obs_names)

    tmp=i.replace("lam_treatment_", "")
    embed_name=tmp.replace(".csv", "")
    embed_name="X_"+embed_name
    # Assign to obsm
    adata.obsm[embed_name] = umap_df.values


ori_batch_col="DataID"
adata.obs["celltype"] = adata.obs["celltype_011625"].astype("category")

adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.X = adata.X.astype("float32")

bm = Benchmarker(
    adata,
    batch_key="batch_id",
    label_key="celltype",
    bio_conservation_metrics=BioConservation(),
    batch_correction_metrics=BatchCorrection(),
    embedding_obsm_keys=["X_scGPT", "X_seurat_harmony", "X_seurat_integrated.cca", "X_seurat_integrated.rpca", "X_seurat_integrated.scvi", "X_seurat_pca"],
    n_jobs=6,
)
bm.benchmark()
bm.plot_results_table()
# Access the current figure and save it
fig = plt.gcf() 
fig.savefig("../../../LAM_Treatment/Outputs/scGPT_benchmark_results.png", dpi=300, bbox_inches='tight')

# Extract the results as a Pandas DataFrame
results_df = bm.get_results(min_max_scale=True)

# Save to CSV
results_df.to_csv("../../../LAM_Treatment/Outputs/integration_benchmarking_scores.csv")

# Display the first few rows to check
print(results_df.head())
