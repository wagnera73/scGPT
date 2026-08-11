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
adata = sc.read_h5ad("../data/lam_ctrl_final.h5ad")
treat_meta = pd.read_csv("../../Data/LAM_Samples_Rapa_Treatment.csv")
treat_map = dict(zip(treat_meta['Rapa'], treat_meta['Treatment']))
adata.obs['Rapa_Treat']  = adata.obs['DataID'].map(treat_map)
adata = adata[adata.obs['Rapa_Treat'].notna(), :].copy()
ori_batch_col = "Technology"
adata.obs["celltype"] = adata.obs["celltype_011625"].astype("category")
adata.obs["condition"] = adata.obs["Rapa_Treat"].astype("category")

raw_data = adata.raw.X.copy()
if issparse(raw_data):
    # Only modify the non-zero data points for efficiency
    raw_data.data = np.round(raw_data.data).astype(int)
else:
    raw_data = np.round(raw_data).astype(int)
adata.layers["counts"] = raw_data
adata.X = adata.layers["counts"].copy()

if sum(1 for gene in adata.var_names if gene.startswith("ENSG")) > 0:
    gene_map_df  = pd.read_csv("../data/Preprocessing/LAM.genes_rds.txt", sep = "\t")
    adata.var['gene_symbol'] = gene_map_df.loc[adata.var_names, 'symbol'].values
    if adata.var['gene_symbol'].is_unique:
        adata.var_names = adata.var['gene_symbol']
    if '_index' in adata.var.columns:
        adata.var = adata.var.drop(columns=['_index'])
#adata.var = adata.var.set_index("var.features")

model_dir = Path("./save/dev_LAM-Apr13-13-42/")

# This step is much faster than training
embed_adata = scg.tasks.embed_data(
    adata, 
    model_dir, 
    gene_col="index",      # or the column name where your gene symbols are
    batch_size=32,
    device="cuda"          # or "cpu" if no GPU is available
)

# The embeddings are now in embed_adata.obsm["X_scGPT"]
print(adata.obsm.keys())
# 1. Compute neighbors in the scGPT space
sc.pp.neighbors(embed_adata, use_rep="X_scGPT")

# 2. Run UMAP for visualization
sc.tl.umap(embed_adata)

# 3. Cluster to get 'Identities' (Leiden is standard)
sc.tl.leiden(embed_adata, resolution=0.5)

# 4. Plot
sc.pl.umap(embed_adata, color=["leiden", "celltype", "condition"], frameon=False)



# 1. Check if the column exists in .raw.var
if '_index' in embed_adata.raw.var.columns:
    print("Found '_index' column in raw.var, renaming it to 'raw_index'...")
    embed_adata.raw.var.rename(columns={'_index': 'raw_index'}, inplace=True)

# 2. Also check the main .var just in case
if '_index' in embed_adata.var.columns:
    print("Found '_index' column in .var, renaming it to 'var_index'...")
    embed_adata.var.rename(columns={'_index': 'var_index'}, inplace=True)

# 3. Now attempt the write
embed_adata.write("../data/finetune_lam_treatment_20260428.h5ad")