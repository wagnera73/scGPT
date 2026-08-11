import json
from pathlib import Path
from typing import Tuple

import scanpy as sc
from anndata import AnnData
import torch
from torchtext.vocab import Vocab

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained

REPO_ROOT = Path(__file__).resolve().parent.parent

# Maps a dataset name to where its h5ad lives and which obs columns hold
# which kind of per-cell metadata label. Each label_columns entry is a list
# of obs columns; a single column is used as-is, multiple columns are
# joined with "_" to build a composite label (e.g. celltype_condition).
DATASET_REGISTRY = {
    "AML": {
        "path": REPO_ROOT / "data" / "AML_All_Samples.h5ad",
        "label_columns": {
            "celltype": ["AnnoCellType"],
            "condition": ["disease"],
            "celltype_condition": ["AnnoCellType", "disease"],
        },
        "batch_column": "ID",
    },
}


def load_dataset(dataset_name: str, label_column: str) -> Tuple[AnnData, str, str]:
    """
    Load a registered dataset and prepare the obs columns expected by the
    fine-tuning pipeline ("celltype", "str_batch", "batch_id", and
    var["gene_name"]).

    Returns the prepared AnnData, the label obs column actually used (i.e.
    "celltype", which is what the rest of the pipeline reads), and the raw
    obs column used to build "str_batch"/"batch_id" (None if the dataset has
    no registered batch_column, in which case every cell is batch "0").
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Registered datasets: "
            f"{list(DATASET_REGISTRY.keys())}"
        )
    entry = DATASET_REGISTRY[dataset_name]

    if label_column not in entry["label_columns"]:
        raise ValueError(
            f"Dataset '{dataset_name}' has no '{label_column}' label column "
            f"registered. Available: {list(entry['label_columns'].keys())}"
        )
    obs_source_columns = entry["label_columns"][label_column]

    adata = sc.read_h5ad(entry["path"])
    label_values = adata.obs[obs_source_columns[0]].astype(str)
    for col in obs_source_columns[1:]:
        label_values = label_values + "_" + adata.obs[col].astype(str)
    adata.obs["celltype"] = label_values.astype("category")

    batch_column = entry.get("batch_column")
    if batch_column is not None and batch_column in adata.obs:
        adata.obs["str_batch"] = adata.obs[batch_column].astype(str)
    else:
        adata.obs["str_batch"] = "0"
        batch_column = None
    adata.obs["batch_id"] = adata.obs["str_batch"].astype("category").cat.codes.values

    adata.var["gene_name"] = adata.var.index.tolist()

    return adata, "celltype", batch_column


def load_model(
    config,
    vocab: GeneVocab,
    num_types: int,
    num_batch_types: int,
    device: torch.device,
) -> Tuple[TransformerModel, GeneVocab]:
    """
    Build a TransformerModel with a classification head (n_cls=num_types)
    and, if config.load_model is set, load pretrained weights into it.
    """
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    model_file = None
    if config.load_model is not None:
        model_dir = Path(config.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
    else:
        embsize = config.layer_size
        nhead = config.nhead
        nlayers = config.nlayers
        d_hid = config.layer_size

    vocab.set_default_index(vocab[pad_token])

    ntokens = len(vocab)
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=3,
        n_cls=num_types,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=pad_token,
        pad_value=-2,
        do_mvc=config.MVC,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=config.DSBN,
        n_input_bins=config.n_bins,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=False,
        use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm,
    )

    if model_file is not None:
        load_pretrained(model, torch.load(model_file, map_location=device), verbose=False)

    model.to(device)
    return model, vocab
