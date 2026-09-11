import copy
import gc
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

import scgpt as scg
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import random_mask_value, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import add_file_handler, set_seed

from scripts.plotting import generate_run_plots
from scripts.utils import REPO_ROOT, load_dataset, load_model

logger = scg.logger


def load_default_hyperparameters():
    hyperparameter_defaults = {}

    # Add defaults for the batch correction task
    hyperparameter_defaults["batch_correction"] = dict(
        seed=42,
        do_train=True,
        mask_ratio=0.4,
        epochs=30,
        n_bins=20,
        GEPC=True,  # Masked value prediction for cell embedding
        ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=1.0,
        lr=1e-4,
        batch_size=32,
        layer_size=128,
        nlayers=4,
        nhead=4,
        # if load model, batch_size, layer_size, nlayers, nhead will be ignored
        dropout=0.2,
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        log_interval=100,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision
    )
    # Add defaults for the perturbation task
    hyperparameter_defaults["perturbation"] = dict(
        seed=0,
        do_train=True,
        mask_ratio=0.0,
        epochs=10,
        n_bins=51,
        MVC=False,  # Masked value prediction for cell embedding
        ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=0.0,
        lr=1e-4,
        batch_size=32,
        layer_size=128,
        nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead=4,  # number of heads in nn.MultiheadAttention
        dropout=0.2,  # dropout probability
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision
        include_zero_gene=False,
        freeze=False,  # freeze
        DSBN=False,  # Domain-spec batchnorm
    )
    # Add defaults for the metadata (celltype / condition) prediction finetune task
    hyperparameter_defaults["finetune"] = dict(
        seed=0,
        do_train=True,
        mask_ratio=0.0,
        epochs=15,
        n_bins=51,
        n_hvg=1200,  # number of highly variable genes
        #max_seq_len=3001,  # matches tutorials/Tutorial_Annotation.ipynb
        balance_classes=False,  # stratified split + inverse-frequency class-weighted loss
        MVC=False,  # Masked value prediction for cell embedding
        ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=0.0,
        lr=1e-4,
        batch_size=32,
        layer_size=128,
        nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead=4,  # number of heads in nn.MultiheadAttention
        dropout=0.2,  # dropout probability
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        log_interval=100,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision
        include_zero_gene=False,
        DSBN=False,  # Domain-spec batchnorm
    )

    return hyperparameter_defaults


def run_batch(config):
    print("--- Running Batch Integration ---")


def run_perturbation(config):
    print("--- Running Cell Perturbation ---")
    print(" -- Setup -- ")
    print("  - Summarize ")


def run_finetune(config):
    """
    Fine-tune a pretrained scGPT model to predict a per-cell metadata label
    (e.g. celltype or condition) via the model's CLS classification head.
    Adapted from tutorials/Tutorial_Annotation.ipynb.
    """
    print("--------------------------------")
    print("--- Running Model Finetuning ---")
    print("--------------------------------")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device available: {device}")

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = config.mask_ratio
    mask_value = -1
    pad_value = -2
    n_input_bins = config.n_bins
    include_zero_gene = config.include_zero_gene

    save_dir = (
        REPO_ROOT
        / "save"
        / "finetuned"
        / f"dev_{config.dataset_name}-{config.label_column}-{time.strftime('%b%d-%H-%M')}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"save to {save_dir}")
    add_file_handler(logger, save_dir / "run.log")

    # ------------------------------------------------------------------
    # Load and pre-process data
    # ------------------------------------------------------------------
    adata, label_obs_col, batch_column = load_dataset(config.dataset_name, config.label_column)
    config.update({"batch_column": batch_column})

    with open(save_dir / "config.json", "w") as f:
        json.dump(dict(config), f, indent=2, default=str)

    if config.load_model is not None:
        model_dir = Path(config.load_model)
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
    else:
        genes = adata.var["gene_name"].tolist()
        vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab[pad_token])


    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=3,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=config.n_hvg,
        hvg_flavor="seurat_v3",
        binning=config.n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key="str_batch")

    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].toarray()
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()
    # Cap the sequence length instead of using every gene in the dataset
    # (len(genes) + 1): with include_zero_gene=False, pad_batch pads each
    # batch up to this cap, and standard (non flash-attn) attention is
    # O(seq_len^2) in memory, so leaving this uncapped causes CUDA OOM on
    # datasets with many genes matched to the vocab.
    #max_seq_len = min(len(genes) + 1, config.max_seq_len)
    max_seq_len = config.n_hvg + 1


    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
    num_types = len(id2type)
    celltypes_labels = np.array(celltype_id_labels)

    batch_ids = np.array(adata.obs["batch_id"].tolist())
    str_batch_ids = np.array(adata.obs["str_batch"].tolist())
    num_batch_types = len(set(batch_ids))
    cell_indices = np.arange(adata.n_obs)

    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
        train_str_batch,
        valid_str_batch,
        train_idx,
        valid_idx,
    ) = train_test_split(
        all_counts,
        celltypes_labels,
        batch_ids,
        str_batch_ids,
        cell_indices,
        test_size=0.1,
        shuffle=True,
        stratify=celltypes_labels if config.balance_classes else None,
    )

    # Every categorical/string obs column (not just "celltype" and
    # "str_batch") gets its own UMAP later on, so later analyses have all of
    # them on hand rather than just the two the run happens to train on.
    # Columns that are unique per cell (e.g. a barcode) aren't meaningful as
    # a coloring and are skipped.
    metadata_columns = [
        c
        for c in adata.obs.select_dtypes(include=["category", "object", "bool"]).columns
        if adata.obs[c].nunique(dropna=True) < adata.n_obs
    ]
    metadata_fields = {
        c: adata.obs[c].astype(str).to_numpy()[valid_idx] for c in metadata_columns
    }

    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )

    def prepare_data() -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )

        train_data_pt = {
            "gene_ids": tokenized_train["genes"],
            "values": masked_values_train,
            "target_values": tokenized_train["values"],
            "batch_labels": torch.from_numpy(train_batch_labels).long(),
            "celltype_labels": torch.from_numpy(train_celltype_labels).long(),
        }
        valid_data_pt = {
            "gene_ids": tokenized_valid["genes"],
            "values": masked_values_valid,
            "target_values": tokenized_valid["values"],
            "batch_labels": torch.from_numpy(valid_batch_labels).long(),
            "celltype_labels": torch.from_numpy(valid_celltype_labels).long(),
        }
        return train_data_pt, valid_data_pt

    class SeqDataset(Dataset):
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.data = data

        def __len__(self):
            return self.data["gene_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    def prepare_dataloader(
        data_pt: Dict[str, torch.Tensor], batch_size: int, shuffle: bool
    ) -> DataLoader:
        return DataLoader(
            dataset=SeqDataset(data_pt),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Load the pre-trained scGPT model with a classification head
    # ------------------------------------------------------------------
    model, vocab = load_model(config, vocab, num_types, num_batch_types, device)

    if config.balance_classes:
        # Inverse-frequency weights (sklearn's "balanced" scheme) computed
        # from the training split only, so rare classes contribute more
        # per-example loss without duplicating/oversampling any cells.
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_types),
            y=train_celltype_labels,
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion_cls = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    def train_one_epoch(model: nn.Module, loader: DataLoader, epoch: int) -> None:
        model.train()
        total_loss, total_error = 0.0, 0.0
        log_interval = config.log_interval
        start_time = time.time()

        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    CLS=True,
                    MVC=config.MVC,
                    ECS=config.ecs_thres > 0,
                )
                loss = criterion_cls(output_dict["cls_output"], celltype_labels)
                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        "Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}."
                    )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_error += error_rate
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{len(loader):3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {total_loss / log_interval:5.2f} | "
                    f"err {total_error / log_interval:5.2f} |"
                )
                total_loss, total_error = 0.0, 0.0
                start_time = time.time()

    def evaluate(
        model: nn.Module,
        loader: DataLoader,
        return_raw: bool = False,
        collect_extra: bool = False,
    ):
        """collect_extra=True also gathers per-cell softmax probabilities and
        CLS cell embeddings (used for the confusion/ROC/UMAP plots below), at
        the cost of extra memory - so it's only turned on for the one-off
        base-model and final-best-model passes, not every epoch."""
        model.eval()
        total_loss, total_error, total_num = 0.0, 0.0, 0
        predictions = []
        probs_list = []
        cell_embs = []
        true_labels_list = []
        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                celltype_labels = batch_data["celltype_labels"].to(device)

                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                with torch.cuda.amp.autocast(enabled=config.amp):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=None,
                        CLS=True,
                        MVC=False,
                        ECS=False,
                    )
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls(output_values, celltype_labels)

                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_loss += loss.item() * len(input_gene_ids)
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_num += len(input_gene_ids)
                predictions.append(output_values.argmax(1).cpu().numpy())
                if collect_extra:
                    probs_list.append(
                        torch.softmax(output_values.float(), dim=1).cpu().numpy()
                    )
                    cell_embs.append(output_dict["cell_emb"].float().cpu().numpy())
                    true_labels_list.append(celltype_labels.cpu().numpy())

        if collect_extra:
            return {
                "predictions": np.concatenate(predictions, axis=0),
                "probs": np.concatenate(probs_list, axis=0),
                "cell_emb": np.concatenate(cell_embs, axis=0),
                "true_labels": np.concatenate(true_labels_list, axis=0),
            }
        if return_raw:
            return np.concatenate(predictions, axis=0)
        return total_loss / total_num, total_error / total_num

    # ------------------------------------------------------------------
    # Cell embeddings from the base (pre-finetuning) model, kept aside for
    # the base-vs-finetuned UMAP comparison plots generated below. Must run
    # before any training step touches `model`'s weights.
    # ------------------------------------------------------------------
    base_valid_loader = prepare_dataloader(
        prepare_data()[1], config.batch_size, shuffle=False
    )
    base_eval = evaluate(model, base_valid_loader, collect_extra=True)

    # ------------------------------------------------------------------
    # Finetune scGPT with the CLS classification objective
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_model = None
    best_model_epoch = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data()
        train_loader = prepare_dataloader(train_data_pt, config.batch_size, shuffle=True)
        valid_loader = prepare_dataloader(valid_data_pt, config.batch_size, shuffle=False)

        if config.do_train:
            train_one_epoch(model, train_loader, epoch)
        val_loss, val_err = evaluate(model, valid_loader)
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:5.4f} | err {val_err:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        scheduler.step()

    # ------------------------------------------------------------------
    # Evaluate + save
    # ------------------------------------------------------------------
    valid_data_pt = prepare_data()[1]
    valid_loader = prepare_dataloader(valid_data_pt, config.batch_size, shuffle=False)
    final_eval = evaluate(best_model, valid_loader, collect_extra=True)
    predictions = final_eval["predictions"]
    valid_labels = final_eval["true_labels"]

    results = {
        "test/accuracy": accuracy_score(valid_labels, predictions),
        "test/precision": precision_score(
            valid_labels, predictions, average="macro", zero_division=0
        ),
        "test/recall": recall_score(
            valid_labels, predictions, average="macro", zero_division=0
        ),
        "test/macro_f1": f1_score(
            valid_labels, predictions, average="macro", zero_division=0
        ),
        "test/best_model_epoch": best_model_epoch,
    }
    logger.info(
        f"Accuracy: {results['test/accuracy']:.3f}, "
        f"Precision: {results['test/precision']:.3f}, "
        f"Recall: {results['test/recall']:.3f}, "
        f"Macro F1: {results['test/macro_f1']:.3f}"
    )

    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    vocab.save_json(save_dir / "vocab.json")
    with open(save_dir / "id2type.json", "w") as f:
        json.dump({str(k): v for k, v in id2type.items()}, f)

    try:
        generate_run_plots(
            save_dir=save_dir,
            id2type=id2type,
            train_celltype_labels=train_celltype_labels,
            valid_celltype_labels=valid_labels,
            valid_batch_labels=valid_str_batch,
            metadata_fields=metadata_fields,
            predictions=predictions,
            probs=final_eval["probs"],
            base_cell_emb=base_eval["cell_emb"],
            finetuned_cell_emb=final_eval["cell_emb"],
        )
    except Exception:
        # Diagnostic plots are a nice-to-have on top of an already-saved
        # checkpoint; a plotting bug shouldn't take down an otherwise-
        # successful (and expensive) training run.
        logger.exception("Failed to generate diagnostic plots; continuing.")

    gc.collect()
    return results
