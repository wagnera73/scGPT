"""Diagnostic plots generated at the end of a scripts.analyses.run_finetune run.

Everything here writes PNGs (plus a manifest.json describing them) into
<run_save_dir>/plots/, which the webapp/ FastAPI app serves on each run's
detail page. Kept separate from scripts/analyses.py so the plotting code -
and its matplotlib/umap imports - stay out of the main training logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe: these run on GPU cluster nodes with no display
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scib.metrics as scib_metrics
import umap
from anndata import AnnData
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _one_hot_labels(y_true: np.ndarray, n_classes: int) -> np.ndarray:
    """label_binarize returns a single column for exactly 2 classes; normalize
    that away so downstream one-vs-rest code can treat binary and multiclass
    the same."""
    if n_classes == 2:
        return np.stack([(y_true == 0).astype(int), (y_true == 1).astype(int)], axis=1)
    return label_binarize(y_true, classes=list(range(n_classes)))


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], out_path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * n + 2), max(4, 1.1 * n + 1.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (validation set)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str], out_path: Path
) -> None:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6.0, 1.0 * len(class_names)), 4.5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall / F1 (validation set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_class_distribution(
    train_labels: np.ndarray,
    valid_labels: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
) -> None:
    n = len(class_names)
    train_counts = np.bincount(train_labels, minlength=n)[:n]
    valid_counts = np.bincount(valid_labels, minlength=n)[:n]
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6.0, 1.0 * n), 4.5))
    ax.bar(x - width / 2, train_counts, width, label="Train")
    ax.bar(x + width / 2, valid_counts, width, label="Valid")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Number of cells")
    ax.set_title("Class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    roc_out_path: Path,
    pr_out_path: Path,
) -> None:
    y_true_bin = _one_hot_labels(y_true, len(class_names))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest, validation set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(roc_out_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], probs[:, i])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (one-vs-rest, validation set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pr_out_path, dpi=150)
    plt.close(fig)


def compute_umap(embeddings: np.ndarray, seed: int = 0) -> np.ndarray:
    reducer = umap.UMAP(random_state=seed)
    return reducer.fit_transform(embeddings)


def plot_umap_scatter(
    coords: np.ndarray, labels: Sequence, out_path: Path, title: str
) -> None:
    labels = np.asarray(labels, dtype=object)
    uniq = sorted(set(labels.tolist()), key=str)
    cmap = plt.get_cmap("tab20", max(len(uniq), 1))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, lab in enumerate(uniq):
        mask = labels == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4,
            color=cmap(i % 20),
            label=str(lab),
        )
    # A legend with many entries (e.g. one per subject/batch) is more noise
    # than signal, so only draw it when it stays readable.
    if len(uniq) <= 25:
        ax.legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_scib_scores(
    embedding: np.ndarray, label_names: Sequence[str], batch_names: Sequence[str]
) -> Dict[str, float]:
    """scib integration metrics summarizing what a UMAP built from this
    embedding is showing.

    Bio signal (0-1, higher is better - true label structure is preserved):
      - bio_score: average silhouette width (ASW) by true label.
      - graph_connectivity: how connected each label's cells are in the kNN
        graph (fragmented clusters score lower).
      - nmi / ari: agreement between an unsupervised Leiden clustering of
        the embedding and the true labels.

    Batch signal (how much batch structure remains):
      - batch_score (0-1, higher is better): ASW of batch within each label
        - close to 1 means batches are well mixed.
      - pcr: fraction of embedding variance explained by batch. NOT 0-1
        "higher is better" like the others - lower means less residual
        batch structure. Only meaningful compared across embeddings (e.g.
        base vs. finetuned), not in isolation.
    """
    adata = AnnData(X=np.zeros((embedding.shape[0], 1), dtype=np.float32))
    adata.obs["label"] = list(label_names)
    adata.obs["label"] = adata.obs["label"].astype("category")
    adata.obs["batch"] = list(batch_names)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obsm["X_emb"] = np.asarray(embedding)

    bio_score = scib_metrics.silhouette(adata, label_key="label", embed="X_emb")
    batch_score = scib_metrics.silhouette_batch(
        adata, batch_key="batch", label_key="label", embed="X_emb", verbose=False
    )

    # Shared kNN graph on the embedding, reused by graph_connectivity and by
    # the Leiden clustering cluster_optimal_resolution sweeps internally to
    # find the resolution that best matches the true labels (for NMI/ARI).
    sc.pp.neighbors(adata, use_rep="X_emb")
    graph_conn = scib_metrics.graph_connectivity(adata, label_key="label")
    scib_metrics.cluster_optimal_resolution(
        adata, label_key="label", cluster_key="cluster", verbose=False
    )
    nmi_score = scib_metrics.nmi(adata, cluster_key="cluster", label_key="label")
    ari_score = scib_metrics.ari(adata, cluster_key="cluster", label_key="label")

    n_comps = max(2, min(50, embedding.shape[1] - 1, embedding.shape[0] - 1))
    pcr_value = scib_metrics.pcr(adata, covariate="batch", embed="X_emb", n_comps=n_comps)

    return {
        "bio_score": float(bio_score),
        "batch_score": float(batch_score),
        "graph_connectivity": float(graph_conn),
        "nmi": float(nmi_score),
        "ari": float(ari_score),
        "pcr": float(pcr_value),
    }


_SCIB_HIGHER_BETTER_METRICS = [
    ("bio_score", "Bio ASW"),
    ("batch_score", "Batch ASW\n(mixing)"),
    ("graph_connectivity", "Graph\nconnectivity"),
    ("nmi", "NMI"),
    ("ari", "ARI"),
]


def plot_scib_scores(scores: Dict[str, Dict[str, float]], out_path: Path) -> None:
    stages = list(scores.keys())
    metric_keys = [k for k, _ in _SCIB_HIGHER_BETTER_METRICS]
    metric_labels = [label for _, label in _SCIB_HIGHER_BETTER_METRICS]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(stages)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, stage in enumerate(stages):
        values = [scores[stage][k] for k in metric_keys]
        offset = (i - (len(stages) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=stage)
        for xi, v in zip(x + offset, values):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("scib score (0-1, higher is better)")
    ax.set_title("scib integration metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pcr_scores(scores: Dict[str, Dict[str, float]], out_path: Path) -> None:
    stages = list(scores.keys())
    values = [scores[s]["pcr"] for s in stages]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.bar(x, values, width=0.5, color="#C44E52")
    pad = max(values) * 0.03 if max(values) > 0 else 0.001
    for xi, v in zip(x, values):
        ax.text(xi, v + pad, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Variance explained by batch (PCR)")
    ax.set_title("Batch-explained variance (PCR) — lower is better")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_run_plots(
    save_dir: Path,
    id2type: Dict[int, str],
    train_celltype_labels: np.ndarray,
    valid_celltype_labels: np.ndarray,
    valid_batch_labels: np.ndarray,
    predictions: np.ndarray,
    probs: np.ndarray,
    base_cell_emb: np.ndarray,
    finetuned_cell_emb: np.ndarray,
) -> List[Dict[str, str]]:
    """Render all diagnostic plots for one run into save_dir/plots/ and return
    (and persist as plots/manifest.json) the list describing them, so the
    webapp doesn't need to guess titles/grouping from filenames."""
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    class_names = [id2type[i] for i in range(len(id2type))]

    manifest: List[Dict[str, str]] = []

    def add(filename: str, title: str, section: str) -> None:
        manifest.append({"filename": filename, "title": title, "section": section})

    plot_confusion_matrix(
        valid_celltype_labels, predictions, class_names, plots_dir / "confusion_matrix.png"
    )
    add("confusion_matrix.png", "Confusion matrix", "Classification")

    plot_per_class_metrics(
        valid_celltype_labels, predictions, class_names, plots_dir / "per_class_metrics.png"
    )
    add("per_class_metrics.png", "Per-class precision / recall / F1", "Classification")

    plot_class_distribution(
        train_celltype_labels,
        valid_celltype_labels,
        class_names,
        plots_dir / "class_distribution.png",
    )
    add("class_distribution.png", "Class distribution (train vs valid)", "Classification")

    plot_roc_pr_curves(
        valid_celltype_labels,
        probs,
        class_names,
        plots_dir / "roc_curves.png",
        plots_dir / "pr_curves.png",
    )
    add("roc_curves.png", "ROC curves (one-vs-rest)", "Classification")
    add("pr_curves.png", "Precision-recall curves (one-vs-rest)", "Classification")

    true_names = [class_names[i] for i in valid_celltype_labels]
    pred_names = [class_names[i] for i in predictions]

    base_coords = compute_umap(base_cell_emb)
    plot_umap_scatter(
        base_coords,
        true_names,
        plots_dir / "umap_base_by_label.png",
        "Base (pre-finetuning) embeddings — colored by true label",
    )
    add("umap_base_by_label.png", "Base model UMAP, colored by true label", "Embeddings")

    plot_umap_scatter(
        base_coords,
        valid_batch_labels,
        plots_dir / "umap_base_by_batch.png",
        "Base (pre-finetuning) embeddings — colored by batch",
    )
    add("umap_base_by_batch.png", "Base model UMAP, colored by batch", "Embeddings")

    finetuned_coords = compute_umap(finetuned_cell_emb)
    plot_umap_scatter(
        finetuned_coords,
        true_names,
        plots_dir / "umap_finetuned_by_label.png",
        "Finetuned embeddings — colored by true label",
    )
    add(
        "umap_finetuned_by_label.png",
        "Finetuned model UMAP, colored by true label",
        "Embeddings",
    )

    plot_umap_scatter(
        finetuned_coords,
        pred_names,
        plots_dir / "umap_finetuned_by_prediction.png",
        "Finetuned embeddings — colored by predicted label",
    )
    add(
        "umap_finetuned_by_prediction.png",
        "Finetuned model UMAP, colored by predicted label",
        "Embeddings",
    )

    plot_umap_scatter(
        finetuned_coords,
        valid_batch_labels,
        plots_dir / "umap_finetuned_by_batch.png",
        "Finetuned embeddings — colored by batch",
    )
    add("umap_finetuned_by_batch.png", "Finetuned model UMAP, colored by batch", "Embeddings")

    scib_scores = {
        "Base": compute_scib_scores(base_cell_emb, true_names, valid_batch_labels),
        "Finetuned": compute_scib_scores(finetuned_cell_emb, true_names, valid_batch_labels),
    }
    plot_scib_scores(scib_scores, plots_dir / "scib_scores.png")
    add(
        "scib_scores.png",
        "scib integration metrics (base vs. finetuned)",
        "Embeddings",
    )
    plot_pcr_scores(scib_scores, plots_dir / "scib_pcr.png")
    add(
        "scib_pcr.png",
        "scib PCR: batch-explained variance (base vs. finetuned)",
        "Embeddings",
    )
    with open(plots_dir / "scib_metrics.json", "w") as f:
        json.dump(scib_scores, f, indent=2)

    with open(plots_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
