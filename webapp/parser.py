"""Parses scGPT fine-tuning run directories (as written by scripts.analyses.run_finetune)
into structured summaries the web app can render, without importing torch/scanpy/etc.

A run directory (e.g. save/finetuned/dev_AML-condition-Aug20-12-45/) contains:
  - run.log          plain-text log from the scGPT logger (always present if a run started)
  - best_model.pt    checkpoint of the best epoch (only present once training reaches it)
  - vocab.json        gene vocabulary used
  - id2type.json       {label_id: label_name} mapping for the classification head
"""

from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

RUN_DIR_RE = re.compile(
    r"^dev_(?P<dataset>[^-]+)-(?P<label>.+)-(?P<ts>[A-Za-z]{3}\d{2}-\d{2}-\d{2})$"
)
GENES_RE = re.compile(r"match (\d+)/(\d+) genes in vocabulary of size (\d+)")
DEVICE_RE = re.compile(r"Device available: (\S+)")
SAMPLES_RE = re.compile(
    r"(train|valid) set number of samples: (\d+),\s*\n\s*feature length: (\d+)"
)
EPOCH_RE = re.compile(
    r"end of epoch\s*(\d+)\s*\|\s*time:\s*([\d.]+)s\s*\|\s*valid loss\s*([\d.]+)\s*\|\s*err\s*([\d.]+)"
)
BEST_RE = re.compile(r"Best model with score ([\d.]+)")
FINAL_RE = re.compile(
    r"Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), Macro F1: ([\d.]+)"
)
TRAIN_BATCH_RE = re.compile(
    r"\| epoch\s*(\d+) \| \s*(\d+)/\s*(\d+) batches \| lr [\d.]+ \| ms/batch [\d.]+ \|"
    r" loss\s*([\d.]+) \| err\s*([\d.]+) \|"
)

# A run whose log hasn't been touched in this long, but has no final metrics, is
# considered stalled/failed rather than still running.
RUNNING_STALE_AFTER = timedelta(minutes=15)


@dataclass
class EpochMetric:
    epoch: int
    time_s: float
    valid_loss: float
    valid_err: float
    is_best: bool = False
    best_score: Optional[float] = None
    train_loss: Optional[float] = None
    train_err: Optional[float] = None


@dataclass
class RunSummary:
    run_id: str
    path: Path
    dataset_name: Optional[str]
    label_column: Optional[str]
    timestamp_str: Optional[str]
    device: Optional[str]
    genes_matched: Optional[int]
    genes_total: Optional[int]
    vocab_size: Optional[int]
    train_samples: Optional[int]
    valid_samples: Optional[int]
    feature_length: Optional[int]
    epochs: List[EpochMetric]
    best_epoch: Optional[int]
    best_valid_loss: Optional[float]
    final_accuracy: Optional[float]
    final_precision: Optional[float]
    final_recall: Optional[float]
    final_macro_f1: Optional[float]
    label_classes: Optional[List[str]]
    status: str
    last_modified: datetime
    log_size_bytes: int
    has_model: bool
    config: Optional[Dict[str, object]]
    base_model: Optional[str]
    balance_classes: Optional[bool]
    plots: List[Dict[str, str]]


def _load_config(run_dir: Path) -> Optional[Dict[str, object]]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None


def _load_plots(run_dir: Path) -> List[Dict[str, str]]:
    manifest_path = run_dir / "plots" / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError):
        return []
    # Only keep entries whose image file is actually present, in case a run
    # crashed partway through plot generation.
    return [
        p
        for p in manifest
        if isinstance(p, dict)
        and p.get("filename")
        and (run_dir / "plots" / p["filename"]).exists()
    ]


def _parse_run_dir_name(name: str):
    m = RUN_DIR_RE.match(name)
    if not m:
        return None, None, None
    return m.group("dataset"), m.group("label"), m.group("ts")


def _parse_epochs(text: str) -> List[EpochMetric]:
    epoch_matches = list(EPOCH_RE.finditer(text))
    epochs = [
        EpochMetric(
            epoch=int(m.group(1)),
            time_s=float(m.group(2)),
            valid_loss=float(m.group(3)),
            valid_err=float(m.group(4)),
        )
        for m in epoch_matches
    ]
    positions = [m.start() for m in epoch_matches]

    for bm in BEST_RE.finditer(text):
        idx = bisect.bisect_right(positions, bm.start()) - 1
        if idx >= 0:
            epochs[idx].is_best = True
            epochs[idx].best_score = float(bm.group(1))

    train_loss_sum: Dict[int, float] = {}
    train_err_sum: Dict[int, float] = {}
    train_count: Dict[int, int] = {}
    for tm in TRAIN_BATCH_RE.finditer(text):
        ep = int(tm.group(1))
        train_loss_sum[ep] = train_loss_sum.get(ep, 0.0) + float(tm.group(4))
        train_err_sum[ep] = train_err_sum.get(ep, 0.0) + float(tm.group(5))
        train_count[ep] = train_count.get(ep, 0) + 1
    for e in epochs:
        c = train_count.get(e.epoch)
        if c:
            e.train_loss = train_loss_sum[e.epoch] / c
            e.train_err = train_err_sum[e.epoch] / c

    return epochs


def parse_run(run_dir: Path) -> RunSummary:
    log_path = run_dir / "run.log"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    stat = log_path.stat()

    dataset_name, label_column, timestamp_str = _parse_run_dir_name(run_dir.name)

    genes_matched = genes_total = vocab_size = None
    gm = GENES_RE.search(text)
    if gm:
        genes_matched, genes_total, vocab_size = (int(x) for x in gm.groups())

    device = None
    dm = DEVICE_RE.search(text)
    if dm:
        device = dm.group(1)

    train_samples = train_feature_len = None
    valid_samples = valid_feature_len = None
    for sm in SAMPLES_RE.finditer(text):
        kind, n, flen = sm.group(1), int(sm.group(2)), int(sm.group(3))
        if kind == "train":
            train_samples, train_feature_len = n, flen
        else:
            valid_samples, valid_feature_len = n, flen

    epochs = _parse_epochs(text)
    best_epoch = None
    best_valid_loss = None
    for e in epochs:
        if e.is_best:
            best_epoch = e.epoch
            best_valid_loss = e.best_score

    final_accuracy = final_precision = final_recall = final_macro_f1 = None
    fm = FINAL_RE.search(text)
    if fm:
        final_accuracy, final_precision, final_recall, final_macro_f1 = (
            float(x) for x in fm.groups()
        )

    label_classes = None
    id2type_path = run_dir / "id2type.json"
    if id2type_path.exists():
        try:
            data = json.loads(id2type_path.read_text(encoding="utf-8"))
            label_classes = [data[k] for k in sorted(data, key=int)]
        except (ValueError, KeyError, json.JSONDecodeError):
            label_classes = None

    has_model = (run_dir / "best_model.pt").exists()

    config = _load_config(run_dir)
    base_model = None
    balance_classes = None
    if config is not None:
        load_model_path = config.get("load_model")
        if load_model_path:
            base_model = Path(str(load_model_path)).name
        balance_classes = config.get("balance_classes")

    plots = _load_plots(run_dir)

    if final_accuracy is not None:
        status = "completed"
    elif epochs:
        age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
        status = "running" if age < RUNNING_STALE_AFTER else "incomplete"
    else:
        age = datetime.now() - datetime.fromtimestamp(stat.st_mtime)
        status = "starting" if age < RUNNING_STALE_AFTER else "failed"

    return RunSummary(
        run_id=run_dir.name,
        path=run_dir,
        dataset_name=dataset_name,
        label_column=label_column,
        timestamp_str=timestamp_str,
        device=device,
        genes_matched=genes_matched,
        genes_total=genes_total,
        vocab_size=vocab_size,
        train_samples=train_samples,
        valid_samples=valid_samples,
        feature_length=train_feature_len or valid_feature_len,
        epochs=epochs,
        best_epoch=best_epoch,
        best_valid_loss=best_valid_loss,
        final_accuracy=final_accuracy,
        final_precision=final_precision,
        final_recall=final_recall,
        final_macro_f1=final_macro_f1,
        label_classes=label_classes,
        status=status,
        last_modified=datetime.fromtimestamp(stat.st_mtime),
        log_size_bytes=stat.st_size,
        has_model=has_model,
        config=config,
        base_model=base_model,
        balance_classes=balance_classes,
        plots=plots,
    )


def discover_runs(runs_dir: Path) -> List[RunSummary]:
    if not runs_dir.exists():
        return []
    runs = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or not (d / "run.log").exists():
            continue
        runs.append(parse_run(d))
    runs.sort(key=lambda r: r.last_modified, reverse=True)
    return runs
