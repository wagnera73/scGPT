"""FastAPI app that summarizes and compares scGPT fine-tuning runs.

Run from the repo root with:
    python -m uvicorn webapp.app:app --reload --port 8000

By default it scans save/finetuned/. Point it elsewhere with:
    SCGPT_RUNS_DIR=/path/to/runs python -m uvicorn webapp.app:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.charts import PALETTE, series_to_points
from webapp.parser import RunSummary, discover_runs

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = REPO_ROOT / "save" / "finetuned"
RUNS_DIR = Path(os.environ.get("SCGPT_RUNS_DIR", str(DEFAULT_RUNS_DIR)))

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(title="scGPT Fine-tuning Run Explorer")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


SORT_KEYS = {
    "name": lambda r: r.run_id,
    "dataset": lambda r: r.dataset_name or "",
    "label": lambda r: r.label_column or "",
    "status": lambda r: r.status,
    "base_model": lambda r: r.base_model or "",
    "accuracy": lambda r: r.final_accuracy if r.final_accuracy is not None else -1,
    "macro_f1": lambda r: r.final_macro_f1 if r.final_macro_f1 is not None else -1,
    "best_loss": lambda r: (
        r.best_valid_loss if r.best_valid_loss is not None else float("inf")
    ),
    "modified": lambda r: r.last_modified,
}


def get_runs() -> List[RunSummary]:
    return discover_runs(RUNS_DIR)


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    sort: str = "modified",
    order: str = "desc",
    dataset: Optional[str] = None,
    label: Optional[str] = None,
    status: Optional[str] = None,
    base_model: Optional[str] = None,
):
    all_runs = get_runs()
    datasets = sorted({r.dataset_name for r in all_runs if r.dataset_name})
    labels = sorted({r.label_column for r in all_runs if r.label_column})
    statuses = sorted({r.status for r in all_runs})
    base_models = sorted({r.base_model for r in all_runs if r.base_model})

    runs = all_runs
    if dataset:
        runs = [r for r in runs if r.dataset_name == dataset]
    if label:
        runs = [r for r in runs if r.label_column == label]
    if status:
        runs = [r for r in runs if r.status == status]
    if base_model:
        runs = [r for r in runs if r.base_model == base_model]

    key_fn = SORT_KEYS.get(sort, SORT_KEYS["modified"])
    runs = sorted(runs, key=key_fn, reverse=(order != "asc"))

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "runs": runs,
            "runs_dir": str(RUNS_DIR),
            "sort": sort,
            "order": order,
            "dataset": dataset,
            "label": label,
            "status": status,
            "base_model": base_model,
            "datasets": datasets,
            "labels": labels,
            "statuses": statuses,
            "base_models": base_models,
        },
    )


@app.get("/run/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str):
    runs_by_id = {r.run_id: r for r in get_runs()}
    run = runs_by_id.get(run_id)
    if run is None:
        return templates.TemplateResponse(
            request,
            "missing.html",
            {"run_id": run_id},
            status_code=404,
        )

    valid_loss = [e.valid_loss for e in run.epochs]
    valid_err = [e.valid_err for e in run.epochs]
    train_loss = [e.train_loss for e in run.epochs if e.train_loss is not None]
    train_err = [e.train_err for e in run.epochs if e.train_err is not None]

    loss_values = valid_loss + train_loss
    err_values = valid_err + train_err
    loss_range = (min(loss_values), max(loss_values)) if loss_values else None
    err_range = (min(err_values), max(err_values)) if err_values else None

    loss_chart = {
        "valid": series_to_points(valid_loss, y_range=loss_range),
        "train": series_to_points(train_loss, y_range=loss_range) if train_loss else None,
    }
    err_chart = {
        "valid": series_to_points(valid_err, y_range=err_range),
        "train": series_to_points(train_err, y_range=err_range) if train_err else None,
    }

    log_text = (run.path / "run.log").read_text(encoding="utf-8", errors="replace")
    log_tail = "\n".join(log_text.splitlines()[-200:])

    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "run": run,
            "loss_chart": loss_chart,
            "err_chart": err_chart,
            "loss_range": loss_range,
            "err_range": err_range,
            "log_tail": log_tail,
        },
    )


@app.get("/compare", response_class=HTMLResponse)
def compare(request: Request, runs: List[str] = Query(default=[])):
    all_runs_by_id = {r.run_id: r for r in get_runs()}
    selected = [all_runs_by_id[r] for r in runs if r in all_runs_by_id]

    all_valid_loss = [e.valid_loss for r in selected for e in r.epochs]
    y_range = (min(all_valid_loss), max(all_valid_loss)) if all_valid_loss else None

    series = []
    for i, r in enumerate(selected):
        values = [e.valid_loss for e in r.epochs]
        points = series_to_points(values, y_range=y_range)
        if points:
            series.append({"run": r, "points": points, "color": PALETTE[i % len(PALETTE)]})

    return templates.TemplateResponse(
        request,
        "compare.html",
        {
            "selected": selected,
            "series": series,
            "available_runs": sorted(
                all_runs_by_id.values(), key=lambda r: r.last_modified, reverse=True
            ),
            "selected_ids": set(runs),
        },
    )
