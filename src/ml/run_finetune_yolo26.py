"""
Finetune YOLO26 (Ultralytics) on ``datasets/fyp_merged`` (hand + tool).

Loads ``--model`` (default ``yolo26n.pt`` from the Ultralytics hub, or a local ``.pt``).
Requires an Ultralytics build that defines YOLO26 (newer than some vendored trees).

Multi-scale training is enabled by default (``multi_scale=0.25``) so the model sees
resolutions from 240 to 400 px during training, covering all deployment targets
(256, 288, 320, 384).

Run from the repository root:
    ./project.py finetune-yolo26
    ./project.py finetune-yolo26 -- --model yolo26n.pt
    ./project.py finetune-yolo26 -- --model /path/to/best.pt

By default training **resumes** from ``last.pt`` under the run dir if present (passed
as the ``resume`` path; boolean ``resume=True`` does not work after loading hub ``.pt``
weights — see Ultralytics ``Model.train``). Pass ``--no-resume`` to always start a new
run from the loaded model checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.paths import get_results_dir
from src.dataset.dataset_common import materialize_fyp_merged_data_yaml

PROJECT = get_results_dir() / "model"

FINETUNE_EPOCHS = 220
FINETUNE_LR0 = 0.0008
FINETUNE_BATCH = 24
FINETUNE_NBS = 64
FINETUNE_IMGSZ = 320
FINETUNE_MULTI_SCALE = 0.2
FINETUNE_PATIENCE = 60


def run_name_for(size: int) -> str:
    return f"yolo26_{size}"


app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option(
        "yolo26n.pt", help="Ultralytics hub name or path to a .pt checkpoint"
    ),
    device: str | None = typer.Option(
        None, help="Ultralytics device (e.g. 0, cpu); default is auto"
    ),
    workers: int | None = typer.Option(
        None, help="Data loader workers; omit for Ultralytics default"
    ),
    cache: str | None = typer.Option(
        None, help="Dataset cache mode (none, disk, ram); omit for default"
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help="Do not resume: start a new run from the loaded weights (ignore last.pt in the output dir).",
    ),
    size: int = typer.Option(
        FINETUNE_IMGSZ,
        "--size",
        help="Training image size (deployment target is usually 320).",
    ),
    epochs: int = typer.Option(
        FINETUNE_EPOCHS,
        "--epochs",
        help="Max finetune epochs.",
    ),
    lr0: float = typer.Option(
        FINETUNE_LR0,
        "--lr0",
        help="Initial learning rate.",
    ),
    patience: int = typer.Option(
        FINETUNE_PATIENCE,
        "--patience",
        help="Early-stopping patience.",
    ),
):
    run_name = run_name_for(size)
    local_run_dir = PROJECT / run_name
    weights_dir = local_run_dir / "weights"

    from ultralytics import YOLO

    raw = model.strip()
    cand = Path(raw).expanduser()
    model_path = str(cand.resolve()) if cand.is_file() else raw

    print(f"Loading {model_path!r} ...")
    yolo = YOLO(model_path)

    data_yaml = materialize_fyp_merged_data_yaml()
    with open(data_yaml, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    print(f"Using dataset YAML: {data_yaml}")
    print(f"Dataset root: {data_cfg.get('path')}")

    cache_norm: str | None = None
    if cache is not None:
        cache_norm = cache.strip().lower()
        if cache_norm not in {"none", "disk", "ram"}:
            typer.echo("Error: cache must be one of [none, disk, ram]", err=True)
            raise typer.Exit(1)

    last_pt = weights_dir / "last.pt"

    resume_val: bool | str = False
    if not no_resume and last_pt.is_file():
        resume_val = str(last_pt)

    train_kw: dict = {
        "data": data_yaml,
        "imgsz": size,
        "epochs": epochs,
        "optimizer": "AdamW",
        "batch": FINETUNE_BATCH,
        "nbs": FINETUNE_NBS,
        "lr0": lr0,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_bias_lr": 0.01,
        "warmup_momentum": 0.8,
        "cos_lr": True,
        "amp": True,
        "hsv_h": 0.015,
        "hsv_s": 0.4,
        "hsv_v": 0.2,
        "fliplr": 0.5,
        "translate": 0.05,
        "scale": 0.2,
        "multi_scale": FINETUNE_MULTI_SCALE,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 15,
        "cls": 1.2,
        "box": 8.5,
        "flipud": 0.0,
        "deterministic": False,
        "project": str(PROJECT),
        "name": run_name,
        "exist_ok": True,
        "patience": patience,
        "resume": resume_val,
    }
    if device:
        train_kw["device"] = device
    if workers is not None:
        train_kw["workers"] = workers
    if cache_norm is not None:
        train_kw["cache"] = False if cache_norm == "none" else cache_norm

    try:
        yolo.train(**train_kw)
    except KeyboardInterrupt:
        print("Interrupted.")

    print(f"Finetune done. Weights under {weights_dir}")


if __name__ == "__main__":
    app()
