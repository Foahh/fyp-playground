"""
Finetune TinyissimoYOLO v8 on ``datasets/fyp_merged`` (hand + tool) from a checkpoint.

Loads ``best.pt`` then ``last.pt`` from the 320px run under ``results/model/tinyissimoyolo_v8_320``
(unless ``--weights`` is set), then trains at **288px** into ``tinyissimoyolo_v8_288`` with
``resume=False``.

Run from the repository root:
    python src/ml/run_finetune_tinyissimoyolo.py
    python src/ml/run_finetune_tinyissimoyolo.py --weights /path/to/best.pt
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

TINY = ROOT / "external" / "TinyissimoYOLO"
PROJECT = get_results_dir() / "model"

FINETUNE_EPOCHS = 300
FINETUNE_LR0 = 0.0005
FINETUNE_BATCH = 64
FINETUNE_NBS = 64

SOURCE_RUN_NAME = "tinyissimoyolo_v8_320"
FINETUNE_IMGSZ = 320


def run_name_for(size: int) -> str:
    return f"tinyissimoyolo_v8_{size}"


def _resolve_weights(source_run_name: str, weights: Path | None) -> Path:
    if weights is not None:
        p = weights.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"No checkpoint at {p}")
        return p
    base = PROJECT / source_run_name / "weights"
    for name in ("best.pt", "last.pt"):
        cand = base / name
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"No weights found under {base} (expected best.pt or last.pt). "
        "Train first or pass --weights."
    )


app = typer.Typer()


@app.command()
def main(
    weights: Path | None = typer.Option(
        None,
        help=f"Checkpoint .pt; default: best.pt then last.pt under {SOURCE_RUN_NAME}/weights",
    ),
    device: str | None = typer.Option(None, help="Ultralytics device (e.g. 0, cpu); default is auto"),
    workers: int | None = typer.Option(None, help="Data loader workers; omit for Ultralytics default"),
    cache: str | None = typer.Option(None, help="Dataset cache mode (none, disk, ram); omit for default"),
):
    if not TINY.is_dir():
        raise FileNotFoundError(f"Expected TinyissimoYOLO at {TINY}")

    run_name = run_name_for(FINETUNE_IMGSZ)
    local_run_dir = PROJECT / run_name
    weights_dir = local_run_dir / "weights"

    ckpt = _resolve_weights(SOURCE_RUN_NAME, weights)
    print(f"Finetuning from {ckpt} ...")

    from ultralytics import YOLO

    model = YOLO(str(ckpt))

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

    print(
        "Runtime profile: "
        f"workers={workers if workers is not None else 'auto'}, "
        f"cache={cache_norm if cache_norm is not None else 'auto'}, "
        f"device={device or 'auto'}"
    )

    train_kw: dict = {
        "data": data_yaml,
        "imgsz": FINETUNE_IMGSZ,
        "epochs": FINETUNE_EPOCHS,
        "optimizer": "SGD",
        "batch": FINETUNE_BATCH,
        "nbs": FINETUNE_NBS,
        "lr0": FINETUNE_LR0,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_bias_lr": 0.01,
        "warmup_momentum": 0.8,
        "cos_lr": True,
        "amp": True,
        "label_smoothing": 0.05,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "fliplr": 0.5,
        "translate": 0.1,
        "scale": 0.4,
        "mosaic": 1.0,
        "mixup": 0.15,
        "close_mosaic": 15,
        "deterministic": False,
        "project": str(PROJECT),
        "name": run_name,
        "exist_ok": True,
        "patience": 50,
        "resume": True,
    }
    if device:
        train_kw["device"] = device
    if workers is not None:
        train_kw["workers"] = workers
    if cache_norm is not None:
        train_kw["cache"] = False if cache_norm == "none" else cache_norm

    try:
        model.train(**train_kw)
    except KeyboardInterrupt:
        print("Interrupted.")

    print(f"Finetune done. Weights under {weights_dir}")


if __name__ == "__main__":
    app()
