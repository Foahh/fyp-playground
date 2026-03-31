"""
Train TinyissimoYOLO v8 on COCO Person (single class).

Run from the repository root (outputs under $FYP_RESULTS_DIR/model/ or results/model/):
    python src/ml/run_train_tinyissimo_coco_person.py --size 192
    python src/ml/run_train_tinyissimo_coco_person.py --size 192 --no-resume

Quantization to INT8 TFLite is handled separately by run_quantize.py.
"""

from pathlib import Path

import typer
import yaml

ROOT = Path(__file__).resolve().parents[2]

from src.common.paths import get_results_dir
from src.dataset.dataset_common import materialize_coco_data_yaml
from ultralytics import YOLO

TINY = ROOT / "external" / "TinyissimoYOLO"
MODEL_YAML = str(TINY / "ultralytics/cfg/models/tinyissimo/tinyissimo-v8.yaml")
PROJECT = str(get_results_dir() / "model")


def run_name_for(size: int) -> str:
    return f"tinyissimoyolo_v8_{size}"


app = typer.Typer()


@app.command()
def main(
    size: int = typer.Option(..., help="Input resolution (192, 256, 288, or 320)"),
    no_resume: bool = typer.Option(False, help="Start a fresh run instead of resuming from last checkpoint"),
    device: str | None = typer.Option(None, help="Ultralytics device (e.g. 0, 0,1 for multi-GPU, cpu); default is auto"),
    workers: int | None = typer.Option(None, help="Data loader workers; omit to use Ultralytics default"),
    cache: str | None = typer.Option(None, help="Dataset cache mode (none, disk, ram); omit to use Ultralytics default"),
):
    if size not in [192, 256, 288, 320]:
        typer.echo(f"Error: size must be one of [192, 256, 288, 320]", err=True)
        raise typer.Exit(1)
    if not TINY.is_dir():
        raise FileNotFoundError(f"Expected TinyissimoYOLO at {TINY}")

    run_name = run_name_for(size)
    weights_dir = Path(PROJECT) / run_name / "weights"
    resume = not no_resume

    if resume:
        last_pt = weights_dir / "last.pt"
        if last_pt.exists():
            print(f"Resuming from {last_pt} ...")
            model = YOLO(str(last_pt))
        else:
            print(f"No checkpoint found at {last_pt}; starting a new run ...")
            resume = False
            model = YOLO(MODEL_YAML)
    else:
        print(f"Creating new model from {MODEL_YAML} ...")
        model = YOLO(MODEL_YAML)

    data_yaml = materialize_coco_data_yaml(require_person=True)
    with open(data_yaml, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    print(f"Using dataset YAML: {data_yaml}")
    print(f"Dataset root: {data_cfg.get('path')}")
    print("Training profile: paper (fixed)")
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
        "classes": [0],
        "single_cls": True,
        "imgsz": size,
        "epochs": 1000,
        "optimizer": "SGD",
        "batch": 64,
        "lr0": 0.001,
        "warmup_epochs": 3.0,
        "warmup_bias_lr": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "cos_lr": True,
        "warmup_momentum": 0.8,
        "weight_decay": 0.0005,
        "amp": True,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "translate": 0.1,
        "scale": 0.5,
        "mosaic": 1.0,
        "deterministic": False,
        "project": PROJECT,
        "name": run_name,
        "exist_ok": True,
        "patience": 0,
        "resume": resume,
    }
    if device:
        train_kw["device"] = device
    if workers is not None:
        train_kw["workers"] = workers
    if cache_norm is not None:
        train_kw["cache"] = False if cache_norm == "none" else cache_norm

    model.train(**train_kw)

    print(f"Training done. Weights under {weights_dir}")


if __name__ == "__main__":
    app()
