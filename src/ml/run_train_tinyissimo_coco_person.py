"""
Train TinyissimoYOLO v8 on COCO Person (single class).

Run from the repository root (outputs under $RESULTS_DIR/model/ or results/model/):
    python src/ml/run_train_tinyissimo_coco_person.py --size 192
    python src/ml/run_train_tinyissimo_coco_person.py --size 192 --profile paper
    python src/ml/run_train_tinyissimo_coco_person.py --size 192 --profile powerful
    python src/ml/run_train_tinyissimo_coco_person.py --size 192 --no-resume

Quantization to INT8 TFLite is handled separately by run_quantize.py.
"""

import sys
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


def run_name_for(size: int, profile: str) -> str:
    base = f"tinyissimoyolo_v8_{size}"
    if profile == "paper":
        return base
    return f"{base}_{profile}"


def train_profile_kwargs(profile: str) -> dict:
    """Hyperparameters and infra for model.train(); profile-specific only."""
    if profile == "paper":
        return {
            "batch": 64,
            "lr0": 0.001,
            "warmup_epochs": 3.0,
            "warmup_bias_lr": 0.01,
        }
    if profile == "powerful":
        return {
            "batch": 256,
            "lr0": 0.004,
            "warmup_epochs": 5.0,
            "warmup_bias_lr": 0.04,
            "workers": 8,
            "cache": "ram",
        }
    raise ValueError(f"Unknown profile: {profile!r}")


app = typer.Typer()


@app.command()
def main(
    size: int = typer.Option(..., help="Input resolution (192, 256, 288, or 320)"),
    no_resume: bool = typer.Option(False, help="Start a fresh run instead of resuming from last checkpoint"),
    optimizer: str = typer.Option("SGD"),
    profile: str = typer.Option("paper", help="Training preset: paper (batch 64) or powerful (batch 256, workers=8, cache=ram)"),
    device: str | None = typer.Option(None, help="Ultralytics device (e.g. 0, 0,1 for multi-GPU, cpu); default is auto"),
):
    if size not in [192, 256, 288, 320]:
        typer.echo(f"Error: size must be one of [192, 256, 288, 320]", err=True)
        raise typer.Exit(1)
    if profile not in ("paper", "powerful"):
        typer.echo(f"Error: profile must be 'paper' or 'powerful'", err=True)
        raise typer.Exit(1)
    if not TINY.is_dir():
        raise FileNotFoundError(f"Expected TinyissimoYOLO at {TINY}")

    run_name = run_name_for(size, profile)
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
    print(f"Training profile: {profile}")

    train_kw: dict = {
        "data": data_yaml,
        "classes": [0],
        "single_cls": True,
        "imgsz": size,
        "epochs": 1000,
        "optimizer": "SGD",
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
        **train_profile_kwargs(profile),
    }
    if device:
        train_kw["device"] = device

    model.train(**train_kw)

    print(f"Training done. Weights under {weights_dir}")


if __name__ == "__main__":
    app()
