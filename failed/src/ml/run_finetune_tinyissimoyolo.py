"""
Finetune TinyissimoYOLO v8 on ``datasets/fyp_merged`` (hand + tool) from a checkpoint.

Loads ``best.pt`` then ``last.pt`` from ``results/model/tinyissimoyolo_v8_320`` (unless
``--weights`` is set), then runs a fresh finetune at **320px** with balanced defaults into
``tinyissimoyolo_v8_320``.

Run from the repository root:
    ./project.py finetune-tinyissimoyolo
    ./project.py finetune-tinyissimoyolo -- --weights /path/to/best.pt

By default training **resumes** from ``last.pt`` under the run dir if present (passes
that path as ``resume``; boolean ``resume=True`` does not work when ``YOLO()`` was loaded
from arbitrary weights). Pass ``--no-resume`` to always start a new run from the resolved
checkpoint (e.g. after changing hyperparameters or when you want a clean slate).

Saves ``epoch*.pt`` every 10 epochs and prunes older epoch checkpoints (keeps the last 5).
"""

from __future__ import annotations

import re
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

FINETUNE_EPOCHS = 260
FINETUNE_LR0 = 0.0008
FINETUNE_BATCH = 32
FINETUNE_NBS = 64
FINETUNE_PATIENCE = 70

SOURCE_RUN_NAME = "tinyissimoyolo_v8_320"
FINETUNE_IMGSZ = 320


def _lr_tag(lr: float) -> str:
    return f"lr{int(round(lr * 1_000_000))}"


def run_name_for(
    size: int,
    epochs: int = FINETUNE_EPOCHS,
    lr0: float = FINETUNE_LR0,
    patience: int = FINETUNE_PATIENCE,
) -> str:
    return f"tinyissimoyolo_v8_{size}_e{epochs}_{_lr_tag(lr0)}_p{patience}"


def prune_epoch_checkpoints(trainer, keep: int = 5) -> None:
    wdir = trainer.wdir

    def epoch_key(p: Path) -> int:
        m = re.match(r"epoch(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    epoch_pts = sorted(wdir.glob("epoch*.pt"), key=epoch_key)
    for p in epoch_pts[:-keep] if keep > 0 else epoch_pts:
        p.unlink(missing_ok=True)


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
        help="Do not resume: start a new run from the resolved weights (ignore last.pt in the output dir).",
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
    if not TINY.is_dir():
        raise FileNotFoundError(f"Expected TinyissimoYOLO at {TINY}")

    run_name = run_name_for(size=size, epochs=epochs, lr0=lr0, patience=patience)
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
        "hsv_s": 0.35,
        "hsv_v": 0.15,
        "fliplr": 0.5,
        "translate": 0.03,
        "scale": 0.15,
        "mosaic": 0.4,
        "mixup": 0.0,
        "close_mosaic": 15,
        "box": 8.5,
        "cls": 1.2,
        "flipud": 0.0,
        "single_cls": False,
        "deterministic": False,
        "project": str(PROJECT),
        "name": run_name,
        "exist_ok": True,
        "patience": patience,
        "resume": resume_val,
        "save_period": 10,
    }
    if device:
        train_kw["device"] = device
    if workers is not None:
        train_kw["workers"] = workers
    if cache_norm is not None:
        train_kw["cache"] = False if cache_norm == "none" else cache_norm

    model.add_callback("on_model_save", lambda tr: prune_epoch_checkpoints(tr, keep=5))
    try:
        model.train(**train_kw)
    except KeyboardInterrupt:
        print("Interrupted.")

    print(f"Finetune done. Weights under {weights_dir}")


if __name__ == "__main__":
    app()
