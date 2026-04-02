"""
Train TinyissimoYOLO v8 on full COCO-80 at 320px (pretraining for lower-res finetuning).

Run from repository root:
    python src/ml/run_train_tinyissimo_coco80_320.py
    python src/ml/run_train_tinyissimo_coco80_320.py --no-resume
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
from src.dataset.dataset_common import materialize_coco_80_data_yaml
TINY = ROOT / "external" / "TinyissimoYOLO"
MODEL_YAML = str(TINY / "ultralytics/cfg/models/tinyissimo/tinyissimo-v8.yaml")
PROJECT = get_results_dir() / "model"
DEFAULT_IMGSZ = 320
DEFAULT_EPOCHS = 300
DEFAULT_LR0 = 0.001
DEFAULT_BATCH = 64
DEFAULT_NBS = 64
def run_name_for(size: int = DEFAULT_IMGSZ) -> str:
    return f"tinyissimoyolo_v8_{size}_coco80"
def prune_epoch_checkpoints(trainer, keep: int = 3) -> None:
    wdir = trainer.wdir
    def epoch_key(path: Path) -> int:
        match = re.match(r"epoch(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1
    epoch_pts = sorted(wdir.glob("epoch*.pt"), key=epoch_key)
    for path in epoch_pts[:-keep] if keep > 0 else epoch_pts:
        path.unlink(missing_ok=True)
def build_train_kwargs(
    *,
    data_yaml: str,
    run_name: str,
    project: str,
    resume: bool,
    device: str | None,
    workers: int | None,
    cache_norm: str | None,
) -> dict:
    train_kw: dict = {
        "data": data_yaml,
        "imgsz": DEFAULT_IMGSZ,
        "epochs": DEFAULT_EPOCHS,
        "optimizer": "SGD",
        "batch": DEFAULT_BATCH,
        "nbs": DEFAULT_NBS,
        "lr0": DEFAULT_LR0,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_bias_lr": 0.01,
        "warmup_momentum": 0.8,
        "cos_lr": True,
        "amp": True,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "fliplr": 0.5,
        "translate": 0.1,
        "scale": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "close_mosaic": 30,
        "deterministic": False,
        "project": project,
        "name": run_name,
        "exist_ok": True,
        "patience": 30,
        "resume": resume,
        "save_period": 10,
    }
    if device:
        train_kw["device"] = device
    if workers is not None:
        train_kw["workers"] = workers
    if cache_norm is not None:
        train_kw["cache"] = False if cache_norm == "none" else cache_norm
    return train_kw
app = typer.Typer()
@app.command()
def main(
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help="Start a fresh run instead of resuming from last checkpoint.",
    ),
    device: str | None = typer.Option(
        None,
        help="Ultralytics device (e.g. 0, 0,1, cpu); default is auto.",
    ),
    workers: int | None = typer.Option(
        None, help="Data loader workers; omit to use Ultralytics default."
    ),
    cache: str | None = typer.Option(
        None, help="Dataset cache mode (none, disk, ram); omit for default."
    ),
):
    if not TINY.is_dir():
        raise FileNotFoundError(f"Expected TinyissimoYOLO at {TINY}")
    run_name = run_name_for()
    local_run_dir = PROJECT / run_name
    weights_dir = local_run_dir / "weights"
    resume = not no_resume
    if resume:
        last_pt = weights_dir / "last.pt"
        if last_pt.exists():
            print(f"Resuming from {last_pt} ...")
            from ultralytics import YOLO
            model = YOLO(str(last_pt))
        else:
            print(f"No checkpoint found at {last_pt}; starting a new run ...")
            resume = False
            from ultralytics import YOLO
            model = YOLO(MODEL_YAML)
    else:
        print(f"Creating new model from {MODEL_YAML} ...")
        from ultralytics import YOLO
        model = YOLO(MODEL_YAML)
    data_yaml = materialize_coco_80_data_yaml()
    with open(data_yaml, encoding="utf-8") as file:
        data_cfg = yaml.safe_load(file)
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
    print(
        "Training profile: "
        f"imgsz={DEFAULT_IMGSZ}, batch={DEFAULT_BATCH}, epochs={DEFAULT_EPOCHS}, "
        f"lr0={DEFAULT_LR0}, nbs={DEFAULT_NBS}"
    )
    train_kw = build_train_kwargs(
        data_yaml=data_yaml,
        run_name=run_name,
        project=str(PROJECT),
        resume=resume,
        device=device,
        workers=workers,
        cache_norm=cache_norm,
    )
    model.add_callback("on_model_save", lambda trainer: prune_epoch_checkpoints(trainer, keep=3))
    try:
        model.train(**train_kw)
    except KeyboardInterrupt:
        print("Interrupted.")
    print(f"Training done. Weights under {weights_dir}")
if __name__ == "__main__":
    app()
