"""
Train TinyissimoYOLO v8 on COCO Person (single class).

Run from the repository root (outputs under $FYP_RESULTS_DIR/model/ or results/model/):
    python src/ml/run_train_tinyissimo_coco_person.py --size 192
    python src/ml/run_train_tinyissimo_coco_person.py --size 192 --no-resume

Quantization to INT8 TFLite is handled separately by run_quantize.py.
"""

import atexit
import signal
import sys
from pathlib import Path

import typer
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.paths import get_results_dir
from src.dataset.dataset_common import materialize_coco_data_yaml
from src.ml.backup_rclone import BackupManager, build_dest

TINY = ROOT / "external" / "TinyissimoYOLO"
MODEL_YAML = str(TINY / "ultralytics/cfg/models/tinyissimo/tinyissimo-v8.yaml")
PROJECT = get_results_dir() / "model"


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
    backup_dest: str | None = typer.Option(None, "--backup-dest", help="rclone destination base (enables backups when set)"),
    backup_every_n_epochs: int = typer.Option(1, "--backup-every-n-epochs", help="Backup cadence in epochs"),
    backup_timeout_s: int = typer.Option(600, "--backup-timeout-s", help="Per-sync timeout in seconds"),
):
    if size not in [192, 256, 288, 320]:
        typer.echo(f"Error: size must be one of [192, 256, 288, 320]", err=True)
        raise typer.Exit(1)
    if not TINY.is_dir():
        raise FileNotFoundError(f"Expected TinyissimoYOLO at {TINY}")

    run_name = run_name_for(size)
    local_run_dir = PROJECT / run_name
    weights_dir = local_run_dir / "weights"
    resume = not no_resume

    if backup_every_n_epochs <= 0:
        typer.echo("Error: --backup-every-n-epochs must be > 0", err=True)
        raise typer.Exit(1)
    if backup_timeout_s <= 0:
        typer.echo("Error: --backup-timeout-s must be > 0", err=True)
        raise typer.Exit(1)

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

    backup_mgr: BackupManager | None = None
    termination_requested = False

    if backup_dest:
        dest = build_dest(backup_dest, run_name)
        backup_mgr = BackupManager(local_run_dir=local_run_dir, dest=dest, timeout_s=backup_timeout_s)

        def _on_exit() -> None:
            backup_mgr.request_sync("atexit")
            backup_mgr.maybe_run_sync()

        atexit.register(_on_exit)

        def _handle_signal(signum, _frame) -> None:  # type: ignore[no-untyped-def]
            nonlocal termination_requested
            termination_requested = True
            backup_mgr.request_sync(f"signal_{signum}")

        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)

        warned_save_dir = False

        def on_train_epoch_end(trainer):  # type: ignore[no-untyped-def]
            nonlocal warned_save_dir
            if not warned_save_dir:
                save_dir = getattr(trainer, "save_dir", None)
                if save_dir is not None:
                    try:
                        save_dir_p = Path(str(save_dir)).resolve()
                        if save_dir_p != local_run_dir.resolve():
                            print(f"[backup] warning: trainer.save_dir={save_dir_p} != expected={local_run_dir.resolve()}")
                    except Exception:
                        print(f"[backup] warning: could not parse trainer.save_dir={save_dir!r}")
                warned_save_dir = True

            epoch_idx = int(getattr(trainer, "epoch", 0))
            if termination_requested or ((epoch_idx + 1) % backup_every_n_epochs == 0):
                backup_mgr.request_sync("epoch_end")
                backup_mgr.maybe_run_sync()

        def on_train_end(_trainer):  # type: ignore[no-untyped-def]
            backup_mgr.request_sync("train_end")
            backup_mgr.maybe_run_sync()

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
        "project": str(PROJECT),
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

    if backup_mgr:
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_train_end", on_train_end)

    try:
        model.train(**train_kw)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if backup_mgr:
            backup_mgr.request_sync("finalize")
            backup_mgr.maybe_run_sync()

    print(f"Training done. Weights under {weights_dir}")


if __name__ == "__main__":
    app()
