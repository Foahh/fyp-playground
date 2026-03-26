"""
Export TinyissimoYOLO checkpoints to TensorFlow SavedModel only.

This script performs stage-1 of a split pipeline:
  1) PyTorch/Ultralytics export: .pt -> SavedModel
  2) (separate script) TensorFlow quantization: SavedModel -> TFLite

Usage:
    python run_export.py --img_size 192
    python run_export.py --img_size 256 --skip-sync
    python run_export.py --img_size 320 --weights /path/to/best.pt
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.coco_yolo_data import materialize_coco_data_yaml

RESULTS_SRC = ROOT / "external" / "TinyissimoYOLO" / "results"
MODELS = ROOT / "results" / "model"


@contextmanager
def working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def sync_results_to_models() -> None:
    """Copy external/TinyissimoYOLO/results/* into results/model/ (merge, overwrite files)."""
    if not RESULTS_SRC.is_dir():
        raise FileNotFoundError(f"No training results at {RESULTS_SRC}")
    MODELS.mkdir(parents=True, exist_ok=True)
    for item in RESULTS_SRC.iterdir():
        dest = MODELS / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def parse_args():
    p = argparse.ArgumentParser(description="Export TinyissimoYOLO checkpoint to SavedModel")
    p.add_argument(
        "--img_size",
        type=int,
        required=True,
        choices=[192, 256, 288, 320],
        help="Must match training resolution",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to best.pt (default: results/model/tinyissimoyolo_v8_<img_size>/weights/best.pt)",
    )
    p.add_argument(
        "--skip-sync",
        action="store_true",
        help="Do not copy external/TinyissimoYOLO/results into results/model/ (use existing export tree only)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_name = f"tinyissimoyolo_v8_{args.img_size}"
    weights_dir = MODELS / run_name / "weights"
    data_yaml = Path(materialize_coco_data_yaml())

    if not args.skip_sync:
        print(f"Syncing {RESULTS_SRC} -> {MODELS} ...")
        sync_results_to_models()

    ckpt = args.weights or (weights_dir / "best.pt")
    ckpt = ckpt.resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    from ultralytics import YOLO

    print(f"Loading {ckpt} ...")
    model = YOLO(str(ckpt))
    ckpt_stem = Path(model.ckpt_path).stem if model.ckpt_path else "best"

    print(f"Exporting SavedModel in {MODELS} (imgsz={args.img_size}, data={data_yaml}) ...")
    with working_directory(MODELS):
        model.export(
            format="saved_model",
            int8=False,
            data=str(data_yaml),
            imgsz=[args.img_size, args.img_size],
        )

    saved_model_dir = weights_dir / f"{ckpt_stem}_saved_model"
    if not saved_model_dir.is_dir():
        raise FileNotFoundError(f"SavedModel export not found at {saved_model_dir}")
    print(f"Done. SavedModel: {saved_model_dir}")


if __name__ == "__main__":
    main()
