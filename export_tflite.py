"""
Export TinyissimoYOLO checkpoints using the **installed** Ultralytics package (PyPI).

Copies `external/TinyissimoYOLO/results/*` into `results/model/`, then runs TFLite export from there
(per-channel INT8 patch: `scripts/conda/patch_ultralytics_tflite_quant.py` + `conda_setup_export.py`).

Usage:
    python export_tflite.py --img_size 192
"""

from __future__ import annotations

import argparse
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RESULTS_SRC = ROOT / "external" / "TinyissimoYOLO" / "results"
MODELS = ROOT / "results" / "model"
DATA_YAML = ROOT / "external" / "TinyissimoYOLO" / "ultralytics" / "cfg" / "datasets" / "coco.yaml"


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
    p = argparse.ArgumentParser(description="Export TinyissimoYOLO checkpoint to TFLite INT8")
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
        help=f"Path to best.pt (default: results/model/tinyissimoyolo_v8_<img_size>/weights/best.pt)",
    )
    p.add_argument(
        "--skip-sync",
        action="store_true",
        help="Do not copy external/TinyissimoYOLO/results into results/model/ (use existing export tree only)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.skip_sync:
        print(f"Syncing {RESULTS_SRC} -> {MODELS} ...")
        sync_results_to_models()

    run_name = f"tinyissimoyolo_v8_{args.img_size}"
    weights_dir = MODELS / run_name / "weights"
    ckpt = args.weights or (weights_dir / "best.pt")
    ckpt = ckpt.resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    data_yaml = DATA_YAML.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    print(f"Loading {ckpt} ...")
    model = YOLO(str(ckpt))
    ckpt_stem = Path(model.ckpt_path).stem if model.ckpt_path else "best"

    print(
        f"Exporting TFLite INT8 in {MODELS} (PTQ, imgsz={args.img_size}, data={data_yaml}) ..."
    )
    with working_directory(MODELS):
        model.export(
            format="tflite",
            int8=True,
            data=str(data_yaml),
            imgsz=[args.img_size, args.img_size],
            simplify=True,
        )

    tflite_path = weights_dir / f"{ckpt_stem}_saved_model" / f"{ckpt_stem}_int8.tflite"
    print(f"Done. TFLite INT8: {tflite_path}")


if __name__ == "__main__":
    main()
