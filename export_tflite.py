"""
Export TinyissimoYOLO checkpoints using the **installed** Ultralytics package (PyPI).

Copies `external/TinyissimoYOLO/results/*` into `results/model/`, then runs TFLite export from there
(per-channel INT8 patch: `scripts/conda/patch_ultralytics_tflite_quant.py` + `conda_setup_export.py`).

After Ultralytics produces the SavedModel, a second pass re-quantizes with configurable
input/output dtypes (default: uint8 input / int8 output) matching STM32 deployment requirements.

Usage:
    python export_tflite.py --img_size 192
    python export_tflite.py --img_size 256 --quant-input uint8 --quant-output int8
    python export_tflite.py --img_size 192 --calib-dir /path/to/calibration/images
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
RESULTS_SRC = ROOT / "external" / "TinyissimoYOLO" / "results"
MODELS = ROOT / "results" / "model"
DATA_YAML = ROOT / "external" / "TinyissimoYOLO" / "ultralytics" / "cfg" / "datasets" / "coco.yaml"

QUANT_DTYPES = {
    "int8": tf.int8,
    "uint8": tf.uint8,
    "float": None,
}
NUM_CALIB_IMAGES = 200


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


def _resolve_calib_dir(data_yaml: Path) -> Path | None:
    """Discover calibration images directory from an Ultralytics data YAML."""
    if not data_yaml.is_file():
        return None
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    ds_root = Path(cfg.get("path", ""))
    if not ds_root.is_absolute():
        ds_root = data_yaml.parent / ds_root
    for subdir in ("images/val2017", "images/train2017", "images/val", "images/train"):
        candidate = ds_root / subdir
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate.resolve()
    return None


def _representative_data_gen(calib_dir: Path, imgsz: int, num_images: int = NUM_CALIB_IMAGES):
    """Yield float32 [0,1]-normalised images for TFLite PTQ calibration."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in calib_dir.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No images found in {calib_dir}")
    if len(files) > num_images:
        random.seed(42)
        files = random.sample(files, num_images)
    for p in files:
        img = Image.open(p).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        yield [np.expand_dims(arr, 0)]


def _requantize(
    saved_model_dir: Path,
    output_path: Path,
    imgsz: int,
    calib_dir: Path,
    input_type: str,
    output_type: str,
) -> Path:
    """Re-quantize a SavedModel with explicit input/output dtypes."""
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if QUANT_DTYPES.get(input_type):
        converter.inference_input_type = QUANT_DTYPES[input_type]
    if QUANT_DTYPES.get(output_type):
        converter.inference_output_type = QUANT_DTYPES[output_type]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_data_gen(calib_dir, imgsz)

    tflite_bytes = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    return output_path


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
        help="Path to best.pt (default: results/model/tinyissimoyolo_v8_<img_size>/weights/best.pt)",
    )
    p.add_argument(
        "--skip-sync",
        action="store_true",
        help="Do not copy external/TinyissimoYOLO/results into results/model/ (use existing export tree only)",
    )
    p.add_argument(
        "--quant-input",
        choices=list(QUANT_DTYPES),
        default="uint8",
        help="Quantized model input dtype (default: uint8)",
    )
    p.add_argument(
        "--quant-output",
        choices=list(QUANT_DTYPES),
        default="int8",
        help="Quantized model output dtype (default: int8)",
    )
    p.add_argument(
        "--calib-dir",
        type=Path,
        default=None,
        help="Directory of calibration images (auto-detected from data YAML if omitted)",
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

    # --- Step 1: Ultralytics export (produces SavedModel + onnx2tf TFLite) ---
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

    saved_model_dir = weights_dir / f"{ckpt_stem}_saved_model"
    onnx2tf_tflite = saved_model_dir / f"{ckpt_stem}_int8.tflite"
    print(f"Ultralytics export done: {onnx2tf_tflite}")

    # --- Step 2: Re-quantize from SavedModel with target input/output dtypes ---
    calib_dir = args.calib_dir or _resolve_calib_dir(data_yaml)
    if calib_dir is None or not calib_dir.is_dir():
        raise FileNotFoundError(
            f"No calibration images found. Supply --calib-dir or ensure COCO images "
            f"exist at the path referenced by {data_yaml}"
        )

    tag = f"{args.quant_input[0]}{args.quant_output[0]}"  # e.g. "ui" for uint8/int8
    out_name = f"{ckpt_stem}_{tag}_int8.tflite"
    out_path = saved_model_dir / out_name

    print(
        f"Re-quantizing: input={args.quant_input}, output={args.quant_output}, "
        f"calib={calib_dir} ..."
    )
    _requantize(
        saved_model_dir=saved_model_dir,
        output_path=out_path,
        imgsz=args.img_size,
        calib_dir=calib_dir,
        input_type=args.quant_input,
        output_type=args.quant_output,
    )
    print(f"Done. TFLite ({args.quant_input} in / {args.quant_output} out): {out_path}")


if __name__ == "__main__":
    main()
