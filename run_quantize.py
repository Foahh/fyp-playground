"""
Quantize a TensorFlow SavedModel to TFLite with configurable input/output dtypes.

This script performs stage-2 of a split pipeline:
  1) (separate script) export: .pt -> SavedModel
  2) TensorFlow quantization: SavedModel -> TFLite

Usage:
    python run_quantize.py --img_size 192 --saved-model-dir results/model/tinyissimoyolo_v8_192/weights/best_saved_model
    python run_quantize.py --img_size 256 --saved-model-dir ... --quant-input uint8 --quant-output int8
    python run_quantize.py --img_size 192 --saved-model-dir ... --calib-dir /path/to/calibration/images
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.coco_yolo_data import materialize_coco_data_yaml

QUANT_DTYPES = {
    "int8": tf.int8,
    "uint8": tf.uint8,
    "float": None,
}
NUM_CALIB_IMAGES = 200


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
    """Yield float32 [0,1]-normalized images for TFLite PTQ calibration."""
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


def _quantize(
    saved_model_dir: Path,
    output_path: Path,
    imgsz: int,
    calib_dir: Path,
    input_type: str,
    output_type: str,
) -> Path:
    """Quantize a SavedModel with explicit input/output dtypes."""
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
    p = argparse.ArgumentParser(description="Quantize SavedModel to TFLite")
    p.add_argument(
        "--img_size",
        type=int,
        required=True,
        choices=[192, 256, 288, 320],
        help="Must match training resolution",
    )
    p.add_argument(
        "--saved-model-dir",
        type=Path,
        required=True,
        help="Path to SavedModel directory (e.g. .../best_saved_model)",
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
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .tflite path (default: <saved_model_dir>/<stem>_<tag>_int8.tflite)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    saved_model_dir = args.saved_model_dir.resolve()
    if not saved_model_dir.is_dir():
        raise FileNotFoundError(f"No SavedModel directory at {saved_model_dir}")

    data_yaml = Path(materialize_coco_data_yaml())
    calib_dir = args.calib_dir or _resolve_calib_dir(data_yaml)
    if calib_dir is None or not calib_dir.is_dir():
        raise FileNotFoundError(
            "No calibration images found. Pass --calib-dir or install COCO under "
            "./datasets/coco (symlink to ~/datasets per README)."
        )

    stem = (
        saved_model_dir.name[: -len("_saved_model")]
        if saved_model_dir.name.endswith("_saved_model")
        else saved_model_dir.name
    )
    tag = f"{args.quant_input[0]}{args.quant_output[0]}"
    out_path = args.out.resolve() if args.out else (saved_model_dir / f"{stem}_{tag}_int8.tflite")

    print(
        f"Quantizing SavedModel: input={args.quant_input}, output={args.quant_output}, "
        f"calib={calib_dir} ..."
    )
    _quantize(
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
