"""
Quantize a trained TinyissimoYOLO checkpoint to INT8 TFLite via Ultralytics
PTQ export, then optionally evaluate the quantized model.

Requires the ``fyp-ml`` conda env (ultralytics + tensorflow).

Usage:
    conda activate fyp-ml
    python src/ml/run_quantize.py --size 192
    python src/ml/run_quantize.py --size 192 --no-eval
    python src/ml/run_quantize.py --size 192 --checkpoint /path/to/best.pt

Ultralytics writes several TFLite variants under ``best_saved_model/``. Export returns
``best_int8.tflite`` (float I/O); evaluation uses ``best_full_integer_quant.tflite`` when
present so metrics match ST Edge AI / NPU deployment (int8 I/O). Both paths are printed
at the end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODELS = ROOT / "results" / "model"


def _resolve_checkpoint(img_size: int, override: Path | None) -> Path:
    """Find the best.pt checkpoint for the given image size."""
    if override is not None:
        p = override.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"No checkpoint at {p}")
        return p

    weights_dir = MODELS / f"tinyissimoyolo_v8_{img_size}" / "weights"
    best = weights_dir / "best.pt"
    if best.is_file():
        return best

    raise FileNotFoundError(
        f"No checkpoint at {best}. Train first: "
        f"python project.py train -- --size {img_size}"
    )


def _quantize(img_size: int, pt_path: Path) -> Path:
    """Export quantized INT8 TFLite via Ultralytics; return path Ultralytics wrote."""
    from ultralytics import YOLO

    from dataset.coco_yolo_data import materialize_coco_data_yaml

    data_yaml = materialize_coco_data_yaml()

    print(f"Loading checkpoint {pt_path} ...")
    model = YOLO(str(pt_path))

    print(f"Exporting INT8 TFLite (imgsz={img_size}, data={data_yaml}) ...")
    exported_path = model.export(
        format="tflite",
        int8=True,
        data=str(data_yaml),
        imgsz=[img_size, img_size],
    )

    exported = Path(exported_path).resolve()
    if not exported.is_file():
        raise FileNotFoundError(f"Export did not produce a file at {exported}")
    return exported


def _eval_tflite_path(exported_float_io: Path) -> Path:
    """Prefer full integer I/O TFLite (same graph as ST Edge AI) next to the export artifact."""
    parent = exported_float_io.parent
    if exported_float_io.name.endswith("_int8.tflite"):
        stem_base = exported_float_io.name[: -len("_int8.tflite")]
        full_int = parent / f"{stem_base}_full_integer_quant.tflite"
        if full_int.is_file():
            return full_int
    return exported_float_io


def _evaluate(tflite_path: Path, img_size: int) -> None:
    """Run Ultralytics evaluation on the quantized TFLite model."""
    from ultralytics import YOLO
    from dataset.coco_yolo_data import materialize_coco_data_yaml

    data_yaml = materialize_coco_data_yaml()
    # Keep val runs beside the TFLite (no separate quantized/ tree).
    val_project = tflite_path.parent

    print(f"Evaluating quantized model with Ultralytics (imgsz={img_size}) ...")
    model = YOLO(str(tflite_path))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=img_size,
        split="val",
        project=str(val_project),
        name="val_int8",
        exist_ok=True,
        verbose=True,
    )
    print("Evaluation complete.")
    if hasattr(metrics, "results_dict"):
        print("Metrics:", metrics.results_dict)


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantize model to INT8 TFLite via Ultralytics PTQ export"
    )
    p.add_argument(
        "--size",
        type=int,
        required=True,
        choices=[192, 256, 288, 320],
        help="Must match training resolution",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pt checkpoint (default: auto-detect best.pt from results/model/)",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after quantization",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pt_path = _resolve_checkpoint(args.size, args.checkpoint)

    exported = _quantize(args.size, pt_path)
    eval_path = _eval_tflite_path(exported)

    if not args.no_eval:
        _evaluate(eval_path, args.size)

    print()
    print("Export return path (often float I/O):")
    print(f"  {exported}")
    if eval_path != exported:
        print("Evaluation model (int8 I/O, ST Edge AI–aligned):")
        print(f"  {eval_path}")
    elif exported.name.endswith("_int8.tflite"):
        print(
            "Warning: *_full_integer_quant.tflite not found next to export; "
            "evaluated float-I/O model instead."
        )
    if not args.no_eval:
        print("Ultralytics val output (metrics, plots):")
        print(f"  {eval_path.parent / 'val_int8'}")
    print("Done.")


if __name__ == "__main__":
    main()
