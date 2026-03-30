"""
Quantize a trained TinyissimoYOLO checkpoint to INT8 TFLite via Ultralytics
PTQ export, then optionally evaluate the quantized model.

Requires the 'yolo' conda env (ultralytics + tensorflow).

Usage:
    conda activate yolo
    python scripts/run_quantize.py --img_size 192
    python scripts/run_quantize.py --img_size 192 --no-eval
    python scripts/run_quantize.py --img_size 192 --checkpoint /path/to/best.pt

The INT8 TFLite is written where Ultralytics places export artifacts (typically next
to the checkpoint); the script prints that path explicitly at the end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

    from scripts.coco_yolo_data import materialize_coco_data_yaml

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


def _evaluate(tflite_path: Path, img_size: int) -> None:
    """Run Ultralytics evaluation on the quantized TFLite model."""
    from ultralytics import YOLO
    from scripts.coco_yolo_data import materialize_coco_data_yaml

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
        "--img_size",
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
    pt_path = _resolve_checkpoint(args.img_size, args.checkpoint)

    tflite_path = _quantize(args.img_size, pt_path)

    if not args.no_eval:
        _evaluate(tflite_path, args.img_size)

    print()
    print("Artifact (quantized INT8 TFLite):")
    print(f"  {tflite_path}")
    if not args.no_eval:
        print("Ultralytics val output (metrics, plots):")
        print(f"  {tflite_path.parent / 'val_int8'}")
    print("Done.")


if __name__ == "__main__":
    main()
