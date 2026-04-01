"""
Quantize a trained TinyissimoYOLO checkpoint to INT8 TFLite via Ultralytics
PTQ export only (evaluation is handled separately via STM32 Model Zoo services).

Requires the ``fyp-qtlz`` conda env (ultralytics + tensorflow).

Usage:
    conda activate fyp-qtlz
    python src/ml/run_quantize.py --size 192
    python src/ml/run_quantize.py --size 192 --checkpoint /path/to/best.pt

Ultralytics writes several TFLite variants under ``best_saved_model/``. Export returns
``best_int8.tflite`` (float I/O). For deployment / STM32 Model Zoo evaluation you typically
want ``*_full_integer_quant.tflite`` (int8 I/O); if it exists, this script prints it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import typer

from src.common.paths import get_results_dir

MODELS = get_results_dir() / "model"


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

    from src.dataset.dataset_common import materialize_coco_data_yaml

    data_yaml = materialize_coco_data_yaml(require_person=True)

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


def _newest(paths: Iterable[Path]) -> Path | None:
    ps = [p for p in paths if p.is_file()]
    if not ps:
        return None
    return max(ps, key=lambda p: p.stat().st_mtime)


def _find_existing_tflite(model_root: Path) -> Path | None:
    """
    Find an existing quantized TFLite export under the model's results directory.

    Preference order:
    1) *_full_integer_quant.tflite (int8 I/O; ST Edge AI–aligned)
    2) *_int8.tflite
    3) any *.tflite (fallback)

    If multiple candidates exist, choose the newest by mtime.
    """
    if not model_root.is_dir():
        return None

    full_int = _newest(model_root.rglob("*_full_integer_quant.tflite"))
    if full_int is not None:
        return full_int

    int8 = _newest(model_root.rglob("*_int8.tflite"))
    if int8 is not None:
        return int8

    any_tflite = _newest(model_root.rglob("*.tflite"))
    return any_tflite


def _eval_tflite_path(exported_float_io: Path) -> Path:
    """Prefer full integer I/O TFLite (same graph as ST Edge AI) next to the export artifact."""
    parent = exported_float_io.parent
    if exported_float_io.name.endswith("_int8.tflite"):
        stem_base = exported_float_io.name[: -len("_int8.tflite")]
        full_int = parent / f"{stem_base}_full_integer_quant.tflite"
        if full_int.is_file():
            return full_int
    return exported_float_io


app = typer.Typer()


@app.command()
def main(
    size: int = typer.Option(..., help="Must match training resolution"),
    checkpoint: Path | None = typer.Option(
        None,
        help="Path to .pt checkpoint (default: best.pt under $FYP_RESULTS_DIR/model/ or <repo>/results/model/)",
    ),
    force_quantize: bool = typer.Option(False, help="Re-export INT8 TFLite even if an existing export is found"),
):
    if size not in [192, 256, 288, 320]:
        typer.echo(f"Error: size must be one of [192, 256, 288, 320]", err=True)
        raise typer.Exit(1)

    pt_path = _resolve_checkpoint(size, checkpoint)

    model_root = MODELS / f"tinyissimoyolo_v8_{size}"

    exported: Path | None = None
    existing = None if force_quantize else _find_existing_tflite(model_root)
    if existing is not None:
        exported = existing
        print(f"Found existing TFLite; skipping export:\n  {exported}")
    else:
        exported = _quantize(size, pt_path)

    print()
    print("Export return path (often float I/O):")
    print(f"  {exported}")
    eval_path = _eval_tflite_path(exported)
    if eval_path != exported:
        print("Preferred evaluation / deployment model (int8 I/O, ST Edge AI–aligned):")
        print(f"  {eval_path}")
    elif exported.name.endswith("_int8.tflite"):
        print(
            "Warning: *_full_integer_quant.tflite not found next to export artifact; "
            "you may want to locate the int8 I/O model for STM32 Model Zoo evaluation."
        )
    print("Done.")


if __name__ == "__main__":
    app()
