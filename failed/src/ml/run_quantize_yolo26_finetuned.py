"""
Two-step YOLO26 → INT8 TFLite quantization for STM32N6 deployment.

Step 1: Ultralytics export → SavedModel (with quantization-friendly output normalization)
Step 2: TF Lite Converter with ST-compatible settings (per-channel, uint8 input)

The finetuned YOLO26 model is trained with ``multi_scale`` on ``imgsz`` from
``run_finetune_yolo26`` (currently 368 px), so it supports deployment resolutions
including 256–384.  Pass ``--size`` to choose the export / deployment resolution.

Requires the ``fyp-ml`` conda env (ultralytics + TensorFlow / export stack).

Usage:
    conda activate fyp-ml
    ./project.py quantize-yolo26 --size 320
    ./project.py quantize-yolo26 --size 256 --input-type uint8 --output-type int8
    ./project.py quantize-yolo26 --size 384 --checkpoint /path/to/best.pt --force
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import typer

from src.common.paths import get_datasets_dir, get_results_dir
from src.ml.run_finetune_yolo26 import FINETUNE_IMGSZ, run_name_for

MODELS = get_results_dir() / "model"

IO_TAG = {"uint8": "u", "int8": "i", "float": "f"}
VALID_IO_TYPES = tuple(IO_TAG)
VALID_QUANT_TYPES = ("per_channel", "per_tensor")
VALID_SIZES = (256, 288, 320, 384)

RUN_NAME_BASE = run_name_for(FINETUNE_IMGSZ)


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def _resolve_checkpoint(override: Path | None) -> Path:
    if override is not None:
        p = override.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"No checkpoint at {p}")
        return p

    weights_dir = MODELS / RUN_NAME_BASE / "weights"
    best = weights_dir / "best.pt"
    if best.is_file():
        return best

    raise FileNotFoundError(
        f"No checkpoint at {best}. Finetune first: "
        f"python project.py finetune-yolo26"
    )


# ---------------------------------------------------------------------------
# Step 1 – Ultralytics export → SavedModel
# ---------------------------------------------------------------------------


def _remove_stale_ultralytics_exports(pt_path: Path) -> None:
    """Remove Ultralytics artifacts next to the checkpoint (SavedModel dir, ONNX)."""
    parent = pt_path.parent
    stem = pt_path.stem
    saved_model = parent / f"{stem}_saved_model"
    onnx_path = parent / f"{stem}.onnx"
    if saved_model.is_dir():
        shutil.rmtree(saved_model)
        print(f"Removed stale SavedModel: {saved_model}")
    if onnx_path.is_file():
        onnx_path.unlink()
        print(f"Removed stale ONNX: {onnx_path}")


def _find_saved_model(pt_path: Path, img_size: int) -> Path | None:
    """Locate a SavedModel directory from a previous Ultralytics export at the
    requested resolution."""
    candidate = pt_path.parent / f"{pt_path.stem}_{img_size}_saved_model"
    if candidate.is_dir() and (candidate / "saved_model.pb").is_file():
        return candidate
    # Fallback: standard Ultralytics naming (no size suffix)
    fallback = pt_path.parent / f"{pt_path.stem}_saved_model"
    if fallback.is_dir() and (fallback / "saved_model.pb").is_file():
        return fallback
    return None


def _export_saved_model(img_size: int, pt_path: Path) -> Path:
    """Run Ultralytics export to produce a SavedModel with quantization-friendly
    output normalization.  The ``int8=True`` flag is critical — it triggers the
    normalised-output graph that Step 2 needs.
    """
    from ultralytics import YOLO

    from src.dataset.dataset_common import materialize_fyp_merged_data_yaml

    data_yaml = materialize_fyp_merged_data_yaml()

    print(f"Loading checkpoint {pt_path} ...")
    model = YOLO(str(pt_path))

    print(f"[Step 1] Exporting SavedModel via Ultralytics (imgsz={img_size}) ...")
    model.export(
        format="tflite",
        int8=True,
        data=str(data_yaml),
        imgsz=[img_size, img_size],
        end2end=False,  # YOLO26 has end2end=True by default in >=8.3; TFLite/STEdgeAI
        # don't support the Gather/TopK ops it introduces. The exporter doesn't
        # auto-disable end2end for tflite (only for rknn/ncnn/paddle/etc.).
    )

    saved_model_dir = pt_path.parent / f"{pt_path.stem}_saved_model"
    if not saved_model_dir.is_dir():
        raise FileNotFoundError(
            f"Ultralytics did not produce a SavedModel at {saved_model_dir}. "
            "Check Ultralytics export output above."
        )

    # Rename to include resolution so multiple sizes can coexist
    sized_dir = pt_path.parent / f"{pt_path.stem}_{img_size}_saved_model"
    if sized_dir != saved_model_dir:
        if sized_dir.is_dir():
            shutil.rmtree(sized_dir)
        saved_model_dir.rename(sized_dir)
    print(f"  SavedModel: {sized_dir}")
    return sized_dir


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _resolve_calib_dir(override: Path | None) -> Path | None:
    """Find a directory of calibration images, preferring an explicit override."""
    if override is not None:
        if not override.is_dir():
            raise FileNotFoundError(f"--calib-dir is not a directory: {override}")
        return override

    datasets = get_datasets_dir()
    candidates = [
        datasets / "fyp_merged" / "val",
        datasets / "fyp_merged" / "test",
    ]
    for d in candidates:
        if d.is_dir() and any(d.glob("*.jpg")):
            return d
    return None


def _representative_data_gen(
    calib_dir: Path | None,
    img_size: int,
    max_samples: int,
):
    """Yield ``[np.float32 (1,H,W,3)]`` samples for TFLite representative-dataset
    calibration.  Falls back to random data when no images are available."""
    import numpy as np

    if calib_dir is None or not calib_dir.is_dir():
        print("  WARNING: no calibration images found — using random data")
        for _ in range(5):
            yield [np.random.rand(1, img_size, img_size, 3).astype(np.float32)]
        return

    import cv2

    images = sorted(calib_dir.glob("*.jpg"))[:max_samples]
    print(f"  Calibrating with {len(images)} images from {calib_dir}")
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            image,
            (img_size, img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        normalized = resized / 255.0
        yield [np.expand_dims(normalized.astype(np.float32), 0)]


# ---------------------------------------------------------------------------
# Step 2 – TF Lite quantization with ST-compatible settings
# ---------------------------------------------------------------------------


def _st_quantize(
    saved_model_dir: Path,
    img_size: int,
    model_name: str,
    output_dir: Path,
    input_type: str,
    output_type: str,
    quant_type: str,
    calib_dir: Path | None,
    max_calib: int,
) -> Path:
    """Quantize a SavedModel to INT8 TFLite via ``tf.lite.TFLiteConverter``."""
    import tensorflow as tf

    qt_label = quant_type.replace("_", "-")
    print(
        f"[Step 2] Quantizing SavedModel → TFLite "
        f"(input={input_type}, output={output_type}, {qt_label}) ..."
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    tf_type_map = {"int8": tf.int8, "uint8": tf.uint8}
    if input_type in tf_type_map:
        converter.inference_input_type = tf_type_map[input_type]
    if output_type in tf_type_map:
        converter.inference_output_type = tf_type_map[output_type]

    if quant_type == "per_tensor":
        converter._experimental_disable_per_channel = True  # noqa: SLF001

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_data_gen(
        calib_dir,
        img_size,
        max_calib,
    )

    tflite_bytes = converter.convert()

    qt_tag = "pc" if quant_type == "per_channel" else "pt"
    io_tag = IO_TAG[input_type] + IO_TAG[output_type]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}_quant_{qt_tag}_{io_tag}_fyp_merged.tflite"
    out_path.write_bytes(tflite_bytes)
    return out_path


# ---------------------------------------------------------------------------
# Existing-artifact helpers
# ---------------------------------------------------------------------------


def _newest(paths: Iterable[Path]) -> Path | None:
    ps = [p for p in paths if p.is_file()]
    if not ps:
        return None
    return max(ps, key=lambda p: p.stat().st_mtime)


def _find_existing_tflite(
    output_dir: Path,
    model_name: str,
    qt_tag: str,
    io_tag: str,
) -> Path | None:
    """Find an existing ST-quantized TFLite matching the requested config."""
    exact = output_dir / f"{model_name}_quant_{qt_tag}_{io_tag}_fyp_merged.tflite"
    if exact.is_file():
        return exact
    return _newest(
        output_dir.rglob(f"{model_name}_quant_{qt_tag}_{io_tag}_fyp_merged.tflite")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def main(
    size: int = typer.Option(
        ..., help=f"Target deployment resolution; one of {list(VALID_SIZES)}"
    ),
    checkpoint: Path | None = typer.Option(
        None,
        help=f"Path to .pt checkpoint (default: best.pt under results/model/{RUN_NAME_BASE}/)",
    ),
    input_type: str = typer.Option(
        "uint8",
        help="Model input type: uint8 (STM32N6 camera), int8, or float",
    ),
    output_type: str = typer.Option(
        "int8",
        help="Model output type: float or int8",
    ),
    quant_type: str = typer.Option(
        "per_channel",
        help="Quantization granularity: per_channel (recommended) or per_tensor",
    ),
    calib_dir: Path | None = typer.Option(
        None,
        help="Directory of .jpg calibration images (default: auto-detect fyp_merged val set)",
    ),
    max_calib: int = typer.Option(
        200,
        help="Max calibration images to use",
    ),
    force: bool = typer.Option(
        False,
        help="Re-export SavedModel and re-quantize from scratch",
    ),
):
    if size not in VALID_SIZES:
        typer.echo(f"Error: --size must be one of {list(VALID_SIZES)}", err=True)
        raise typer.Exit(1)
    if input_type not in VALID_IO_TYPES:
        typer.echo(
            f"Error: --input-type must be one of {list(VALID_IO_TYPES)}", err=True
        )
        raise typer.Exit(1)
    if output_type not in VALID_IO_TYPES:
        typer.echo(
            f"Error: --output-type must be one of {list(VALID_IO_TYPES)}", err=True
        )
        raise typer.Exit(1)
    if quant_type not in VALID_QUANT_TYPES:
        typer.echo(
            f"Error: --quant-type must be one of {list(VALID_QUANT_TYPES)}", err=True
        )
        raise typer.Exit(1)

    pt_path = _resolve_checkpoint(checkpoint)
    model_name = f"yolo26_{size}"

    qt_tag = "pc" if quant_type == "per_channel" else "pt"
    io_tag = IO_TAG[input_type] + IO_TAG[output_type]

    # --- fast-path: already quantized with the same settings ---------------
    if not force:
        existing = _find_existing_tflite(MODELS, model_name, qt_tag, io_tag)
        if existing is not None:
            print(f"Found existing TFLite matching requested config:\n  {existing}")
            print("Use --force to re-quantize.")
            return

    if force:
        _remove_stale_ultralytics_exports(pt_path)

    # --- Step 1: SavedModel ------------------------------------------------
    saved_model_dir = None if force else _find_saved_model(pt_path, size)
    if saved_model_dir is not None:
        print(f"Reusing existing SavedModel:\n  {saved_model_dir}")
    else:
        saved_model_dir = _export_saved_model(size, pt_path)

    # --- Step 2: TFLite quantization (ST-compatible) -----------------------
    resolved_calib = _resolve_calib_dir(calib_dir)
    result = _st_quantize(
        saved_model_dir=saved_model_dir,
        img_size=size,
        model_name=model_name,
        output_dir=MODELS,
        input_type=input_type,
        output_type=output_type,
        quant_type=quant_type,
        calib_dir=resolved_calib,
        max_calib=max_calib,
    )

    print()
    print(f"Quantized model written to:\n  {result}")
    print(f"  Input:  {input_type}")
    print(f"  Output: {output_type}")
    print(f"  Quant:  {quant_type}")
    print("Done.")


if __name__ == "__main__":
    app()
