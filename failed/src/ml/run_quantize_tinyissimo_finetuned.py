"""
Two-step TinyissimoYOLOv8 finetuned → INT8 TFLite quantization for STM32N6.

Step 1: Ultralytics export → SavedModel (with quantization-friendly output normalization)
Step 2: TF Lite Converter with ST-compatible settings (per-channel, uint8 input)

This variant is for Tinyissimo models finetuned on ``datasets/fyp_merged``.
Compared to ``run_quantize_tinyissimo_coco_person.py``, it:
  - exports with ``materialize_fyp_merged_data_yaml()``
  - calibrates from ``datasets/fyp_merged/{val,test}``
  - writes ``*_fyp_merged.tflite`` outputs
  - prefers finetuned run checkpoints by default

Usage:
    conda activate fyp-ml
    ./project.py quantize-tiny-finetuned -- --size 320
    ./project.py quantize-tiny-finetuned -- --size 320 --input-type uint8 --output-type int8
    ./project.py quantize-tiny-finetuned -- --size 320 --run-name tinyissimoyolo_v8_320_e260_lr800_p70
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import typer

from src.common.paths import get_datasets_dir, get_results_dir

MODELS = get_results_dir() / "model"

IO_TAG = {"uint8": "u", "int8": "i", "float": "f"}
VALID_IO_TYPES = tuple(IO_TAG)
VALID_QUANT_TYPES = ("per_channel", "per_tensor")
VALID_SIZES = (192, 256, 288, 320)


def _newest(paths: Iterable[Path]) -> Path | None:
    ps = [p for p in paths if p.is_file()]
    if not ps:
        return None
    return max(ps, key=lambda p: p.stat().st_mtime)


def _resolve_checkpoint(
    img_size: int,
    override: Path | None,
    run_name: str | None,
) -> Path:
    if override is not None:
        p = override.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"No checkpoint at {p}")
        return p

    if run_name:
        best = MODELS / run_name / "weights" / "best.pt"
        if best.is_file():
            return best
        raise FileNotFoundError(f"No checkpoint at {best}")

    # Prefer finetune-style runs for this size, then fallback to base run name.
    size_prefix = f"tinyissimoyolo_v8_{img_size}_"
    candidates = list(MODELS.glob(f"{size_prefix}*/weights/best.pt"))
    newest = _newest(candidates)
    if newest is not None:
        return newest

    fallback = MODELS / f"tinyissimoyolo_v8_{img_size}" / "weights" / "best.pt"
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        f"No finetuned checkpoint found for size {img_size}. "
        "Provide --checkpoint or --run-name."
    )


def _remove_stale_ultralytics_exports(pt_path: Path) -> None:
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


def _find_saved_model(pt_path: Path) -> Path | None:
    candidate = pt_path.parent / f"{pt_path.stem}_saved_model"
    if candidate.is_dir() and (candidate / "saved_model.pb").is_file():
        return candidate
    return None


def _export_saved_model(img_size: int, pt_path: Path) -> Path:
    from ultralytics import YOLO

    from src.dataset.dataset_common import materialize_fyp_merged_data_yaml

    data_yaml = materialize_fyp_merged_data_yaml()

    print(f"Loading checkpoint {pt_path} ...")
    model = YOLO(str(pt_path))

    print("[Step 1] Exporting SavedModel via Ultralytics...")
    model.export(
        format="tflite",
        int8=True,
        data=str(data_yaml),
        imgsz=[img_size, img_size],
    )

    saved_model_dir = pt_path.parent / f"{pt_path.stem}_saved_model"
    if not saved_model_dir.is_dir():
        raise FileNotFoundError(
            f"Ultralytics did not produce a SavedModel at {saved_model_dir}. "
            "Check Ultralytics export output above."
        )
    print(f"  SavedModel: {saved_model_dir}")
    return saved_model_dir


def _resolve_calib_dir(override: Path | None) -> Path | None:
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


def _find_existing_tflite(
    output_dir: Path,
    model_name: str,
    qt_tag: str,
    io_tag: str,
) -> Path | None:
    exact = output_dir / f"{model_name}_quant_{qt_tag}_{io_tag}_fyp_merged.tflite"
    if exact.is_file():
        return exact
    return _newest(
        output_dir.rglob(f"{model_name}_quant_{qt_tag}_{io_tag}_fyp_merged.tflite")
    )


app = typer.Typer()


@app.command()
def main(
    size: int = typer.Option(..., help=f"Image size; one of {list(VALID_SIZES)}"),
    checkpoint: Path | None = typer.Option(
        None,
        help="Path to .pt checkpoint. If omitted, auto-resolves newest finetuned checkpoint.",
    ),
    run_name: str | None = typer.Option(
        None,
        help="Optional run name under results/model to resolve weights from.",
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
        help="Directory of .jpg calibration images (default: auto-detect fyp_merged val/test)",
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

    pt_path = _resolve_checkpoint(size, checkpoint, run_name)
    model_name = f"tinyissimoyolo_v8_{size}"

    qt_tag = "pc" if quant_type == "per_channel" else "pt"
    io_tag = IO_TAG[input_type] + IO_TAG[output_type]

    if not force:
        existing = _find_existing_tflite(MODELS, model_name, qt_tag, io_tag)
        if existing is not None:
            print(f"Found existing TFLite matching requested config:\n  {existing}")
            print("Use --force to re-quantize.")
            return

    if force:
        _remove_stale_ultralytics_exports(pt_path)

    saved_model_dir = None if force else _find_saved_model(pt_path)
    if saved_model_dir is not None:
        print(f"Reusing existing SavedModel:\n  {saved_model_dir}")
    else:
        saved_model_dir = _export_saved_model(size, pt_path)

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
