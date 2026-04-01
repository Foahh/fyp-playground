"""Compare float (.pt) vs quantized INT8 (.tflite) AP using the STM32 Model Zoo
services host evaluator.  Results go to ``<results>/model/<variant>/float_int8_ap_comparison.csv``
(not the shared ``results/evaluation_result.csv``).  Legacy
``quant_eval/ap_comparison.csv`` is copied there once if present and the new file
does not exist yet.

Usage:
  python project.py eval-quantize -- --size 192
  python project.py eval-quantize -- --size 192 --force
  python project.py eval-quantize -- --size 288 --no-float
"""

from __future__ import annotations

import csv
import datetime
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import typer

from src.benchmark.core.config import build_eval_config
from src.benchmark.core.models import ModelEntry
from src.benchmark.io.parsing import parse_metrics
from src.benchmark.paths import SERVICES_DIR
from src.conda.conda_setup_common import (
    conda_cli_available,
    conda_run_argv,
    qtlz_conda_env_name,
 )
from src.common.paths import get_results_dir

MODELS = get_results_dir() / "model"
VARIANT_PREFIX = "tinyissimoyolo_v8"
FAMILY = "tinyissimoyolo_v8"
DATASET_NAME = "COCO-Person"
# Under ``results/model/<variant>/`` (same level as training ``results.csv``).
COMPARISON_CSV_NAME = "float_int8_ap_comparison.csv"

CSV_HEADER = [
    "timestamp_utc",
    "variant",
    "format",
    "model_path",
    "ap_50",
    "elapsed_s",
]


def _config_path_for_size(size: int) -> Path:
    from src.common.paths import get_repo_root

    return get_repo_root() / "configs" / f"{VARIANT_PREFIX}_{size}_config.yaml"


def _find_float_model(size: int) -> Path | None:
    p = MODELS / f"{VARIANT_PREFIX}_{size}" / "weights" / "best.pt"
    return p if p.is_file() else None


def _find_float_onnx_model(size: int) -> Path | None:
    weights_dir = MODELS / f"{VARIANT_PREFIX}_{size}" / "weights"
    p = weights_dir / "best.onnx"
    return p if p.is_file() else None


def _export_float_pt_to_onnx(pt_path: Path, *, size: int) -> Path | None:
    """
    Export a float ONNX from an Ultralytics `.pt` checkpoint.

    We run this in the quantization/export conda env (ultralytics installed),
    then prefer a stable output location: `<weights>/best.onnx`.
    """
    if not conda_cli_available():
        return None

    out_path = pt_path.parent / "best.onnx"
    if out_path.is_file():
        return out_path

    # Use a tiny inline export script so `eval-quantize` doesn't depend on ultralytics.
    code = "\n".join(
        [
            "from ultralytics import YOLO",
            "from pathlib import Path",
            "import sys",
            "pt = Path(sys.argv[1]).resolve()",
            "size = int(sys.argv[2])",
            "out = Path(sys.argv[3]).resolve()",
            "m = YOLO(str(pt))",
            "p = m.export(format='onnx', imgsz=[size, size], simplify=False)",
            "p = Path(p).resolve()",
            "out.parent.mkdir(parents=True, exist_ok=True)",
            "if p != out:",
            "    out.write_bytes(p.read_bytes())",
            "print(str(out))",
        ]
    )

    cmd = conda_run_argv(
        qtlz_conda_env_name(),
        [
            "python",
            "-c",
            code,
            str(pt_path),
            str(size),
            str(out_path),
        ],
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return out_path if out_path.is_file() else None


def _find_int8_model(size: int) -> Path | None:
    root = MODELS / f"{VARIANT_PREFIX}_{size}" / "weights" / "best_saved_model"
    full_int = root / "best_full_integer_quant.tflite"
    if full_int.is_file():
        return full_int
    int8 = root / "best_int8.tflite"
    return int8 if int8.is_file() else None


def _make_entry(
    size: int,
    model_path: Path,
    config_path: Path,
    *,
    fmt: str,
    framework: str,
    input_data_type: str,
    output_data_type: str,
) -> ModelEntry:
    return ModelEntry(
        family=FAMILY,
        variant=f"{VARIANT_PREFIX}_{size}",
        hyperparameters="",
        dataset=DATASET_NAME,
        num_classes=1,
        fmt=fmt,
        resolution=size,
        model_path=str(model_path.resolve()),
        config_path=str(config_path.resolve()),
        framework=framework,
        input_data_type=input_data_type,
        output_data_type=output_data_type,
    )


def _run_zoo_evaluator(entry: ModelEntry) -> tuple[str, str, int]:
    """Run stm32ai_main.py and return (stdout, stderr, returncode)."""
    build_eval_config(entry)

    cmd = [
        sys.executable,
        "stm32ai_main.py",
        "--config-path",
        ".",
        "--config-name",
        "_benchmark_temp_config",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["TQDM_DISABLE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SERVICES_DIR),
            capture_output=True,
            text=True,
            timeout=3600,
            env=env,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if e.stdout else ""
        stderr = e.stderr.decode() if e.stderr else ""
        return stdout, stderr, -1


def _comparison_csv_paths(size: int) -> tuple[Path, Path]:
    """Return (new_csv_under_variant, legacy_quant_eval_csv)."""
    variant_dir = MODELS / f"{VARIANT_PREFIX}_{size}"
    return variant_dir / COMPARISON_CSV_NAME, variant_dir / "quant_eval" / "ap_comparison.csv"


def _ensure_comparison_csv(size: int, *, force: bool) -> Path:
    """Resolve CSV path; one-time copy from legacy ``quant_eval/ap_comparison.csv``."""
    csv_path, legacy = _comparison_csv_paths(size)
    if not force and not csv_path.is_file() and legacy.is_file():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, csv_path)
    return csv_path


def _append_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            w.writeheader()
        w.writerow(row)


def _run_and_record(
    label: str,
    entry: ModelEntry,
    csv_path: Path,
) -> dict | None:
    print(f"\n{'='*60}")
    print(f"  [{label}] {entry.fmt}  model: {entry.model_path}")
    print(f"{'='*60}")

    t0 = time.monotonic()
    stdout, stderr, rc = _run_zoo_evaluator(entry)
    elapsed = time.monotonic() - t0

    if rc != 0:
        print(f"  FAILED (rc={rc}, {elapsed:.1f}s)")
        if stderr:
            print(stderr[-800:])
        return None

    met = parse_metrics(stdout, stderr)
    ap_50 = met.get("ap_50", "")

    row = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "variant": entry.variant,
        "format": entry.fmt,
        "model_path": entry.model_path,
        "ap_50": ap_50,
        "elapsed_s": f"{elapsed:.1f}",
    }
    _append_csv(csv_path, row)

    print(f"  AP@50 = {ap_50 or 'N/A'}  ({elapsed:.1f}s)")
    return row


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    size: int = typer.Option(..., help="Image resolution (must match training), e.g. 192, 256, 288"),
    no_float: bool = typer.Option(False, help="Skip float (.pt) model"),
    no_int8: bool = typer.Option(False, help="Skip INT8 (.tflite) model"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-run even if results already exist in the CSV"),
) -> None:
    if size not in [192, 256, 288, 320]:
        typer.echo("Error: size must be one of [192, 256, 288, 320]", err=True)
        raise typer.Exit(1)

    config_path = _config_path_for_size(size)
    if not config_path.is_file():
        typer.echo(f"Error: config not found at {config_path}", err=True)
        raise typer.Exit(1)

    csv_path = _ensure_comparison_csv(size, force=force)

    existing_fmts: set[str] = set()
    if not force and csv_path.is_file():
        with csv_path.open(newline="") as f:
            for r in csv.DictReader(f):
                existing_fmts.add(r.get("format", ""))

    results: list[dict | None] = []

    if not no_float:
        pt = _find_float_model(size)
        if pt is None:
            typer.echo(f"Warning: no float .pt checkpoint for size {size}, skipping", err=True)
        elif "Float" in existing_fmts:
            print("Float already present in CSV (use --force to re-run)")
        else:
            # STM32AI Model Zoo services does not provide a YOLOv8 torch wrapper for
            # object_detection; evaluate float via ONNX evaluator instead.
            onnx = _find_float_onnx_model(size) or _export_float_pt_to_onnx(pt, size=size)
            if onnx is None:
                typer.echo(
                    "Warning: could not find/export float ONNX; skipping float eval. "
                    "Tip: ensure the quantization env exists (project.py setup-env-qtlz).",
                    err=True,
                )
                pt = None
            else:
                pt = onnx
            entry = _make_entry(
                size,
                pt,
                config_path,
                fmt="Float",
                framework="tf",
                input_data_type="float32",
                output_data_type="float32",
            )
            results.append(_run_and_record("Float .pt", entry, csv_path))

    if not no_int8:
        tflite = _find_int8_model(size)
        if tflite is None:
            typer.echo(f"Warning: no INT8 .tflite for size {size}, skipping", err=True)
        elif "Int8" in existing_fmts:
            print("Int8 already present in CSV (use --force to re-run)")
        else:
            entry = _make_entry(
                size,
                tflite,
                config_path,
                fmt="Int8",
                framework="tf",
                input_data_type="int8",
                output_data_type="int8",
            )
            results.append(_run_and_record("INT8 TFLite", entry, csv_path))

    print(f"\n{'='*60}")
    print(f"  Results: {csv_path}")
    for r in results:
        if r:
            print(f"  {r['format']:>6}  AP@50 = {r['ap_50'] or 'N/A'}")
    print(f"{'='*60}")


def _entrypoint() -> int:
    try:
        app(args=sys.argv[1:])
    except typer.Exit as e:
        return int(e.exit_code) if e.exit_code is not None else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(_entrypoint())

