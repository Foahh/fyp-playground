"""
Quantize a TF SavedModel to TFLite via stm32ai-modelzoo-services, then
optionally evaluate the quantized model.

Quantization uses the modelzoo's tflite_quant.py (from_saved_model, Hydra config).
Evaluation uses the modelzoo's stm32ai_main.py (operation_mode: evaluation).

Usage:
    python scripts/run_quantize.py --img_size 192
    python scripts/run_quantize.py --img_size 192 --no-eval
    python scripts/run_quantize.py --img_size 192 --saved-model /path/to/saved_model
    python scripts/run_quantize.py --img_size 192 --out /path/to/output.tflite
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "results" / "model"
MODELZOO = ROOT / "external" / "stm32ai-modelzoo-services"
TFLITE_QUANT = MODELZOO / "tutorials" / "scripts" / "yolov8_quantization" / "tflite_quant.py"
STM32AI_MAIN = MODELZOO / "object_detection" / "stm32ai_main.py"
CONFIG_DIR = ROOT / "configs"
CALIB_DIR = ROOT / "datasets" / "coco" / "images" / "val2017"
EVAL_TEST_DIR = ROOT / "datasets" / "coco_2017_person" / "test"


def _resolve_saved_model(img_size: int, override: Path | None) -> Path:
    if override is not None:
        p = override.resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"No SavedModel at {p}")
        return p

    weights_dir = MODELS / f"tinyissimoyolo_v8_{img_size}" / "weights"
    default = weights_dir / "best_saved_model"
    if default.is_dir():
        return default

    candidates = sorted(p for p in weights_dir.glob("*_saved_model") if p.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"Multiple SavedModel dirs under {weights_dir}: {names}. Pass --saved-model explicitly."
        )

    raise FileNotFoundError(
        f"No SavedModel under {weights_dir}. Run export first: "
        f"python project.py export --img_size {img_size}"
    )


def _quantize(img_size: int, saved_model_dir: Path, output_dir: Path) -> Path:
    """Run modelzoo tflite_quant.py and return path to the quantized .tflite."""
    config_name = f"tinyissimoyolo_v8_{img_size}_quant"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TFLITE_QUANT),
        f"--config-path={CONFIG_DIR.resolve()}",
        f"--config-name={config_name}",
        f"model.model_path={saved_model_dir.resolve()}",
        f"model.input_shape=[{img_size},{img_size},3]",
        f"quantization.calib_dataset_path={CALIB_DIR.resolve()}",
        f"quantization.export_path={output_dir.resolve()}",
        f"hydra.run.dir={output_dir.resolve()}",
    ]

    print("Quantizing SavedModel ...")
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    tflite_files = sorted(output_dir.glob("*.tflite"))
    if not tflite_files:
        raise FileNotFoundError(f"tflite_quant.py produced no .tflite under {output_dir}")
    return tflite_files[0]


def _evaluate(tflite_path: Path, img_size: int, output_dir: Path) -> None:
    """Run modelzoo stm32ai_main.py evaluation on the quantized model."""
    config_name = f"tinyissimoyolo_v8_{img_size}_config"
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(STM32AI_MAIN),
        f"--config-path={CONFIG_DIR.resolve()}",
        f"--config-name={config_name}",
        f"++operation_mode=evaluation",
        f"++model.model_path={tflite_path.resolve()}",
        f"++dataset.test_path={EVAL_TEST_DIR.resolve()}",
        f"++hydra.run.dir={eval_dir.resolve()}",
        f"++mlflow.uri={eval_dir.resolve() / 'mlruns'}",
    ]

    print("Evaluating quantized model ...")
    print("+", " ".join(cmd))
    env = os.environ.copy()
    # Keep evaluation on CPU to avoid TensorFlow/CUDA runtime issues on host GPUs.
    env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Quantize model via stm32ai-modelzoo-services")
    p.add_argument(
        "--img_size",
        type=int,
        required=True,
        choices=[192, 256, 288, 320],
        help="Must match training resolution",
    )
    p.add_argument(
        "--saved-model",
        type=Path,
        default=None,
        help="Path to SavedModel dir (default: auto-detect from results/model/)",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after quantization",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Copy quantized .tflite to this path",
    )
    return p.parse_args()


def main():
    args = parse_args()
    saved_model_dir = _resolve_saved_model(args.img_size, args.saved_model)
    run_name = f"tinyissimoyolo_v8_{args.img_size}"
    output_dir = MODELS / run_name / "quantized"

    tflite_path = _quantize(args.img_size, saved_model_dir, output_dir)
    print(f"Quantized TFLite: {tflite_path}")

    if not args.no_eval:
        _evaluate(tflite_path, args.img_size, output_dir)

    if args.out:
        dest = args.out.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tflite_path, dest)
        print(f"Copied to: {dest}")

    print("Done.")


if __name__ == "__main__":
    main()
