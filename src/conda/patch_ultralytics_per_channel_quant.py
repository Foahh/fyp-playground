#!/usr/bin/env python3
"""Force Ultralytics onnx2tf TFLite INT8 to use per-channel quantization.

STM32 tooling expects per-channel weights; Ultralytics/onnx2tf default is per-tensor.
See stm32ai-modelzoo-services/object_detection/docs/tuto/How_to_deploy_yolov8_yolov5_object_detection.md

Targets (whichever exists in the installed Ultralytics version):
  - ultralytics.utils.export.tensorflow (current)
  - ultralytics.engine.exporter (older)

Run inside the ``fyp-ml`` conda env (e.g. after ``conda_setup_ml.py``), or let setup invoke this script.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Long single-line form in some Ultralytics / Tinyissimo forks
OLD_QUANT_LINE = (
    'quant_type="per-tensor",  # "per-tensor" (faster) or '
    '"per-channel" (slower but more accurate)'
)
NEW_QUANT_LINE = (
    'quant_type="per-channel",  # "per-tensor" (faster) or '
    '"per-channel" (slower but more accurate)'
)

# Newer Ultralytics: no quant_type kwarg → onnx2tf defaults to per-tensor
INSERT_AFTER = "        output_integer_quantized_tflite=int8,\n"
INSERT_FULL = (
    "        output_integer_quantized_tflite=int8,\n"
    '        quant_type="per-channel",\n'
)

MODULES = (
    "ultralytics.utils.export.tensorflow",
    "ultralytics.engine.exporter",
)


def _patch_one_file(path: Path) -> str | None:
    """Return 'patched', 'ok', or None if file is not an onnx2tf.convert host."""
    text = path.read_text(encoding="utf-8")
    if "onnx2tf.convert" not in text:
        return None

    if OLD_QUANT_LINE in text:
        path.write_text(text.replace(OLD_QUANT_LINE, NEW_QUANT_LINE, 1), encoding="utf-8")
        return "patched"
    if 'quant_type="per-tensor"' in text:
        path.write_text(
            text.replace('quant_type="per-tensor"', 'quant_type="per-channel"', 1),
            encoding="utf-8",
        )
        return "patched"
    if 'quant_type="per-channel"' in text:
        return "ok"
    if INSERT_AFTER in text and INSERT_FULL not in text:
        path.write_text(text.replace(INSERT_AFTER, INSERT_FULL, 1), encoding="utf-8")
        return "patched"
    return None


def main() -> int:
    any_handled = False
    for name in MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            continue
        path = Path(spec.origin)
        result = _patch_one_file(path)
        if result == "patched":
            print("Patched onnx2tf quant to per-channel:", path)
            any_handled = True
        elif result == "ok":
            print("Ultralytics already uses per-channel:", path)
            any_handled = True

    if not any_handled:
        print(
            "ERROR: could not patch Ultralytics onnx2tf quant "
            "(expected tensorflow.py or exporter.py with onnx2tf.convert).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
