#!/usr/bin/env python3
"""
Post-install: ensure onnx2tf INT8 weights use per-channel quantization (STM32 NPU).

Run after `pip install -U ultralytics`. Patches the installed `ultralytics/utils/export/tensorflow.py`
only if missing or wrong `quant_type` on the `onnx2tf.convert(...)` call.

Idempotent: safe to run multiple times.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _tensorflow_path() -> Path:
    import ultralytics

    return Path(ultralytics.__file__).resolve().parent / "utils" / "export" / "tensorflow.py"


def patch(path: Path) -> int:
    if not path.is_file():
        print(f"patch_ultralytics_tflite_quant: missing {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    if re.search(r"quant_type\s*=\s*[\"']per-channel[\"']", text):
        print(f"patch_ultralytics_tflite_quant: OK (already per-channel) — {path}")
        return 0

    m = re.search(r"^(\s*)output_integer_quantized_tflite=int8,\s*$", text, re.MULTILINE)
    if not m:
        print(
            "patch_ultralytics_tflite_quant: could not find "
            "`output_integer_quantized_tflite=int8,` — update this script for your ultralytics version.",
            file=sys.stderr,
        )
        return 1

    indent = m.group(1)
    insert = f'{indent}quant_type="per-channel",  # STM32 NPU / onnx2tf INT8 weights'

    if re.search(r"^\s*quant_type\s*=", text, re.MULTILINE):
        new_text, n = re.subn(
            r"^\s*quant_type\s*=\s*[\"'][^\"']+[\"']\s*,?\s*$",
            insert,
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if n != 1:
            print("patch_ultralytics_tflite_quant: found quant_type but replace failed", file=sys.stderr)
            return 1
        path.write_text(new_text, encoding="utf-8")
        print(f"patch_ultralytics_tflite_quant: set quant_type=per-channel — {path}")
        return 0

    line = m.group(0)
    path.write_text(text.replace(line, line + "\n" + insert, 1), encoding="utf-8")
    print(f"patch_ultralytics_tflite_quant: inserted quant_type=per-channel — {path}")
    return 0


if __name__ == "__main__":
    sys.exit(patch(_tensorflow_path()))
