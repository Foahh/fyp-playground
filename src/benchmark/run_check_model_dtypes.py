#!/usr/bin/env python3
"""Check actual input/output data types of every model in MODEL_REGISTRY.

Parses TFLite FlatBuffers to extract tensor types; ONNX QDQ models are
reported as uint8/int8 (the standard stedgeai mapping).

Usage:
    python src/benchmark/run_check_model_dtypes.py
"""

import struct
import sys
from pathlib import Path

from src.benchmark.core.registry import load_model_registry

_TFLITE_TYPE = {
    0: "float32",
    1: "float16",
    2: "int32",
    3: "uint8",
    4: "int64",
    5: "string",
    6: "bool",
    7: "int16",
    8: "complex64",
    9: "int8",
}


def _tflite_io(path: str) -> tuple[str, str]:
    """Return (input_dtype, output_dtype) by parsing the TFLite FlatBuffer."""
    with open(path, "rb") as f:
        buf = f.read()

    root = struct.unpack_from("<I", buf, 0)[0]
    vt = root - struct.unpack_from("<i", buf, root)[0]

    def _field(table: int, vt_start: int, idx: int) -> int:
        off = vt_start + 4 + 2 * idx
        if off + 2 > vt_start + struct.unpack_from("<H", buf, vt_start)[0]:
            return 0
        return struct.unpack_from("<H", buf, off)[0]

    def _vec(table: int, foff: int) -> int:
        rel = struct.unpack_from("<I", buf, table + foff)[0]
        return table + foff + rel

    def _tensor_type(sg: int, sg_vt: int, tensor_list_idx: int) -> str:
        vec = _vec(sg, _field(sg, sg_vt, tensor_list_idx))
        idx_val = struct.unpack_from("<i", buf, vec + 4)[0]
        tensors_vec = _vec(sg, _field(sg, sg_vt, 0))
        t_off = tensors_vec + 4 + 4 * idx_val
        tensor = t_off + struct.unpack_from("<I", buf, t_off)[0]
        t_vt = tensor - struct.unpack_from("<i", buf, tensor)[0]
        tf = _field(tensor, t_vt, 1)
        if not tf:
            return "float32"
        raw = struct.unpack_from("<B", buf, tensor + tf)[0]
        return _TFLITE_TYPE.get(raw, f"unknown({raw})")

    sg_vec = _vec(root, _field(root, vt, 2))
    sg = sg_vec + 4 + struct.unpack_from("<I", buf, sg_vec + 4)[0]
    sg_vt = sg - struct.unpack_from("<i", buf, sg)[0]
    return _tensor_type(sg, sg_vt, 1), _tensor_type(sg, sg_vt, 2)


def main() -> None:
    hdr = f"{'Variant':<45} {'Ext':<7} {'Actual In':<10} {'Actual Out':<11} {'Reg In':<10} {'Reg Out':<10} {'OK?'}"
    print(hdr)
    print("-" * len(hdr))

    mismatches = 0
    missing = 0

    for reg in load_model_registry():
        path = Path(str(reg["model"]))
        ext = path.suffix
        variant = reg["variant"]
        fmt = reg.get("fmt", "")
        label = f"{variant} {fmt}"

        reg_in = reg.get("input_data_type", "uint8")
        reg_out = reg.get("output_data_type", "int8")

        if not path.exists():
            print(f"{label:<45} {ext:<7} {'FILE NOT FOUND'}")
            missing += 1
            continue

        if ext == ".tflite":
            actual_in, actual_out = _tflite_io(str(path))
        else:
            actual_in, actual_out = "uint8", "int8"

        ok = reg_in == actual_in and reg_out == actual_out
        mark = "YES" if ok else "MISMATCH"
        if not ok:
            mismatches += 1

        print(
            f"{label:<45} {ext:<7} {actual_in:<10} {actual_out:<11} {reg_in:<10} {reg_out:<10} {mark}"
        )

    print()
    total = len(load_model_registry())
    print(f"Total: {total}  |  OK: {total - mismatches - missing}  |  Mismatches: {mismatches}  |  Missing files: {missing}")


if __name__ == "__main__":
    main()
