"""ModelEntry dataclass and model discovery logic."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .constants import (
    IN_SCOPE_FAMILIES,
    MODELZOO_DIR,
    REMOTE_MODELS,
    SSD_FAMILIES,
    YOLOD_FAMILIES,
)


@dataclass
class ModelEntry:
    family: str
    variant: str
    hyperparameters: str
    dataset: str  # "COCO-Person" or "COCO-80"
    num_classes: int
    fmt: str  # "Int8" or "W4A8"
    resolution: int
    model_path: str  # absolute or URL
    model_type: str  # ssd, st_yolod, st_yoloxn, st_yololcv1, yolov8n, yolo11n, yolo26
    model_name: str = ""  # for PT models: e.g. ssdlite_mobilenetv1_pt


def _parse_resolution(variant_name: str) -> int:
    """Extract the last numeric segment from variant directory name."""
    m = re.findall(r"(\d+)", variant_name)
    return int(m[-1]) if m else 0


def _parse_hyperparams(family: str, variant_name: str) -> str:
    """Extract hyperparameters from variant name."""
    if family == "st_yoloxn":
        m = re.search(r"(d\d+_w\d+)", variant_name)
        return m.group(1) if m else ""
    if family in YOLOD_FAMILIES:
        if "actrelu" in variant_name:
            return "actrelu"
    return ""


def _detect_dataset(path_parts: list[str]) -> tuple[str, int]:
    """Detect dataset from directory path components."""
    path_str = "/".join(path_parts).lower()
    if "coco_2017_person" in path_str or "coco_person" in path_str:
        return "COCO-Person", 1
    if "coco_2017_80_classes" in path_str or "coco" in path_str:
        return "COCO-80", 80
    return "", 0


def _pick_w4a8(files: list[Path]) -> Optional[Path]:
    """Pick the W4A8 ONNX with highest mAP in filename."""
    w4_files = [
        f for f in files if re.search(r"_qdq_w4_", f.name) and f.suffix == ".onnx"
    ]
    if not w4_files:
        return None
    if len(w4_files) == 1:
        return w4_files[0]
    # pick highest map
    best, best_map = w4_files[0], 0.0
    for f in w4_files:
        m = re.search(r"_map_(\d+(?:\.\d+)?)", f.name)
        if m and float(m.group(1)) > best_map:
            best_map = float(m.group(1))
            best = f
    return best


def _model_type_for_family(family: str) -> str:
    if family in SSD_FAMILIES:
        return "ssd"
    if family in YOLOD_FAMILIES:
        return "st_yolod"
    if family == "st_yoloxn":
        return "st_yoloxn"
    if family == "st_yololcv1":
        return "st_yololcv1"
    if family == "yolov8n":
        return "yolov8n"
    if family == "yolo11n":
        return "yolo11n"
    if family == "yolo26":
        return "yolo26"
    return family


def _model_name_for_ssd(family: str) -> str:
    """Return the model_name string expected by the SSD framework."""
    return family  # e.g. "ssdlite_mobilenetv1_pt"


def _model_name_for_yolod(variant_name: str) -> str:
    """e.g. st_yolodv2milli_actrelu_pt from st_yolodv2milli_actrelu_pt_coco_person_192."""
    m = re.match(r"(st_yolodv2\w+_actrelu_pt)", variant_name)
    return m.group(1) if m else variant_name


def discover_models() -> list[ModelEntry]:
    """Walk modelzoo tree and return list of ModelEntry."""
    entries = []

    for family in IN_SCOPE_FAMILIES:
        family_dir = MODELZOO_DIR / family

        # Handle predefined local/remote models (e.g. yolov8n/yolo11n/yolo26)
        if family in REMOTE_MODELS:
            for info in REMOTE_MODELS[family]:
                entries.append(
                    ModelEntry(
                        family=family,
                        variant=f"{family}_{info['resolution']}",
                        hyperparameters="",
                        dataset=info["dataset"],
                        num_classes=info["num_classes"],
                        fmt="Int8",
                        resolution=info["resolution"],
                        model_path=info["model_path"],
                        model_type=info["model_type"],
                    )
                )
            continue

        if not family_dir.is_dir():
            print(f"[WARN] Family dir not found: {family_dir}")
            continue

        # Walk to find variant directories (leaf dirs containing model files)
        for root, dirs, files in os.walk(family_dir):
            root_path = Path(root)
            path_parts = root_path.relative_to(MODELZOO_DIR).parts

            # Skip VOC variants
            if any(p.lower() == "voc" for p in path_parts):
                continue

            # Skip custom dataset variants (st_person)
            if any("custom_dataset" in p.lower() for p in path_parts):
                continue

            # Check if this directory has model files
            all_files = [root_path / f for f in files]
            onnx_files = [f for f in all_files if f.suffix == ".onnx"]
            tflite_files = [f for f in all_files if f.suffix == ".tflite"]

            if not onnx_files and not tflite_files:
                continue

            variant_name = root_path.name
            dataset, num_classes = _detect_dataset(list(path_parts))
            if not dataset:
                continue

            resolution = _parse_resolution(variant_name)
            hyperparams = _parse_hyperparams(family, variant_name)
            model_type = _model_type_for_family(family)

            # Determine model_name for PT models
            model_name = ""
            if family in SSD_FAMILIES:
                model_name = _model_name_for_ssd(family)
            elif family in YOLOD_FAMILIES:
                model_name = _model_name_for_yolod(variant_name)

            # W4A8 entry
            w4_file = _pick_w4a8(all_files)
            if w4_file:
                entries.append(
                    ModelEntry(
                        family=family,
                        variant=variant_name,
                        hyperparameters=hyperparams,
                        dataset=dataset,
                        num_classes=num_classes,
                        fmt="W4A8",
                        resolution=resolution,
                        model_path=str(w4_file),
                        model_type=model_type,
                        model_name=model_name,
                    )
                )

            # Int8 entry: prefer tflite, else qdq_int8.onnx
            int8_tflite = [f for f in tflite_files if "_int8" in f.stem]
            int8_onnx = [f for f in onnx_files if "_qdq_int8" in f.name]

            if int8_tflite:
                entries.append(
                    ModelEntry(
                        family=family,
                        variant=variant_name,
                        hyperparameters=hyperparams,
                        dataset=dataset,
                        num_classes=num_classes,
                        fmt="Int8",
                        resolution=resolution,
                        model_path=str(int8_tflite[0]),
                        model_type=model_type,
                        model_name=model_name,
                    )
                )
            elif int8_onnx:
                entries.append(
                    ModelEntry(
                        family=family,
                        variant=variant_name,
                        hyperparameters=hyperparams,
                        dataset=dataset,
                        num_classes=num_classes,
                        fmt="Int8",
                        resolution=resolution,
                        model_path=str(int8_onnx[0]),
                        model_type=model_type,
                        model_name=model_name,
                    )
                )

    return entries
