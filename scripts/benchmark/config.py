"""Evaluation config builder: loads a model's base YAML and applies benchmark overrides."""

from copy import deepcopy
from pathlib import Path

import yaml

from .constants import (
    COCO_80_ANNOTATIONS,
    COCO_80_TFS_TEST,
    COCO_IMAGES,
    COCO_PERSON_ANNOTATIONS,
    COCO_PERSON_TFS_TEST,
    SERVICES_DIR,
    STEDGEAI_PATH,
)
from .models import ModelEntry


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def build_eval_config(entry: ModelEntry) -> Path:
    """Load the model's base config, apply evaluation overrides, write a temp file.

    Returns the path to the written temporary config YAML (inside SERVICES_DIR).
    """
    with open(entry.config_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    overrides: dict = {
        "operation_mode": "evaluation",
        "model": {
            "model_path": entry.model_path,
        },
        "evaluation": {
            "profile": "profile_O3",
            "input_type": "uint8",
            "output_type": "int8",
            "input_chpos": "chlast",
            "output_chpos": "chlast",
            "target": "host",
        },
        "postprocessing": {
            "plot_metrics": False,
        },
        "tools": {
            "stedgeai": {
                "path_to_stedgeai": STEDGEAI_PATH,
            },
        },
    }

    if entry.dataset in ("COCO-Person", "ST-Person"):
        tfs_test = COCO_PERSON_TFS_TEST
        coco_annotations = COCO_PERSON_ANNOTATIONS
    else:
        tfs_test = COCO_80_TFS_TEST
        coco_annotations = COCO_80_ANNOTATIONS

    ds_format = base.get("dataset", {}).get("format", "tfs")
    if ds_format == "coco":
        overrides["dataset"] = {
            "test_images_path": COCO_IMAGES,
            "test_annotations_path": coco_annotations,
            # Null out paths unused during evaluation; the base config may
            # contain relative paths that do not resolve from the benchmark CWD.
            "quantization_path": None,
            "prediction_path": None,
        }
    else:
        overrides["dataset"] = {
            "test_path": tfs_test,
        }

    # Ensure model_name is set — some base configs omit it, but the
    # modelzoo dataloader dispatcher (combined.py) requires it.
    base_model = base.get("model", {})
    if not base_model.get("model_name"):
        overrides["model"]["model_name"] = base_model.get("model_type", entry.family)

    overrides = _deep_merge(overrides, entry.overrides)
    merged = _deep_merge(base, overrides)

    config_path = SERVICES_DIR / "_benchmark_temp_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    return config_path
