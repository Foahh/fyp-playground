"""Evaluation config builder: loads a model's base YAML and applies benchmark overrides."""

from pathlib import Path

import yaml

from ...common.utils import deep_merge
from ..constants import SSD_FAMILIES
from ..paths import (
    COCO_80_ANNOTATIONS,
    COCO_80_TFS_TEST,
    COCO_IMAGES,
    COCO_PERSON_ANNOTATIONS,
    COCO_PERSON_TFS_TEST,
    SERVICES_DIR,
    STEDGEAI_PATH,
)
from .models import ModelEntry


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
            "input_type": entry.input_data_type,
            "output_type": entry.output_data_type,
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

    if entry.dataset == "COCO-Person":
        tfs_test = COCO_PERSON_TFS_TEST
        coco_annotations = COCO_PERSON_ANNOTATIONS
    elif entry.dataset == "COCO-80":
        tfs_test = COCO_80_TFS_TEST
        coco_annotations = COCO_80_ANNOTATIONS
    else:
        raise ValueError(f"Invalid dataset: {entry.dataset}")

    ds_format = base.get("dataset", {}).get("format", "tfs")
    if ds_format == "coco":
        overrides["dataset"] = {
            "test_images_path": COCO_IMAGES,
            "test_annotations_path": coco_annotations,
            "quantization_path": None,
            "prediction_path": None,
        }
    else:
        overrides["dataset"] = {
            "test_path": tfs_test,
            "test_images_path": tfs_test,
            "test_annotations_path": tfs_test,
        }

    base_model = base.get("model", {})

    if entry.framework == "torch":
        if not base_model.get("model_name"):
            overrides["model"]["model_name"] = (
                base_model.get("model_type") or entry.family
            )
    else:
        overrides["model"]["model_name"] = None

    if not base_model.get("input_shape"):
        overrides["model"]["input_shape"] = [entry.resolution, entry.resolution, 3]

    overrides = deep_merge(overrides, entry.overrides)
    merged = deep_merge(base, overrides)

    config_path = SERVICES_DIR / "_benchmark_temp_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    return config_path
