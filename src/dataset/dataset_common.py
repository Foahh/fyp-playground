"""Shared dataset helpers (COCO and YAMLs with absolute paths for Ultralytics)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import yaml

from ..common.paths import get_datasets_dir, get_repo_root, get_results_dir

REPO_ROOT = get_repo_root()
COCO_TEMPLATE_YAML = (
    REPO_ROOT
    / "external"
    / "TinyissimoYOLO"
    / "ultralytics"
    / "cfg"
    / "datasets"
    / "coco.yaml"
)


def _is_person_split(root: Path) -> bool:
    return (
        (root / "train2017.txt").is_file()
        and (root / "val2017.txt").is_file()
        and (root / "labels" / "train").is_dir()
        and (root / "labels" / "val").is_dir()
    )


def _write_coco_data_yaml_file(cfg: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="coco_data_",
        delete=False,
        encoding="utf-8",
    )
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    return tmp.name


def materialize_coco_80_data_yaml() -> str:
    """Write a data YAML for full COCO (80 classes); absolute root, not global Ultralytics datasets_dir."""
    coco_root = (get_datasets_dir() / "coco").resolve()
    if (
        not (coco_root / "train2017.txt").is_file()
        or not (coco_root / "val2017.txt").is_file()
    ):
        raise FileNotFoundError(
            "Missing COCO split files. Checked: "
            f"{coco_root / 'train2017.txt'} and {coco_root / 'val2017.txt'}"
            ". Set FYP_DATASETS_DIR (optional) and run src/dataset/download_dataset.py."
        )
    with COCO_TEMPLATE_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(coco_root)
    return _write_coco_data_yaml_file(cfg)


def materialize_coco_person_data_yaml() -> str:
    """Write a data YAML for person-only COCO under coco_2017_person."""
    person_root = (get_datasets_dir() / "coco_2017_person").resolve()
    if not _is_person_split(person_root):
        raise FileNotFoundError(
            "Missing person-only COCO split files under coco_2017_person. Checked: "
            f"{person_root / 'train2017.txt'} and {person_root / 'val2017.txt'}"
            ". Re-run src/dataset/download_dataset.py (or ``python project.py download-dataset``) to regenerate person splits."
        )
    with COCO_TEMPLATE_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(person_root)
    cfg["train"] = "train2017.txt"
    cfg["val"] = "val2017.txt"
    cfg["names"] = ["person"]
    cfg.pop("test", None)
    return _write_coco_data_yaml_file(cfg)
