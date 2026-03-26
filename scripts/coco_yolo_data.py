"""COCO data YAML with absolute paths for Ultralytics."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COCO_TEMPLATE_YAML = (
    REPO_ROOT
    / "external"
    / "TinyissimoYOLO"
    / "ultralytics"
    / "cfg"
    / "datasets"
    / "coco.yaml"
)
DATASETS_ROOT = Path(
    os.environ.get("DATASETS_DIR", str(REPO_ROOT / "datasets"))
).expanduser()
_COCO_ROOT = (DATASETS_ROOT / "coco").resolve()


def materialize_coco_data_yaml() -> str:
    """Write a data YAML with absolute COCO root (not global Ultralytics datasets_dir)."""
    val_list = _COCO_ROOT / "val2017.txt"
    if not val_list.is_file():
        raise FileNotFoundError(
            f"Missing {val_list}. Set DATASETS_DIR (optional) and run load_coco.py."
        )
    with COCO_TEMPLATE_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(_COCO_ROOT)
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
