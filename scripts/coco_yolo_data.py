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
DEFAULT_DATASETS_ROOT = (REPO_ROOT / "datasets").resolve()


def _candidate_coco_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("DATASETS_DIR")
    if env_root:
        roots.append((Path(env_root).expanduser() / "coco").resolve())
    roots.append((DEFAULT_DATASETS_ROOT / "coco").resolve())
    return roots


def materialize_coco_data_yaml() -> str:
    """Write a data YAML with absolute COCO root (not global Ultralytics datasets_dir)."""
    coco_root = None
    tried: list[Path] = []
    for candidate in _candidate_coco_roots():
        tried.append(candidate)
        if (candidate / "val2017.txt").is_file():
            coco_root = candidate
            break
    if coco_root is None:
        raise FileNotFoundError(
            "Missing COCO val2017.txt. Checked: "
            + ", ".join(str(p / "val2017.txt") for p in tried)
            + ". Set DATASETS_DIR (optional) and run load_coco.py."
        )
    with COCO_TEMPLATE_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(coco_root)
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
