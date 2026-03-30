"""Shared dataset helpers (e.g. COCO data YAML with absolute paths for Ultralytics)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from ..common.paths import get_datasets_dir, get_repo_root

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


def _candidate_coco_roots() -> list[Path]:
    datasets_dir = get_datasets_dir()
    return [
        (datasets_dir / "coco_2017_person").resolve(),
        (datasets_dir / "coco").resolve(),
    ]


def _is_person_split(root: Path) -> bool:
    return (
        (root / "train2017.txt").is_file()
        and (root / "val2017.txt").is_file()
        and (root / "labels" / "train").is_dir()
        and (root / "labels" / "val").is_dir()
    )


def materialize_coco_data_yaml(require_person: bool = False) -> str:
    """Write a data YAML with absolute COCO root (not global Ultralytics datasets_dir)."""
    coco_root = None
    tried: list[Path] = []
    is_person_root = False
    person_candidates = [
        p for p in _candidate_coco_roots() if p.name == "coco_2017_person"
    ]
    non_person_candidates = [
        p for p in _candidate_coco_roots() if p.name != "coco_2017_person"
    ]
    candidates = person_candidates + ([] if require_person else non_person_candidates)

    for candidate in candidates:
        tried.append(candidate)
        if _is_person_split(candidate):
            coco_root = candidate
            is_person_root = True
            break
        if (candidate / "val2017.txt").is_file() and (candidate / "train2017.txt").is_file():
            coco_root = candidate
            break
    if coco_root is None:
        if require_person:
            raise FileNotFoundError(
                "Missing person-only COCO split files under coco_2017_person. Checked: "
                + ", ".join(
                    f"{p / 'train2017.txt'} and {p / 'val2017.txt'}" for p in tried
                )
                + ". Re-run src/dataset/run_download_coco_dataset.py (or ``python project.py download-coco``) to regenerate person splits."
            )
        raise FileNotFoundError(
            "Missing COCO split files. Checked: "
            + ", ".join(f"{p / 'train2017.txt'} and {p / 'val2017.txt'}" for p in tried)
            + ". Set DATASETS_DIR (optional) and run src/dataset/run_download_coco_dataset.py."
        )
    with COCO_TEMPLATE_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(coco_root)
    if is_person_root:
        cfg["train"] = "train2017.txt"
        cfg["val"] = "val2017.txt"
        cfg["names"] = ["person"]
        cfg.pop("test", None)
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
