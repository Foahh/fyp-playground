"""Shared dataset helpers (COCO and fyp_merged YAMLs with absolute paths for Ultralytics)."""

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
            ". Set FYP_DATASETS_DIR (optional) and run src/dataset/run_download_coco_dataset.py."
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
            ". Re-run src/dataset/run_download_coco_dataset.py (or ``python project.py download-coco``) to regenerate person splits."
        )
    with COCO_TEMPLATE_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(person_root)
    cfg["train"] = "train2017.txt"
    cfg["val"] = "val2017.txt"
    cfg["names"] = ["person"]
    cfg.pop("test", None)
    return _write_coco_data_yaml_file(cfg)


def _dir_has_split_images(split_dir: Path) -> bool:
    if not split_dir.is_dir():
        return False
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return any(p.suffix.lower() in exts for p in split_dir.iterdir() if p.is_file())


def get_finetune_yolo_dir(name: str) -> Path:
    """Path to a *converted* finetune dataset (YOLO layout), never ``*_raw``.

    These trees are produced by ``run_download_finetune_dataset`` (per-dataset
    conversion and optional merge). Use this in tooling so scripts read
    ``ego2hands/``, ``construction_tools/``, etc., not the download-only raw folders.

    *name* can be ``ego2hands``, ``construction_tools``, ``metu_alet``, or
    ``fyp_merged`` (aliases with hyphens are accepted).
    """
    key = name.lower().replace("-", "_")
    root = get_datasets_dir().resolve()
    mapping = {
        "ego2hands": "ego2hands",
        "construction_tools": "construction_tools",
        "metu_alet": "metu_alet",
        "fyp_merged": "fyp_merged",
    }
    sub = mapping.get(key)
    if sub is None:
        raise ValueError(
            f"Unknown finetune YOLO dataset {name!r}; "
            f"expected one of: {', '.join(sorted(mapping))}."
        )
    return root / sub


_IMG_EXT_FYP = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fyp_merged_pseudo_root() -> Path:
    """``results/fyp_merged_pseudo`` (overlay + small metadata; safe to commit ``overlay/``)."""
    return (get_results_dir() / "fyp_merged_pseudo").resolve()


def fyp_merged_overlay_root() -> Path:
    """Label overrides only: ``overlay/{train,val}/*.txt`` (commit-friendly)."""
    return fyp_merged_pseudo_root() / "overlay"


def fyp_merged_overlay_active() -> bool:
    """True when at least one overlay label exists (pseudo or hand-corrected)."""
    root = fyp_merged_overlay_root()
    for split in ("train", "val"):
        d = root / split
        if d.is_dir() and any(d.glob("*.txt")):
            return True
    return False


def fyp_merged_runtime_root() -> Path:
    """Generated full YOLO tree (gitignored): base merge + overlay."""
    return (get_results_dir() / "fyp_merged_runtime").resolve()


def sync_fyp_merged_runtime() -> Path:
    """Build ``results/fyp_merged_runtime`` with symlinks to images and merged labels.

    For each image in ``datasets/fyp_merged/{train,val}``, the label is
    ``overlay/...`` if present, otherwise the original ``fyp_merged`` ``.txt``.
    """
    base = (get_datasets_dir() / "fyp_merged").resolve()
    overlay = fyp_merged_overlay_root()
    runtime = fyp_merged_runtime_root()
    if runtime.is_dir():
        shutil.rmtree(runtime)

    for split in ("train", "val"):
        src_split = base / split
        if not src_split.is_dir():
            continue
        img_out = runtime / "images" / split
        lbl_out = runtime / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        ovr_split = overlay / split

        for img_path in sorted(src_split.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXT_FYP:
                continue
            stem = img_path.stem
            lbl_base = src_split / f"{stem}.txt"
            if not lbl_base.is_file():
                continue
            dst_img = img_out / img_path.name
            try:
                rel = os.path.relpath(img_path.resolve(), start=dst_img.parent.resolve())
                os.symlink(rel, dst_img)
            except OSError:
                shutil.copy2(img_path, dst_img)

            lbl_ov = ovr_split / f"{stem}.txt"
            src_lbl = lbl_ov if lbl_ov.is_file() else lbl_base
            shutil.copy2(src_lbl, lbl_out / f"{stem}.txt")

    return runtime


def materialize_fyp_merged_data_yaml() -> str:
    """Write a data YAML for merged finetune data (hand + tool).

    If ``results/fyp_merged_pseudo/overlay`` has any ``.txt`` files, builds
    ``results/fyp_merged_runtime`` (base ``datasets/fyp_merged`` + overlay) and points
    the YAML there. Otherwise uses ``datasets/fyp_merged/{train,val}/`` directly.
    """
    if fyp_merged_overlay_active():
        root = sync_fyp_merged_runtime()
        cfg = {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "nc": 2,
            "names": ["hand", "tool"],
        }
        return _write_coco_data_yaml_file(cfg)

    merged_root = (get_datasets_dir() / "fyp_merged").resolve()
    train_dir = merged_root / "train"
    val_dir = merged_root / "val"
    if not _dir_has_split_images(train_dir) or not _dir_has_split_images(val_dir):
        raise FileNotFoundError(
            "Expected YOLO images under fyp_merged/train/ and fyp_merged/val/ "
            f"(resolved root: {merged_root}). "
            "Run ``python project.py download-finetune`` (merge step) or build fyp_merged there."
        )
    cfg = {
        "path": str(merged_root),
        "train": "train",
        "val": "val",
        "nc": 2,
        "names": ["hand", "tool"],
    }
    return _write_coco_data_yaml_file(cfg)
