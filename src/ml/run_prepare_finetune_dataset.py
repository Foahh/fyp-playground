#!/usr/bin/env python3
"""Prepare an ST Model Zoo object-detection dataset for finetuning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OD_ROOT = ROOT / "external" / "stm32ai-modelzoo-services" / "object_detection"

CONVERTER = OD_ROOT / "datasets" / "dataset_converter" / "converter.py"
CREATE_TFS = OD_ROOT / "datasets" / "dataset_create_tfs" / "dataset_create_tfs.py"
ANALYSIS = OD_ROOT / "datasets" / "dataset_analysis" / "dataset_analysis.py"


def _load_config(config_file: Path) -> dict:
    with config_file.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_repo_path(path_value: str) -> Path:
    return (ROOT / path_value).resolve()


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }


def _convert_split_yolo_to_coco(
    images_dir: Path,
    output_json: Path,
    class_names: list[str],
) -> tuple[int, int]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected images directory not found: {images_dir}")

    image_id = 1
    annotation_id = 1
    images: list[dict] = []
    annotations: list[dict] = []

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or not _is_image_file(image_path):
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = image_path.with_suffix(".txt")
        if label_path.is_file():
            with label_path.open(encoding="utf-8") as f:
                for raw_line in f:
                    parts = raw_line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= len(class_names):
                        continue

                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height
                    x_min = x_center - (box_width / 2.0)
                    y_min = y_center - (box_height / 2.0)

                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + 1,
                            "bbox": [x_min, y_min, box_width, box_height],
                            "area": box_width * box_height,
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )
                    annotation_id += 1

        image_id += 1

    categories = [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    return len(images), len(annotations)


def _ensure_coco_annotations(config_file: Path, force: bool = False) -> None:
    """Generate COCO JSON annotations from YOLO txt labels for fyp_merged-like layouts."""
    data = _load_config(config_file)
    ds = data.get("dataset") or {}
    if ds.get("format") != "coco":
        return

    class_names = ds.get("class_names") or []
    if not class_names:
        raise ValueError(
            "dataset.class_names is required for COCO annotation generation"
        )

    train_images_rel = ds.get("train_images_path")
    val_images_rel = ds.get("val_images_path")
    test_images_rel = ds.get("test_images_path")
    train_ann_rel = ds.get("train_annotations_path")
    val_ann_rel = ds.get("val_annotations_path")
    test_ann_rel = ds.get("test_annotations_path")
    if (
        not train_images_rel
        or not val_images_rel
        or not train_ann_rel
        or not val_ann_rel
    ):
        raise ValueError(
            "dataset.train_images_path, dataset.val_images_path, dataset.train_annotations_path, "
            "and dataset.val_annotations_path are required for COCO annotation generation"
        )
    if bool(test_images_rel) != bool(test_ann_rel):
        raise ValueError(
            "dataset.test_images_path and dataset.test_annotations_path must either both be set "
            "or both be omitted for COCO annotation generation"
        )

    train_images_dir = _resolve_repo_path(train_images_rel)
    val_images_dir = _resolve_repo_path(val_images_rel)
    train_json = _resolve_repo_path(train_ann_rel)
    val_json = _resolve_repo_path(val_ann_rel)
    test_images_dir = _resolve_repo_path(test_images_rel) if test_images_rel else None
    test_json = _resolve_repo_path(test_ann_rel) if test_ann_rel else None

    required_jsons = [train_json, val_json]
    if test_json is not None:
        required_jsons.append(test_json)
    if not force and all(path.is_file() for path in required_jsons):
        print(
            "[prepare-finetune-dataset] COCO annotation JSON files already exist; skipping regeneration."
        )
        return

    train_counts = _convert_split_yolo_to_coco(
        train_images_dir, train_json, class_names
    )
    val_counts = _convert_split_yolo_to_coco(val_images_dir, val_json, class_names)
    test_counts: tuple[int, int] | None = None
    if test_images_dir is not None and test_json is not None:
        test_counts = _convert_split_yolo_to_coco(
            test_images_dir, test_json, class_names
        )

    log_parts = [
        "[prepare-finetune-dataset] Wrote COCO annotations "
        f"train={train_json} ({train_counts[0]} images, {train_counts[1]} boxes), "
        f"val={val_json} ({val_counts[0]} images, {val_counts[1]} boxes)"
    ]
    if test_counts is not None and test_json is not None:
        log_parts.append(
            f", test={test_json} ({test_counts[0]} images, {test_counts[1]} boxes)"
        )
    print("".join(log_parts) + ".")


def _skip_st_converter(config_file: Path) -> bool:
    """Skip ST converter when input already uses COCO or VOC xml folder is unavailable."""
    data = _load_config(config_file)
    ds = data.get("dataset") or {}
    if ds.get("format") == "coco":
        print("[prepare-finetune-dataset] dataset.format=coco; skipping converter.py.")
        return True

    # Keep existing VOC fallback behavior.
    if ds.get("format") != "pascal_voc_format":
        return False
    pvf = data.get("pascal_voc_format") or {}
    xml_rel = pvf.get("xml_files_path")
    if not xml_rel:
        return False
    xml_dir = (ROOT / xml_rel).resolve()
    if xml_dir.is_dir():
        return False
    print(
        f"[prepare-finetune-dataset] Pascal VOC xml folder missing ({xml_dir}); "
        "skipping converter.py (use YOLO .txt labels already under dataset paths). "
        "To convert from VOC, create that folder with annotations or fix xml_files_path in the config.",
        file=sys.stderr,
    )
    return True


def _hydra_config_parts(config_file: Path) -> tuple[str, str]:
    resolved = config_file.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    return str(resolved.parent), resolved.stem


def _run(script: Path, config_file: Path, overrides: list[str]) -> None:
    if not script.is_file():
        raise FileNotFoundError(f"Expected script not found: {script}")

    config_path, config_name = _hydra_config_parts(config_file)
    cmd = [
        sys.executable,
        str(script),
        f"--config-path={config_path}",
        f"--config-name={config_name}",
        *overrides,
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


app = typer.Typer()


@app.command()
def main(
    config: Path = typer.Option(
        ..., help="Path to the dataset config YAML used by ST Model Zoo dataset tools."
    ),
    skip_convert: bool = typer.Option(
        False, help="Skip converter.py and only run dataset_create_tfs.py."
    ),
    force_coco: bool = typer.Option(
        False, help="Force regeneration of COCO JSON annotations from YOLO txt labels."
    ),
    analyze: bool = typer.Option(
        False, help="Run dataset_analysis.py after TFS creation."
    ),
    override: list[str] = typer.Option(
        [],
        help="Hydra override passed through to each invoked dataset tool. Repeat as needed.",
    ),
) -> int:
    overrides = list(override)

    _ensure_coco_annotations(config, force=force_coco)

    if not skip_convert and not _skip_st_converter(config):
        _run(CONVERTER, config, overrides)

    _run(CREATE_TFS, config, overrides)

    if analyze:
        _run(ANALYSIS, config, overrides)

    print("Dataset preparation done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
