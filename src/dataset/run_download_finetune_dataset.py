#!/usr/bin/env python3
"""Download and prepare finetune datasets for hand / hazardous-tool detection.

Narrow hazard definition: sharp contact / cutting (incl. snips, hand planes),
thermal burns (soldering iron), and staple guns — not crush-first tools (hammers,
wrenches, clamps, etc.).  All kept tool labels are remapped to class 0 = "tool".

Datasets
--------
ego2hands          Ego2Hands hand segmentation/detection (Box.com, ~2 k eval images)
construction_tools Zenodo small construction-tool detection (hazard subset of 12 classes)

Environment
-----------
Use the ``fyp-ml`` conda env (same as COCO prep / training)::

    ./project.py setup-env-ml
    conda activate fyp-ml    # or ``FYP_YOLO_ENV``

Usage
-----
./project.py download-finetune                                           # all datasets
./project.py download-finetune -- --dataset ego2hands                      # single dataset
./project.py download-finetune -- --skip-download                          # convert only (pre-downloaded)
./project.py download-finetune -- --wget --no-check-certificate
./project.py download-finetune -- --clear   # remove merged + dataset outputs (see --help)

Inspect *converted* labels (not ``*_raw``)::

    ./project.py view-finetune-labels -- --preset ego2hands
    ./project.py view-finetune-labels -- --preset construction_tools
    ./project.py view-finetune-labels -- --preset fyp_merged --filename-prefix ct_
    ./project.py view-finetune-labels -- --preset ego2hands --mask
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import typer
from PIL import Image

from ..common.paths import get_datasets_dir

DATASETS_DIR = get_datasets_dir()
ZIPS_DIR_DEFAULT = DATASETS_DIR / "_zips"

# ── Raw (downloaded) and output (YOLO) directories ──────────────────────────
EGO2HANDS_RAW = DATASETS_DIR / "ego2hands_raw"
EGO2HANDS_YOLO = DATASETS_DIR / "ego2hands"

CONSTRUCTION_TOOLS_RAW = DATASETS_DIR / "construction_tools_raw"
CONSTRUCTION_TOOLS_YOLO = DATASETS_DIR / "construction_tools"

# ── Download URLs ────────────────────────────────────────────────────────────
EGO2HANDS_EVAL_URL = "https://app.box.com/s/gd1uywmyeodpwcyyi3dnyfrb8oybe8nx"

ZENODO_DOWNLOADS: list[tuple[str, str]] = [
    (
        f"https://zenodo.org/records/6530106/files/DATA{i}.zip?download=1",
        f"DATA{i}.zip",
    )
    for i in range(1, 5)
]

# ── Class lists ──────────────────────────────────────────────────────────────
EGO2HANDS_CLASSES = ["hand"]
OUTPUT_TOOL_CLASSES = ["tool"]

_ZENODO_ALL_CLASSES = [
    "bucket",
    "cutter",
    "drill",
    "grinder",
    "hammer",
    "knife",
    "saw",
    "shovel",
    "spanner",
    "tacker",
    "trowel",
    "wrench",
]

ZENODO_HAZARD_IDS: set[int] = {
    _ZENODO_ALL_CLASSES.index(n) for n in ("knife",)
}
ZENODO_HAZARD_NAMES: set[str] = {_ZENODO_ALL_CLASSES[i] for i in ZENODO_HAZARD_IDS}

VAL_RATIO = 0.2
RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _resumable_download(
    url: str,
    dest_dir: Path,
    filename: str | None = None,
    *,
    use_wget: bool = False,
    ca_certificate: str | None = None,
    check_certificate: bool = True,
) -> Path:
    """Download a file with resume support via aria2c; wget does not resume partials."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.rsplit("/", 1)[-1].split("?")[0]
    dest_file = dest_dir / filename

    if dest_file.exists() and zipfile.is_zipfile(dest_file):
        print(f"  Already downloaded and valid: {dest_file}")
        return dest_file

    if use_wget:
        print(f"  Downloading (wget) {url} -> {dest_file}")
        cmd: list[str] = [
            "wget",
            "--tries=5",
            "--timeout=60",
            "-O",
            str(dest_file),
            url,
        ]
        if ca_certificate:
            cmd[1:1] = [f"--ca-certificate={ca_certificate}"]
        if not check_certificate:
            cmd[1:1] = ["--no-check-certificate"]
    else:
        print(f"  Downloading (aria2c) {url} -> {dest_file}")
        cmd = [
            "aria2c",
            "--continue=true",
            "--max-connection-per-server=8",
            "--split=8",
            "--min-split-size=10M",
            "--max-tries=5",
            "--timeout=60",
            "--dir",
            str(dest_dir),
            "--out",
            filename,
            url,
        ]
        if ca_certificate:
            cmd[1:1] = [f"--ca-certificate={ca_certificate}"]
        if not check_certificate:
            cmd[1:1] = ["--check-certificate=false"]

    subprocess.run(cmd, check=True)

    if not dest_file.exists() or not zipfile.is_zipfile(dest_file):
        raise RuntimeError(
            f"Download appears incomplete or corrupt: {dest_file}. "
            "Delete it and re-run. (aria2c can resume partial downloads; wget does not.)"
        )
    return dest_file


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip archive, skipping files that already exist."""
    print(f"  Extracting {zip_path.name} -> {extract_to}")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        existing = 0
        for member in members:
            target = extract_to / member
            if target.exists():
                existing += 1
                continue
            zf.extract(member, extract_to)
        if existing:
            print(f"    Skipped {existing}/{len(members)} already-extracted entries")


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    os.symlink(rel, dst)


def _stable_unique_stem(path: Path, *, root: Path | None = None) -> str:
    """Create a deterministic, collision-resistant stem from a file path."""
    resolved = path.resolve()
    base_root = DATASETS_DIR if root is None else root
    try:
        rel = resolved.relative_to(base_root.resolve())
    except ValueError:
        rel = Path(path.name)

    base = rel.with_suffix("").as_posix()

    safe = base.replace("/", "__").replace("\\", "__").replace(":", "_").lstrip("._-")
    if not safe:
        safe = resolved.stem
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"{safe}__{digest}"


def _train_val_split(
    items: list, val_ratio: float = VAL_RATIO, seed: int = RANDOM_SEED
) -> tuple[list, list]:
    """Deterministic random split into (train, val)."""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    split_idx = int(len(items) * (1 - val_ratio))
    return items[:split_idx], items[split_idx:]


def _write_classes_txt(out_dir: Path, class_names: list[str]) -> None:
    path = out_dir / "classes.txt"
    with path.open("w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"  Class names written to {path}")


def _collect_images(root: Path) -> list[Path]:
    """Recursively collect image files under *root*."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ego2Hands  —  hand detection from segmentation masks
# ═══════════════════════════════════════════════════════════════════════════════


def _mask_to_yolo_boxes(mask_array: np.ndarray) -> list[list[float]]:
    """Derive YOLO bounding boxes from an Ego2Hands segmentation mask.

    Pixel convention: 0 = background, ~50 = left hand, ~100 = right hand.
    Both hand classes are mapped to class 0 ("hand").
    Returns list of [class_id, cx, cy, w, h] (normalised).
    """
    img_h, img_w = mask_array.shape
    boxes: list[list[float]] = []
    for hand_val in (50, 100):
        hand_mask = mask_array == hand_val
        if not hand_mask.any():
            continue
        rows = np.where(hand_mask.any(axis=1))[0]
        cols = np.where(hand_mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        y_min, y_max = int(rows[0]), int(rows[-1])
        x_min, x_max = int(cols[0]), int(cols[-1])
        bw = (x_max - x_min) / img_w
        bh = (y_max - y_min) / img_h
        if bw < 0.01 or bh < 0.01:
            continue
        cx = (x_min + x_max) / 2.0 / img_w
        cy = (y_min + y_max) / 2.0 / img_h
        boxes.append([0, cx, cy, bw, bh])
    return boxes


def _write_yolo_label(path: Path, boxes: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for box in boxes:
            cls_id = int(box[0])
            coords = " ".join(f"{v:.6f}" for v in box[1:])
            f.write(f"{cls_id} {coords}\n")


def _extract_tars(directory: Path) -> None:
    """Extract all .tar files found in *directory* (non-recursive)."""
    for tar_path in sorted(directory.rglob("*.tar")):
        marker = tar_path.with_suffix(".tar.extracted")
        if marker.exists():
            print(f"  {tar_path.name} already extracted")
            continue
        print(f"  Extracting {tar_path.name} …")
        with tarfile.open(tar_path) as tf:
            tf.extractall(path=tar_path.parent, filter="data")
        marker.touch()


def download_ego2hands(*, use_wget: bool = False, **dl_kwargs: object) -> None:
    """Download Ego2Hands eval set from Box.com.

    Box.com shared-folder links cannot be fetched directly with aria2c / wget.
    The function attempts a download and, if it fails, prints manual instructions.
    """
    eval_dirs = list(EGO2HANDS_RAW.glob("**/eval_seq*_imgs"))
    if eval_dirs:
        print(f"  Ego2Hands eval data already present ({len(eval_dirs)} sequences)")
        return

    tar_files = list(EGO2HANDS_RAW.rglob("*.tar"))
    if tar_files:
        _extract_tars(EGO2HANDS_RAW)
        return

    zips_dir: Path = Path(dl_kwargs.pop("zips_dir", ZIPS_DIR_DEFAULT))
    zips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zips_dir / "ego2hands_eval.zip"
    if zip_path.exists() and zipfile.is_zipfile(zip_path):
        _extract_zip(zip_path, EGO2HANDS_RAW)
        _extract_tars(EGO2HANDS_RAW)
        return

    print(
        "\n  Box.com shared-folder links require a browser to download.\n"
        "  Please download the Ego2Hands eval set manually:\n"
        f"    1. Open  : {EGO2HANDS_EVAL_URL}\n"
        f"    2. Click 'Download' and save the zip to:\n"
        f"             {zip_path}\n"
        "    3. Re-run this script (or use --skip-download after extracting).\n"
    )


def convert_ego2hands() -> None:
    """Convert Ego2Hands eval segmentation masks to YOLO bounding boxes."""
    out_dir = EGO2HANDS_YOLO

    all_items: list[tuple[Path, Path]] = []
    for seq_dir in sorted(EGO2HANDS_RAW.glob("**/eval_seq*_imgs")):
        for img_path in sorted(seq_dir.glob("*.png")):
            stem = img_path.stem
            if any(s in stem for s in ("_seg", "_e_l", "_e_r", "_vis")):
                continue
            mask_path = img_path.with_name(f"{stem}_seg.png")
            if mask_path.exists():
                all_items.append((img_path, mask_path))

    if not all_items:
        print("  No Ego2Hands eval images found — skipping conversion.")
        return

    train_items, val_items = _train_val_split(all_items)
    written = 0

    for split, items in [("train", train_items), ("val", val_items)]:
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in items:
            mask = np.array(Image.open(mask_path).convert("L"))
            boxes = _mask_to_yolo_boxes(mask)
            if not boxes:
                continue

            unique_stem = _stable_unique_stem(img_path, root=EGO2HANDS_RAW)
            _safe_symlink(img_path, img_out / f"{unique_stem}.png")
            _write_yolo_label(lbl_out / f"{unique_stem}.txt", boxes)
            written += 1

    _write_classes_txt(out_dir, EGO2HANDS_CLASSES)
    print(
        f"  Ego2Hands YOLO dataset written to {out_dir}: "
        f"{written} images (train={len(train_items)}, val={len(val_items)})"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Construction Tools (Zenodo)  —  hazardous subset of 12-class YOLO labels
# ═══════════════════════════════════════════════════════════════════════════════


def _all_construction_tools_zenodo_extracted(raw_dir: Path) -> bool:
    """True only when all four Zenodo parts have been extracted (per-zip markers)."""
    return all((raw_dir / f".DATA{i}.zip.extracted").exists() for i in range(1, 5))


def download_construction_tools(*, use_wget: bool = False, **dl_kwargs: object) -> None:
    """Download construction-tool images + YOLO labels from Zenodo (4 zips, ~110 GB)."""
    raw_dir = CONSTRUCTION_TOOLS_RAW
    zips_dir: Path = Path(dl_kwargs.pop("zips_dir", ZIPS_DIR_DEFAULT))
    zips_dir.mkdir(parents=True, exist_ok=True)

    if _all_construction_tools_zenodo_extracted(raw_dir):
        print(
            f"  Construction Tools: all Zenodo archives already extracted under {raw_dir}, "
            "skipping download."
        )
        return

    for url, filename in ZENODO_DOWNLOADS:
        zip_path = zips_dir / filename
        marker = raw_dir / f".{filename}.extracted"
        if marker.exists():
            print(f"  {filename} already extracted")
            continue

        _resumable_download(url, zips_dir, filename, use_wget=use_wget, **dl_kwargs)

        if zip_path.exists() and zipfile.is_zipfile(zip_path):
            _extract_zip(zip_path, raw_dir)
            marker.touch()


def _normalise_zenodo_class_name(raw_name: str) -> str:
    """Normalise Zenodo class names (e.g. 'drill_test' -> 'drill')."""
    name = raw_name.strip().lower()
    for suffix in ("_train", "_test", "_val", "_valid"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _xywh_to_yolo_box(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> list[float] | None:
    """Convert absolute XYWH (top-left anchored) to clipped YOLO XYWH."""
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None

    x1 = max(0.0, min(x, float(img_w)))
    y1 = max(0.0, min(y, float(img_h)))
    x2 = max(0.0, min(x + w, float(img_w)))
    y2 = max(0.0, min(y + h, float(img_h)))
    if x2 <= x1 or y2 <= y1:
        return None

    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    return [0.0, cx, cy, bw, bh]


def _filter_zenodo_labels(
    label_path: Path, *, img_size: tuple[int, int] | None = None
) -> list[list[float]] | None:
    """Read Zenodo labels, keep hazardous classes only, remapped to class 0.

    Supports both:
    - YOLO txt format: ``cls cx cy w h`` (already normalised), and
    - Zenodo CSV-ish format: ``x,y,w,h,class`` or ``class,x,y,w,h``.
    """
    boxes: list[list[float]] = []
    with label_path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # YOLO-style labels: "cls cx cy w h"
            parts = stripped.split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    cls_id = -1
                if cls_id in ZENODO_HAZARD_IDS:
                    boxes.append([0.0] + [float(v) for v in parts[1:5]])
                    continue

            # Zenodo CSV-like labels: "x,y,w,h,class" or "class,x,y,w,h"
            csv_parts = [p.strip() for p in stripped.split(",") if p.strip()]
            if len(csv_parts) != 5 or img_size is None:
                continue

            if _safe_float(csv_parts[0]) is None:
                cls_name = _normalise_zenodo_class_name(csv_parts[0])
                coords = [_safe_float(v) for v in csv_parts[1:5]]
            else:
                cls_name = _normalise_zenodo_class_name(csv_parts[-1])
                coords = [_safe_float(v) for v in csv_parts[0:4]]

            if cls_name not in ZENODO_HAZARD_NAMES or any(v is None for v in coords):
                continue

            x, y, w, h = coords  # type: ignore[misc]
            img_w, img_h = img_size
            box = _xywh_to_yolo_box(x, y, w, h, img_w, img_h)
            if box is not None:
                boxes.append(box)
    return boxes if boxes else None


def convert_construction_tools() -> None:
    """Filter Zenodo labels to hazardous tools only and write train/val splits."""
    raw_dir = CONSTRUCTION_TOOLS_RAW
    out_dir = CONSTRUCTION_TOOLS_YOLO

    all_items: list[tuple[Path, list[list[float]]]] = []
    for img_path in _collect_images(raw_dir):
        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        with Image.open(img_path) as img:
            boxes = _filter_zenodo_labels(label_path, img_size=img.size)
        if boxes is None:
            continue
        all_items.append((img_path, boxes))

    if not all_items:
        print("  No construction-tools images with hazardous labels found — skipping.")
        return

    hazard_names = sorted(_ZENODO_ALL_CLASSES[i] for i in ZENODO_HAZARD_IDS)
    print(f"  Keeping hazardous classes: {hazard_names}")

    train_items, val_items = _train_val_split(all_items)
    written = 0

    for split, items in [("train", train_items), ("val", val_items)]:
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, boxes in items:
            _safe_symlink(img_path, img_out / img_path.name)
            _write_yolo_label(lbl_out / (img_path.stem + ".txt"), boxes)
            written += 1

    _write_classes_txt(out_dir, OUTPUT_TOOL_CLASSES)
    print(
        f"  Construction Tools YOLO dataset written to {out_dir}: "
        f"{written} images (train={len(train_items)}, val={len(val_items)})"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Merge — combine per-dataset YOLO outputs into fyp_merged/ for ST pipeline
# ═══════════════════════════════════════════════════════════════════════════════

MERGED_DIR = DATASETS_DIR / "fyp_merged"
MERGED_CLASSES = ["hand", "tool"]

# Per-dataset YOLO output dirs under DATASETS_DIR (for --clear; *_raw is kept).
_DATASET_YOLO_OUTPUTS: dict[str, list[Path]] = {
    "ego2hands": [EGO2HANDS_YOLO],
    "construction_tools": [CONSTRUCTION_TOOLS_YOLO],
}


def _clear_script_artifacts(*, dataset: str) -> None:
    """Remove merged output and per-dataset YOLO trees before a rebuild (not *_raw or _zips)."""
    paths: list[Path] = [MERGED_DIR]
    if dataset == "all":
        for sub in _DATASET_YOLO_OUTPUTS.values():
            paths.extend(sub)
    else:
        paths.extend(_DATASET_YOLO_OUTPUTS[dataset])
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if p.exists():
            shutil.rmtree(p)
            print(f"  Removed existing {p}")


# hand=0, tool=1  (source layouts: hand datasets use class 0 = hand; tool datasets use 0 = tool)
_DATASET_SPECS: list[tuple[str, Path, str, dict[int, int]]] = [
    ("ego2hands", EGO2HANDS_YOLO, "eh", {0: 0}),
    ("construction_tools", CONSTRUCTION_TOOLS_YOLO, "ct", {0: 1}),
]


def _ensure_jpg(src: Path, dst: Path) -> None:
    """Write *dst* as JPEG.  Symlinks if already JPEG, converts otherwise."""
    if dst.exists() or dst.is_symlink():
        return
    real_src = src.resolve() if src.is_symlink() else src
    if real_src.suffix.lower() in (".jpg", ".jpeg"):
        rel = os.path.relpath(real_src.resolve(), start=dst.parent.resolve())
        os.symlink(rel, dst)
    else:
        Image.open(real_src).convert("RGB").save(dst, "JPEG", quality=95)


def _remap_label(src: Path, dst: Path, cls_map: dict[int, int]) -> bool:
    """Read YOLO label, remap class IDs, write to *dst*.  Returns False if empty."""
    if dst.exists():
        return True
    lines: list[str] = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            new_cls = cls_map.get(int(parts[0]))
            if new_cls is None:
                continue
            lines.append(f"{new_cls} {' '.join(parts[1:5])}\n")
    if not lines:
        return False
    with dst.open("w", encoding="utf-8") as f:
        f.writelines(lines)
    return True


TEST_RATIO = 0.1


def _prepare_merged_dir(*, clear: bool) -> None:
    """If ``fyp_merged`` exists, exit unless *clear* was used (artefacts removed at startup)."""
    if not MERGED_DIR.exists():
        return
    if clear:
        return
    typer.echo(
        f"Error: merged output already exists: {MERGED_DIR}\n"
        "  Re-run with --clear to delete it and rebuild.",
        err=True,
    )
    raise typer.Exit(1)


def _populate_split(
    split: str,
    items: list[tuple[Path, Path, str, dict[int, int]]],
    counts: dict[str, int],
) -> int:
    """Write images + remapped labels for one split. Returns count written."""
    out_dir = MERGED_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path, lbl_src, prefix, cls_map in items:
        stem = f"{prefix}_{_stable_unique_stem(img_path)}"
        dst_jpg = out_dir / (stem + ".jpg")
        dst_txt = out_dir / (stem + ".txt")
        _ensure_jpg(img_path, dst_jpg)
        if _remap_label(lbl_src, dst_txt, cls_map):
            n += 1
    counts[split] = n
    return n


def _dataset_premerge_stats(
    ds_dir: Path, cls_map: dict[int, int]
) -> tuple[int, int, dict[int, int]]:
    """Count image/label pairs and remapped instances for train+val splits."""
    pair_count = 0
    remapped_image_count = 0
    class_counts: dict[int, int] = {}

    for split in ("train", "val"):
        img_dir = ds_dir / "images" / split
        lbl_dir = ds_dir / "labels" / split
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            lbl_src = lbl_dir / (img_path.stem + ".txt")
            if not lbl_src.exists():
                continue

            pair_count += 1
            remapped_in_file = 0
            with lbl_src.open(encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        src_cls = int(parts[0])
                    except ValueError:
                        continue
                    dst_cls = cls_map.get(src_cls)
                    if dst_cls is None:
                        continue
                    class_counts[dst_cls] = class_counts.get(dst_cls, 0) + 1
                    remapped_in_file += 1
            if remapped_in_file > 0:
                remapped_image_count += 1

    return pair_count, remapped_image_count, class_counts


def _print_premerge_stats() -> None:
    """Print per-dataset stats right before merged-dataset creation."""
    print("\n=== Pre-merge dataset stats ===")
    total_pairs = 0
    total_remapped_images = 0
    total_class_counts: dict[int, int] = {}

    for logical, base, _, cls_map in _DATASET_SPECS:
        pair_count, remapped_image_count, class_counts = _dataset_premerge_stats(
            base, cls_map
        )
        total_pairs += pair_count
        total_remapped_images += remapped_image_count
        for cls_id, cnt in class_counts.items():
            total_class_counts[cls_id] = total_class_counts.get(cls_id, 0) + cnt

        cls_parts = [
            f"{MERGED_CLASSES[cls_id]}={class_counts.get(cls_id, 0)}"
            for cls_id in range(len(MERGED_CLASSES))
        ]
        print(
            f"  {logical}: pairs={pair_count}, "
            f"images_with_remapped_boxes={remapped_image_count}, "
            f"instances({', '.join(cls_parts)})"
        )

    total_cls_parts = [
        f"{MERGED_CLASSES[cls_id]}={total_class_counts.get(cls_id, 0)}"
        for cls_id in range(len(MERGED_CLASSES))
    ]
    print(
        f"  total: pairs={total_pairs}, "
        f"images_with_remapped_boxes={total_remapped_images}, "
        f"instances({', '.join(total_cls_parts)})"
    )


def merge_for_finetune(*, balance: bool = False) -> None:
    """Combine all per-dataset YOLO outputs into ``fyp_merged/{train,val,test}``.

    When *balance* is True, hand-only images are subsampled to match the total
    number of tool-only images, reducing class imbalance.
    """
    hand_items: list[tuple[Path, Path, str, dict[int, int]]] = []
    tool_items: list[tuple[Path, Path, str, dict[int, int]]] = []

    for logical, base, prefix, cls_map in _DATASET_SPECS:
        is_hand_pool = logical == "ego2hands"
        for split in ("train", "val"):
            img_dir = base / "images" / split
            lbl_dir = base / "labels" / split
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                lbl_src = lbl_dir / (img_path.stem + ".txt")
                if lbl_src.exists():
                    item = (img_path, lbl_src, prefix, cls_map)
                    if is_hand_pool:
                        hand_items.append(item)
                    else:
                        tool_items.append(item)

    if balance and len(hand_items) > len(tool_items):
        original = len(hand_items)
        rng = random.Random(RANDOM_SEED)
        hand_items = rng.sample(hand_items, len(tool_items))
        print(
            f"  Balanced: subsampled hand-only images {original} → {len(hand_items)} "
            f"(to match {len(tool_items)} tool images)"
        )

    all_items = hand_items + tool_items

    if not all_items:
        print("  No data to merge.")
        return

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_items)
    n_test = max(1, int(len(all_items) * TEST_RATIO))
    n_val = max(1, int(len(all_items) * VAL_RATIO))
    test_items = all_items[:n_test]
    val_items = all_items[n_test : n_test + n_val]
    train_items = all_items[n_test + n_val :]

    counts: dict[str, int] = {}
    for name, items in [
        ("train", train_items),
        ("val", val_items),
        ("test", test_items),
    ]:
        _populate_split(name, items, counts)

    total = sum(counts.values())
    print(f"\n  Merged dataset written to {MERGED_DIR}")
    for k in ("train", "val", "test"):
        print(f"    {k}: {counts.get(k, 0)}")
    print(f"    total: {total}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

app = typer.Typer()


@app.command()
def main(
    dataset: str = typer.Option(
        "all",
        help="Dataset(s) to process (ego2hands, construction_tools, all)",
    ),
    wget: bool = typer.Option(False, help="Use wget instead of aria2c for downloads"),
    ca_certificate: str | None = typer.Option(
        None, help="Path to CA certificate bundle"
    ),
    no_check_certificate: bool = typer.Option(
        False, help="Disable server certificate verification"
    ),
    skip_download: bool = typer.Option(
        False, help="Skip download step; only run conversion on pre-downloaded data"
    ),
    skip_merge: bool = typer.Option(
        False, help="Skip the final merge into fyp_merged/"
    ),
    zips_dir: Path = typer.Option(
        ZIPS_DIR_DEFAULT,
        help="Directory to store downloaded zip files (can be deleted later)",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Delete fyp_merged/ and this run's dataset YOLO dirs (not *_raw, not zips cache) — before rebuild",
    ),
    balance: bool = typer.Option(
        True,
        "--balance",
        help="Subsample hand-only images to match tool image count, reducing class imbalance",
    ),
) -> int:
    valid_datasets = ["ego2hands", "construction_tools", "all"]
    if dataset not in valid_datasets:
        typer.echo(
            f"Error: dataset must be one of [{', '.join(valid_datasets)}]",
            err=True,
        )
        raise typer.Exit(1)

    dl_kwargs = dict(
        use_wget=wget,
        ca_certificate=ca_certificate,
        check_certificate=not no_check_certificate,
        zips_dir=zips_dir,
    )

    if clear:
        print("\n=== Clearing previous outputs ===")
        _clear_script_artifacts(dataset=dataset)

    if dataset in ("ego2hands", "all"):
        print("\n=== Ego2Hands (hand detection) ===")
        if not skip_download:
            download_ego2hands(**dl_kwargs)
        convert_ego2hands()

    if dataset in ("construction_tools", "all"):
        print("\n=== Construction Tools — Zenodo ===")
        if not skip_download:
            download_construction_tools(**dl_kwargs)
        convert_construction_tools()

    if not skip_merge:
        print("\n=== Merging into fyp_merged/ ===")
        _print_premerge_stats()
        _prepare_merged_dir(clear=clear)
        merge_for_finetune(balance=balance)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
