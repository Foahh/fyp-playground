#!/usr/bin/env python3
"""Download and prepare finetune datasets for hand / hazardous-tool detection.

Only tools that can cause hand injury (cutters, grinders, knives, saws, etc.)
are kept.  All hazardous-tool labels are remapped to a single class 0 = "tool".

Datasets
--------
ego2hands          Ego2Hands hand segmentation/detection (Box.com, ~2 k eval images)
construction_tools Zenodo small construction-tool detection (hazard subset of 12 classes)
metu_alet          METU-ALET tool detection in the wild (hazard subset of 49 classes)

Environment
-----------
Use the ``yolo`` conda env (same as COCO prep / training)::

    python project.py conda-yolo
    conda activate yolo    # or $ST_YOLO_ENV

Usage
-----
python load_finetune_data.py                          # all datasets
python load_finetune_data.py --dataset ego2hands      # single dataset
python load_finetune_data.py --skip-download          # convert only (pre-downloaded)
python load_finetune_data.py --wget --no-check-certificate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tarfile
import typing
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path(
    os.environ.get("DATASETS_DIR", str(REPO_ROOT / "datasets"))
).expanduser()

# ── Raw (downloaded) and output (YOLO) directories ──────────────────────────
EGO2HANDS_RAW = DATASETS_DIR / "ego2hands_raw"
EGO2HANDS_YOLO = DATASETS_DIR / "ego2hands"

CONSTRUCTION_TOOLS_RAW = DATASETS_DIR / "construction_tools_raw"
CONSTRUCTION_TOOLS_YOLO = DATASETS_DIR / "construction_tools"

METU_ALET_RAW = DATASETS_DIR / "metu_alet_raw"
METU_ALET_YOLO = DATASETS_DIR / "metu_alet"

# ── Download URLs / dataset names ────────────────────────────────────────────
EGO2HANDS_EVAL_URL = "https://app.box.com/s/gd1uywmyeodpwcyyi3dnyfrb8oybe8nx"

DTOOLS_CONSTRUCTION_TOOLS = "Detection of Small Size Construction Tools"
DTOOLS_METU_ALET = "METU-ALET"

# ── Class lists ──────────────────────────────────────────────────────────────
EGO2HANDS_CLASSES = ["hand"]
OUTPUT_TOOL_CLASSES = ["tool"]

# A class name containing any of these tokens (case-insensitive) is kept as
# hazardous.  Used for both Construction Tools and METU-ALET filtering.
HAZARD_KEYWORDS: set[str] = {
    "axe", "blade", "chisel", "cleaver", "cutter", "drill",
    "grinder", "hatchet", "knife", "machete", "plier", "saw",
    "scissor", "scythe", "shear", "sickle", "snip", "staple", "tacker",
}

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
    """Download a file with resume support. Uses aria2c by default, wget as fallback."""
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
            "wget", "-c", "--tries=5", "--timeout=60",
            "-O", str(dest_file), url,
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
            "--dir", str(dest_dir),
            "--out", filename,
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
            "Delete it and re-run to start fresh, or re-run to resume."
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
    os.symlink(src.resolve(), dst)


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

    zip_path = EGO2HANDS_RAW / "ego2hands_eval.zip"
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

            _safe_symlink(img_path, img_out / img_path.name)
            _write_yolo_label(lbl_out / (img_path.stem + ".txt"), boxes)
            written += 1

    _write_classes_txt(out_dir, EGO2HANDS_CLASSES)
    print(
        f"  Ego2Hands YOLO dataset written to {out_dir}: "
        f"{written} images (train={len(train_items)}, val={len(val_items)})"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset Ninja (Supervisely format) — shared download + conversion
#  Optional: ``pip install dataset-tools`` if using Ninja downloads.
# ═══════════════════════════════════════════════════════════════════════════════

def _is_hazard_class(name: str) -> bool:
    """Return True if *name* matches any hazard keyword (substring, case-insensitive)."""
    lower = name.lower()
    return any(kw in lower for kw in HAZARD_KEYWORDS)


def _dtools_download(dataset_name: str, dst_dir: Path) -> Path:
    """Download a dataset via ``dataset-tools`` (Dataset Ninja).

    Returns the Supervisely root directory (the one containing ``meta.json``).
    """
    import dataset_tools as dtools

    meta_files = list(dst_dir.rglob("meta.json"))
    if meta_files:
        print(f"  Already downloaded: {meta_files[0].parent}")
        return meta_files[0].parent

    dst_dir.mkdir(parents=True, exist_ok=True)
    dtools.download(dataset=dataset_name, dst_dir=str(dst_dir))

    meta_files = list(dst_dir.rglob("meta.json"))
    if not meta_files:
        raise RuntimeError(
            f"Download succeeded but no meta.json found under {dst_dir}. "
            "Check the dataset-tools output."
        )
    return meta_files[0].parent


def _convert_supervisely(
    raw_dir: Path,
    out_dir: Path,
    class_filter: typing.Callable[[str], bool],
    label: str,
) -> None:
    """Convert a Supervisely-format dataset to YOLO, keeping only filtered classes.

    All kept classes are remapped to class 0 ("tool").
    """
    meta_files = list(raw_dir.rglob("meta.json"))
    if not meta_files:
        print(f"  No meta.json found in {raw_dir} — skipping conversion.")
        return
    sly_root = meta_files[0].parent

    with meta_files[0].open(encoding="utf-8") as f:
        meta = json.load(f)
    all_classes = [c["title"] for c in meta.get("classes", [])]
    kept = sorted(c for c in all_classes if class_filter(c))
    skipped = sorted(c for c in all_classes if not class_filter(c))
    if kept:
        print(f"  Kept hazardous classes : {kept}")
    if skipped:
        print(f"  Skipped non-hazardous  : {skipped}")

    total = 0
    for split_dir in sorted(sly_root.iterdir()):
        if not split_dir.is_dir():
            continue
        img_dir = split_dir / "img"
        ann_dir = split_dir / "ann"
        if not img_dir.exists() or not ann_dir.exists():
            continue

        split_name = split_dir.name
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        written = 0
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            ann_path = ann_dir / (img_path.name + ".json")
            if not ann_path.exists():
                continue

            with ann_path.open(encoding="utf-8") as f:
                ann = json.load(f)

            img_w = ann["size"]["width"]
            img_h = ann["size"]["height"]
            if img_w == 0 or img_h == 0:
                continue

            boxes: list[list[float]] = []
            for obj in ann.get("objects", []):
                if obj.get("geometryType") != "rectangle":
                    continue
                if not class_filter(obj.get("classTitle", "")):
                    continue
                pts = obj["points"]["exterior"]
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                w = abs(x2 - x1) / img_w
                h = abs(y2 - y1) / img_h
                boxes.append([0.0, cx, cy, w, h])

            if not boxes:
                continue

            _safe_symlink(img_path, img_out / img_path.name)
            _write_yolo_label(lbl_out / (img_path.stem + ".txt"), boxes)
            written += 1

        if written:
            print(f"    {split_name}: {written} images")
        total += written

    _write_classes_txt(out_dir, OUTPUT_TOOL_CLASSES)
    print(f"  {label} YOLO dataset written to {out_dir}: {total} images")


# ═══════════════════════════════════════════════════════════════════════════════
#  Construction Tools  —  hazardous subset of 12-class tool detection
# ═══════════════════════════════════════════════════════════════════════════════

def download_construction_tools(**_dl_kwargs: object) -> None:
    """Download construction-tool dataset via Dataset Ninja (dataset-tools)."""
    _dtools_download(DTOOLS_CONSTRUCTION_TOOLS, CONSTRUCTION_TOOLS_RAW)


def convert_construction_tools() -> None:
    """Convert Construction Tools (Supervisely) to YOLO, hazardous classes only."""
    _convert_supervisely(
        CONSTRUCTION_TOOLS_RAW, CONSTRUCTION_TOOLS_YOLO,
        _is_hazard_class, "Construction Tools",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  METU-ALET  —  hazardous tool detection (filtered from 49 classes)
# ═══════════════════════════════════════════════════════════════════════════════

def download_metu_alet(**_dl_kwargs: object) -> None:
    """Download METU-ALET dataset via Dataset Ninja (dataset-tools)."""
    _dtools_download(DTOOLS_METU_ALET, METU_ALET_RAW)


def convert_metu_alet() -> None:
    """Convert METU-ALET (Supervisely) to YOLO, hazardous classes only."""
    _convert_supervisely(
        METU_ALET_RAW, METU_ALET_YOLO,
        _is_hazard_class, "METU-ALET",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Merge — combine per-dataset YOLO outputs into fyp_merged/ for ST pipeline
# ═══════════════════════════════════════════════════════════════════════════════

MERGED_DIR = DATASETS_DIR / "fyp_merged"
MERGED_CLASSES = ["hand", "tool"]

# hand=0, tool=1
_DATASET_REMAP: list[tuple[Path, str, dict[int, int]]] = [
    (EGO2HANDS_YOLO, "eh", {0: 0}),
    (CONSTRUCTION_TOOLS_YOLO, "ct", {0: 1}),
    (METU_ALET_YOLO, "al", {0: 1}),
]


def _ensure_jpg(src: Path, dst: Path) -> None:
    """Write *dst* as JPEG.  Symlinks if already JPEG, converts otherwise."""
    if dst.exists() or dst.is_symlink():
        return
    real_src = src.resolve() if src.is_symlink() else src
    if real_src.suffix.lower() in (".jpg", ".jpeg"):
        os.symlink(real_src, dst)
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
        stem = f"{prefix}_{img_path.stem}"
        dst_jpg = out_dir / (stem + ".jpg")
        dst_txt = out_dir / (stem + ".txt")
        _ensure_jpg(img_path, dst_jpg)
        if _remap_label(lbl_src, dst_txt, cls_map):
            n += 1
    counts[split] = n
    return n


def merge_for_finetune() -> None:
    """Combine all per-dataset YOLO outputs into ``fyp_merged/{train,val,test}``."""
    all_items: list[tuple[Path, Path, str, dict[int, int]]] = []

    for ds_dir, prefix, cls_map in _DATASET_REMAP:
        for split in ("train", "val"):
            img_dir = ds_dir / "images" / split
            lbl_dir = ds_dir / "labels" / split
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                lbl_src = lbl_dir / (img_path.stem + ".txt")
                if lbl_src.exists():
                    all_items.append((img_path, lbl_src, prefix, cls_map))

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
    for name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        _populate_split(name, items, counts)

    total = sum(counts.values())
    print(f"\n  Merged dataset written to {MERGED_DIR}")
    for k in ("train", "val", "test"):
        print(f"    {k}: {counts.get(k, 0)}")
    print(f"    total: {total}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_CHOICES = ["ego2hands", "construction_tools", "metu_alet", "all"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare finetune datasets "
        "(Ego2Hands, Construction Tools, METU-ALET)",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="all",
        help="Dataset(s) to process (default: all)",
    )
    parser.add_argument(
        "--wget",
        action="store_true",
        help="Use wget instead of aria2c for downloads",
    )
    parser.add_argument(
        "--ca-certificate",
        metavar="FILE",
        help="Path to CA certificate bundle",
    )
    parser.add_argument(
        "--no-check-certificate",
        action="store_true",
        help="Disable server certificate verification",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step; only run conversion on pre-downloaded data",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the final merge into fyp_merged/",
    )
    args = parser.parse_args()

    ego2hands_dl_kwargs = dict(
        use_wget=args.wget,
        ca_certificate=args.ca_certificate,
        check_certificate=not args.no_check_certificate,
    )

    ds = args.dataset

    if ds in ("ego2hands", "all"):
        print("\n=== Ego2Hands (hand detection) ===")
        if not args.skip_download:
            download_ego2hands(**ego2hands_dl_kwargs)
        convert_ego2hands()

    if ds in ("construction_tools", "all"):
        print("\n=== Construction Tools — Dataset Ninja ===")
        if not args.skip_download:
            download_construction_tools()
        convert_construction_tools()

    if ds in ("metu_alet", "all"):
        print("\n=== METU-ALET — Dataset Ninja ===")
        if not args.skip_download:
            download_metu_alet()
        convert_metu_alet()

    if not args.skip_merge:
        print("\n=== Merging into fyp_merged/ ===")
        merge_for_finetune()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
