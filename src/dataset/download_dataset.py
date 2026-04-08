"""Download and prepare COCO datasets (YOLO layout, person subset, TFS exports).

Run in the same ``fyp-ml`` conda env as training / quantization::

    ./project.py setup-env-ml
    conda activate fyp-ml    # or $ST_YOLO_ENV
"""

import json
import os
import subprocess
import zipfile
from pathlib import Path

import typer

from ..common.paths import get_datasets_dir

DATASETS_DIR = get_datasets_dir()
ZIPS_DIR_DEFAULT = DATASETS_DIR / "_zips"
DEST = DATASETS_DIR / "coco"
PERSON_YOLO_DIR = DATASETS_DIR / "coco_2017_person"

COCO_DOWNLOADS: list[tuple[str, Path]] = [
    (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip",
        DEST.parent,
    ),
    ("http://images.cocodataset.org/zips/train2017.zip", DEST / "images"),
    ("http://images.cocodataset.org/zips/val2017.zip", DEST / "images"),
]


def _resumable_download(
    url: str,
    dest_dir: Path,
    use_wget: bool = False,
    ca_certificate: str | None = None,
    check_certificate: bool = True,
) -> Path:
    """Download a file with resume support. Uses aria2c by default, wget as fallback."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    dest_file = dest_dir / filename

    if dest_file.exists() and zipfile.is_zipfile(dest_file):
        print(f"  Already downloaded and valid: {dest_file}")
        return dest_file

    if use_wget:
        print(f"  Downloading (wget) {url} -> {dest_file}")
        cmd = ["wget", "-c", "--tries=5", "--timeout=60", "-O", str(dest_file), url]
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
            "Delete it and re-run to start fresh, or re-run to resume."
        )
    return dest_file


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip, skipping files that already exist on disk."""
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


def download_coco(
    use_wget: bool = False,
    ca_certificate: str | None = None,
    check_certificate: bool = True,
    zips_dir: Path = ZIPS_DIR_DEFAULT,
):
    ann_dir = DEST / "annotations"
    train_img_dir = DEST / "images" / "train2017"
    val_img_dir = DEST / "images" / "val2017"

    has_annotations = (ann_dir / "instances_train2017.json").is_file() and (
        ann_dir / "instances_val2017.json"
    ).is_file()
    has_train_images = train_img_dir.is_dir() and any(train_img_dir.iterdir())
    has_val_images = val_img_dir.is_dir() and any(val_img_dir.iterdir())

    if has_annotations and has_train_images and has_val_images:
        print(f"COCO data already present under {DEST}, skipping download.")
        return

    zips_dir.mkdir(parents=True, exist_ok=True)
    for url, dest_dir in COCO_DOWNLOADS:
        name = url.rsplit("/", 1)[-1]

        if name == "coco2017labels.zip" and has_annotations:
            print(f"COCO annotations already present, skipping {name}.")
            continue
        if name == "train2017.zip" and has_train_images:
            print(f"COCO train2017 images already present, skipping {name}.")
            continue
        if name == "val2017.zip" and has_val_images:
            print(f"COCO val2017 images already present, skipping {name}.")
            continue

        zip_path = _resumable_download(
            url,
            zips_dir,
            use_wget=use_wget,
            ca_certificate=ca_certificate,
            check_certificate=check_certificate,
        )
        _extract_zip(zip_path, dest_dir)


def _resolve_coco_root() -> Path:
    from ..common.paths import resolve_coco_root

    return resolve_coco_root()


def generate_person_annotations():
    """Filter COCO val2017 annotations to person-only using pycocotools."""
    from pycocotools.coco import COCO

    coco_root = _resolve_coco_root()
    ann_val_path = coco_root / "annotations" / "instances_val2017.json"
    out_path = coco_root / "annotations" / "instances_val2017_person.json"

    coco = COCO(str(ann_val_path))
    person_cat_ids = coco.getCatIds(catNms=["person"])
    person_img_ids = coco.getImgIds(catIds=person_cat_ids)
    person_ann_ids = coco.getAnnIds(imgIds=person_img_ids, catIds=person_cat_ids)

    out = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": coco.loadImgs(person_img_ids),
        "annotations": coco.loadAnns(person_ann_ids),
        "categories": coco.loadCats(person_cat_ids),
    }

    with open(out_path, "w") as f:
        json.dump(out, f)

    print(
        f"Written {out_path} ({len(out['images'])} images, {len(out['annotations'])} annotations)"
    )


def _coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] (top-left) to normalized [cx, cy, w, h]."""
    x, y, w, h = bbox
    return [(x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h]


def generate_tfs_dataset(
    category_names: list[str], out_dir: Path, max_detections: int = 100
):
    import tensorflow as tf
    from pycocotools.coco import COCO

    coco_root = _resolve_coco_root()
    ann_val_path = coco_root / "annotations" / "instances_val2017.json"
    val_images_dir = coco_root / "images" / "val2017"
    coco = COCO(str(ann_val_path))
    cat_ids = coco.getCatIds(catNms=category_names)
    cat_id_to_idx = {cid: i for i, cid in enumerate(sorted(cat_ids))}

    img_ids = sorted(
        {img_id for cat_id in cat_ids for img_id in coco.getImgIds(catIds=[cat_id])}
    )
    imgs = coco.loadImgs(img_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    discarded = 0

    for img_info in imgs:
        ann_ids = coco.getAnnIds(imgIds=img_info["id"], catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        if not anns:
            continue
        if len(anns) > max_detections:
            discarded += 1
            continue

        labels = [
            [float(cat_id_to_idx[a["category_id"]])]
            + _coco_bbox_to_yolo(a["bbox"], img_info["width"], img_info["height"])
            for a in anns
        ]
        labels += [[0.0] * 5] * (max_detections - len(labels))

        src_img = val_images_dir / img_info["file_name"]
        dst_img = out_dir / img_info["file_name"]
        _safe_symlink(src_img, dst_img)

        stem = Path(img_info["file_name"]).stem
        tfs_path = out_dir / (stem + ".tfs")
        data = tf.io.serialize_tensor(tf.convert_to_tensor(labels, dtype=tf.float32))
        tf.io.write_file(str(tfs_path), data)

        written += 1

    print(f"TFS dataset written to {out_dir}: {written} images, {discarded} discarded")


def _all_coco_category_names() -> list[str]:
    """Load all 80 COCO category names from the annotation file."""
    from pycocotools.coco import COCO

    coco_root = _resolve_coco_root()
    ann_val_path = coco_root / "annotations" / "instances_val2017.json"
    coco = COCO(str(ann_val_path))
    return [c["name"] for c in coco.loadCats(coco.getCatIds())]


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    os.symlink(rel, dst)


def _write_person_yolo_split(
    src_labels_dir: Path, images_dir: Path, split_name: str, out_root: Path
) -> int:
    """
    Materialize person-only YOLO split using symlinks for images.
    """
    images_out = out_root / "images" / split_name
    labels_out = out_root / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    list_path = out_root / f"{split_name}2017.txt"
    written = 0
    with list_path.open("w", encoding="utf-8") as list_f:
        for src_label in sorted(src_labels_dir.glob("*.txt")):
            person_lines: list[str] = []
            with src_label.open("r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    if parts[0] != "0":
                        continue
                    person_lines.append(" ".join(parts))

            if not person_lines:
                continue

            stem = src_label.stem
            src_img = images_dir / f"{stem}.jpg"
            dst_img = images_out / f"{stem}.jpg"
            if not src_img.exists():
                src_img = images_dir / f"{stem}.png"
                dst_img = images_out / f"{stem}.png"
            if not src_img.exists():
                continue

            _safe_symlink(src_img, dst_img)

            label_path = labels_out / src_label.name
            with label_path.open("w", encoding="utf-8") as lf:
                for pl in person_lines:
                    lf.write(f"{pl}\n")

            # Keep symlink path inside coco_2017_person, do not resolve to source COCO path.
            list_f.write(f"{dst_img}\n")
            written += 1
    return written


def generate_person_yolo_dataset() -> None:
    coco_root = _resolve_coco_root()
    src_train_labels_dir = coco_root / "labels" / "train2017"
    src_val_labels_dir = coco_root / "labels" / "val2017"
    train_images_dir = coco_root / "images" / "train2017"
    val_images_dir = coco_root / "images" / "val2017"

    train_count = _write_person_yolo_split(
        src_train_labels_dir, train_images_dir, "train", PERSON_YOLO_DIR
    )
    val_count = _write_person_yolo_split(
        src_val_labels_dir, val_images_dir, "val", PERSON_YOLO_DIR
    )
    print(
        f"YOLO person dataset written to {PERSON_YOLO_DIR}: "
        f"train={train_count}, val={val_count}"
    )


app = typer.Typer()


@app.command()
def main(
    wget: bool = typer.Option(
        False, help="Use wget instead of aria2c for downloads (single-connection)"
    ),
    ca_certificate: str | None = typer.Option(
        None, help="Path to CA certificate bundle (forwarded to aria2c/wget)"
    ),
    no_check_certificate: bool = typer.Option(
        False, help="Disable server certificate verification (forwarded to aria2c/wget)"
    ),
    zips_dir: Path = typer.Option(
        ZIPS_DIR_DEFAULT,
        help="Directory to store downloaded zip files (can be deleted later)",
    ),
):
    download_coco(
        use_wget=wget,
        ca_certificate=ca_certificate,
        check_certificate=not no_check_certificate,
        zips_dir=zips_dir,
    )
    generate_person_yolo_dataset()
    generate_person_annotations()

    tfs_person_dir = DATASETS_DIR / "coco_2017_person" / "test"
    if not tfs_person_dir.exists():
        generate_tfs_dataset(["person"], tfs_person_dir)

    tfs_80_dir = DATASETS_DIR / "coco_2017_80_classes" / "test"
    if not tfs_80_dir.exists():
        generate_tfs_dataset(_all_coco_category_names(), tfs_80_dir)


if __name__ == "__main__":
    app()
