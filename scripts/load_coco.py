import json
import os
from pathlib import Path

import tensorflow as tf
from pycocotools.coco import COCO
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path(
    os.environ.get("DATASETS_DIR", str(REPO_ROOT / "datasets"))
).expanduser()
DEST = DATASETS_DIR / "coco"
PERSON_YOLO_DIR = DATASETS_DIR / "coco_2017_person"


def download_coco():
    urls = [ASSETS_URL + "/coco2017labels.zip"]
    download(urls, dir=DEST.parent, exist_ok=True)
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
    ]
    download(urls, dir=DEST / "images", threads=8)


def _resolve_coco_root() -> Path:
    candidates = [
        DEST,
        (Path.home() / "datasets" / "coco"),
    ]
    for root in candidates:
        if (root / "annotations" / "instances_val2017.json").is_file():
            return root
    raise FileNotFoundError(
        "Unable to locate COCO root with annotations. Checked: "
        + ", ".join(str(c / "annotations" / "instances_val2017.json") for c in candidates)
    )


def generate_person_annotations():
    """Filter COCO val2017 annotations to person-only using pycocotools."""
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
        if not dst_img.exists():
            os.symlink(src_img.resolve(), dst_img)

        stem = Path(img_info["file_name"]).stem
        tfs_path = out_dir / (stem + ".tfs")
        data = tf.io.serialize_tensor(tf.convert_to_tensor(labels, dtype=tf.float32))
        tf.io.write_file(str(tfs_path), data)

        written += 1

    print(f"TFS dataset written to {out_dir}: {written} images, {discarded} discarded")


def _all_coco_category_names() -> list[str]:
    """Load all 80 COCO category names from the annotation file."""
    coco_root = _resolve_coco_root()
    ann_val_path = coco_root / "annotations" / "instances_val2017.json"
    coco = COCO(str(ann_val_path))
    return [c["name"] for c in coco.loadCats(coco.getCatIds())]


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


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

            list_f.write(f"{dst_img.resolve()}\n")
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


def main():
    download_coco()
    generate_person_yolo_dataset()
    generate_person_annotations()

    tfs_person_dir = DATASETS_DIR / "coco_2017_person" / "test"
    if not tfs_person_dir.exists():
        generate_tfs_dataset(["person"], tfs_person_dir)

    tfs_80_dir = DATASETS_DIR / "coco_2017_80_classes" / "test"
    if not tfs_80_dir.exists():
        generate_tfs_dataset(_all_coco_category_names(), tfs_80_dir)


if __name__ == "__main__":
    main()
