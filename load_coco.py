import json
import os
from pathlib import Path

import tensorflow as tf
from pycocotools.coco import COCO
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

DEST = Path("datasets/coco")
ANN_PATH = DEST / "annotations" / "instances_val2017.json"
IMAGES_DIR = DEST / "images" / "val2017"


def download_coco():
    urls = [ASSETS_URL + "/coco2017labels.zip"]
    download(urls, dir=DEST.parent, exist_ok=True)
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
    ]
    download(urls, dir=DEST / "images", threads=8)


def generate_person_annotations():
    """Filter COCO val2017 annotations to person-only using pycocotools."""
    out_path = DEST / "annotations" / "instances_val2017_person.json"

    coco = COCO(str(ANN_PATH))
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
    coco = COCO(str(ANN_PATH))
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

        src_img = IMAGES_DIR / img_info["file_name"]
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
    coco = COCO(str(ANN_PATH))
    return [c["name"] for c in coco.loadCats(coco.getCatIds())]


def main():
    download_coco()
    generate_person_annotations()

    tfs_person_dir = Path("datasets/coco_2017_person/test")
    if not tfs_person_dir.exists():
        generate_tfs_dataset(["person"], tfs_person_dir)

    tfs_80_dir = Path("datasets/coco_2017_80_classes/test")
    if not tfs_80_dir.exists():
        generate_tfs_dataset(_all_coco_category_names(), tfs_80_dir)


if __name__ == "__main__":
    main()
