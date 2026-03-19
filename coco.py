import os
import zipfile
import urllib.request
from tqdm import tqdm
from pycocotools.coco import COCO

dest = "datasets/coco"
os.makedirs(dest, exist_ok=True)

files = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


class DownloadProgressBar(tqdm):
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download(url, path):
    if os.path.exists(path):
        print(f"Zip already exists, skipping download: {path}")
        return

    print(f"Downloading {url} ...")
    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(path),
    ) as t:
        urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)

    print(f"Saved to {path}")


def already_extracted(filename, out_dir):
    if filename == "train2017.zip":
        return os.path.isdir(os.path.join(out_dir, "train2017"))
    elif filename == "val2017.zip":
        return os.path.isdir(os.path.join(out_dir, "val2017"))
    elif filename == "annotations_trainval2017.zip":
        return os.path.isdir(os.path.join(out_dir, "annotations"))
    return False


def extract(zip_path, out_dir, filename):
    if already_extracted(filename, out_dir):
        print(f"Already extracted, skipping: {filename}")
        return

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


for filename, url in files.items():
    zip_path = os.path.join(dest, filename)

    download(url, zip_path)

    if os.path.exists(zip_path):
        extract(zip_path, dest, filename)
    else:
        print(f"Missing zip file, cannot extract: {zip_path}")

ann_dir = os.path.join(dest, "annotations")
for split in ["train2017", "val2017"]:
    ann_file = os.path.join(ann_dir, f"instances_{split}.json")

    if not os.path.exists(ann_file):
        print(f"Annotation file not found, skipping {split}: {ann_file}")
        continue

    print(f"\nLoading {ann_file} ...")
    coco = COCO(ann_file)

    person_cat_id = coco.getCatIds(catNms=["person"])
    img_ids = coco.getImgIds(catIds=person_cat_id)
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=person_cat_id, iscrowd=False)

    print(f"{split}: {len(img_ids)} images, {len(ann_ids)} person annotations")
