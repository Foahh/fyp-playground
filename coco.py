from pathlib import Path
from pycocotools.coco import COCO
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

DEST = Path("datasets/coco")


def main():
    urls = [ASSETS_URL + "/coco2017labels.zip"]
    download(urls, dir=DEST.parent, exist_ok=True)
    urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
    ]
    download(urls, dir=DEST / "images", threads=3, exist_ok=True)

if __name__ == "__main__":
    main()
