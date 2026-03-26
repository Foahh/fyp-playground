"""
Train TinyissimoYOLO v8 on COCO Person (single class).

Run from the parent repository root (paths point into external/TinyissimoYOLO/):
    python train_coco_person.py --img_size 192
    python train_coco_person.py --img_size 192 --resume

Dependencies are provided by the training Docker image (`docker/train.Dockerfile`).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.coco_yolo_data import materialize_coco_data_yaml
from ultralytics import YOLO
TINY = ROOT / "external" / "TinyissimoYOLO"
MODEL_YAML = str(TINY / "ultralytics/cfg/models/tinyissimo/tinyissimo-v8.yaml")
PROJECT = str(TINY / "results")

EPOCHS = 1000
BATCH = -1  # -1 = AutoBatch


def parse_args():
    p = argparse.ArgumentParser(description="Train TinyissimoYOLO v8 on COCO Person")
    p.add_argument(
        "--img_size",
        type=int,
        required=True,
        choices=[192, 256, 288, 320],
        help="Input resolution (192, 256, 288, or 320)",
    )
    p.add_argument(
        "--resume", action="store_true", help="Resume training from last checkpoint"
    )
    p.add_argument("--optimizer", type=str, default="SGD")
    return p.parse_args()


def main():
    args = parse_args()
    if not TINY.is_dir():
        raise FileNotFoundError(f"Expected TinyissimoYOLO at {TINY}")

    run_name = f"tinyissimoyolo_v8_{args.img_size}"
    weights_dir = Path(PROJECT) / run_name / "weights"

    if args.resume:
        last_pt = weights_dir / "last.pt"
        if not last_pt.exists():
            raise FileNotFoundError(f"No checkpoint to resume from at {last_pt}")
        print(f"Resuming from {last_pt} ...")
        model = YOLO(str(last_pt))
    else:
        print(f"Creating new model from {MODEL_YAML} ...")
        model = YOLO(MODEL_YAML)

    data_yaml = materialize_coco_data_yaml()
    model.train(
        data=data_yaml,
        classes=[0],  # filter to person class only
        single_cls=True,  # single-class mode
        imgsz=args.img_size,
        epochs=EPOCHS,
        batch=BATCH,
        optimizer=args.optimizer,
        project=PROJECT,
        name=run_name,
        exist_ok=True,
        patience=100,
        resume=args.resume,
    )
    print(f"Done. Weights under {weights_dir}")


if __name__ == "__main__":
    main()
