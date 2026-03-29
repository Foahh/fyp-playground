"""
Train TinyissimoYOLO v8 on COCO Person (single class).

Run from the parent repository root (outputs under results/model/):
    python scripts/run_train_tinyissimo_coco_person.py --size 256
    python scripts/run_train_tinyissimo_coco_person.py --size 256 --no-resume
    python scripts/run_train_tinyissimo_coco_person.py --size 256 --export

    # Single GPU, maximise throughput:
    python scripts/run_train_tinyissimo_coco_person.py --size 256 \
        --device 0 --batch 512 --workers 16 --cache ram

    # Dual GPU (DDP), maximise throughput:
    python scripts/run_train_tinyissimo_coco_person.py --size 256 \
        --device 0,1 --batch 1024 --workers 12 --cache ram

Dependencies are provided by the training Docker image (`docker/train.Dockerfile`).
"""

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.coco_yolo_data import materialize_coco_data_yaml
from ultralytics import YOLO

TINY = ROOT / "external" / "TinyissimoYOLO"
MODEL_YAML = str(TINY / "ultralytics/cfg/models/tinyissimo/tinyissimo-v8.yaml")
PROJECT = str(ROOT / "results" / "model")

EPOCHS = 1000


@contextmanager
def working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def export_saved_model(size: int, weights: Path | None = None) -> Path:
    run_name = f"tinyissimoyolo_v8_{size}"
    weights_dir = Path(PROJECT) / run_name / "weights"
    data_yaml = Path(materialize_coco_data_yaml())
    ckpt = (weights or (weights_dir / "best.pt")).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    print(f"Loading {ckpt} ...")
    model = YOLO(str(ckpt))
    ckpt_stem = Path(model.ckpt_path).stem if model.ckpt_path else "best"

    print(f"Exporting SavedModel in {PROJECT} (imgsz={size}, data={data_yaml}) ...")
    with working_directory(Path(PROJECT)):
        model.export(
            format="saved_model",
            int8=False,
            data=str(data_yaml),
            imgsz=[size, size],
        )

    saved_model_dir = weights_dir / f"{ckpt_stem}_saved_model"
    if not saved_model_dir.is_dir():
        raise FileNotFoundError(f"SavedModel export not found at {saved_model_dir}")
    return saved_model_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train TinyissimoYOLO v8 on COCO Person")
    p.add_argument(
        "--size",
        type=int,
        required=True,
        choices=[192, 256, 288, 320],
        help="Input resolution (192, 256, 288, or 320)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh run instead of resuming from last checkpoint",
    )
    p.add_argument("--optimizer", type=str, default="SGD")
    p.add_argument(
        "--export",
        action="store_true",
        help="Export SavedModel only (skip training); useful to force export from latest checkpoint",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size per training step. Use -1 for AutoBatch (single-GPU only). "
        "Must be set explicitly for multi-GPU (e.g. --batch 1024 for two GPUs).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader worker threads per GPU rank (default: 8).",
    )
    p.add_argument(
        "--cache",
        type=str,
        default="False",
        choices=["False", "ram", "disk"],
        help="Dataset caching strategy: False (no cache), ram, or disk (default: False).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="CUDA device(s) to use, e.g. '0', '0,1', or 'cpu' (default: auto-select).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not TINY.is_dir():
        raise FileNotFoundError(f"Expected TinyissimoYOLO at {TINY}")

    run_name = f"tinyissimoyolo_v8_{args.size}"
    weights_dir = Path(PROJECT) / run_name / "weights"
    resume = not args.no_resume

    if args.export:
        saved_model_dir = export_saved_model(size=args.size)
        print(f"Done. Exported SavedModel at {saved_model_dir}")
        return

    if resume:
        last_pt = weights_dir / "last.pt"
        if last_pt.exists():
            print(f"Resuming from {last_pt} ...")
            model = YOLO(str(last_pt))
        else:
            print(f"No checkpoint found at {last_pt}; starting a new run ...")
            resume = False
            model = YOLO(MODEL_YAML)
    else:
        print(f"Creating new model from {MODEL_YAML} ...")
        model = YOLO(MODEL_YAML)

    data_yaml = materialize_coco_data_yaml(require_person=True)
    with open(data_yaml, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    print(f"Using dataset YAML: {data_yaml}")
    print(f"Dataset root: {data_cfg.get('path')}")

    num_gpus = len(args.device.split(",")) if args.device else 1
    if num_gpus > 1 and args.batch == -1:
        print(
            "WARNING: AutoBatch (--batch -1) is incompatible with multi-GPU training. "
            "Ultralytics will fall back to batch=16. "
            "Pass --batch explicitly, e.g. --batch 1024."
        )

    cache = False if args.cache == "False" else args.cache

    model.train(
        data=data_yaml,
        classes=[0],  # filter to person class only
        single_cls=True,  # single-class mode
        imgsz=args.size,
        epochs=EPOCHS,
        batch=args.batch,
        optimizer=args.optimizer,
        device=args.device if args.device else None,
        workers=args.workers,
        cache=cache,
        deterministic=False,
        project=PROJECT,
        name=run_name,
        exist_ok=True,
        patience=100,
        resume=resume,
    )
    print(f"Training done. Weights under {weights_dir}")

    saved_model_dir = export_saved_model(size=args.size)
    print(f"Done. Exported SavedModel at {saved_model_dir}")


if __name__ == "__main__":
    main()
