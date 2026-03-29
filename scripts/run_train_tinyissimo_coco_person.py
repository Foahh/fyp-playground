"""
Train TinyissimoYOLO v8 on COCO Person (single class).

Run from the parent repository root (outputs under results/model/):
    python scripts/run_train_tinyissimo_coco_person.py --size 192
    python scripts/run_train_tinyissimo_coco_person.py --size 192 --profile paper
    python scripts/run_train_tinyissimo_coco_person.py --size 192 --profile powerful
    python scripts/run_train_tinyissimo_coco_person.py --size 192 --no-resume
    python scripts/run_train_tinyissimo_coco_person.py --size 192 --export

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

def run_name_for(size: int, profile: str) -> str:
    base = f"tinyissimoyolo_v8_{size}"
    if profile == "paper":
        return base
    return f"{base}_{profile}"


def train_profile_kwargs(profile: str) -> dict:
    """Hyperparameters and infra for model.train(); profile-specific only."""
    if profile == "paper":
        return {
            "batch": 64,
            "lr0": 0.001,
            "warmup_epochs": 3.0,
            "warmup_bias_lr": 0.01,
        }
    if profile == "powerful":
        return {
            "batch": 256,
            "lr0": 0.004,
            "warmup_epochs": 5.0,
            "warmup_bias_lr": 0.04,
            "workers": 8,
            "cache": "ram",
        }
    raise ValueError(f"Unknown profile: {profile!r}")


@contextmanager
def working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def export_saved_model(
    size: int, weights: Path | None = None, profile: str = "paper"
) -> Path:
    run_name = run_name_for(size, profile)
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
        "--profile",
        choices=("paper", "powerful"),
        default="paper",
        help=(
            "Training preset: TinyissimoYOLO paper (batch 64, paper-aligned LR/schedule) or "
            "powerful (batch 256, linearly scaled LR, workers=8, cache=ram)"
        ),
    )
    p.add_argument(
        "--device",
        default=None,
        help="Ultralytics device (e.g. 0, 0,1 for multi-GPU, cpu); default is auto",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not TINY.is_dir():
        raise FileNotFoundError(f"Expected TinyissimoYOLO at {TINY}")

    run_name = run_name_for(args.size, args.profile)
    weights_dir = Path(PROJECT) / run_name / "weights"
    resume = not args.no_resume

    if args.export:
        saved_model_dir = export_saved_model(size=args.size, profile=args.profile)
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
    print(f"Training profile: {args.profile}")

    train_kw: dict = {
        "data": data_yaml,
        "classes": [0],
        "single_cls": True,
        "imgsz": args.size,
        "epochs": 1000,
        "optimizer": "SGD",
        "lrf": 0.01,
        "momentum": 0.937,
        "cos_lr": True,
        "warmup_momentum": 0.8,
        "weight_decay": 0.0005,
        "amp": True,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "translate": 0.1,
        "scale": 0.5,
        "mosaic": 1.0,
        "deterministic": False,
        "project": PROJECT,
        "name": run_name,
        "exist_ok": True,
        "patience": 0,
        "resume": resume,
        **train_profile_kwargs(args.profile),
    }
    if args.device:
        train_kw["device"] = args.device

    model.train(**train_kw)

    print(f"Training done. Weights under {weights_dir}")

    saved_model_dir = export_saved_model(size=args.size, profile=args.profile)
    print(f"Done. Exported SavedModel at {saved_model_dir}")


if __name__ == "__main__":
    main()
