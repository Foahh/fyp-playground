#!/usr/bin/env python3
"""Browse YOLO labels on *converted* finetune datasets (not ``*_raw``).

Defaults use paths from ``get_finetune_yolo_dir`` in ``dataset_common`` — the same
trees written by ``run_download_finetune_dataset`` (e.g. ``ego2hands/``).
For Ego2Hands, press ``m`` to overlay ``*_seg.png`` when the RGB
image resolves next to that mask (see ``convert_ego2hands``).

Run::

    ./project.py view-finetune-labels -- --preset ego2hands
    ./project.py view-finetune-labels -- --preset construction_tools
    ./project.py view-finetune-labels -- --preset metu_alet
    ./project.py view-finetune-labels -- --preset fyp_merged --split val
    ./project.py view-finetune-labels -- --preset merged-al   # METU-ALET rows in merge (prefix al_)
    ./project.py view-finetune-labels -- --preset merged-ct

Keys: ``n``/→ next | ``p``/← prev | ``m`` mask toggle | ``q``/Esc quit
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import typer
from PIL import Image

from src.dataset.dataset_common import get_finetune_yolo_dir

app = typer.Typer(add_completion=False)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EDGE_COLORS = ["yellow", "cyan", "magenta", "lime", "orange"]

# preset -> (get_finetune_yolo_dir key, layout, optional default filename prefix)
_PRESETS: dict[str, tuple[str, str, str | None]] = {
    # Per-dataset converted trees (YOLO images/{split} + labels/{split})
    "ego2hands": ("ego2hands", "yolo_splits", None),
    "construction-tools": ("construction_tools", "yolo_splits", None),
    "construction_tools": ("construction_tools", "yolo_splits", None),
    "metu-alet": ("metu_alet", "yolo_splits", None),
    "metu_alet": ("metu_alet", "yolo_splits", None),
    # Merged hand+tool (flat train/val/test); prefixes match merge_for_finetune
    "merged": ("fyp_merged", "flat", None),
    "fyp_merged": ("fyp_merged", "flat", None),
    "fyp-merged": ("fyp_merged", "flat", None),
    "merged-eh": ("fyp_merged", "flat", "eh_"),
    "merged-ego2hands": ("fyp_merged", "flat", "eh_"),
    "merged-ct": ("fyp_merged", "flat", "ct_"),
    "merged-construction": ("fyp_merged", "flat", "ct_"),
    "merged-tools": ("fyp_merged", "flat", "ct_"),
    "merged-al": ("fyp_merged", "flat", "al_"),
    "merged-metu": ("fyp_merged", "flat", "al_"),
}


def _collect_pairs_yolo_splits(
    images_dir: Path, labels_dir: Path
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if not images_dir.is_dir():
        return pairs
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = labels_dir / f"{img_path.stem}.txt"
        if lbl.is_file():
            pairs.append((img_path, lbl))
    return pairs


def _collect_pairs_flat(split_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if not split_dir.is_dir():
        return pairs
    for img_path in sorted(split_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = split_dir / f"{img_path.stem}.txt"
        if lbl.is_file():
            pairs.append((img_path, lbl))
    return pairs


def _read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    lines = (
        label_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    )
    out: list[tuple[int, float, float, float, float]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        out.append((cls, cx, cy, w, h))
    return out


def _yolo_norm_to_xyxy(
    boxes: list[tuple[int, float, float, float, float]], iw: int, ih: int
) -> list[tuple[int, float, float, float, float]]:
    xyxy: list[tuple[int, float, float, float, float]] = []
    for cls, cx, cy, bw, bh in boxes:
        x1 = (cx - bw / 2) * iw
        y1 = (cy - bh / 2) * ih
        x2 = (cx + bw / 2) * iw
        y2 = (cy + bh / 2) * ih
        xyxy.append((cls, x1, y1, x2, y2))
    return xyxy


def _seg_mask_path_for_image(img_path: Path) -> Path | None:
    resolved = img_path.resolve()
    cand = resolved.with_name(f"{resolved.stem}_seg.png")
    return cand if cand.is_file() else None


def _mask_overlay_rgba(mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    left = mask == 50
    right = mask == 100
    rgba[left] = [0, 200, 80, int(255 * alpha)]
    rgba[right] = [220, 60, 200, int(255 * alpha)]
    return rgba


def _resolve_root_and_pairs(
    *,
    preset: str | None,
    dataset_root: Path | None,
    split: str,
    filename_prefix: str | None,
) -> tuple[Path, list[tuple[Path, Path, str]]]:
    layout: str
    root: Path
    preset_prefix: str | None = None
    if dataset_root is not None:
        root = dataset_root.expanduser().resolve()
        layout = (
            "flat"
            if (root / "train").is_dir() and not (root / "images").is_dir()
            else "yolo_splits"
        )
    elif preset is not None:
        pk = preset.strip().lower()
        if pk not in _PRESETS:
            alt = pk.replace("-", "_")
            if alt in _PRESETS:
                pk = alt
            else:
                alt2 = pk.replace("_", "-")
                if alt2 in _PRESETS:
                    pk = alt2
        if pk not in _PRESETS:
            opts = ", ".join(sorted(_PRESETS))
            raise typer.BadParameter(
                f"unknown preset {preset!r}; choose one of: {opts}"
            )
        ds_key, layout, preset_prefix = _PRESETS[pk]
        root = get_finetune_yolo_dir(ds_key)
    else:
        root = get_finetune_yolo_dir("ego2hands")
        layout = "yolo_splits"

    split_l = split.strip().lower()
    if split_l not in ("train", "val", "test", "both"):
        raise typer.BadParameter("split must be train, val, test, or both")

    splits = ["train", "val", "test"] if split_l == "both" else [split_l]

    pairs: list[tuple[Path, Path, str]] = []
    if layout == "flat":
        for sp in splits:
            sdir = root / sp
            for ip, lp in _collect_pairs_flat(sdir):
                pairs.append((ip, lp, sp))
    else:
        for sp in splits:
            idir = root / "images" / sp
            ldir = root / "labels" / sp
            for ip, lp in _collect_pairs_yolo_splits(idir, ldir):
                pairs.append((ip, lp, sp))

    user_pref = filename_prefix.strip() if filename_prefix else None
    eff_pref = user_pref or preset_prefix
    if eff_pref:
        pairs = [(a, b, sp) for a, b, sp in pairs if a.name.startswith(eff_pref)]

    return root, pairs


@app.command()
def main(
    preset: str | None = typer.Option(
        None,
        "--preset",
        "-p",
        help=(
            "Converted dataset: ego2hands; construction-tools|construction_tools; "
            "metu-alet|metu_alet; merged|fyp_merged; "
            "merged-eh|merged-ct|merged-al (+aliases)"
        ),
    ),
    dataset_root: Path | None = typer.Option(
        None,
        "--dataset-root",
        "-d",
        help="Override root (YOLO splits or flat merged layout); use converted trees, not *_raw",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        "-s",
        help="train, val, test, or both (flat layout uses train/val/test dirs)",
    ),
    start: int = typer.Option(0, "--start", help="Start index in the file list"),
    mask_overlay: bool = typer.Option(
        False,
        "--mask/--no-mask",
        help="Start with Ego2Hands-style *_seg.png overlay when available",
    ),
    filename_prefix: str | None = typer.Option(
        None,
        "--filename-prefix",
        help="Only images whose filename starts with this (e.g. eh_ on merged for Ego2Hands)",
    ),
) -> None:
    root, pairs = _resolve_root_and_pairs(
        preset=preset,
        dataset_root=dataset_root,
        split=split,
        filename_prefix=filename_prefix,
    )

    if not pairs:
        typer.echo(
            f"No labelled images under {root} (after filters). "
            "Run download-finetune conversion first; use --preset or --dataset-root.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as e:
        typer.echo(
            "matplotlib is required (e.g. pip install matplotlib in fyp-ml).",
            err=True,
        )
        raise typer.Exit(code=1) from e

    idx = max(0, min(start, len(pairs) - 1))
    show_mask = mask_overlay

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.08)

    class_names: list[str] = []
    classes_txt = root / "classes.txt"
    if classes_txt.is_file():
        class_names = [
            ln.strip()
            for ln in classes_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    elif (root / "train").is_dir() and not (root / "images").is_dir():
        class_names = ["hand", "tool"]

    def draw_frame() -> None:
        ax.clear()
        img_path, label_path, sp = pairs[idx]
        im = np.asarray(Image.open(img_path).convert("RGB"))
        ih, iw = im.shape[:2]
        ax.imshow(im)
        mask_note = ""
        if show_mask:
            mp = _seg_mask_path_for_image(img_path)
            if mp is None:
                mask_note = " (no *_seg.png beside resolved RGB)"
            else:
                m = np.array(Image.open(mp).convert("L"))
                if m.shape[:2] == (ih, iw):
                    ax.imshow(_mask_overlay_rgba(m), interpolation="nearest")
                else:
                    mask_note = f" (mask shape {m.shape} != image {(ih, iw)})"

        boxes = _read_yolo_boxes(label_path)
        xyxy = _yolo_norm_to_xyxy(boxes, iw, ih)
        for cls, x1, y1, x2, y2 in xyxy:
            name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
            color = EDGE_COLORS[cls % len(EDGE_COLORS)]
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                max(0, y1 - 4),
                name,
                color=color,
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1),
            )

        title = (
            f"[{idx + 1}/{len(pairs)}] split={sp}  boxes={len(boxes)}  "
            f"mask={'on' if show_mask else 'off'}{mask_note}\n{img_path.name}"
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event: object) -> None:
        nonlocal idx, show_mask
        key = getattr(event, "key", None)
        if key is None:
            return
        k = key.lower()
        if k in ("q", "escape"):
            plt.close(fig)
            return
        if k in ("n", "right"):
            idx = (idx + 1) % len(pairs)
            draw_frame()
        elif k in ("p", "left"):
            idx = (idx - 1) % len(pairs)
            draw_frame()
        elif k == "m":
            show_mask = not show_mask
            draw_frame()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_frame()
    fig.suptitle(
        "Keys: n/→ next | p/← prev | m mask | q quit",
        fontsize=10,
        y=0.02,
    )
    plt.show()


if __name__ == "__main__":
    app()
