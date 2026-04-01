#!/usr/bin/env python3
"""Collect dataset archives into datasets/_zips for easy cleanup.

Moves matching archives found anywhere under the datasets directory into
``datasets/_zips/`` while preserving their relative path (to avoid name clashes).

Also reports whether each archive filename is one that our download scripts manage.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import typer

from ..common.paths import get_datasets_dir


@dataclass(frozen=True)
class MovePlan:
    src: Path
    dst: Path


def _expected_archive_filenames() -> set[str]:
    # From src/dataset/run_download_coco_dataset.py and src/dataset/run_download_finetune_dataset.py
    return {
        # COCO
        "coco2017labels.zip",
        "train2017.zip",
        "val2017.zip",
        # Finetune datasets
        "DATA1.zip",
        "DATA2.zip",
        "DATA3.zip",
        "DATA4.zip",
        "metu_alet.zip",
        "ego2hands_eval.zip",
    }


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _plan_moves(
    datasets_dir: Path,
    zips_dir: Path,
    extensions: set[str],
) -> list[MovePlan]:
    plans: list[MovePlan] = []
    # Avoid walking the entire datasets tree (can be massive). Instead, glob for each extension.
    seen: set[Path] = set()
    for ext in sorted(extensions):
        pattern = f"*{ext}"
        for p in datasets_dir.rglob(pattern):
            if p in seen:
                continue
            seen.add(p)
            if not p.is_file():
                continue
            if _is_under(p, zips_dir):
                continue
            rel = p.relative_to(datasets_dir)
            dst = zips_dir / rel
            plans.append(MovePlan(src=p, dst=dst))
    plans.sort(key=lambda mp: str(mp.src))
    return plans


app = typer.Typer(add_completion=False)


@app.command()
def main(
    dry_run: bool = typer.Option(True, help="Print actions but do not move files"),
    zips_dir: Path = typer.Option(None, help="Where to collect archives (default: <datasets>/_zips)"),
    ext: list[str] = typer.Option([".zip"], help="File extensions to collect (case-insensitive)"),
) -> int:
    datasets_dir = get_datasets_dir()
    zips_dir = (datasets_dir / "_zips") if zips_dir is None else zips_dir.expanduser()
    extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in ext}

    expected = _expected_archive_filenames()
    plans = _plan_moves(datasets_dir, zips_dir, extensions)

    if not plans:
        print(f"No archives found under {datasets_dir} matching {sorted(extensions)} outside {zips_dir}.")
        return 0

    print(f"Collecting {len(plans)} archive(s) into: {zips_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'MOVE'}")

    managed = 0
    for plan in plans:
        is_managed = plan.src.name in expected
        managed += int(is_managed)
        tag = "managed" if is_managed else "unknown"
        print(f"- [{tag}] {plan.src} -> {plan.dst}")

    print(f"\nSummary: managed={managed}, unknown={len(plans) - managed}")

    if dry_run:
        print("\nDry-run only. Re-run with --no-dry-run to actually move files.")
        return 0

    zips_dir.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        plan.dst.parent.mkdir(parents=True, exist_ok=True)
        if plan.dst.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing file: {plan.dst}\n"
                f"Source was: {plan.src}"
            )
        shutil.move(str(plan.src), str(plan.dst))

        try:
            parent = plan.src.parent
            while parent != datasets_dir and parent.exists():
                parent.rmdir()
                parent = parent.parent
        except OSError:
            pass

    print("\nDone.")
    print(f"To free space later, you can delete: {zips_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(app())

