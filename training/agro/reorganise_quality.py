#!/usr/bin/env python3
"""
training/agro/reorganise_quality.py — Reshape the Kaggle Fruits Fresh/Rotten
dataset into the train/{good,defective} layout train_quality.py expects.

Source layout (Kaggle "Fruits Fresh and Rotten for Classification"):
    <source>/
    ├── train/
    │   ├── freshapples/    rottenapples/
    │   ├── freshbanana/    rottenbanana/
    │   └── freshoranges/   rottenoranges/
    └── test/
        ├── freshapples/    rottenapples/
        ├── freshbanana/    rottenbanana/
        └── freshoranges/   rottenoranges/

Output layout (what train_quality.py expects):
    <output>/
    ├── train/
    │   ├── good/        ← all "fresh*" images from source/train/
    │   └── defective/   ← all "rotten*" images from source/train/
    └── val/
        ├── good/        ← all "fresh*" images from source/test/
        └── defective/   ← all "rotten*" images from source/test/

We respect the dataset's existing train/test split (which was hand-picked
by the dataset authors) instead of doing a random shuffle. The original
6 fruit-specific folders are collapsed into 2 quality folders — the model
learns to recognise quality, not which fruit it's looking at.

Usage (from inside the training/ folder):
    python agro/reorganise_quality.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

GOOD_PREFIX = "fresh"
DEFECTIVE_PREFIX = "rotten"
IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def collect_files(folder: Path, prefix: str) -> list[Path]:
    """Find all images inside subfolders of `folder` whose name starts with `prefix`."""
    files: list[Path] = []
    for sub in sorted(folder.iterdir()):
        if sub.is_dir() and sub.name.lower().startswith(prefix):
            for pattern in IMAGE_GLOBS:
                files.extend(sub.glob(pattern))
    return files


def copy_split(src_folder: Path, dst_folder: Path) -> dict[str, int]:
    """Copy images from <src>/{fresh*, rotten*}/ into <dst>/{good, defective}/."""
    dst_good = dst_folder / "good"
    dst_defective = dst_folder / "defective"
    dst_good.mkdir(parents=True, exist_ok=True)
    dst_defective.mkdir(parents=True, exist_ok=True)

    good_files = collect_files(src_folder, GOOD_PREFIX)
    defective_files = collect_files(src_folder, DEFECTIVE_PREFIX)

    for f in good_files:
        shutil.copy2(f, dst_good / f.name)
    for f in defective_files:
        shutil.copy2(f, dst_defective / f.name)

    return {"good": len(good_files), "defective": len(defective_files)}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/fruits"),
        help="Folder containing train/ and test/ subfolders with fresh*/rotten* inside",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/agro_quality"),
        help="Where to write the train/{good,defective} + val/{good,defective} layout",
    )
    args = p.parse_args()

    src_train = args.source / "train"
    src_test = args.source / "test"
    if not src_train.exists():
        raise SystemExit(f"ERROR: {src_train} does not exist. Pass --source pointing at a folder with train/ and test/ inside.")
    if not src_test.exists():
        raise SystemExit(f"ERROR: {src_test} does not exist. Pass --source pointing at a folder with train/ and test/ inside.")

    print(f"[1/2] Copying {src_train}/  ->  {args.output}/train/")
    train_counts = copy_split(src_train, args.output / "train")
    print(f"      good={train_counts['good']}  defective={train_counts['defective']}")

    print(f"[2/2] Copying {src_test}/   ->  {args.output}/val/")
    val_counts = copy_split(src_test, args.output / "val")
    print(f"      good={val_counts['good']}  defective={val_counts['defective']}")

    print()
    print(f"Done. Reorganised dataset is at {args.output}")
    print(f"  total train images: {train_counts['good'] + train_counts['defective']}")
    print(f"  total val images:   {val_counts['good'] + val_counts['defective']}")


if __name__ == "__main__":
    main()
