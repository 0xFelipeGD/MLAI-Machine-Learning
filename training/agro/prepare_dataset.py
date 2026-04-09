#!/usr/bin/env python3
"""
training/agro/prepare_dataset.py — Convert Fruits-360 to a clean train/val split.

Fruits-360 has dozens of fruit varieties. We only want apple, orange and
tomato so we copy just those, then split 80/20 into train/val and write a
labels.txt that TF Model Maker understands.

Output structure:
    <output>/
        labels.txt
        train/
            apple/
            orange/
            tomato/
        val/
            apple/
            orange/
            tomato/

Usage:
    python prepare_dataset.py --source ~/Downloads/fruits-360 --output dataset/agro
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

logger = logging.getLogger("prepare_dataset")

# Map our 3 target classes to keywords likely to appear in Fruits-360 folder names.
TARGETS = {
    "apple": ["apple"],
    "orange": ["orange"],
    "tomato": ["tomato"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="Root of unzipped Fruits-360 dataset")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_images(source: Path, keywords: list[str]) -> list[Path]:
    out: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        name = path.parent.name.lower()
        if any(k in name for k in keywords):
            out.append(path)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    random.seed(args.seed)

    if not args.source.exists():
        logger.error("Source folder %s does not exist", args.source)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    labels_path = args.output / "labels.txt"
    labels_path.write_text("\n".join(TARGETS.keys()) + "\n")

    summary: dict[str, dict[str, int]] = {}
    for cls, keywords in TARGETS.items():
        imgs = collect_images(args.source, keywords)
        if not imgs:
            logger.warning("No images found for class '%s'", cls)
            continue
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * args.val_split))
        val = imgs[:n_val]
        train = imgs[n_val:]
        summary[cls] = {"train": len(train), "val": len(val)}
        for split_name, items in (("train", train), ("val", val)):
            target_dir = args.output / split_name / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in items:
                dst = target_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
        logger.info("  %-8s  train=%d  val=%d", cls, len(train), len(val))

    logger.info("Wrote labels to %s", labels_path)
    logger.info("Summary: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
