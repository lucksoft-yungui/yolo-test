#!/usr/bin/env python3
"""
Utility to restructure the play-phone dataset into a YOLO-compatible layout.

Source layout (datasets/play-phone-source):
  正样本/
    images/*.jpg
    labels/*.txt
  负样本/
    images/*.jpg
    labels/*.txt (can be empty to mark background images)

The script combines both subsets, performs a train/val split, copies the data
into datasets/play-phone/{images,labels}/{train,val}, and emits play-phone.yaml.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


POSITIVE_SUBDIR = "正样本"
NEGATIVE_SUBDIR = "负样本"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    image_path: Path
    label_path: Path | None
    split: str
    prefix: str
    index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert play-phone-source into a YOLO formatted dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/play-phone-source"),
        help="Root directory of the raw dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/play-phone"),
        help="Destination directory for the YOLO-formatted dataset.",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=Path("play-phone.yaml"),
        help="Path of the dataset YAML to create.",
    )
    parser.add_argument(
        "--class-name",
        default="play_phone",
        help="Class name to use in the generated YAML file.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for validation (per subset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling before splitting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow deleting the existing output directory when present.",
    )
    return parser.parse_args()


def collect_samples(subset_dir: Path) -> List[Tuple[Path, Path | None]]:
    images_dir = subset_dir / "images"
    labels_dir = subset_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    samples: List[Tuple[Path, Path | None]] = []
    for image_path in images_dir.iterdir():
        if image_path.is_dir():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        samples.append((image_path, label_path if label_path.exists() else None))
    return samples


def split_samples(
    samples: Iterable[Tuple[Path, Path | None]],
    val_ratio: float,
    seed: int,
    prefix: str,
) -> List[Sample]:
    items = list(samples)
    random.Random(seed).shuffle(items)
    val_count = int(len(items) * val_ratio)
    # Ensure at least one example lands in each split if possible
    if len(items) > 0 and val_ratio > 0 and val_count == 0:
        val_count = 1
    split_markers = ["val"] * val_count + ["train"] * (len(items) - val_count)
    # Use a second shuffle to mix val/train markers
    random.Random(seed + 1).shuffle(split_markers)

    prepared: List[Sample] = []
    train_idx = 0
    val_idx = 0
    for (image_path, label_path), split in zip(items, split_markers):
        if split == "train":
            train_idx += 1
            index = train_idx
        else:
            val_idx += 1
            index = val_idx
        prepared.append(
            Sample(
                image_path=image_path,
                label_path=label_path,
                split=split,
                prefix=prefix,
                index=index,
            )
        )
    return prepared


def ensure_empty_label(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("", encoding="utf-8")


def copy_samples(samples: Iterable[Sample], output: Path) -> None:
    for sample in samples:
        split = sample.split
        base_name = f"{sample.prefix}_{sample.index:06d}"
        image_dest = output / "images" / split / f"{base_name}{sample.image_path.suffix.lower()}"
        label_dest = output / "labels" / split / f"{base_name}.txt"

        image_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, image_dest)

        if sample.label_path and sample.label_path.exists():
            label_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.label_path, label_dest)
        else:
            ensure_empty_label(label_dest)


def write_yaml(yaml_path: Path, dataset_dir: Path, class_name: str) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    relative_dataset_path = os.path.relpath(dataset_dir, yaml_path.parent)
    content_lines = [
        "# Play-phone dataset configuration generated by prepare_play_phone_dataset.py",
        f"path: {relative_dataset_path}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
        f"  0: {class_name}",
        "",
    ]
    yaml_path.write_text("\n".join(content_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = args.source
    output = args.output

    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    if output.exists():
        if args.overwrite:
            shutil.rmtree(output)
        else:
            raise FileExistsError(
                f"Output directory {output} exists. Use --overwrite to replace it."
            )

    subsets = [
        (source / POSITIVE_SUBDIR, "pos"),
        (source / NEGATIVE_SUBDIR, "neg"),
    ]

    all_samples: List[Sample] = []
    for offset, (subset_dir, prefix) in enumerate(subsets):
        subset_samples = collect_samples(subset_dir)
        splitted = split_samples(
            subset_samples, args.val_ratio, args.seed + offset, prefix
        )
        all_samples.extend(splitted)

    copy_samples(all_samples, output)
    write_yaml(args.yaml_path, output, args.class_name)

    train_count = sum(1 for s in all_samples if s.split == "train")
    val_count = sum(1 for s in all_samples if s.split == "val")
    print(
        f"Dataset prepared at {output} "
        f"(train: {train_count}, val: {val_count}, total: {len(all_samples)})"
    )
    print(f"YAML saved to {args.yaml_path}")


if __name__ == "__main__":
    main()
