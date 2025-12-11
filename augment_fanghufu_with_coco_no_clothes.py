#!/usr/bin/env python3
"""
Merge the existing fanghufu clothes-only YOLO dataset with COCO person samples
as a new class "no_clothes", producing a two-class dataset:
  class 0: clothes      (from datasets/fanghufu-clothes)
  class 1: no_clothes   (from COCO person)

Assumptions:
- Existing clothes dataset is in datasets/fanghufu-clothes/{images,labels}/{train,val}
- COCO annotations are in COCO format (e.g., annotations/instances_train2017.json)
- COCO images directory matches the annotation split (e.g., train2017/)

Usage example:
  python augment_fanghufu_with_coco_no_clothes.py \
      --coco-ann /path/to/instances_train2017.json \
      --coco-images /path/to/train2017 \
      --output datasets/fanghufu-clothes-2cls \
      --coco-max 500 \
      --val-ratio 0.2 \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class YoloLabel:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class Sample:
    stem: str
    image_path: Path
    labels: List[YoloLabel]
    split: str | None = None
    prefix: str | None = None  # optional prefix to avoid name collisions

    def image_name(self) -> str:
        suffix = self.image_path.suffix.lower()
        name = self.stem
        if self.prefix:
            name = f"{self.prefix}_{name}"
        return f"{name}{suffix}"

    def label_name(self) -> str:
        name = self.stem
        if self.prefix:
            name = f"{self.prefix}_{name}"
        return f"{name}.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment fanghufu clothes dataset with COCO person as no_clothes"
    )
    parser.add_argument(
        "--clothes-dataset",
        type=Path,
        default=Path("datasets/fanghufu-clothes"),
        help="Existing clothes-only YOLO dataset root",
    )
    parser.add_argument(
        "--coco-ann",
        type=Path,
        required=True,
        help="COCO annotation json file (e.g., instances_train2017.json)",
    )
    parser.add_argument(
        "--coco-images",
        type=Path,
        required=True,
        help="Directory containing COCO images for the given annotation split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fanghufu-clothes-2cls"),
        help="Output directory for the merged two-class dataset",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=Path("fanghufu-clothes-2cls.yaml"),
        help="Path of the generated YAML file",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio for COCO-derived samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--coco-max",
        type=int,
        default=1000,
        help="Maximum number of COCO images to use (shuffle before sampling)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it exists",
    )
    return parser.parse_args()


def load_clothes_samples(root: Path) -> List[Sample]:
    """
    Load existing clothes dataset (already YOLO formatted, class 0).
    """
    samples: List[Sample] = []
    images_dir = root / "images"
    labels_dir = root / "labels"
    for split in ("train", "val"):
        for img_path in (images_dir / split).iterdir():
            if img_path.is_dir() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = labels_dir / split / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            lines = label_path.read_text(encoding="utf-8").splitlines()
            labels: List[YoloLabel] = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = parts
                labels.append(
                    YoloLabel(
                        cls=int(float(cls)),
                        cx=float(cx),
                        cy=float(cy),
                        w=float(w),
                        h=float(h),
                    )
                )
            samples.append(
                Sample(
                    stem=img_path.stem,
                    image_path=img_path,
                    labels=labels,
                    split=split,
                    prefix="hf",  # avoid collision with COCO names
                )
            )
    return samples


def load_coco_person_samples(
    ann_path: Path, images_dir: Path, max_images: int, seed: int
) -> List[Sample]:
    data = json.loads(ann_path.read_text())
    images = {img["id"]: img for img in data["images"]}
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in data["annotations"]:
        if ann.get("category_id") != 1:  # COCO person id
            continue
        if ann.get("iscrowd", 0):
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Shuffle image ids to randomize sampling
    rng = random.Random(seed)
    image_ids = list(anns_by_image.keys())
    rng.shuffle(image_ids)
    if max_images > 0:
        image_ids = image_ids[:max_images]

    samples: List[Sample] = []
    for image_id in image_ids:
        info = images.get(image_id)
        if not info:
            continue
        file_name = info["file_name"]
        width = float(info["width"])
        height = float(info["height"])
        img_path = images_dir / file_name
        if not img_path.exists():
            continue
        labels: List[YoloLabel] = []
        for ann in anns_by_image[image_id]:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2.0) / width
            cy = (y + h / 2.0) / height
            labels.append(YoloLabel(cls=1, cx=cx, cy=cy, w=w / width, h=h / height))
        if labels:
            samples.append(
                Sample(
                    stem=Path(file_name).stem,
                    image_path=img_path,
                    labels=labels,
                    prefix="coco",
                )
            )
    return samples


def split_samples(
    samples: Iterable[Sample], val_ratio: float, seed: int
) -> List[Sample]:
    items = list(samples)
    rng = random.Random(seed)
    rng.shuffle(items)
    val_count = int(len(items) * val_ratio)
    if len(items) > 0 and val_ratio > 0 and val_count == 0:
        val_count = 1
    for idx, sample in enumerate(items):
        sample.split = "val" if idx < val_count else "train"
    return items


def copy_samples(samples: Iterable[Sample], output: Path) -> None:
    for sample in samples:
        assert sample.split is not None
        image_dest = output / "images" / sample.split / sample.image_name()
        label_dest = output / "labels" / sample.split / sample.label_name()
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, image_dest)
        label_lines = [label.to_line() for label in sample.labels]
        label_dest.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def write_yaml(yaml_path: Path, dataset_dir: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    rel_path = os.path.relpath(dataset_dir, yaml_path.parent)
    lines = [
        "# Clothes (class 0) + No_clothes/person (class 1)",
        f"path: {rel_path}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
        "  0: clothes",
        "  1: no_clothes",
        "",
    ]
    yaml_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.clothes_dataset.is_dir():
        raise FileNotFoundError(
            f"Clothes dataset not found: {args.clothes_dataset}. "
            "Run prepare_clothes_yolo.py first."
        )
    if not args.coco_ann.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {args.coco_ann}")
    if not args.coco_images.is_dir():
        raise FileNotFoundError(f"COCO images directory not found: {args.coco_images}")

    if args.output.exists():
        if args.overwrite:
            shutil.rmtree(args.output)
        else:
            raise FileExistsError(
                f"Output {args.output} exists. Use --overwrite to replace."
            )

    # Load datasets
    clothes_samples = load_clothes_samples(args.clothes_dataset)
    coco_samples = load_coco_person_samples(
        args.coco_ann, args.coco_images, args.coco_max, args.seed
    )
    coco_samples = split_samples(coco_samples, args.val_ratio, args.seed)

    # Copy
    copy_samples(clothes_samples, args.output)
    copy_samples(coco_samples, args.output)
    write_yaml(args.yaml_path, args.output)

    train_count = sum(1 for s in clothes_samples + coco_samples if s.split == "train")
    val_count = sum(1 for s in clothes_samples + coco_samples if s.split == "val")
    print(
        f"Done. Output: {args.output} "
        f"(train: {train_count}, val: {val_count}, total: {len(clothes_samples + coco_samples)})"
    )
    print(f"Dataset YAML: {args.yaml_path}")
    print(
        "Classes -> 0: clothes (fanghufu), 1: no_clothes (COCO person); "
        "COCO person count may be limited by --coco-max."
    )


if __name__ == "__main__":
    main()
