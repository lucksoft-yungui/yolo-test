#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from PIL import ImageDraw
from ultralytics import YOLO

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_names(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    raise ValueError(f"Unsupported names in {yaml_path}")


def resolve_class_id(
    names: dict[int, str],
    class_name: str | None,
    class_id: int | None,
    yaml_path: Path,
) -> int:
    if class_id is not None:
        return class_id
    if not class_name:
        raise ValueError(f"Missing class name for {yaml_path}")
    for idx, name in names.items():
        if name == class_name:
            return idx
    raise ValueError(f'Class "{class_name}" not found in {yaml_path}')


def iter_images(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return [root]
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )


def clear_dir_contents(root: Path) -> None:
    if not root.exists():
        return
    for path in root.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], label: str, color: str) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if label:
        text_pos = (max(0, x1), max(0, y1 - 14))
        draw.text(text_pos, label, fill=color)


def save_plot(result, dest_path: Path) -> None:
    plotted = result.plot()
    if plotted is None:
        return
    rgb = plotted[:, :, ::-1]
    Image.fromarray(rgb.astype(np.uint8)).save(dest_path)


def format_detections(result, class_names: dict[int, str]) -> str:
    if result.boxes is None or len(result.boxes) == 0:
        return "no detections"
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy().tolist()
    parts = []
    for cls_id, conf in zip(classes, confs):
        name = class_names.get(int(cls_id), str(int(cls_id)))
        parts.append(f"{name}:{conf:.4f}")
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find an image with no labcoat + glove and copy it to target folder."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/labcoat"),
        help="Dataset root containing images/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kafuka/labcoat/test/images"),
        help="Destination directory for matched image.",
    )
    parser.add_argument(
        "--labcoat-model",
        type=Path,
        default=Path("model/labcoat/best.pt"),
        help="Labcoat detection model path.",
    )
    parser.add_argument(
        "--glove-model",
        type=Path,
        default=Path("model/glove/best.pt"),
        help="Glove detection model path.",
    )
    parser.add_argument(
        "--labcoat-yaml",
        type=Path,
        default=Path("labcoat.yaml"),
        help="Labcoat dataset yaml.",
    )
    parser.add_argument(
        "--glove-yaml",
        type=Path,
        default=Path("glove.yaml"),
        help="Glove dataset yaml.",
    )
    parser.add_argument(
        "--labcoat-class-name",
        default="no labcoat",
        help='Class name for "no labcoat".',
    )
    parser.add_argument(
        "--labcoat-class-id",
        type=int,
        default=None,
        help="Override class id for no labcoat.",
    )
    parser.add_argument(
        "--glove-class-name",
        default="with glove",
        help='Class name for "with glove".',
    )
    parser.add_argument(
        "--glove-class-id",
        type=int,
        default=None,
        help="Override class id for glove.",
    )
    parser.add_argument("--labcoat-conf", type=float, default=0.85)
    parser.add_argument("--glove-conf", type=float, default=0.7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-count",
        type=int,
        default=23,
        help="Max matched images to copy.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory to save debug images with plotted boxes.",
    )
    parser.add_argument(
        "--debug-target",
        type=str,
        default=None,
        help="Only save debug plots for the specified image filename.",
    )
    parser.add_argument(
        "--only-image",
        type=str,
        default=None,
        help="Only process the specified image (filename or path).",
    )
    parser.add_argument(
        "--tagged-dir",
        type=Path,
        default=Path("kafuka/labcoat/test/image-with-tag"),
        help="Directory to save tagged images.",
    )
    args = parser.parse_args()

    labcoat_names = load_names(args.labcoat_yaml)
    glove_names = load_names(args.glove_yaml)
    no_labcoat_id = resolve_class_id(
        labcoat_names, args.labcoat_class_name, args.labcoat_class_id, args.labcoat_yaml
    )
    glove_id = resolve_class_id(
        glove_names, args.glove_class_name, args.glove_class_id, args.glove_yaml
    )

    image_root = args.dataset_dir / "images"
    image_paths = iter_images(image_root)
    if not image_paths:
        print(f"No images found under {image_root}", file=sys.stderr)
        return 1
    if args.only_image:
        only_path = Path(args.only_image)
        if only_path.exists():
            image_paths = [only_path]
        else:
            image_paths = [p for p in image_paths if p.name == args.only_image]
        if not image_paths:
            print(f"Only-image target not found: {args.only_image}", file=sys.stderr)
            return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clear_dir_contents(args.output_dir)
    args.tagged_dir.mkdir(parents=True, exist_ok=True)
    clear_dir_contents(args.tagged_dir)
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        clear_dir_contents(args.debug_dir)

    labcoat_model = YOLO(str(args.labcoat_model))
    glove_model = YOLO(str(args.glove_model))

    matched = 0
    for image_path in image_paths:
        debug_this = args.debug_dir and (
            args.debug_target is None or image_path.name == args.debug_target
        )
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            labcoat_results = labcoat_model.predict(
                source=img,
                conf=args.labcoat_conf,
                verbose=False,
                device=args.device,
            )[0]
            if debug_this:
                save_plot(
                    labcoat_results,
                    args.debug_dir / f"{image_path.stem}_labcoat.jpg",
                )
                print(
                    f"[debug] labcoat {image_path.name}: "
                    f"{format_detections(labcoat_results, labcoat_names)}"
                )
            if labcoat_results.boxes is None or len(labcoat_results.boxes) == 0:
                continue

            boxes = labcoat_results.boxes
            classes = boxes.cls.cpu().numpy().astype(int)
            if no_labcoat_id not in classes:
                continue

            width, height = img.size
            for idx, cls_id in enumerate(classes):
                if cls_id != no_labcoat_id:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().tolist()
                x1 = max(0, min(width, int(x1)))
                y1 = max(0, min(height, int(y1)))
                x2 = max(0, min(width, int(x2)))
                y2 = max(0, min(height, int(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                glove_results = glove_model.predict(
                    source=crop,
                    conf=args.glove_conf,
                    verbose=False,
                    device=args.device,
                )[0]
                if debug_this:
                    save_plot(
                        glove_results,
                        args.debug_dir
                        / f"{image_path.stem}_glove_{idx}.jpg",
                    )
                    print(
                        f"[debug] glove {image_path.name} crop#{idx}: "
                        f"{format_detections(glove_results, glove_names)}"
                    )
                if glove_results.boxes is None or len(glove_results.boxes) == 0:
                    continue
                glove_classes = glove_results.boxes.cls.cpu().numpy().astype(int)
                if glove_id in glove_classes:
                    dest = args.output_dir / image_path.name
                    shutil.copy2(image_path, dest)
                    tagged = img.copy()
                    draw = ImageDraw.Draw(tagged)
                    draw_box(draw, [x1, y1, x2, y2], "no labcoat", "red")
                    glove_boxes = glove_results.boxes.xyxy.cpu().numpy().tolist()
                    for glove_idx, glove_cls in enumerate(glove_classes):
                        if glove_cls != glove_id:
                            continue
                        gx1, gy1, gx2, gy2 = glove_boxes[glove_idx]
                        gx1 = x1 + max(0, int(gx1))
                        gy1 = y1 + max(0, int(gy1))
                        gx2 = x1 + max(0, int(gx2))
                        gy2 = y1 + max(0, int(gy2))
                        draw_box(
                            draw,
                            [gx1, gy1, gx2, gy2],
                            "with glove",
                            "green",
                        )
                    tagged.save(args.tagged_dir / image_path.name)
                    matched += 1
                    print(f"Matched image copied to {dest}")
                    if matched >= args.max_count:
                        return 0

    if matched == 0:
        print("No matching image found.", file=sys.stderr)
        return 1
    print(f"Matched images copied: {matched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
