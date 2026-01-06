import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从负样本视频抽帧并写入 YOLO 数据集")
    parser.add_argument("--video", type=Path, required=True, help="负样本视频路径")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="数据集 yaml 路径")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="写入的数据集划分，默认 train",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.15,
        help="负样本目标占比（新增后），默认 0.15",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最大抽取帧数，>0 时覆盖 ratio 计算",
    )
    parser.add_argument(
        "--every-seconds",
        type=float,
        default=0.0,
        help="按秒抽帧间隔，>0 时按该间隔抽帧",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="输出文件名前缀，默认使用视频文件名",
    )
    return parser.parse_args()


def resolve_root(yaml_path: Path, raw_path: str) -> Path:
    if not raw_path:
        return yaml_path.parent
    root = Path(raw_path).expanduser()
    if root.is_absolute():
        return root
    candidate = (yaml_path.parent / root).resolve()
    if candidate.exists():
        return candidate
    candidate2 = root.resolve()
    if candidate2.exists():
        return candidate2
    return candidate


def parse_names_from_yaml(yaml_path: Path) -> Dict[int, str]:
    names: Dict[int, str] = {}
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "names:":
            in_names = True
            continue
        if in_names and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if not key.isdigit():
                continue
            name = value.strip().strip('"').strip("'")
            if name:
                names[int(key)] = name
    return names


def parse_dataset_yaml(yaml_path: Path) -> Tuple[Path, Dict[str, str], Dict[int, str]]:
    data = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.split("#", 1)[0].strip()
        data[key.strip()] = value

    root = resolve_root(yaml_path, data.get("path", ""))
    names = parse_names_from_yaml(yaml_path)
    return root, data, names


def resolve_split_dirs(root: Path, data: Dict[str, str], split: str) -> Tuple[Path, Path]:
    rel = data.get(split, f"images/{split}")
    split_path = Path(rel)
    images_dir = split_path if split_path.is_absolute() else (root / rel)
    if "images" in rel:
        labels_rel = rel.replace("images", "labels", 1)
        labels_path = Path(labels_rel)
        labels_dir = labels_path if labels_path.is_absolute() else (root / labels_rel)
    else:
        labels_dir = root / "labels" / Path(rel).name
    return images_dir, labels_dir


def list_images(images_dir: Path) -> int:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    if not images_dir.exists():
        return 0
    return len([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def count_empty_labels(labels_dir: Path, images_dir: Path) -> int:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    if not images_dir.exists():
        return 0
    empty = 0
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in exts:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            empty += 1
    return empty


def compute_target_add(total_images: int, empty_labels: int, ratio: float) -> int:
    ratio = max(0.0, min(ratio, 0.95))
    if ratio <= 0.0:
        return 0
    target = (ratio * total_images - empty_labels) / (1 - ratio)
    return max(0, int(math.ceil(target)))


def next_available_name(images_dir: Path, base: str) -> str:
    candidate = base
    if not (images_dir / f"{candidate}.jpg").exists() and not (images_dir / f"{candidate}.png").exists():
        return candidate
    idx = 1
    while True:
        candidate = f"{base}_{idx}"
        if not (images_dir / f"{candidate}.jpg").exists() and not (images_dir / f"{candidate}.png").exists():
            return candidate
        idx += 1


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"找不到视频: {args.video}")
    if not args.dataset_yaml.exists():
        raise FileNotFoundError(f"找不到数据集配置: {args.dataset_yaml}")

    root, data, _ = parse_dataset_yaml(args.dataset_yaml)
    images_dir, labels_dir = resolve_split_dirs(root, data, args.split)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    total_images = list_images(images_dir)
    empty_labels = count_empty_labels(labels_dir, images_dir)

    if args.max_frames > 0:
        target_add = args.max_frames
    else:
        target_add = compute_target_add(total_images, empty_labels, args.ratio)

    if target_add <= 0:
        print("无需新增负样本（目标占比已满足）。")
        return

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if args.every_seconds > 0 and fps > 0:
        step = max(1, int(round(args.every_seconds * fps)))
    elif total_frames > 0:
        step = max(1, total_frames // target_add)
    else:
        step = 1

    prefix = args.prefix or args.video.stem
    saved = 0
    frame_idx = 0

    while saved < target_add:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            base = next_available_name(images_dir, f"{prefix}_neg_{frame_idx:06d}")
            img_path = images_dir / f"{base}.jpg"
            label_path = labels_dir / f"{base}.txt"
            cv2.imwrite(str(img_path), frame)
            label_path.write_text("", encoding="utf-8")
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"完成：新增负样本 {saved} 张，输出到 {images_dir}")


if __name__ == "__main__":
    main()
