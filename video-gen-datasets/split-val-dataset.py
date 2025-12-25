import argparse
import random
import shutil
from pathlib import Path
from typing import Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从训练集复制出验证集（保留原始数据集）")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="数据集 yaml 路径")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认0.2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认42")
    parser.add_argument("--update-yaml", action="store_true", help="更新 yaml 中的 val 指向 images/val")
    parser.add_argument("--no-backup", action="store_true", help="不备份原始数据集")
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


def parse_dataset_yaml(yaml_path: Path) -> Tuple[Path, Path, Path, Path]:
    data = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()

    root = resolve_root(yaml_path, data.get("path", ""))
    train_rel = data.get("train", "images/train")
    train_path = Path(train_rel)
    images_dir = train_path if train_path.is_absolute() else (root / train_rel)
    if "images" in train_rel:
        labels_rel = train_rel.replace("images", "labels", 1)
        labels_path = Path(labels_rel)
        labels_dir = labels_path if labels_path.is_absolute() else (root / labels_rel)
    else:
        labels_dir = root / "labels" / Path(train_rel).name

    val_images_dir = root / "images" / "val"
    val_labels_dir = root / "labels" / "val"
    return images_dir, labels_dir, val_images_dir, val_labels_dir


def backup_dataset(root: Path) -> Path:
    bak_root = root.parent / "bak"
    bak_root.mkdir(parents=True, exist_ok=True)
    dst = bak_root / root.name
    if dst.exists():
        idx = 1
        while (bak_root / f"{root.name}_{idx}").exists():
            idx += 1
        dst = bak_root / f"{root.name}_{idx}"
    shutil.copytree(root, dst)
    return dst


def list_images(images_dir: Path) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def update_yaml_val(yaml_path: Path) -> None:
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    updated = False
    for i, line in enumerate(lines):
        if line.strip().startswith("val:"):
            lines[i] = "val: images/val"
            updated = True
            break
    if not updated:
        lines.append("val: images/val")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.dataset_yaml.exists():
        raise FileNotFoundError(f"找不到数据集配置: {args.dataset_yaml}")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("val-ratio 必须在 0 到 1 之间")

    images_dir, labels_dir, val_images_dir, val_labels_dir = parse_dataset_yaml(args.dataset_yaml)
    root = images_dir.parents[1] if images_dir.name == "train" else images_dir.parent.parent
    if not args.no_backup:
        dst = backup_dataset(root)
        print(f"已备份原始数据集到: {dst}")
    if not images_dir.exists():
        raise FileNotFoundError(f"训练图片目录不存在: {images_dir}")
    if not labels_dir.exists():
        print(f"标签目录不存在: {labels_dir}，将仅复制图片并生成空标签")

    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    if not images:
        print("训练集图片为空，未进行分割。")
        return

    rng = random.Random(args.seed)
    rng.shuffle(images)
    val_count = max(1, int(round(len(images) * args.val_ratio)))
    selected = images[:val_count]

    copied = 0
    for img_path in selected:
        dst_img = val_images_dir / img_path.name
        if dst_img.exists():
            continue
        shutil.copy2(img_path, dst_img)
        label_src = labels_dir / f"{img_path.stem}.txt"
        label_dst = val_labels_dir / f"{img_path.stem}.txt"
        if label_src.exists():
            shutil.copy2(label_src, label_dst)
        else:
            label_dst.write_text("", encoding="utf-8")
        copied += 1

    if args.update_yaml:
        update_yaml_val(args.dataset_yaml)

    print(f"共扫描 {len(images)} 张图片")
    print(f"验证集复制 {copied} 张 -> {val_images_dir}")
    if args.update_yaml:
        print(f"已更新 yaml: {args.dataset_yaml}")
    else:
        print("未修改 yaml，可手动设置 val: images/val")


if __name__ == "__main__":
    main()
