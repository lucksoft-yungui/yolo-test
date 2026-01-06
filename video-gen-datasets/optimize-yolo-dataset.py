import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class LabelIssue:
    reason: str
    image_path: Path
    label_path: Path
    class_id: str
    class_name: str
    x: str
    y: str
    w: str
    h: str
    area: str
    line: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描 YOLO 数据集并输出可优化项")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="数据集 yaml 路径")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录，默认 <dataset_root>/analysis",
    )
    parser.add_argument(
        "--big-area",
        type=float,
        default=0.6,
        help="标记过大框阈值 (w*h)，默认 0.6",
    )
    parser.add_argument(
        "--small-area",
        type=float,
        default=0.002,
        help="标记过小框阈值 (w*h)，默认 0.002",
    )
    parser.add_argument(
        "--export-review",
        action="store_true",
        help="导出可疑样本到 review 目录便于复核",
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
        raw = line
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "names:":
            in_names = True
            continue
        if in_names:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if not key.isdigit():
                continue
            name = value.strip().strip('"').strip("'")
            if name:
                names[int(key)] = name
    return names


def parse_dataset_yaml(yaml_path: Path) -> Tuple[Path, Path, Path, Dict[int, str]]:
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
    names = parse_names_from_yaml(yaml_path)
    return root, images_dir, labels_dir, names


def list_images(images_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    if not images_dir.exists():
        return []
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def parse_label_lines(label_path: Path) -> Iterable[Tuple[str, float, float, float, float, str]]:
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            yield ("", float("nan"), float("nan"), float("nan"), float("nan"), raw_line)
            continue
        try:
            cls = parts[0]
            x, y, w, h = map(float, parts[1:])
            yield (cls, x, y, w, h, raw_line)
        except ValueError:
            yield ("", float("nan"), float("nan"), float("nan"), float("nan"), raw_line)


def is_empty_label(label_path: Path) -> bool:
    if not label_path.exists():
        return True
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return False
    return True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_list(path: Path, items: Iterable[str]) -> None:
    path.write_text("\n".join(items), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.dataset_yaml.exists():
        raise FileNotFoundError(f"找不到数据集配置: {args.dataset_yaml}")

    root, images_dir, labels_dir, names = parse_dataset_yaml(args.dataset_yaml)
    out_dir = args.out_dir or (root / "analysis")
    ensure_dir(out_dir)

    images = list_images(images_dir)
    total_images = len(images)

    missing_labels: List[str] = []
    empty_labels: List[str] = []
    issues: List[LabelIssue] = []
    invalid_lines: List[LabelIssue] = []
    class_counts: Counter[str] = Counter()
    area_values: List[float] = []
    issue_counts: Counter[str] = Counter()
    suspicious_images: set = set()

    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(str(img_path))
            continue
        if is_empty_label(label_path):
            empty_labels.append(str(img_path))
            continue
        for cls, x, y, w, h, raw_line in parse_label_lines(label_path):
            if not cls or any(v != v for v in (x, y, w, h)):
                invalid_lines.append(
                    LabelIssue(
                        reason="invalid_line",
                        image_path=img_path,
                        label_path=label_path,
                        class_id="",
                        class_name="",
                        x="",
                        y="",
                        w="",
                        h="",
                        area="",
                        line=raw_line,
                    )
                )
                suspicious_images.add(img_path.stem)
                issue_counts["invalid_line"] += 1
                continue

            class_counts[cls] += 1
            area = w * h
            area_values.append(area)

            reasons = []
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                reasons.append("out_of_bounds")
            if area >= args.big_area:
                reasons.append("large_box")
            if area <= args.small_area:
                reasons.append("small_box")

            if reasons:
                suspicious_images.add(img_path.stem)
            for reason in reasons:
                issues.append(
                    LabelIssue(
                        reason=reason,
                        image_path=img_path,
                        label_path=label_path,
                        class_id=cls,
                        class_name=names.get(int(cls), ""),
                        x=f"{x:.6f}",
                        y=f"{y:.6f}",
                        w=f"{w:.6f}",
                        h=f"{h:.6f}",
                        area=f"{area:.6f}",
                        line=raw_line,
                    )
                )
                issue_counts[reason] += 1

    area_min = min(area_values) if area_values else 0.0
    area_max = max(area_values) if area_values else 0.0
    area_mean = sum(area_values) / len(area_values) if area_values else 0.0

    write_list(out_dir / "missing_labels.txt", missing_labels)
    write_list(out_dir / "empty_labels.txt", empty_labels)

    suspicious_csv = out_dir / "suspicious.csv"
    with suspicious_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "reason",
                "image",
                "label",
                "class_id",
                "class_name",
                "x",
                "y",
                "w",
                "h",
                "area",
                "line",
            ]
        )
        for item in issues + invalid_lines:
            writer.writerow(
                [
                    item.reason,
                    str(item.image_path),
                    str(item.label_path),
                    item.class_id,
                    item.class_name,
                    item.x,
                    item.y,
                    item.w,
                    item.h,
                    item.area,
                    item.line,
                ]
            )

    summary = {
        "dataset_root": str(root),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "total_images": total_images,
        "missing_labels": len(missing_labels),
        "empty_labels": len(empty_labels),
        "total_boxes": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "area_stats": {
            "min": area_min,
            "max": area_max,
            "mean": area_mean,
        },
        "thresholds": {
            "big_area": args.big_area,
            "small_area": args.small_area,
        },
        "issue_counts": dict(issue_counts),
        "outputs": {
            "missing_labels": str(out_dir / "missing_labels.txt"),
            "empty_labels": str(out_dir / "empty_labels.txt"),
            "suspicious_csv": str(suspicious_csv),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    summary_lines = [
        f"dataset_root: {root}",
        f"images_dir: {images_dir}",
        f"labels_dir: {labels_dir}",
        f"total_images: {total_images}",
        f"missing_labels: {len(missing_labels)}",
        f"empty_labels: {len(empty_labels)}",
        f"total_boxes: {sum(class_counts.values())}",
        "class_counts:",
    ]
    for cls_id, count in sorted(class_counts.items()):
        name = names.get(int(cls_id), "")
        label = f"{cls_id} ({name})" if name else cls_id
        summary_lines.append(f"  {label}: {count}")
    summary_lines.extend(
        [
            f"area_min: {area_min:.6f}",
            f"area_max: {area_max:.6f}",
            f"area_mean: {area_mean:.6f}",
            f"big_area_threshold: {args.big_area}",
            f"small_area_threshold: {args.small_area}",
            "issue_counts:",
        ]
    )
    for reason, count in sorted(issue_counts.items()):
        summary_lines.append(f"  {reason}: {count}")
    summary_lines.extend(
        [
            f"outputs: {out_dir}",
            f"  missing_labels.txt",
            f"  empty_labels.txt",
            f"  suspicious.csv",
            f"  summary.json",
        ]
    )
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    if args.export_review and suspicious_images:
        review_images_dir = out_dir / "review" / "images"
        review_labels_dir = out_dir / "review" / "labels"
        ensure_dir(review_images_dir)
        ensure_dir(review_labels_dir)
        for stem in sorted(suspicious_images):
            img_src = images_dir / f"{stem}.jpg"
            if not img_src.exists():
                for ext in (".jpeg", ".png", ".bmp"):
                    alt = images_dir / f"{stem}{ext}"
                    if alt.exists():
                        img_src = alt
                        break
            if img_src.exists():
                img_dst = review_images_dir / img_src.name
                img_dst.write_bytes(img_src.read_bytes())
            label_src = labels_dir / f"{stem}.txt"
            label_dst = review_labels_dir / f"{stem}.txt"
            if label_src.exists():
                label_dst.write_text(label_src.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                label_dst.write_text("", encoding="utf-8")

    print(f"完成: 输出目录 {out_dir}")


if __name__ == "__main__":
    main()
