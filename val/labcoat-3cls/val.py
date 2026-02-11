import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Any

from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 labcoat-3cls 模型对图片目录进行验证，并输出标准化类别结果。"
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("val/labcoat-3cls/images"),
        help="待检测图片目录，默认 val/labcoat-3cls/images",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model/labcoat-3cls/best.pt"),
        help="模型权重路径，默认 model/labcoat-3cls/best.pt",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("labcoat-3cls.yaml"),
        help="类别配置 YAML，默认 labcoat-3cls.yaml",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.6,
        help="置信度阈值，默认 0.6",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备，例如 cpu / cuda / mps，不填则自动选择",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("val/labcoat-3cls/predictions.json"),
        help="输出 JSON 文件路径，默认 val/labcoat-3cls/predictions.json",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="是否保存可视化结果到 val/labcoat-3cls/annotated",
    )
    return parser.parse_args()


def load_class_names(yaml_path: Path) -> list[str]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"未找到类别配置文件: {yaml_path}")

    names: dict[int, str] = {}
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not (line.startswith(" ") or line.startswith("\t")):
                break
            match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if not match:
                continue
            idx = int(match.group(1))
            name = match.group(2).strip()
            if len(name) >= 2 and name[0] in {"'", '"'} and name[-1] == name[0]:
                name = name[1:-1]
            names[idx] = name

    if not names:
        raise ValueError(f"未能从 {yaml_path} 解析到类别名称")

    return [names[i] for i in sorted(names)]


def collect_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"未找到图片目录: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"图片路径不是目录: {images_dir}")
    images = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def save_json(output_path: Path, data: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    class_names = load_class_names(args.yaml)
    images = collect_images(args.images)
    if not images:
        raise FileNotFoundError(f"目录中未找到图片: {args.images}")

    if not args.model.exists():
        raise FileNotFoundError(f"未找到模型文件: {args.model}")

    model = YOLO(str(args.model))
    model.model.names = {i: name for i, name in enumerate(class_names)}
    if args.device:
        model.to(args.device)

    vis_dir = Path("val/labcoat-3cls/annotated")
    if args.save_vis:
        if vis_dir.exists():
            shutil.rmtree(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    results_payload: list[dict[str, Any]] = []
    summary_counts = {name: 0 for name in class_names}

    for image_path in images:
        results = model(str(image_path), conf=args.conf, verbose=False)
        if not results:
            continue
        result = results[0]

        detections: list[dict[str, Any]] = []
        unique_classes: set[str] = set()

        if result.boxes is not None and len(result.boxes):
            for box, cls, conf in zip(
                result.boxes.xyxy, result.boxes.cls.int(), result.boxes.conf
            ):
                class_id = int(cls)
                class_name = (
                    class_names[class_id] if class_id < len(class_names) else str(class_id)
                )
                unique_classes.add(class_name)
                summary_counts[class_name] = summary_counts.get(class_name, 0) + 1
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(conf),
                        "box": [x1, y1, x2, y2],
                    }
                )

        results_payload.append(
            {
                "image": str(image_path),
                "classes": sorted(unique_classes),
                "detections": detections,
            }
        )

        if args.save_vis:
            annotated = result.plot()
            relative_path = image_path.relative_to(args.images)
            out_path = vis_dir / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # 通过结果自带的 BGR 图直接写入
            import cv2

            cv2.imwrite(str(out_path), annotated)

        class_list = ", ".join(sorted(unique_classes)) if unique_classes else "无检测"
        print(f"{image_path.name}: {class_list}")

    output_data = {
        "model": str(args.model),
        "images_dir": str(args.images),
        "class_names": class_names,
        "summary": summary_counts,
        "results": results_payload,
    }
    save_json(args.output, output_data)
    print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()
