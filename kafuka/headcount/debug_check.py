import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调试单张图片人员检测结果")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="待检测图片路径",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolo11n.pt"),
        help="模型权重路径，默认 yolo11n.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="置信度阈值，默认 0.5",
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=0,
        help="人员类别索引，默认 0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备（cpu / cuda / mps），默认自动",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("kafuka/headcount/debug"),
        help="标注结果保存目录，默认 kafuka/headcount/debug",
    )
    return parser.parse_args()


def draw_boxes(image, boxes, color=(0, 200, 0)) -> object:
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box["xyxy"]]
        label = f"person:{box['conf']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"图片不存在: {args.image}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"读取图片失败: {args.image}")

    if not args.model.exists():
        print(f"模型不存在，尝试自动下载: {args.model}", flush=True)
    model = YOLO(str(args.model))
    if args.device:
        model.to(args.device)

    result = model(image, conf=args.conf, verbose=False)[0]
    person_boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != args.person_class_id:
                continue
            conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            coords = [float(v) for v in box.xyxy[0]]
            person_boxes.append({"conf": conf, "xyxy": coords})

    print(f"检测到人员数量: {len(person_boxes)}")
    for idx, box in enumerate(person_boxes, start=1):
        print(f"#{idx} conf={box['conf']:.3f} xyxy={box['xyxy']}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.save_dir / f"debug_{args.image.name}"
    image = draw_boxes(image, person_boxes)
    cv2.imwrite(str(output_path), image)
    print(f"已保存标注图: {output_path}")


if __name__ == "__main__":
    main()
