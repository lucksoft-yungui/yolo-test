import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="播放视频并用训练好的火焰模型绘制目标框")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("videos") / "fire.mp4",
        help="源视频路径，默认 videos/fire.mp4",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model/fire/weights/best.pt"),
        help="训练好的模型权重，默认 model/fire/weights/best.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="置信度阈值，默认0.01（可调高降低误检）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备，例如 cpu / cuda / mps，不填则自动选择",
    )
    return parser.parse_args()


def load_model(weight_path: Path, device: str | None, class_names: list[str]) -> YOLO:
    if not weight_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {weight_path}")
    model = YOLO(str(weight_path))
    if class_names:
        model.model.names = {i: name for i, name in enumerate(class_names)}
    if device:
        model.to(device)
    return model


def main() -> None:
    args = parse_args()
    names = ["fire"]
    model = load_model(args.model, args.device, names)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {args.video}")

    window_name = "Fire Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("按 q 或 ESC 退出播放。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=args.conf, verbose=False)
            annotated = frame.copy()
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{names[cls_id] if cls_id < len(names) else cls_id} {conf:.2f}"
                    color = (0, 0, 255)
                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
