import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通用视频验证脚本：加载模型并绘制检测框。")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="源视频路径，例如 videos/shoe.mp4",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="模型权重路径，例如 model/fire-store/weights/best.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.6,
        help="置信度阈值，默认0.6",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备，例如 cpu / cuda / mps，不填则自动选择",
    )
    return parser.parse_args()


def load_model(weight_path: Path, device: str | None) -> YOLO:
    if not weight_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {weight_path}")
    model = YOLO(str(weight_path))
    if device:
        model.to(device)
    return model


def main() -> None:
    args = parse_args()
    model = load_model(args.model, args.device)

    if not args.video.exists():
        raise FileNotFoundError(f"未找到视频文件: {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {args.video}")

    window_name = "Video Validation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("按 q 或 ESC 退出播放。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=args.conf, verbose=False)
            annotated = results[0].plot() if results else frame

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
