import argparse
from pathlib import Path
from typing import Union

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用摄像头实时视频流进行检测并显示识别框。")
    parser.add_argument(
        "--model",
        default="model/best.pt",
        help="模型文件路径，默认使用训练得到的 best.pt。",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="视频源。默认 0 表示系统默认摄像头，也可以传入视频文件路径。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="最小置信度阈值，低于该阈值的检测将被过滤，且不会低于 0.5。",
    )
    return parser.parse_args()


def resolve_source(source: str) -> Union[int, str]:
    if source.isdigit():
        return int(source)
    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"未找到视频源：{source_path}")
    return str(source_path)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser()

    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")

    video_source = resolve_source(args.source)

    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        raise RuntimeError("无法打开视频源，请检查摄像头或视频文件。")

    window_name = "YOLO Camera Detection"
    conf_threshold = max(args.conf, 0.6)

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                print("未能从视频源读取到帧，退出。")
                break

            results = model(frame, conf=conf_threshold, verbose=False)
            annotated = results[0].plot() if results else frame

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
