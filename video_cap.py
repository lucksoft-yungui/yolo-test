import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认0")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="抓取间隔（秒），默认2秒",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train-img"),
        help="图片存储目录，默认 train-img",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    capturing = False
    last_capture = time.monotonic()
    window_name = "Camera Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("按 s 开始抓取，按 e 停止抓取，按 q 退出。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头数据，退出。")
                break

            status_text = f"状态: {'抓取中' if capturing else '等待'} | 间隔: {args.interval:.1f}s"
            cv2.putText(
                frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if capturing else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, frame)

            if capturing:
                now = time.monotonic()
                if now - last_capture >= args.interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = output_dir / f"{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    last_capture = now
                    print(f"已保存: {filename}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if not capturing:
                    capturing = True
                    last_capture = time.monotonic() - args.interval
                    print("开始抓取")
            if key == ord("e"):
                if capturing:
                    capturing = False
                    print("停止抓取")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
