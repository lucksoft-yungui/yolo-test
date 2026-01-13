import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 HTTP 流录制为 MP4")
    parser.add_argument(
        "--url",
        default="http://ai-tim.zju-qz.edu.cn/media0/proxy/ipc-hg7i64qn.live.flv",
        help="HTTP 流媒体地址",
    )
    parser.add_argument(
        "--output-dir",
        default="videos",
        help="输出目录，默认保存到 videos/ 目录",
    )
    return parser.parse_args()


def open_capture(url: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def resolve_fps(cap: cv2.VideoCapture, fallback: float = 25.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        return fallback
    return fps


def build_output_path(output_dir: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir).expanduser()
    return out_dir / f"http_{ts}.mp4"


def prepare_writer(
    output_path: Path, fps: float, frame_size: Tuple[int, int]
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {output_path}")
    return writer


def record_stream(url: str, output_dir: str) -> None:
    cap = open_capture(url)
    if cap is None:
        print("无法连接流媒体，结束。")
        return

    writer = None
    output_path = build_output_path(output_dir)

    window = "HTTP Recorder"
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取帧失败，停止录制。")
                break

            if writer is None:
                fps = resolve_fps(cap)
                frame_size = (frame.shape[1], frame.shape[0])
                writer = prepare_writer(output_path, fps, frame_size)
                print(f"开始录制: {output_path} (fps={fps:.2f}, size={frame_size})")

            writer.write(frame)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("手动停止录制。")
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    record_stream(args.url, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到中断信号，已退出。")
        sys.exit(0)
