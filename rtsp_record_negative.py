import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 RTSP 流录制负样本视频")
    parser.add_argument(
        "--url",
        default="rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1",
        help="RTSP 流地址",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出视频路径，默认保存到 videos/neg_YYYYmmdd_HHMMSS.mp4",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="视频编码 fourcc，默认 mp4v",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="当流 FPS 获取失败时使用的默认帧率",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="录制时长（秒），0 表示手动停止",
    )
    parser.add_argument(
        "--window",
        default="RTSP Recorder",
        help="展示窗口名称",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不显示窗口（适合后台录制）",
    )
    parser.add_argument(
        "--reconnect",
        type=int,
        default=5,
        help="读取失败时重新连接的尝试次数，0 表示不重试",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=2.0,
        help="重新连接前的等待秒数",
    )
    return parser.parse_args()


def open_capture(url: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def build_output_path(output: Optional[str]) -> Path:
    if output:
        return Path(output).expanduser()
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("videos") / f"neg_{ts}.mp4"


def resolve_fps(cap: cv2.VideoCapture, fallback: float) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        return fallback
    return fps


def prepare_writer(
    output_path: Path, codec: str, fps: float, frame_size: Tuple[int, int]
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {output_path}")
    return writer


def record_stream(args: argparse.Namespace) -> None:
    attempts = 0
    cap = open_capture(args.url)
    writer = None
    output_path = build_output_path(args.output)
    start_time = time.monotonic()

    try:
        while attempts <= args.reconnect:
            if cap is None:
                attempts += 1
                if attempts > args.reconnect:
                    print("无法连接 RTSP 流，结束。")
                    break
                print(f"连接失败，{args.wait} 秒后尝试第 {attempts} 次重连...")
                time.sleep(args.wait)
                cap = open_capture(args.url)
                continue

            ret, frame = cap.read()
            if not ret:
                print("读取帧失败，尝试重新连接...")
                cap.release()
                cap = None
                continue

            if writer is None:
                fps = resolve_fps(cap, args.fps)
                frame_size = (frame.shape[1], frame.shape[0])
                writer = prepare_writer(output_path, args.codec, fps, frame_size)
                print(f"开始录制: {output_path} (fps={fps:.2f}, size={frame_size})")

            writer.write(frame)

            if not args.no_show:
                cv2.imshow(args.window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("手动停止录制。")
                    break

            if args.duration > 0 and (time.monotonic() - start_time) >= args.duration:
                print("达到设定时长，停止录制。")
                break
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    record_stream(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到中断信号，已退出。")
        sys.exit(0)
