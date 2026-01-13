import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 RTSP 流录制为 MP4")
    parser.add_argument(
        "--url",
        default="rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1",
        help="RTSP 流地址",
    )
    parser.add_argument(
        "--output-dir",
        default="videos",
        help="输出目录，默认保存到 videos/ 目录",
    )
    return parser.parse_args()


@dataclass
class AutoConfig:
    buffer_size: int
    protocol: str
    timeout_s: float
    max_delay_us: int


def detect_hardware_profile() -> dict:
    uname = platform.uname()
    arch = (uname.machine or "").lower()
    logical_cores = os.cpu_count() or 1
    is_apple_silicon = "arm" in arch and uname.system == "Darwin"
    is_low_power = logical_cores <= 4
    is_mid = 4 < logical_cores <= 8
    return {
        "system": uname.system,
        "arch": arch,
        "cores": logical_cores,
        "is_apple_silicon": is_apple_silicon,
        "is_low_power": is_low_power,
        "is_mid": is_mid,
    }


def auto_tune(profile: dict) -> AutoConfig:
    if profile["is_low_power"]:
        return AutoConfig(
            buffer_size=3,
            protocol="tcp",
            timeout_s=8.0,
            max_delay_us=300000,
        )

    if profile["is_mid"]:
        return AutoConfig(
            buffer_size=2,
            protocol="udp",
            timeout_s=5.0,
            max_delay_us=200000,
        )

    protocol = "udp"
    if profile["is_apple_silicon"]:
        protocol = "udp"
    return AutoConfig(
        buffer_size=1,
        protocol=protocol,
        timeout_s=3.0,
        max_delay_us=120000,
    )


def build_ffmpeg_options(protocol: str, timeout_s: float, max_delay_us: int) -> str:
    options = []
    if protocol:
        options.append(f"rtsp_transport;{protocol}")
    if timeout_s > 0:
        options.append(f"stimeout;{int(timeout_s * 1_000_000)}")
    if max_delay_us > 0:
        options.append(f"max_delay;{max_delay_us}")
    return "|".join(options)


def open_capture(
    url: str,
    protocol: str,
    timeout_s: float,
    max_delay_us: int,
    buffer_size: int,
) -> Optional[cv2.VideoCapture]:
    ffmpeg_options = build_ffmpeg_options(protocol, timeout_s, max_delay_us)
    if ffmpeg_options:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_options

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if buffer_size > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
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
    return out_dir / f"rtsp_{ts}.mp4"


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
    profile = detect_hardware_profile()
    auto_cfg = auto_tune(profile)

    cap = open_capture(
        url,
        protocol=auto_cfg.protocol,
        timeout_s=auto_cfg.timeout_s,
        max_delay_us=auto_cfg.max_delay_us,
        buffer_size=auto_cfg.buffer_size,
    )
    if cap is None:
        print("无法连接 RTSP 流，结束。")
        return

    writer = None
    output_path = build_output_path(output_dir)

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
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def main() -> None:
    args = parse_args()
    record_stream(args.url, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到中断信号，已退出。")
        sys.exit(0)
