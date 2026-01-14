import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTSP 抓取性能评估")
    parser.add_argument(
        "--url",
        default="rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1",
        help="RTSP 流地址",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/rtsp-cap",
        help="抓取图片输出目录",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="连续抓取次数",
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


def benchmark_rtsp(url: str, output_dir: str, count: int) -> None:
    start_ts = time.perf_counter()
    profile = detect_hardware_profile()
    auto_cfg = auto_tune(profile)
    startup_ts = time.perf_counter()

    connect_start = time.perf_counter()
    cap = open_capture(
        url,
        protocol=auto_cfg.protocol,
        timeout_s=auto_cfg.timeout_s,
        max_delay_us=auto_cfg.max_delay_us,
        buffer_size=auto_cfg.buffer_size,
    )
    connect_end = time.perf_counter()

    if cap is None:
        print("无法连接 RTSP 流，结束。")
        return

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_ts = time.strftime("%Y%m%d_%H%M%S")

    print(f"启动耗时: {(startup_ts - start_ts) * 1000:.2f} ms")
    print(f"建立连接耗时: {(connect_end - connect_start) * 1000:.2f} ms")

    first_read_logged = False
    try:
        for idx in range(count):
            grab_start = time.perf_counter()
            ret, frame = cap.read()
            grab_end = time.perf_counter()
            if not ret:
                print(f"第 {idx + 1} 次读取失败，停止。")
                break

            if not first_read_logged:
                first_read_logged = True
                print(f"首次读取耗时: {(grab_end - connect_end) * 1000:.2f} ms")

            save_start = time.perf_counter()
            output_path = out_dir / f"rtsp_cap_{base_ts}_{idx + 1:02d}.jpg"
            cv2.imwrite(str(output_path), frame)
            save_end = time.perf_counter()

            grab_ms = (grab_end - grab_start) * 1000
            save_ms = (save_end - save_start) * 1000
            print(
                f"[{idx + 1}/{count}] 读取耗时 {grab_ms:.2f} ms, "
                f"保存耗时 {save_ms:.2f} ms -> {output_path}"
            )
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    benchmark_rtsp(args.url, args.output_dir, args.count)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到中断信号，已退出。")
        sys.exit(0)
