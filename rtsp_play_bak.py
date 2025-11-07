import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="播放 RTSP 视频流")
    parser.add_argument(
        "--url",
        default="rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1",
        help="RTSP 流地址",
    )
    parser.add_argument("--window", default="RTSP Stream", help="展示窗口名称")
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
    parser.add_argument("--buffer", type=int, default=None, help="OpenCV 缓冲帧数量")
    parser.add_argument(
        "--protocol",
        choices=("udp", "tcp"),
        default=None,
        help="RTSP 传输协议，默认根据硬件自动选择",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="FFmpeg stimeout（秒），默认由自动策略决定",
    )
    parser.add_argument(
        "--max-delay",
        type=int,
        default=None,
        help="FFmpeg max_delay（微秒），默认由自动策略决定",
    )
    parser.add_argument(
        "--keep-all",
        dest="keep_all",
        action="store_true",
        help="保留缓冲区所有帧（延迟更大）",
    )
    parser.add_argument(
        "--drop-old",
        dest="keep_all",
        action="store_false",
        help="丢弃旧帧（延迟更低）",
    )
    parser.set_defaults(keep_all=None)
    parser.add_argument(
        "--flush-window",
        type=float,
        default=None,
        help="丢弃旧帧时的抓帧时间窗口（秒），默认由自动策略决定",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="打印最终使用的解码参数，便于排查",
    )
    return parser.parse_args()


@dataclass
class AutoConfig:
    buffer_size: int
    protocol: str
    timeout_s: float
    max_delay_us: int
    drop_old: bool
    flush_window_s: float


def detect_hardware_profile() -> dict:
    uname = platform.uname()
    arch = (uname.machine or "").lower()
    cpu_brand = platform.processor() or uname.processor or ""
    logical_cores = os.cpu_count() or 1
    is_apple_silicon = "arm" in arch and uname.system == "Darwin"
    is_low_power = logical_cores <= 4
    is_mid = 4 < logical_cores <= 8
    return {
        "system": uname.system,
        "arch": arch,
        "cpu_brand": cpu_brand,
        "cores": logical_cores,
        "is_apple_silicon": is_apple_silicon,
        "is_low_power": is_low_power,
        "is_mid": is_mid,
    }


def auto_tune(profile: dict) -> AutoConfig:
    if profile["is_low_power"]:
        # 更保守：增大缓冲、使用 tcp 提升稳定性
        return AutoConfig(
            buffer_size=3,
            protocol="tcp",
            timeout_s=8.0,
            max_delay_us=300000,
            drop_old=False,
            flush_window_s=0.0,
        )

    if profile["is_mid"]:
        return AutoConfig(
            buffer_size=2,
            protocol="udp",
            timeout_s=5.0,
            max_delay_us=200000,
            drop_old=True,
            flush_window_s=0.08,
        )

    # 高性能机器（>= 9 cores），包含 Apple Silicon、桌面工作站
    protocol = "udp"
    if profile["is_apple_silicon"]:
        # macOS 上 UDP 更快，但在复杂网络环境下可退回 tcp
        protocol = "udp"
    return AutoConfig(
        buffer_size=1,
        protocol=protocol,
        timeout_s=3.0,
        max_delay_us=120000,
        drop_old=True,
        flush_window_s=0.05,
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


def read_latest_frame(
    cap: cv2.VideoCapture,
    drop_old: bool,
    flush_window_s: float,
) -> Tuple[bool, Optional[cv2.Mat]]:
    if not drop_old or flush_window_s <= 0:
        return cap.read()

    deadline = time.perf_counter() + flush_window_s
    grabbed_any = False
    # grab() 不解码帧，只向前推进缓冲；最后只解码一次 retrieve()
    while time.perf_counter() < deadline:
        if not cap.grab():
            break
        grabbed_any = True

    if grabbed_any:
        return cap.retrieve()
    return cap.read()


def play_stream(
    url: str,
    window: str,
    reconnect: int,
    wait_seconds: float,
    buffer_size: int,
    protocol: str,
    timeout_s: float,
    max_delay_us: int,
    drop_old: bool,
    flush_window_s: float,
) -> None:
    attempts = 0
    cap = open_capture(url, protocol, timeout_s, max_delay_us, buffer_size)

    while attempts <= reconnect:
        if cap is None:
            attempts += 1
            if attempts > reconnect:
                print("无法连接 RTSP 流，结束。")
                break
            print(f"连接失败，{wait_seconds} 秒后尝试第 {attempts} 次重连...")
            time.sleep(wait_seconds)
            cap = open_capture(url, protocol, timeout_s, max_delay_us, buffer_size)
            continue

        ret, frame = read_latest_frame(cap, drop_old, flush_window_s)
        if not ret:
            print("读取帧失败，尝试重新连接...")
            cap.release()
            cap = None
            continue

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            print("退出播放。")
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    profile = detect_hardware_profile()
    auto_cfg = auto_tune(profile)

    buffer_size = args.buffer if args.buffer is not None else auto_cfg.buffer_size
    protocol = args.protocol if args.protocol is not None else auto_cfg.protocol
    timeout_s = args.timeout if args.timeout is not None else auto_cfg.timeout_s
    max_delay_us = args.max_delay if args.max_delay is not None else auto_cfg.max_delay_us
    flush_window_s = (
        args.flush_window if args.flush_window is not None else auto_cfg.flush_window_s
    )
    if args.keep_all is None:
        drop_old = auto_cfg.drop_old
    else:
        drop_old = not args.keep_all

    if args.show_config:
        print(
            f"硬件: {profile['system']} {profile['arch']} | CPU cores: {profile['cores']} | "
            f"Apple Silicon: {profile['is_apple_silicon']}"
        )
        print(
            "解码参数:"
            f" protocol={protocol}, buffer={buffer_size}, timeout={timeout_s}s,"
            f" max_delay={max_delay_us}us, drop_old={drop_old}, flush_window={flush_window_s}s"
        )

    play_stream(
        url=args.url,
        window=args.window,
        reconnect=args.reconnect,
        wait_seconds=args.wait,
        buffer_size=buffer_size,
        protocol=protocol,
        timeout_s=timeout_s,
        max_delay_us=max_delay_us,
        drop_old=drop_old,
        flush_window_s=flush_window_s,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到中断信号，已退出。")
        sys.exit(0)
