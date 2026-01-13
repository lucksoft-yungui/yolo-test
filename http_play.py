import argparse
import sys
import time
from typing import Optional

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="播放 HTTP 流媒体")
    parser.add_argument(
        "--url",
        default="http://ai-tim.zju-qz.edu.cn/media0/proxy/ipc-hg7i64qn.live.flv",
        help="HTTP 流媒体地址",
    )
    parser.add_argument(
        "--window",
        default="HTTP Stream",
        help="展示窗口名称",
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


def play_stream(url: str, window: str, reconnect: int, wait_seconds: float) -> None:
    attempts = 0
    cap = open_capture(url)

    while attempts <= reconnect:
        if cap is None:
            attempts += 1
            if attempts > reconnect:
                print("无法连接流媒体，结束。")
                break
            print(f"连接失败，{wait_seconds} 秒后尝试第 {attempts} 次重连...")
            time.sleep(wait_seconds)
            cap = open_capture(url)
            continue

        ret, frame = cap.read()
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
    play_stream(args.url, args.window, args.reconnect, args.wait)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到中断信号，已退出。")
        sys.exit(0)
