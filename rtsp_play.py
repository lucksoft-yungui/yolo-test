import argparse
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Optional

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="播放 RTSP 视频流")
    parser.add_argument(
        "--model",
        default="model/best.pt",
        help="YOLO 模型文件路径，用于绘制识别框。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="最小置信度阈值，低于该值的检测将被过滤，且不会低于 0.6。",
    )
    parser.add_argument(
        "--url",
        default="rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1",
        help="RTSP 流地址",
    )
    parser.add_argument(
        "--window",
        default="RTSP Stream",
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


class AsyncAnnotator:
    def __init__(self, model: YOLO, conf_threshold: float) -> None:
        self.model = model
        self.conf_threshold = conf_threshold
        self.frames = Queue(maxsize=1)
        self.latest = None
        self.running = True
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, frame):
        if not self.running:
            return
        try:
            # 丢弃旧帧，始终以最新帧做推理。
            self.frames.get_nowait()
        except Empty:
            pass
        try:
            self.frames.put_nowait(frame.copy())
        except Full:
            pass

    def get_latest(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self) -> None:
        self.running = False
        try:
            self.frames.put_nowait(None)
        except Full:
            pass
        self.worker.join(timeout=1)

    def _loop(self) -> None:
        while self.running:
            try:
                frame = self.frames.get(timeout=0.5)
            except Empty:
                continue
            if frame is None:
                continue
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            annotated = results[0].plot() if results else frame
            with self.lock:
                self.latest = annotated


def play_stream(
    url: str,
    window: str,
    reconnect: int,
    wait_seconds: float,
    model: YOLO,
    conf_threshold: float,
) -> None:
    attempts = 0
    cap = open_capture(url)
    annotator = AsyncAnnotator(model, conf_threshold)

    try:
        while attempts <= reconnect:
            if cap is None:
                attempts += 1
                if attempts > reconnect:
                    print("无法连接 RTSP 流，结束。")
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

            annotator.submit(frame)
            latest = annotator.get_latest()
            annotated = latest if latest is not None else frame

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("退出播放。")
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        annotator.stop()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")

    conf_threshold = max(args.conf, 0.6)
    model = YOLO(str(model_path))

    play_stream(args.url, args.window, args.reconnect, args.wait, model, conf_threshold)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到中断信号，已退出。")
        sys.exit(0)
