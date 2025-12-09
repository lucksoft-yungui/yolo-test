import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2


# OpenCV 默认字体不支持中文，提示改为英文避免乱码
PROMPT_INFO = "Drag to draw fire boxes; r/c clear; s/space save; n skip; q/ESC quit."


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手工标注火焰数据集（1秒一帧，手动框）")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("videos") / "fire.mp4",
        help="输入视频路径，默认 videos/fire.mp4",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/fire"),
        help="数据集输出目录，默认 datasets/fire",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=1.0,
        help="抽帧时间间隔（秒），默认1秒一帧",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=None,
        help="抽帧步长（帧数），设置后优先于 interval-sec；默认自动按 fps 计算",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="起始帧序号，默认0从头开始",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最大处理帧数，默认None处理到视频结束",
    )
    return parser.parse_args()


def prepare_output_dirs(dataset_dir: Path):
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def ensure_dataset_yaml(dataset_dir: Path) -> None:
    yaml_path = dataset_dir / "fire.yaml"
    if yaml_path.exists():
        return
    content = "\n".join(
        [
            "# Auto-generated YOLO dataset config for fire detection",
            f"path: {dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/train  # 如有验证集可改为 images/val",
            "",
            "names:",
            "  0: fire",
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"已生成数据集配置: {yaml_path}")


def yolo_line(box: Box, width: int, height: int) -> str:
    cx = (box.x1 + box.x2) / 2.0 / width
    cy = (box.y1 + box.y2) / 2.0 / height
    bw = (box.x2 - box.x1) / width
    bh = (box.y2 - box.y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def annotate_frame(frame, window_name: str) -> Tuple[List[Box], str]:
    state = {"boxes": [], "drawing": False, "start": (0, 0), "current": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["start"] = (x, y)
            state["current"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            state["current"] = (state["start"][0], state["start"][1], x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            x1, y1, x2, y2 = state["start"][0], state["start"][1], x, y
            if abs(x2 - x1) > 2 and abs(y2 - y1) > 2:
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                state["boxes"].append(Box(x1, y1, x2, y2))
            state["current"] = None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        canvas = frame.copy()
        for b in state["boxes"]:
            cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)
        if state["current"]:
            x1, y1, x2, y2 = state["current"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 1)

        cv2.putText(canvas, PROMPT_INFO, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("s"), 32):  # 保存
            return state["boxes"], "save"
        if key == ord("n"):  # 跳过
            return state["boxes"], "skip"
        if key in (ord("r"), ord("c")):
            state["boxes"].clear()
            state["current"] = None
        if key in (ord("q"), 27):
            return state["boxes"], "quit"


def main() -> None:
    args = parse_args()
    images_dir, labels_dir = prepare_output_dirs(args.dataset)
    ensure_dataset_yaml(args.dataset)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {args.video}")

    frame_step = args.frame_step
    if frame_step is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = fps if fps > 0 else 25
        frame_step = max(1, int(round(fps * args.interval_sec)))
    print(f"抽帧步长: 每 {frame_step} 帧处理一次（fps≈{cap.get(cv2.CAP_PROP_FPS) or '未知'}, interval={args.interval_sec}s）")

    frame_idx = -1
    saved_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx < args.start_frame:
                continue
            if (frame_idx - args.start_frame) % frame_step != 0:
                continue
            if args.max_frames is not None and saved_count >= args.max_frames:
                break

            image_name = f"frame_{frame_idx:06d}.jpg"
            label_name = f"frame_{frame_idx:06d}.txt"

            boxes, action = annotate_frame(frame, "Fire Labeler")
            if action == "quit":
                print("用户退出，结束标注。")
                break
            if action == "skip":
                print(f"跳过帧 {frame_idx}")
                continue

            cv2.imwrite(str(images_dir / image_name), frame)
            width, height = frame.shape[1], frame.shape[0]
            lines = [yolo_line(b, width, height) for b in boxes]
            (labels_dir / label_name).write_text("\n".join(lines), encoding="utf-8")

            saved_count += 1
            print(f"[{saved_count}] 已保存 {image_name}，标注 {len(lines)} 条")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"完成！共处理 {saved_count} 帧，数据集位于: {args.dataset}")


if __name__ == "__main__":
    main()
