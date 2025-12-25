import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2


# OpenCV 默认字体不支持中文，提示改为英文避免乱码
PROMPT_INFO = "Drag to draw boxes; r/c clear; s/space save; n skip; q/ESC quit."


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手工标注视频帧并生成 YOLO 数据集")
    parser.add_argument("--video", type=Path, required=True, help="输入视频路径")
    parser.add_argument("--dataset-name", type=str, required=True, help="数据集名称")
    parser.add_argument("--output-dir", type=Path, required=True, help="数据集输出目录")
    parser.add_argument("--class-name", type=str, required=True, help="类别名称（视频里只有一种物体）")
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
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧序号，默认0从头开始")
    parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数，默认None处理到视频结束")
    return parser.parse_args()


def parse_names_from_yaml(yaml_path: Path) -> dict:
    names = {}
    if not yaml_path.exists():
        return names
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if not key.isdigit():
            continue
        name = value.strip().strip('"').strip("'")
        if name:
            names[int(key)] = name
    return names


def resolve_dataset_dir(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / dataset_name


def prepare_output_dirs(dataset_dir: Path) -> Tuple[Path, Path]:
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def ensure_dataset_yaml(dataset_dir: Path, class_name: str) -> Path:
    yaml_name = f"{dataset_dir.name}.yaml"
    yaml_path = dataset_dir / yaml_name
    if yaml_path.exists():
        return yaml_path
    content = "\n".join(
        [
            f"# Auto-generated YOLO dataset config for {class_name}",
            f"path: {dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/train  # 如有验证集可改为 images/val",
            "",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def ensure_class_in_yaml(yaml_path: Path, class_name: str) -> int:
    names = parse_names_from_yaml(yaml_path)
    for idx, name in names.items():
        if name == class_name:
            return idx
    next_idx = max(names.keys(), default=-1) + 1
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    if "names:" not in [line.strip() for line in lines]:
        lines.append("names:")
    lines.append(f"  {next_idx}: {class_name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return next_idx


def yolo_line(box: Box, width: int, height: int, class_id: int) -> str:
    cx = (box.x1 + box.x2) / 2.0 / width
    cy = (box.y1 + box.y2) / 2.0 / height
    bw = (box.x2 - box.x1) / width
    bh = (box.y2 - box.y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def unique_stem(target_dir: Path, stem: str) -> str:
    candidate = stem
    idx = 1
    while (target_dir / f"{candidate}.jpg").exists() or (target_dir / f"{candidate}.txt").exists():
        candidate = f"{stem}_{idx}"
        idx += 1
    return candidate


def sanitize_prefix(text: str) -> str:
    cleaned = []
    for ch in text.strip():
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    return "".join(cleaned) or "video"


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
        if key in (ord("s"), 32):
            return state["boxes"], "save"
        if key == ord("n"):
            return state["boxes"], "skip"
        if key in (ord("r"), ord("c")):
            state["boxes"].clear()
            state["current"] = None
        if key in (ord("q"), 27):
            return state["boxes"], "quit"


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(args.output_dir, args.dataset_name)
    images_dir, labels_dir = prepare_output_dirs(dataset_dir)
    yaml_path = ensure_dataset_yaml(dataset_dir, args.class_name)
    class_id = ensure_class_in_yaml(yaml_path, args.class_name)
    video_prefix = sanitize_prefix(Path(args.video).stem)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {args.video}")

    frame_step = args.frame_step
    if frame_step is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = fps if fps > 0 else 25
        frame_step = max(1, int(round(fps * args.interval_sec)))
    print(
        f"抽帧步长: 每 {frame_step} 帧处理一次（fps≈{cap.get(cv2.CAP_PROP_FPS) or '未知'}, interval={args.interval_sec}s）"
    )
    print(f"输出目录: {dataset_dir}")
    print(f"数据集配置: {yaml_path}")

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

            stem = unique_stem(images_dir, f"{video_prefix}_frame_{frame_idx:06d}")
            image_name = f"{stem}.jpg"
            label_name = f"{stem}.txt"

            boxes, action = annotate_frame(frame, "Manual Labeler")
            if action == "quit":
                print("用户退出，结束标注。")
                break
            if action == "skip":
                print(f"跳过帧 {frame_idx}")
                continue

            cv2.imwrite(str(images_dir / image_name), frame)
            width, height = frame.shape[1], frame.shape[0]
            lines = [yolo_line(b, width, height, class_id) for b in boxes]
            (labels_dir / label_name).write_text("\n".join(lines), encoding="utf-8")

            saved_count += 1
            print(f"[{saved_count}] 已保存 {image_name}，标注 {len(lines)} 条")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"完成！共处理 {saved_count} 帧，数据集位于: {dataset_dir}")


if __name__ == "__main__":
    main()
