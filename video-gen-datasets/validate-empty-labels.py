import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="检查数据集中缺失或空的标签文件")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="数据集 yaml 路径")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只验证并输出列表，不进入补充标注",
    )
    parser.add_argument("--class-name", type=str, default=None, help="指定要补充标注的类别名称")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="仅输出列表，不打开窗口显示",
    )
    return parser.parse_args()


def resolve_root(yaml_path: Path, raw_path: str) -> Path:
    if not raw_path:
        return yaml_path.parent
    root = Path(raw_path).expanduser()
    if root.is_absolute():
        return root
    candidate = (yaml_path.parent / root).resolve()
    if candidate.exists():
        return candidate
    candidate2 = root.resolve()
    if candidate2.exists():
        return candidate2
    return candidate


def parse_names_from_yaml(yaml_path: Path) -> Dict[int, str]:
    names: Dict[int, str] = {}
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        raw = line
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "names:":
            in_names = True
            continue
        if in_names:
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


def parse_dataset_yaml(yaml_path: Path) -> Tuple[Path, Path, Dict[int, str]]:
    data = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()

    root = resolve_root(yaml_path, data.get("path", ""))

    train_rel = data.get("train", "images/train")
    train_path = Path(train_rel)
    images_dir = train_path if train_path.is_absolute() else (root / train_rel)
    if "images" in train_rel:
        labels_rel = train_rel.replace("images", "labels", 1)
        labels_path = Path(labels_rel)
        labels_dir = labels_path if labels_path.is_absolute() else (root / labels_rel)
    else:
        labels_dir = root / "labels" / Path(train_rel).name
    names = parse_names_from_yaml(yaml_path)
    return images_dir, labels_dir, names


def list_images(images_dir: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def is_empty_label(label_path: Path) -> bool:
    if not label_path.exists():
        return True
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return False
    return True


def show_images(items: List[Tuple[Path, str]]) -> None:
    if not items:
        print("未发现缺失或空标签的图片。")
        return
    idx = 0
    window = "Missing/Empty Labels"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while idx < len(items):
        image_path, reason = items[idx]
        img = cv2.imread(str(image_path))
        if img is None:
            idx += 1
            continue
        cv2.putText(
            img,
            f"{reason}: {image_path.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.imshow(window, img)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        idx += 1
    cv2.destroyAllWindows()


def draw_class_list(canvas, names: Dict[int, str]) -> None:
    y = 60
    for idx in sorted(names.keys()):
        text = f"{idx}: {names[idx]}"
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += 24


def annotate_image(image_path: Path, window_name: str, names: Dict[int, str]) -> Tuple[List[Box], str]:
    image = cv2.imread(str(image_path))
    if image is None:
        return [], "skip"
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
        canvas = image.copy()
        for b in state["boxes"]:
            cv2.rectangle(canvas, (b.x1, b.y1), (b.x2, b.y2), (0, 0, 255), 2)
        if state["current"]:
            x1, y1, x2, y2 = state["current"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 1)
        cv2.putText(canvas, PROMPT_INFO, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        draw_class_list(canvas, names)
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
    if not args.dataset_yaml.exists():
        raise FileNotFoundError(f"找不到数据集配置: {args.dataset_yaml}")

    images_dir, labels_dir, names = parse_dataset_yaml(args.dataset_yaml)
    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")
    if not labels_dir.exists():
        print(f"标签目录不存在: {labels_dir}，将视为全部缺失")
        labels_dir.mkdir(parents=True, exist_ok=True)

    missing_items: List[Tuple[Path, str]] = []
    for img_path in list_images(images_dir):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing_items.append((img_path, "missing"))
        elif is_empty_label(label_path):
            missing_items.append((img_path, "empty"))

    all_images = list_images(images_dir)
    print(f"共扫描 {len(all_images)} 张图片")
    print(f"缺失/空标签数量: {len(missing_items)}")
    for img_path, reason in missing_items:
        print(f"{reason}\t{img_path}")

    if not args.validate_only:
        default_class_id = None
        if args.class_name:
            for idx, name in names.items():
                if name == args.class_name:
                    default_class_id = idx
                    break
            if default_class_id is None:
                raise ValueError(f"类别不存在于 yaml: {args.class_name}")
        elif len(names) == 1:
            default_class_id = list(names.keys())[0]

        filled = 0
        window = "Annotate Missing Labels"
        for img_path, reason in missing_items:
            boxes, action = annotate_image(img_path, window, names)
            if action == "quit":
                break
            if action == "skip":
                continue
            if not boxes:
                continue
            class_id = None
            prompt_window = "Select Class ID"
            cv2.namedWindow(prompt_window, cv2.WINDOW_NORMAL)
            while True:
                prompt_canvas = cv2.imread(str(img_path))
                if prompt_canvas is None:
                    break
                draw_class_list(prompt_canvas, names)
                hint = "Press 0-9 to select class; Enter for default; q/ESC skip"
                cv2.putText(prompt_canvas, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(prompt_window, prompt_canvas)
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (10, 13):  # Enter
                    if default_class_id is None:
                        continue
                    class_id = default_class_id
                    break
                if ord("0") <= key <= ord("9"):
                    selected = int(chr(key))
                    if selected in names:
                        class_id = selected
                        break
            cv2.destroyWindow(prompt_window)
            if class_id is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            width, height = img.shape[1], img.shape[0]
            lines = []
            for b in boxes:
                cx = (b.x1 + b.x2) / 2.0 / width
                cy = (b.y1 + b.y2) / 2.0 / height
                bw = (b.x2 - b.x1) / width
                bh = (b.y2 - b.y1) / height
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            label_path = labels_dir / f"{img_path.stem}.txt"
            label_path.write_text("\n".join(lines), encoding="utf-8")
            filled += 1
            print(f"[{filled}] 已补充: {img_path.name} -> class {class_id}")
        cv2.destroyAllWindows()
        print(f"已补充标注: {filled} 张")
        return

    if not args.no_show:
        show_images(missing_items)


if __name__ == "__main__":
    main()
