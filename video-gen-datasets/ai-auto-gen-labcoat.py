import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from openai import OpenAI

MODEL_NAME = "qwen3-vl-plus"
PROMPT = (
    "你是yolo数据集标注助手，识别图片中的人员，返回如下的结构："
    "[{type:int,pos:{x1,y1,x2,y2}}]，如果人员穿了实验服返回1，否则返回0。"
    "仅输出JSON，不要额外解释。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="利用AI自动生成实验服检测YOLO数据集")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("videos") / "lab coat.mp4",
        help="输入视频路径，默认 videos/lab coat.mp4",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/labcoat"),
        help="数据集输出目录，默认 datasets/labcoat",
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
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="是否打开思考过程（需要模型支持），默认开启，可用 --no-thinking 关闭",
    )
    parser.add_argument(
        "--no-thinking",
        dest="enable_thinking",
        action="store_false",
        help="关闭思考过程",
    )
    return parser.parse_args()


def build_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = load_api_key_from_env_file()
    if not api_key:
        raise RuntimeError("未找到 DASHSCOPE_API_KEY（或 OPENAI_API_KEY），请在环境变量或 .env 中设置")

    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def load_api_key_from_env_file() -> Optional[str]:
    # 依次尝试当前工作目录、脚本目录及其父级的 .env
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path(".").resolve() / ".env",
        script_dir / ".env",
        script_dir.parent / ".env",
        script_dir.parent.parent / ".env",
    ]

    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)

        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
    return None


def prepare_output_dirs(dataset_dir: Path) -> Dict[str, Path]:
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return {"images": images_dir, "labels": labels_dir}


def encode_frame(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("帧编码失败，无法发送给模型")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def parse_detections(raw: str) -> List[Dict[str, Any]]:
    raw = clean_response_text(raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return normalize_parsed_detections(data)
    except json.JSONDecodeError:
        pass
    # 尝试截取第一个 JSON 数组
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return normalize_parsed_detections(data)
        except json.JSONDecodeError:
            pass
    # 最后兜底：用正则粗暴提取 type + 四个数
    fallback = regex_extract_detections(raw)
    if fallback:
        return fallback
    return []


def clean_response_text(raw: str) -> str:
    # 去掉Markdown代码块包裹
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        # 去掉可能的语言标识，如 json
        raw = raw.split("\n", 1)[-1]
    # 替换单引号为双引号，修复常见键名错误
    raw = raw.replace("'", '"')
    raw = re.sub(r'"type\s*:', '"type":', raw)
    raw = re.sub(r'"pos"\s*:\s*\[', '"pos":[', raw)
    # 将 {a, b, c, d} 转成 [a, b, c, d]
    raw = re.sub(
        r"\{\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\}",
        r"[ \1, \2, \3, \4 ]",
        raw,
    )
    return raw


def normalize_parsed_detections(data: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        det: Dict[str, Any] = {}
        # 修正键名可能包含空格或冒号错误
        if "type" in item:
            det["type"] = item["type"]
        elif any(k.startswith("type") for k in item.keys()):
            for k in item:
                if k.startswith("type"):
                    det["type"] = item[k]
                    break
        pos_val = item.get("pos")
        if isinstance(pos_val, list) and len(pos_val) == 4:
            pos_val = {"x1": pos_val[0], "y1": pos_val[1], "x2": pos_val[2], "y2": pos_val[3]}
        if isinstance(pos_val, dict):
            det["pos"] = pos_val
        if "type" in det and "pos" in det:
            normalized.append(det)
    return normalized


def regex_extract_detections(raw: str) -> List[Dict[str, Any]]:
    """
    兜底解析：正则提取 type 数字 + 随后的四个数作为 bbox。
    支持 pos 用 [] 或 {}，或缺键名的情况。
    """
    results: List[Dict[str, Any]] = []
    # 匹配 type 后面跟着一段括号包含的四个数字
    pattern = re.compile(
        r"type\s*[:\"]?\s*(\d+)[^\\[\\{]*([\\[\\{])([^\\]\\}]*)[\\]\\}]",
        re.IGNORECASE,
    )
    for match in pattern.finditer(raw):
        cls = match.group(1)
        bbox_part = match.group(3)
        nums = re.findall(r"-?\\d+\\.?\\d*", bbox_part)
        if len(nums) < 4:
            continue
        x1, y1, x2, y2 = map(float, nums[:4])
        results.append({"type": int(cls), "pos": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}})
    return results


def ensure_dataset_yaml(dataset_dir: Path) -> None:
    yaml_path = dataset_dir / "labcoat.yaml"
    if yaml_path.exists():
        return
    content = "\n".join(
        [
            "# Auto-generated YOLO dataset config for lab coat detection",
            f"path: {dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/train  # 如有验证集可改为 images/val",
            "",
            "names:",
            "  0: no_labcoat",
            "  1: labcoat",
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"已生成数据集配置: {yaml_path}")


def request_annotations(client: OpenAI, image_url: str, enable_thinking: bool) -> List[Dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    params: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }
    if enable_thinking:
        params["extra_body"] = {"enable_thinking": True}

    completion = client.chat.completions.create(**params)
    content = completion.choices[0].message.content
    raw_text = extract_text_content(content)
    detections = parse_detections(raw_text)
    return detections, raw_text


def normalize_bbox(pos: Dict[str, Any], width: int, height: int) -> Optional[str]:
    try:
        x1 = float(pos["x1"])
        y1 = float(pos["y1"])
        x2 = float(pos["x2"])
        y2 = float(pos["y2"])
    except (KeyError, TypeError, ValueError):
        return None

    max_coord = max(x1, y1, x2, y2)
    # 1) 归一化 0-1: 直接放大到像素
    if max_coord <= 1.5:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    # 2) 部分模型返回 0-1000 空间的坐标（与分辨率无关），需要按 1000 缩放
    elif (x2 > width or y2 > height) and max_coord <= 1005:
        scale_x = width / 1000.0
        scale_y = height / 1000.0
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
    # 3) 其他情况认为是像素坐标，直接使用

    if x2 <= x1 or y2 <= y1:
        return None

    x1 = max(0.0, min(x1, float(width)))
    x2 = max(0.0, min(x2, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    y2 = max(0.0, min(y2, float(height)))

    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height

    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and bw > 0 and bh > 0):
        return None
    return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def detections_to_yolo(detections: List[Dict[str, Any]], width: int, height: int) -> List[str]:
    lines: List[str] = []
    for det in detections:
        try:
            class_id = int(det.get("type"))
        except (TypeError, ValueError):
            continue
        pos = det.get("pos") or {}
        normalized = normalize_bbox(pos, width, height)
        if normalized:
            lines.append(f"{class_id} {normalized}")
    return lines


def main() -> None:
    args = parse_args()
    client = build_client()
    dirs = prepare_output_dirs(args.dataset)
    ensure_dataset_yaml(args.dataset)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {args.video}")

    # 根据 fps 自动计算抽帧步长，若用户指定 frame-step 则优先使用
    frame_step = args.frame_step
    if frame_step is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        fps = fps if fps > 0 else 25  # 无法读取 fps 时使用默认 25
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

            image_url = encode_frame(frame)
            detections, raw_text = request_annotations(client, image_url, args.enable_thinking)
            yolo_lines = detections_to_yolo(detections, frame.shape[1], frame.shape[0])

            cv2.imwrite(str(dirs["images"] / image_name), frame)
            label_path = dirs["labels"] / label_name
            label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

            saved_count += 1
            print(f"[{saved_count}] 已保存 {image_name}，标注 {len(yolo_lines)} 条")
            if not yolo_lines:
                print(f"AI原始返回（帧 {frame_idx}）：{raw_text}")
    finally:
        cap.release()

    print(f"完成！共处理 {saved_count} 帧，数据集位于: {args.dataset}")


if __name__ == "__main__":
    main()
