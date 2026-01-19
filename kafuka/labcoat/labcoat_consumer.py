import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import yaml
from kafka import KafkaConsumer, KafkaProducer
from ultralytics import YOLO


@dataclass
class AlarmMessage:
    payload: dict
    device_id: str
    area_id: str
    photo_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="订阅 PPE 主题并检测未穿实验服 + 戴手套")
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:9092",
        help="Kafka 地址，默认 localhost:9092",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="ppe_alarm",
        help="Kafka 主题，默认 ppe_alarm",
    )
    parser.add_argument(
        "--alarm-topic",
        type=str,
        default="ppe_alarm_result",
        help="报警消息推送主题，默认 ppe_alarm_result",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="ppe-alarm-consumer",
        help="消费者组 ID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="每次拉取消息数量，默认 10",
    )
    parser.add_argument(
        "--max-wait-sec",
        type=float,
        default=2.0,
        help="等待凑满批次的最长时间（秒），默认 2",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="最多处理的批次数，0 表示不限制",
    )
    parser.add_argument(
        "--auto-offset-reset",
        type=str,
        default="latest",
        choices=["latest", "earliest", "none"],
        help="起始偏移策略，默认 latest",
    )
    parser.add_argument(
        "--labcoat-model",
        type=Path,
        default=Path("model/labcoat/best.pt"),
        help="实验服模型权重路径",
    )
    parser.add_argument(
        "--glove-model",
        type=Path,
        default=Path("model/glove/best.pt"),
        help="手套模型权重路径",
    )
    parser.add_argument(
        "--labcoat-yaml",
        type=Path,
        default=Path("labcoat.yaml"),
        help="实验服数据集 yaml",
    )
    parser.add_argument(
        "--glove-yaml",
        type=Path,
        default=Path("glove.yaml"),
        help="手套数据集 yaml",
    )
    parser.add_argument(
        "--labcoat-class-name",
        type=str,
        default="no labcoat",
        help='未穿实验服的类别名称，默认 "no labcoat"',
    )
    parser.add_argument(
        "--glove-class-name",
        type=str,
        default="with glove",
        help='戴手套的类别名称，默认 "with glove"',
    )
    parser.add_argument(
        "--labcoat-class-id",
        type=int,
        default=None,
        help="未穿实验服的类别索引（不填则按名称解析）",
    )
    parser.add_argument(
        "--glove-class-id",
        type=int,
        default=None,
        help="戴手套的类别索引（不填则按名称解析）",
    )
    parser.add_argument(
        "--labcoat-conf",
        type=float,
        default=0.7,
        help="实验服模型置信度阈值，默认 0.7",
    )
    parser.add_argument(
        "--glove-conf",
        type=float,
        default=0.7,
        help="手套模型置信度阈值，默认 0.7",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备，例如 cpu / cuda / mps，不填则自动选择",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="指定 GPU 序号（如 0），仅在未指定 --device 时生效",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启调试图保存",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("kafuka/labcoat/debug"),
        help="调试图保存目录，默认 kafuka/labcoat/debug",
    )
    return parser.parse_args()


def load_names(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    raise ValueError(f"Unsupported names in {yaml_path}")


def resolve_class_id(
    names: dict[int, str],
    class_name: str | None,
    class_id: int | None,
    yaml_path: Path,
) -> int:
    if class_id is not None:
        return class_id
    if not class_name:
        raise ValueError(f"Missing class name for {yaml_path}")
    for idx, name in names.items():
        if name == class_name:
            return idx
    raise ValueError(f'Class "{class_name}" not found in {yaml_path}')


def load_model(weight_path: Path, device: str | None, class_names: dict[int, str]) -> YOLO:
    if not weight_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {weight_path}")
    model = YOLO(str(weight_path))
    if hasattr(model, "model") and hasattr(model.model, "names"):
        if len(class_names) == len(model.model.names):
            model.model.names = {i: name for i, name in class_names.items()}
    if device:
        model.to(device)
    return model


def normalize_device_label(device_value: str) -> str:
    device_value = device_value.lower()
    if device_value.startswith("cuda"):
        return "cuda"
    if device_value.startswith("mps"):
        return "mps"
    if device_value.startswith("cpu"):
        return "cpu"
    return device_value


def resolve_model_device(model: YOLO, requested_device: str | None) -> str:
    if requested_device:
        return normalize_device_label(requested_device)
    detected_device = getattr(model, "device", None)
    if detected_device is None and hasattr(model, "model"):
        detected_device = getattr(model.model, "device", None)
    if detected_device is None:
        return "unknown"
    return normalize_device_label(str(detected_device))


def decode_messages(raw: str) -> list[AlarmMessage]:
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("消息体必须是对象或数组")
    results: list[AlarmMessage] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        device_id = str(item.get("deviceId", "")).strip()
        area_id = str(item.get("areaId", "")).strip()
        photo_path = str(item.get("photoPath", "")).strip()
        if not device_id or not photo_path:
            continue
        results.append(
            AlarmMessage(
                payload=item,
                device_id=device_id,
                area_id=area_id,
                photo_path=photo_path,
            )
        )
    return results


def poll_batch(consumer: KafkaConsumer, batch_size: int, max_wait_sec: float) -> list[str]:
    messages: list[str] = []
    start = time.monotonic()
    while len(messages) < batch_size:
        remaining = batch_size - len(messages)
        records = consumer.poll(timeout_ms=500, max_records=remaining)
        for _, msgs in records.items():
            for msg in msgs:
                if isinstance(msg.value, bytes):
                    messages.append(msg.value.decode("utf-8"))
                else:
                    messages.append(str(msg.value))
        if messages and (time.monotonic() - start) >= max_wait_sec:
            break
    return messages


def iter_images(entries: Iterable[AlarmMessage]) -> tuple[list[AlarmMessage], list[object]]:
    valid_entries: list[AlarmMessage] = []
    images: list[object] = []
    for entry in entries:
        img = cv2.imread(entry.photo_path)
        if img is None:
            print(f"读取图片失败: {entry.photo_path}")
            continue
        valid_entries.append(entry)
        images.append(img)
    return valid_entries, images


def draw_boxes(image, boxes: list[dict[str, object]]) -> object:
    for box in boxes:
        coords = box.get("xyxy")
        if not isinstance(coords, list) or len(coords) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in coords]
        class_name = str(box.get("className", ""))
        conf = box.get("conf")
        label = class_name
        if isinstance(conf, (int, float)):
            label = f"{class_name}:{conf:.2f}"
        color = (0, 0, 255) if class_name == "no labcoat" else (0, 200, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return image


def main() -> None:
    args = parse_args()
    print(f"运行参数: {vars(args)}", flush=True)
    device = args.device
    if device is None and args.gpu >= 0:
        device = f"cuda:{args.gpu}"

    labcoat_names = load_names(args.labcoat_yaml)
    glove_names = load_names(args.glove_yaml)
    no_labcoat_id = resolve_class_id(
        labcoat_names, args.labcoat_class_name, args.labcoat_class_id, args.labcoat_yaml
    )
    with_glove_id = resolve_class_id(
        glove_names, args.glove_class_name, args.glove_class_id, args.glove_yaml
    )

    labcoat_model = load_model(args.labcoat_model, device, labcoat_names)
    glove_model = load_model(args.glove_model, device, glove_names)
    device_label = resolve_model_device(labcoat_model, device)
    print(f"当前使用算力: {device_label}", flush=True)

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        enable_auto_commit=False,
        auto_offset_reset=args.auto_offset_reset,
        value_deserializer=lambda v: v,
    )
    producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)

    if args.debug:
        if args.debug_dir.exists():
            for path in args.debug_dir.iterdir():
                if path.is_dir():
                    for sub in path.rglob("*"):
                        if sub.is_file():
                            sub.unlink()
                    path.rmdir()
                else:
                    path.unlink()
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"开始订阅 {args.bootstrap_servers} / {args.topic}，批量 {args.batch_size} 条，"
        f"实验服模型 {args.labcoat_model}，手套模型 {args.glove_model}，"
        f"group={args.group_id}，offset={args.auto_offset_reset}",
        flush=True,
    )
    batch_count = 0
    while True:
        print("等待拉取消息...", flush=True)
        raw_messages = poll_batch(consumer, args.batch_size, args.max_wait_sec)
        print(f"拉取完成，原始消息数={len(raw_messages)}", flush=True)
        if not raw_messages:
            continue

        entries: list[AlarmMessage] = []
        for raw in raw_messages:
            try:
                entries.extend(decode_messages(raw))
            except Exception as exc:
                print(f"解析消息失败: {exc}，原始内容: {raw}", flush=True)

        if not entries:
            print("没有有效消息，跳过本批次。", flush=True)
            continue

        valid_entries, images = iter_images(entries)
        if not images:
            print("没有可用图片，跳过本批次。", flush=True)
            continue
        print(f"解析到 {len(valid_entries)} 条有效消息。", flush=True)

        inference_start = time.monotonic()
        labcoat_results = labcoat_model(images, conf=args.labcoat_conf, verbose=False)
        inference_elapsed = time.monotonic() - inference_start

        no_labcoat_boxes_per_entry: list[list[dict[str, object]]] = []
        crop_images: list[object] = []
        crop_mapping: list[tuple[int, int, int, int]] = []

        for entry_idx, (entry, result, img) in enumerate(
            zip(valid_entries, labcoat_results, images)
        ):
            boxes_for_entry: list[dict[str, object]] = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != no_labcoat_id:
                        continue
                    conf = float(box.conf[0]) if hasattr(box, "conf") else None
                    coords = [float(v) for v in box.xyxy[0]]
                    boxes_for_entry.append(
                        {
                            "className": labcoat_names.get(no_labcoat_id, "no labcoat"),
                            "conf": conf,
                            "xyxy": coords,
                        }
                    )

                    x1, y1, x2, y2 = [int(v) for v in coords]
                    x1 = max(0, min(img.shape[1], x1))
                    x2 = max(0, min(img.shape[1], x2))
                    y1 = max(0, min(img.shape[0], y1))
                    y2 = max(0, min(img.shape[0], y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = img[y1:y2, x1:x2]
                    crop_images.append(crop)
                    crop_mapping.append((entry_idx, len(boxes_for_entry) - 1, x1, y1))
            no_labcoat_boxes_per_entry.append(boxes_for_entry)

        glove_boxes_per_entry: list[list[dict[str, object]]] = [[] for _ in valid_entries]
        matched_labcoat_indices: list[set[int]] = [set() for _ in valid_entries]
        if crop_images:
            glove_results = glove_model(
                crop_images, conf=args.glove_conf, verbose=False
            )
            for mapping, result in zip(crop_mapping, glove_results):
                entry_idx, labcoat_box_idx, x_offset, y_offset = mapping
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != with_glove_id:
                        continue
                    conf = float(box.conf[0]) if hasattr(box, "conf") else None
                    gx1, gy1, gx2, gy2 = [float(v) for v in box.xyxy[0]]
                    coords = [
                        gx1 + x_offset,
                        gy1 + y_offset,
                        gx2 + x_offset,
                        gy2 + y_offset,
                    ]
                    glove_boxes_per_entry[entry_idx].append(
                        {
                            "className": glove_names.get(with_glove_id, "with glove"),
                            "conf": conf,
                            "xyxy": coords,
                        }
                    )
                    matched_labcoat_indices[entry_idx].add(labcoat_box_idx)

        hit_count = 0
        for entry_idx, entry in enumerate(valid_entries):
            no_labcoat_boxes = no_labcoat_boxes_per_entry[entry_idx]
            glove_boxes = glove_boxes_per_entry[entry_idx]
            if not no_labcoat_boxes or not glove_boxes:
                continue
            matched_indices = matched_labcoat_indices[entry_idx]
            if not matched_indices:
                continue
            no_labcoat_boxes = [
                box
                for idx, box in enumerate(no_labcoat_boxes)
                if idx in matched_indices
            ]
            if not no_labcoat_boxes:
                continue
            payload = dict(entry.payload)
            payload.setdefault("topic", args.topic)
            merged_boxes: list[dict[str, object]] = []
            existing_boxes = payload.get("boxes")
            if isinstance(existing_boxes, list):
                merged_boxes.extend(existing_boxes)
            merged_boxes.extend(no_labcoat_boxes)
            merged_boxes.extend(glove_boxes)
            payload["boxes"] = merged_boxes
            producer.send(args.alarm_topic, json.dumps(payload).encode("utf-8"))
            if args.debug:
                image = images[entry_idx].copy()
                draw_boxes(image, payload["boxes"])
                output_name = Path(entry.photo_path).name
                output_path = args.debug_dir / output_name
                cv2.imwrite(str(output_path), image)
            hit_count += 1
            print(
                f"匹配报警: deviceId={entry.device_id}, photoPath={entry.photo_path}, "
                f"no_labcoat={len(no_labcoat_boxes)}, glove={len(glove_boxes)}",
                flush=True,
            )

        batch_count += 1
        print(
            f"处理批次: {batch_count}, 原始消息: {len(raw_messages)}, 有效图片: {len(images)}, "
            f"命中: {hit_count}, 推理耗时: {inference_elapsed:.2f}s",
            flush=True,
        )
        producer.flush()
        consumer.commit()
        if args.max_batches > 0 and batch_count >= args.max_batches:
            print("已达到最大批次限制，退出。", flush=True)
            break


if __name__ == "__main__":
    main()
