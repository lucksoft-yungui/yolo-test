import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
from kafka import KafkaConsumer, KafkaProducer
from ultralytics import YOLO


@dataclass
class AlarmMessage:
    payload: dict
    device_id: str
    area_id: str
    photo_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="订阅无人值守主题并检测人员数量")
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:9092",
        help="Kafka 地址，默认 localhost:9092",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="unattended_alarm",
        help="Kafka 主题，默认 unattended_alarm",
    )
    parser.add_argument(
        "--alarm-topic",
        type=str,
        default="unattended_alarm_result",
        help="报警消息推送主题，默认 unattended_alarm_result",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="unattended-alarm-consumer",
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
        "--model",
        type=Path,
        default=Path("yolo11n.pt"),
        help="YOLO11n 模型权重路径，默认 yolo11n.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="置信度阈值，默认 0.5",
    )
    parser.add_argument(
        "--person-class-name",
        type=str,
        default="person",
        help='人员类别名称，默认 "person"',
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=0,
        help="人员类别索引，默认 0",
    )
    parser.add_argument(
        "--time-start",
        type=str,
        default="00:00",
        help="报警开始时间（24h，HH:MM），默认 00:00",
    )
    parser.add_argument(
        "--time-end",
        type=str,
        default="08:00",
        help="报警结束时间（24h，HH:MM），默认 08:00",
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
        default=Path("kafuka/headcount/debug"),
        help="调试图保存目录，默认 kafuka/headcount/debug",
    )
    return parser.parse_args()


def load_model(weight_path: Path, device: str | None) -> YOLO:
    if not weight_path.exists():
        print(f"模型不存在，尝试自动下载: {weight_path}", flush=True)
    model = YOLO(str(weight_path))
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
        color = (0, 200, 0)
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


def parse_time_str(value: str) -> int:
    value = value.strip()
    try:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"时间格式错误: {value}，应为 HH:MM") from exc
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"时间范围错误: {value}，应为 00:00-23:59")
    return hour * 60 + minute


def is_time_in_window(now_minutes: int, start_minutes: int, end_minutes: int) -> bool:
    if start_minutes == end_minutes:
        return True
    if start_minutes < end_minutes:
        return start_minutes <= now_minutes < end_minutes
    return now_minutes >= start_minutes or now_minutes < end_minutes


def main() -> None:
    args = parse_args()
    print(f"运行参数: {vars(args)}", flush=True)
    device = args.device
    if device is None and args.gpu >= 0:
        device = f"cuda:{args.gpu}"

    model = load_model(args.model, device)
    device_label = resolve_model_device(model, device)
    print(f"当前使用算力: {device_label}", flush=True)

    start_minutes = parse_time_str(args.time_start)
    end_minutes = parse_time_str(args.time_end)

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
            args.debug_dir.rmdir()
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"开始订阅 {args.bootstrap_servers} / {args.topic}，批量 {args.batch_size} 条，模型 {args.model}，"
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
        results = model(images, conf=args.conf, verbose=False)
        inference_elapsed = time.monotonic() - inference_start

        now = datetime.now()
        now_minutes = now.hour * 60 + now.minute
        allow_alarm = is_time_in_window(now_minutes, start_minutes, end_minutes)

        hit_count = 0
        for entry_idx, (entry, result) in enumerate(zip(valid_entries, results)):
            person_boxes: list[dict[str, object]] = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != args.person_class_id:
                        continue
                    conf = float(box.conf[0]) if hasattr(box, "conf") else None
                    coords = [float(v) for v in box.xyxy[0]]
                    person_boxes.append(
                        {
                            "className": args.person_class_name,
                            "conf": conf,
                            "xyxy": coords,
                        }
                    )

            if len(person_boxes) != 1:
                print(
                    f"未触发报警: deviceId={entry.device_id}, photoPath={entry.photo_path}, "
                    f"personCount={len(person_boxes)}",
                    flush=True,
                )
                continue

            if not allow_alarm:
                print(
                    f"超出时间范围: deviceId={entry.device_id}, photoPath={entry.photo_path}, "
                    f"now={now.strftime('%H:%M')}",
                    flush=True,
                )
                continue

            payload = dict(entry.payload)
            payload.setdefault("topic", args.topic)
            payload["boxes"] = person_boxes
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
                f"personCount=1",
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
