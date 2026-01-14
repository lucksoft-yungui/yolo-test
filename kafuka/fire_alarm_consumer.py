import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from kafka import KafkaConsumer, KafkaProducer
from ultralytics import YOLO


@dataclass
class AlarmMessage:
    device_id: str
    area_id: str
    photo_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="订阅 Kafka fire-alarm 主题并批量检测烟火")
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:9092",
        help="Kafka 地址，默认 localhost:9092",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="fire-alarm",
        help="Kafka 主题，默认 fire-alarm",
    )
    parser.add_argument(
        "--alarm-topic",
        type=str,
        default="alarm-queue",
        help="报警消息推送主题，默认 alarm-queue",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="fire-alarm-consumer",
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
        default=Path("model/fire-kaggle/weights/best.pt"),
        help="模型权重路径，默认 model/fire-kaggle/weights/best.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.6,
        help="置信度阈值，默认 0.6",
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
        "--fire-class",
        type=str,
        default="fire",
        help="烟火类别名称，默认 fire",
    )
    return parser.parse_args()


def load_model(weight_path: Path, device: str | None, class_names: list[str]) -> YOLO:
    if not weight_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {weight_path}")
    model = YOLO(str(weight_path))
    if class_names:
        model.model.names = {i: name for i, name in enumerate(class_names)}
    if device:
        model.to(device)
    return model


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
        results.append(AlarmMessage(device_id=device_id, area_id=area_id, photo_path=photo_path))
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


def main() -> None:
    args = parse_args()
    print(f"运行参数: {vars(args)}", flush=True)
    device = args.device
    if device is None and args.gpu >= 0:
        device = f"cuda:{args.gpu}"
    model = load_model(args.model, device, [args.fire_class])
    fire_index = 0

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        enable_auto_commit=False,
        auto_offset_reset=args.auto_offset_reset,
        value_deserializer=lambda v: v,
    )
    producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)

    print(
        f"开始订阅 {args.bootstrap_servers} / {args.topic}，批量 {args.batch_size} 条，模型 {args.model}，"
        f"group={args.group_id}，offset={args.auto_offset_reset}",
        flush=True,
    )
    batch_count = 0
    while True:
        print("等待拉取消息...", flush=True)
        batch_start = time.monotonic()
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
        print(f"解析到 {len(entries)} 条消息。", flush=True)

        valid_entries, images = iter_images(entries)
        if not images:
            print("没有可用图片，跳过本批次。", flush=True)
            continue
        print(f"有效图片数量: {len(images)}", flush=True)

        # 使用列表一次性推理，实现批量合并计算
        results = model(images, conf=args.conf, verbose=False)
        for entry, result in zip(valid_entries, results):
            has_fire = False
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == fire_index:
                    has_fire = True
                    break
            if has_fire:
                alarm_payload = [
                    {
                        "topic": args.topic,
                        "deviceId": entry.device_id,
                        "areaId": entry.area_id,
                        "photoPath": entry.photo_path,
                    }
                ]
                producer.send(
                    args.alarm_topic,
                    json.dumps(alarm_payload).encode("utf-8"),
                )
                print(
                    f"检测到烟火: deviceId={entry.device_id}, photoPath={entry.photo_path}",
                    flush=True,
                )
            else:
                print(
                    f"未检测到烟火: deviceId={entry.device_id}, photoPath={entry.photo_path}",
                    flush=True,
                )

        batch_count += 1
        elapsed_sec = time.monotonic() - batch_start
        print(
            f"处理批次: {batch_count}, 消息数: {len(raw_messages)}, 有效图片: {len(images)}, "
            f"耗时: {elapsed_sec:.2f}s",
            flush=True,
        )
        producer.flush()
        consumer.commit()
        if args.max_batches > 0 and batch_count >= args.max_batches:
            print("已达到最大批次限制，退出。", flush=True)
            break


if __name__ == "__main__":
    main()
