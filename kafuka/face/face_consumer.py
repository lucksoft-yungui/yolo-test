import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import face_recognition
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from ultralytics import YOLO


@dataclass
class AlarmMessage:
    payload: dict
    device_id: str
    area_id: str
    photo_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="订阅并校验人脸")
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
        default="face_recognition_alarm_result",
        help="报警主题，默认 face_recognition_alarm_result",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default="face-alarm-consumer",
        help="消费者组 ID",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Kafka 单次拉取数量，默认 10",
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
        "--max-poll-interval-ms",
        type=int,
        default=60000,
        help="最大拉取间隔（毫秒），默认 60000",
    )
    parser.add_argument(
        "--session-timeout-ms",
        type=int,
        default=30000,
        help="会话超时时间（毫秒），默认 30000",
    )
    parser.add_argument(
        "--heartbeat-interval-ms",
        type=int,
        default=10000,
        help="心跳间隔（毫秒），默认 10000",
    )
    parser.add_argument(
        "--auto-offset-reset",
        type=str,
        default="latest",
        choices=["latest", "earliest", "none"],
        help="起始偏移策略，默认 latest",
    )
    parser.add_argument(
        "--people-dir",
        type=Path,
        default=Path("people"),
        help="人脸库目录，默认 people",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="人脸匹配阈值，默认 0.6",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hog",
        choices=["hog", "cnn"],
        help="人脸检测模型，默认 hog",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="不使用 YOLO 预检测（直接进行人脸检测）",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("yolo11s.pt"),
        help="YOLO 模型路径，默认 yolo11s.pt",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO 置信度阈值，默认 0.25",
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=0,
        help="人员类别索引，默认 0",
    )
    parser.add_argument(
        "--num-upsample",
        type=int,
        default=0,
        help="上采样次数，默认 0",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="启用 GPU 批量模式（需使用 cnn 模型）",
    )
    parser.add_argument(
        "--face-batch-size",
        type=int,
        default=128,
        help="GPU 批量检测大小，默认 128",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="保存带人脸框的调试图片",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("kafuka/face/debug"),
        help="调试图片输出目录，默认 kafuka/face/debug",
    )
    return parser.parse_args()


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


def iter_images(entries: Iterable[AlarmMessage]) -> tuple[list[AlarmMessage], list[np.ndarray], list[np.ndarray]]:
    valid_entries: list[AlarmMessage] = []
    images_bgr: list[np.ndarray] = []
    images_rgb: list[np.ndarray] = []
    for entry in entries:
        img = cv2.imread(entry.photo_path)
        if img is None:
            print(f"读取图片失败: {entry.photo_path}")
            continue
        valid_entries.append(entry)
        images_bgr.append(img)
        images_rgb.append(np.ascontiguousarray(img[:, :, ::-1]))
    return valid_entries, images_bgr, images_rgb


def load_known_faces(people_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    if not people_dir.exists():
        print(f"人脸库目录不存在: {people_dir}")
        return [], []
    encodings: list[np.ndarray] = []
    names: list[str] = []
    for image_path in sorted(people_dir.glob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = face_recognition.load_image_file(str(image_path))
        face_encodings = face_recognition.face_encodings(image)
        if not face_encodings:
            print(f"未检测到人脸: {image_path}")
            continue
        encodings.append(face_encodings[0])
        names.append(image_path.stem)
    print(f"已加载人脸库: {len(encodings)}", flush=True)
    return encodings, names


def batch_face_locations_by_shape(
    images_rgb: list[np.ndarray],
    num_upsample: int,
    batch_size: int,
) -> list[list[tuple[int, int, int, int]]]:
    results: list[list[tuple[int, int, int, int]]] = [
        [] for _ in range(len(images_rgb))
    ]
    groups: dict[tuple[int, int, int], list[int]] = {}
    for idx, img in enumerate(images_rgb):
        groups.setdefault(img.shape, []).append(idx)

    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            batch_images = [images_rgb[i] for i in chunk]
            batch_locations = face_recognition.batch_face_locations(
                batch_images, number_of_times_to_upsample=num_upsample
            )
            for offset, face_locations in enumerate(batch_locations):
                results[chunk[offset]] = face_locations
    return results


def detect_face_locations(
    images_rgb: list[np.ndarray],
    model: str,
    num_upsample: int,
    use_gpu_batch: bool,
    batch_size: int,
) -> list[list[tuple[int, int, int, int]]]:
    if use_gpu_batch:
        return batch_face_locations_by_shape(
            images_rgb, num_upsample=num_upsample, batch_size=batch_size
        )
    return [
        face_recognition.face_locations(img, number_of_times_to_upsample=num_upsample, model=model)
        for img in images_rgb
    ]


def load_yolo_model(weight_path: Path) -> YOLO:
    if not weight_path.exists():
        print(f"YOLO 模型不存在，尝试自动下载: {weight_path}", flush=True)
    return YOLO(str(weight_path))


def collect_person_crops(
    results: list[object],
    images_rgb: list[np.ndarray],
    person_class_id: int,
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    crops: list[np.ndarray] = []
    meta: list[tuple[int, int, int]] = []
    for entry_idx, result in enumerate(results):
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != person_class_id:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(images_rgb[entry_idx].shape[1], x2)
            y2 = min(images_rgb[entry_idx].shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = images_rgb[entry_idx][y1:y2, x1:x2]
            crops.append(np.ascontiguousarray(crop))
            meta.append((entry_idx, x1, y1))
    return crops, meta


def draw_faces(image_bgr: np.ndarray, boxes: list[dict[str, object]]) -> np.ndarray:
    for box_info in boxes:
        coords = box_info.get("xyxy")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in coords]
        label = str(box_info.get("className", ""))
        color = (0, 200, 0) if label != "unknown" else (0, 0, 255)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        if label:
            cv2.putText(
                image_bgr,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return image_bgr


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        for entry in path.iterdir():
            if entry.is_dir():
                for sub in entry.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                entry.rmdir()
            else:
                entry.unlink()
        path.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    print(f"运行参数: {vars(args)}", flush=True)

    if args.gpu and args.model != "cnn":
        print("GPU 批量模式建议使用 cnn 模型", flush=True)

    known_encodings, known_names = load_known_faces(args.people_dir)
    use_yolo = not args.no_yolo
    yolo_model = load_yolo_model(args.yolo_model) if use_yolo else None

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        enable_auto_commit=False,
        auto_offset_reset=args.auto_offset_reset,
        max_poll_interval_ms=args.max_poll_interval_ms,
        session_timeout_ms=args.session_timeout_ms,
        heartbeat_interval_ms=args.heartbeat_interval_ms,
        value_deserializer=lambda v: v,
    )
    producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)

    if args.debug:
        ensure_empty_dir(args.debug_dir)

    print(
        f"开始订阅 {args.bootstrap_servers} / {args.topic}，批量 {args.batch_size} 条，"
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

        valid_entries, images_bgr, images_rgb = iter_images(entries)
        if not images_rgb:
            print("没有可用图片，跳过本批次。", flush=True)
            continue
        print(f"解析到 {len(valid_entries)} 条有效消息。", flush=True)

        inference_start = time.monotonic()
        faces_by_entry: dict[int, list[tuple[np.ndarray, tuple[int, int, int, int]]]] = {}
        if use_yolo and yolo_model is not None:
            yolo_results = yolo_model(images_bgr, conf=args.yolo_conf, verbose=False)
            crops, meta = collect_person_crops(
                yolo_results, images_rgb, args.person_class_id
            )
            crop_locations_batch: list[list[tuple[int, int, int, int]]] = []
            if crops:
                crop_locations_batch = detect_face_locations(
                    crops,
                    model=args.model,
                    num_upsample=args.num_upsample,
                    use_gpu_batch=args.gpu and args.model == "cnn",
                    batch_size=args.face_batch_size,
                )
            faces_by_entry = {idx: [] for idx in range(len(valid_entries))}
            for crop_idx, locations in enumerate(crop_locations_batch):
                if not locations:
                    continue
                crop = crops[crop_idx]
                encodings = face_recognition.face_encodings(
                    crop, known_face_locations=locations
                )
                entry_idx, offset_x, offset_y = meta[crop_idx]
                for face_encoding, location in zip(encodings, locations):
                    top, right, bottom, left = location
                    full_location = (
                        top + offset_y,
                        right + offset_x,
                        bottom + offset_y,
                        left + offset_x,
                    )
                    faces_by_entry[entry_idx].append((face_encoding, full_location))
        else:
            face_locations_batch = detect_face_locations(
                images_rgb,
                model=args.model,
                num_upsample=args.num_upsample,
                use_gpu_batch=args.gpu and args.model == "cnn",
                batch_size=args.face_batch_size,
            )
        inference_elapsed = time.monotonic() - inference_start

        hit_count = 0
        for entry_idx, entry in enumerate(valid_entries):
            if use_yolo:
                face_items = faces_by_entry.get(entry_idx, [])
                if not face_items:
                    print(
                        f"未检测到人脸: deviceId={entry.device_id}, photoPath={entry.photo_path}",
                        flush=True,
                    )
                    continue
                encodings = [item[0] for item in face_items]
                locations = [item[1] for item in face_items]
            else:
                locations = face_locations_batch[entry_idx]
            if not locations:
                print(
                    f"未检测到人脸: deviceId={entry.device_id}, photoPath={entry.photo_path}",
                    flush=True,
                )
                continue

            if not use_yolo:
                encodings = face_recognition.face_encodings(
                    images_rgb[entry_idx], known_face_locations=locations
                )
            boxes_payload: list[dict[str, object]] = []
            unknown_count = 0

            for face_encoding, location in zip(encodings, locations):
                match_name = "unknown"
                matched = False
                distance = None
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_idx = int(np.argmin(distances))
                    distance = float(distances[best_idx])
                    matched = distance <= args.tolerance
                    if matched:
                        match_name = known_names[best_idx]
                if not matched:
                    unknown_count += 1
                conf = None
                if distance is not None:
                    conf = max(0.0, min(1.0, 1.0 - distance))
                top, right, bottom, left = [int(v) for v in location]
                boxes_payload.append(
                    {
                        "className": match_name,
                        "conf": conf,
                        "xyxy": [left, top, right, bottom],
                    }
                )

            if unknown_count == 0:
                print(
                    f"人脸库命中: deviceId={entry.device_id}, photoPath={entry.photo_path}, "
                    f"faceCount={len(boxes_payload)}",
                    flush=True,
                )
                if args.debug:
                    image = images_bgr[entry_idx].copy()
                    draw_faces(image, boxes_payload)
                    output_path = args.debug_dir / Path(entry.photo_path).name
                    cv2.imwrite(str(output_path), image)
                continue

            payload = dict(entry.payload)
            payload.setdefault("topic", args.topic)
            payload["boxes"] = boxes_payload
            payload["unknownCount"] = unknown_count
            payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
            producer.send(args.alarm_topic, json.dumps(payload).encode("utf-8"))

            if args.debug:
                image = images_bgr[entry_idx].copy()
                draw_faces(image, boxes_payload)
                output_path = args.debug_dir / Path(entry.photo_path).name
                cv2.imwrite(str(output_path), image)

            hit_count += 1
            print(
                f"触发报警: deviceId={entry.device_id}, photoPath={entry.photo_path}, "
                f"unknownCount={unknown_count}",
                flush=True,
            )

        batch_count += 1
        print(
            f"处理批次: {batch_count}, 原始消息: {len(raw_messages)}, 有效图片: {len(images_rgb)}, "
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
