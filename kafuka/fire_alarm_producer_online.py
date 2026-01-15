import argparse
import json
from pathlib import Path
from uuid import uuid4

from kafka import KafkaProducer


IMAGE_ROOT = Path("/mnt/nfs/datasets")
IMAGE_NAMES = [f"fntr_img_{index}.jpg" for index in range(1000, 1010)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="线上环境推送火警图片消息到 Kafka")
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
        "--limit",
        type=int,
        default=0,
        help="限制发送图片数量，0 表示不限制",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="",
        help="固定 deviceId，不填则每条消息随机生成",
    )
    parser.add_argument(
        "--area-id",
        type=str,
        default="",
        help="固定 areaId，不填则不发送该字段",
    )
    return parser.parse_args()


def collect_images() -> list[Path]:
    return [(IMAGE_ROOT / name) for name in IMAGE_NAMES]


def main() -> None:
    args = parse_args()
    image_paths = collect_images()
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        print("没有可发送的图片，退出。")
        return

    producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)
    try:
        for path in image_paths:
            device_id = args.device_id or str(uuid4())
            item = {"deviceId": device_id, "photoPath": str(path)}
            if args.area_id:
                item["areaId"] = args.area_id
            payload = [item]
            producer.send(args.topic, json.dumps(payload).encode("utf-8"))
        producer.flush()
    finally:
        producer.close()

    print(
        f"已发送 {len(image_paths)} 条消息到 {args.bootstrap_servers} / {args.topic}",
        flush=True,
    )


if __name__ == "__main__":
    main()
