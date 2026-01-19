import argparse
import json
from pathlib import Path
from uuid import uuid4

from kafka import KafkaProducer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="手动推送火警图片消息到 Kafka")
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
        "--image",
        type=Path,
        action="append",
        default=[],
        help="指定单张图片路径，可重复使用",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("datasets/fire/images"),
        help="从目录递归收集 .jpg 图片，默认 datasets/fire/images",
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


def collect_images(paths: list[Path], image_dir: Path) -> list[Path]:
    images: list[Path] = []
    seen = set()
    for path in paths:
        if path.exists() and path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                images.append(resolved)
                seen.add(resolved)
        else:
            print(f"图片不存在，已跳过: {path}")

    if image_dir.exists():
        for path in sorted(image_dir.rglob("*.jpg")):
            resolved = path.resolve()
            if resolved not in seen:
                images.append(resolved)
                seen.add(resolved)
    else:
        print(f"目录不存在，已跳过: {image_dir}")

    return images


def main() -> None:
    args = parse_args()
    image_paths = collect_images(args.image, args.image_dir)
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
