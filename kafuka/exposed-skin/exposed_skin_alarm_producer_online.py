import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

from kafka import KafkaProducer


ONLINE_IMAGE_DIR = Path("/mnt/nfs/collector/test/exposed-skin")
LOCAL_TEST_IMAGE_DIR = Path("kafuka/exposed-skin/test/images")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="线上环境推送皮肤裸露检测图片消息到 Kafka")
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="10.10.6.13:9092",
        help="Kafka 地址，默认 10.10.6.13:9092",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="skin_exposure_alarm",
        help="Kafka 主题，默认 skin_exposure_alarm",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ONLINE_IMAGE_DIR,
        help="线上测试图片目录，默认 /mnt/nfs/collector/test/exposed-skin",
    )
    parser.add_argument(
        "--local-image-dir",
        type=Path,
        default=LOCAL_TEST_IMAGE_DIR,
        help="本地测试图片目录（用于匹配线上同名文件），默认 kafuka/exposed-skin/test/images",
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
    parser.add_argument(
        "--area-no",
        type=str,
        default="",
        help="固定 areaNo，不填则不发送该字段",
    )
    parser.add_argument(
        "--zone-no",
        type=str,
        default="",
        help="固定 zoneNo，不填则不发送该字段",
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        default=0,
        help="时间戳（毫秒），默认使用当前时间",
    )
    parser.add_argument(
        "--check-online-exists",
        action="store_true",
        help="校验线上文件在当前机器可访问且存在（默认不校验）",
    )
    return parser.parse_args()


def collect_images(
    local_image_dir: Path,
    online_image_dir: Path,
    check_online_exists: bool,
) -> tuple[list[Path], list[Path]]:
    if not local_image_dir.exists():
        print(f"本地测试目录不存在: {local_image_dir}")
        return [], []

    local_rel_paths: list[Path] = []
    for path in sorted(local_image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            local_rel_paths.append(path.relative_to(local_image_dir))

    if not local_rel_paths:
        print(f"本地测试目录下未找到图片: {local_image_dir}")
        return [], []

    image_paths: list[Path] = []
    missing_paths: list[Path] = []
    for rel_path in local_rel_paths:
        online_path = online_image_dir / rel_path
        if check_online_exists:
            if online_path.exists() and online_path.is_file():
                image_paths.append(online_path)
            else:
                missing_paths.append(online_path)
            continue
        image_paths.append(online_path)

    return image_paths, missing_paths


def main() -> None:
    args = parse_args()
    image_paths, missing_paths = collect_images(
        args.local_image_dir,
        args.image_dir,
        args.check_online_exists,
    )

    if args.check_online_exists and missing_paths:
        print(f"线上目录缺少 {len(missing_paths)} 张同名图片，示例: {missing_paths[0]}")

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        print("没有可发送的图片，退出。")
        return

    producer = KafkaProducer(bootstrap_servers=args.bootstrap_servers)
    try:
        for path in image_paths:
            item = {
                "deviceId": args.device_id or str(uuid4()),
                "photoPath": str(path),
                "timestamp": args.timestamp or int(time.time() * 1000),
            }
            if args.area_id:
                item["areaId"] = args.area_id
            if args.area_no:
                item["areaNo"] = args.area_no
            if args.zone_no:
                item["zoneNo"] = args.zone_no
            producer.send(args.topic, json.dumps([item]).encode("utf-8"))
        producer.flush()
    finally:
        producer.close()

    print(
        f"已发送 {len(image_paths)} 条消息到 {args.bootstrap_servers} / {args.topic}",
        flush=True,
    )


if __name__ == "__main__":
    main()
