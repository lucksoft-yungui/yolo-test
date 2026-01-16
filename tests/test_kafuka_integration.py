import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from uuid import uuid4

from kafka import KafkaAdminClient, KafkaProducer
from kafka.admin import NewTopic
from kafka.errors import NoBrokersAvailable
from kafka.errors import TopicAlreadyExistsError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSUMER_SCRIPT = PROJECT_ROOT / "kafuka" / "alarm_consumer.py"
MODEL_PATH = PROJECT_ROOT / "model" / "fire-kaggle" / "weights" / "best.pt"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "fire-alarm")


class KafkaFireAlarmIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        if not MODEL_PATH.exists():
            self.skipTest(f"模型文件不存在: {MODEL_PATH}")

        try:
            self.admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
            self.producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
        except NoBrokersAvailable:
            self.skipTest(f"Kafka 未运行在 {KAFKA_BOOTSTRAP_SERVERS}")

        try:
            self.admin.create_topics([NewTopic(name=KAFKA_TOPIC, num_partitions=1, replication_factor=1)])
        except TopicAlreadyExistsError:
            pass

    def tearDown(self) -> None:
        if hasattr(self, "admin"):
            self.admin.close()
        if hasattr(self, "producer"):
            self.producer.close()

    def test_consume_three_batches(self) -> None:
        preferred = [
            PROJECT_ROOT
            / "datasets"
            / "fanghufu-clothes"
            / "images"
            / "train"
            / "20200818_37.jpg",
            PROJECT_ROOT / "datasets" / "fire" / "images" / "train" / "fntr_img_2203.jpg",
            PROJECT_ROOT / "datasets" / "fire" / "images" / "train" / "fntr_img_2231.jpg",
            PROJECT_ROOT / "datasets" / "fire" / "images" / "train" / "fntr_img_2236.jpg",
            PROJECT_ROOT / "datasets" / "fire" / "images" / "train" / "fntr_img_2256.jpg",
            PROJECT_ROOT / "datasets" / "fire" / "images" / "train" / "fntr_img_2264.jpg",
            PROJECT_ROOT / "datasets" / "fire" / "images" / "train" / "fntr_img_2265.jpg",
            PROJECT_ROOT / "datasets" / "fire" / "images" / "train" / "fntr_img_2271.jpg",
            PROJECT_ROOT / "datasets" / "fire" / "images" / "train" / "fntr_img_2412.jpg",
            PROJECT_ROOT / "datasets" / "fire" / "images" / "train" / "fntr_img_2688.jpg",
        ]
        missing = [p for p in preferred if not p.exists()]
        if missing:
            self.skipTest(f"指定图片不存在: {missing[0]}")

        image_dir = PROJECT_ROOT / "datasets" / "fire" / "images"
        all_images = [p for p in sorted(image_dir.rglob("*.jpg")) if p not in preferred]
        image_paths = preferred + all_images
        if len(image_paths) < 23:
            self.skipTest("datasets/fire 图片数量不足 23 张")
        image_paths = image_paths[:23]

        topic = KAFKA_TOPIC
        process = subprocess.Popen(
            [
                sys.executable,
                str(CONSUMER_SCRIPT),
                "--bootstrap-servers",
                KAFKA_BOOTSTRAP_SERVERS,
                "--topic",
                topic,
                "--batch-size",
                "10",
                "--max-wait-sec",
                "1",
                "--max-batches",
                "3",
                "--auto-offset-reset",
                "earliest",
                "--model",
                str(MODEL_PATH),
                "--device",
                "cpu",
            ],
            cwd=str(PROJECT_ROOT),
            env={**dict(os.environ), "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            time.sleep(1.0)
            for path in image_paths:
                payload = [{"deviceId": str(uuid4()), "photoPath": str(path.resolve())}]
                self.producer.send(topic, json.dumps(payload).encode("utf-8"))
            self.producer.flush()

            output, _ = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            self.fail("消费者超时未退出")
        finally:
            if process.poll() is None:
                process.kill()

        if output:
            print(output)
        batch_lines = [line for line in output.splitlines() if line.startswith("处理批次:")]
        self.assertEqual(len(batch_lines), 3, f"期望消费 3 轮，实际输出:\n{output}")


if __name__ == "__main__":
    unittest.main()
