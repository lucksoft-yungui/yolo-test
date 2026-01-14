# Kafka 火警订阅说明

## 功能简介
- 订阅 `fire-alarm` 主题，按批次拉取消息并批量推理。
- 将消息中的图片合并推理，命中烟火类别则打印。

## 消息格式
```json
[
  {"deviceId": "uuid", "photoPath": "/abs/path/to/image.jpg"}
]
```

## 依赖安装
```bash
uv add kafka-python
```

## 运行示例
```bash
uv run python kafuka/fire_alarm_consumer.py \
  --bootstrap-servers localhost:9092 \
  --topic fire-alarm \
  --batch-size 10 \
  --model model/fire-kaggle/weights/best.pt

 uv run python kafuka/fire_alarm_consumer.py \
  --bootstrap-servers 10.10.6.13:9092 \
  --topic fire-alarm \
  --batch-size 10 \
  --model model/fire-kaggle/weights/best.pt 
```

## 手动推送消息

用于向 Kafka 主题手动推送图片消息（只推送，不消费）。

```bash
uv run python kafuka/fire_alarm_producer.py \
  --bootstrap-servers localhost:9092 \
  --topic fire-alarm \
  --image-dir datasets/fire/images \
  --limit 23

uv run python kafuka/fire_alarm_producer.py \
  --bootstrap-servers 10.10.6.13:9092 \
  --topic fire-alarm \
  --image-dir datasets/fire/images \
  --limit 23
```

参数说明：
- `--image` 指定单张图片路径，可多次传入
- `--image-dir` 图片目录（递归收集 `.jpg`）
- `--limit` 限制发送数量，0 表示不限制
- `--device-id` 固定 deviceId（不填则每条随机）

## 参数说明
- `--bootstrap-servers` Kafka 地址，默认 `localhost:9092`
- `--topic` 主题名，默认 `fire-alarm`
- `--group-id` 消费者组 ID
- `--batch-size` 单次拉取数量
- `--max-wait-sec` 等待凑满批次的最长时间
- `--model` 模型权重路径
- `--conf` 置信度阈值
- `--device` 推理设备（cpu / cuda / mps）
- `--fire-class` 烟火类别名称（默认 `fire`）
- `--auto-offset-reset` 起始偏移策略（latest / earliest / none）
