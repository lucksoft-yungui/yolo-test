# Kafka 火警订阅说明

## 功能简介
- 订阅 `fire-alarm` 主题，按批次拉取消息并批量推理。
- 将消息中的图片合并推理，命中烟火类别则打印。
- 命中烟火后推送报警消息到 `fire-alarm-result` 主题（可配置）。

## 消息格式
```json
[
  {
    "deviceId": "uuid",
    "areaId": "uuid",
    "photoPath": "/abs/path/to/image.jpg"
  }
]
```

## 报警消息格式
```json
[
  {
    "topic": "fire-alarm",
    "deviceId": "uuid",
    "areaId": "uuid",
    "photoPath": "/abs/path/to/image.jpg",
    "boxes": [
      {
        "conf": 0.85,
        "xyxy": [10.0, 20.0, 30.0, 40.0]
      }
    ]
  }
]
```

## 依赖安装
```bash
uv add kafka-python
```

## 运行示例
```bash
uv run python kafuka/base/alarm_consumer.py \
  --bootstrap-servers localhost:9092 \
  --topic fire-alarm \
  --alarm-topic fire-alarm-result \
  --batch-size 10 \
  --model model/fire-lab/best.pt \
  --conf 0.6 \
  --imgsz 1920 \
  --debug

  uv run python kafuka/base/alarm_consumer.py \
  --bootstrap-servers localhost:9092 \
  --topic fire-alarm \
  --alarm-topic fire-alarm-result \
  --batch-size 10 \
  --model model/fire-lab/best_1280_68_bak.pt \
  --conf 0.6 \
  --imgsz 1920 \
  --debug

uv run python kafuka/base/alarm_consumer.py \
  --bootstrap-servers 10.10.6.13:9092 \
  --topic fire-alarm \
  --alarm-topic fire-alarm-result \
  --batch-size 10 \
  --model model/fire-kaggle/weights/best.pt \
  --device mps \
  --imgsz 1920 \
  --debug
```

## 手动推送消息

用于向 Kafka 主题手动推送图片消息（只推送，不消费）。

```bash
uv run python kafuka/base/fire_alarm_producer.py \
  --bootstrap-servers localhost:9092 \
  --topic fire-alarm \
  --image-dir datasets/fire-lab/images \
  --limit 23

uv run python kafuka/base/fire_alarm_producer.py \
  --bootstrap-servers 10.10.6.13:9092 \
  --topic fire-alarm \
  --image-dir datasets/fire/images \
  --limit 23
```

## 线上发送测试

固定发送线上目录 `/mnt/nfs/datasets` 下的 `fntr_img_1000.jpg` ~ `fntr_img_1009.jpg`。

```bash
uv run python kafuka/base/fire_alarm_producer_online.py \
  --bootstrap-servers 10.10.6.13:9092 \
  --topic fire-alarm \
  --limit 10
```

参数说明：
- `--image` 指定单张图片路径，可多次传入
- `--image-dir` 图片目录（递归收集 `.jpg`）
- `--limit` 限制发送数量，0 表示不限制
- `--device-id` 固定 deviceId（不填则每条随机）
- `--area-id` 固定 areaId（不填则不发送）

## 参数说明
- `--bootstrap-servers` Kafka 地址，默认 `localhost:9092`
- `--topic` 主题名，默认 `fire-alarm`
- `--alarm-topic` 报警消息推送主题，默认 `fire-alarm-result`
- `--group-id` 消费者组 ID
- `--batch-size` 单次拉取数量
- `--max-wait-sec` 等待凑满批次的最长时间
- `--model` 模型权重路径
- `--conf` 置信度阈值
- `--imgsz` 推理输入尺寸（正方形边长，如 1920/1280/640；不填或 0 表示模型默认）
- `--debug` 调试模式，保存识别结果图到 `kafuka/base/debug`
- `--device` 推理设备（cpu / cuda / mps）
- `--target-class-name` 目标类别名称（默认 `fire`）
- `--target-class-index` 报警触发的类别索引（默认 `0`）
- `--auto-offset-reset` 起始偏移策略（latest / earliest / none）
