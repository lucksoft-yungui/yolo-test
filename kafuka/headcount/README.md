# Kafka 无人值守人数检测

## 功能简介
- 订阅 `unattended_alarm` 主题，批量拉取消息并进行人员检测。
- 使用 YOLO11n 模型统计人员数量，人数为 1 且在时间范围内触发报警。
- 报警消息推送到 `unattended_alarm_result` 主题。
- 传入 `--debug` 时保存带线框的报警图片到 `kafuka/headcount/debug`。

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
    "topic": "unattended_alarm",
    "deviceId": "uuid",
    "areaId": "uuid",
    "photoPath": "/abs/path/to/image.jpg",
    "boxes": [
      {
        "className": "person",
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
uv run python kafuka/headcount/headcount_consumer.py \
  --bootstrap-servers localhost:9092 \
  --topic unattended_alarm \
  --alarm-topic unattended_alarm_result \
  --batch-size 10 \
  --model yolo11n.pt \
  --conf 0.5 \
  --time-start 00:00 \
  --time-end 08:00

uv run python kafuka/headcount/headcount_consumer.py \
  --bootstrap-servers localhost:9092 \
  --topic unattended_alarm \
  --alarm-topic unattended_alarm_result \
  --batch-size 10 \
  --model yolo11s.pt \
  --conf 0.3 \
  --time-start 08:00 \
  --time-end 23:00 \
  --debug

uv run python kafuka/headcount/headcount_consumer.py \
  --bootstrap-servers 10.10.6.13:9092 \
  --topic unattended_alarm \
  --alarm-topic unattended_alarm_result \
  --batch-size 10 \
  --model yolo11n.pt \
  --conf 0.5 \
  --device mps \
  --debug
```

## 手动推送消息

用于向 Kafka 主题手动推送图片消息（只推送，不消费）。

```bash
uv run python kafuka/headcount/headcount_alarm_producer.py \
  --bootstrap-servers localhost:9092 \
  --topic unattended_alarm \
  --image-dir datasets/labcoat/images/train \
  --limit 23

uv run python kafuka/headcount/headcount_alarm_producer.py \
  --bootstrap-servers 10.10.6.13:9092 \
  --topic unattended_alarm \
  --image-dir datasets/labcoat/images/train \
  --limit 23
```

参数说明：
- `--image` 指定单张图片路径，可多次传入
- `--image-dir` 图片目录（递归收集 `.jpg`）
- `--limit` 限制发送数量，0 表示不限制
- `--device-id` 固定 deviceId（不填则每条随机）
- `--area-id` 固定 areaId（不填则不发送）

## 单图调试

用于单张图片调试人员检测结果，会打印人数与坐标，并在 `kafuka/headcount/debug` 输出标注图。

```bash
uv run python kafuka/headcount/debug_check.py \
  --image "kafuka/headcount/debug/003--Neg -gloves with labcoat_20260116125612308_puce1w-compressed-20260116233754.jpg" \
  --model yolo11n.pt \
  --conf 0.5

uv run python kafuka/headcount/debug_check.py \
  --image "kafuka/headcount/debug/0012Neg -gloves with labcoat_20260116125612330_o5t9dl-compressed-20260116233755.jpg" \
  --model yolo11s.pt \
  --conf 0.3
```

参数说明：
- `--image` 待检测图片路径（必填）
- `--model` 模型权重路径（支持自动下载）
- `--conf` 置信度阈值
- `--person-class-id` 人员类别索引
- `--device` 推理设备（cpu / cuda / mps）
- `--save-dir` 标注结果保存目录

## 参数说明
- `--bootstrap-servers` Kafka 地址，默认 `localhost:9092`
- `--topic` 主题名，默认 `unattended_alarm`
- `--alarm-topic` 报警消息推送主题，默认 `unattended_alarm_result`
- `--group-id` 消费者组 ID
- `--batch-size` 单次拉取数量
- `--max-wait-sec` 等待凑满批次的最长时间
- `--max-batches` 最大处理批次数，0 表示不限制
- `--auto-offset-reset` 起始偏移策略（latest / earliest / none）
- `--model` YOLO11n 权重路径
- `--conf` 置信度阈值
- `--person-class-name` 人员类别名称
- `--person-class-id` 人员类别索引
- `--time-start` 报警开始时间（HH:MM）
- `--time-end` 报警结束时间（HH:MM）
- `--device` 推理设备（cpu / cuda / mps）
- `--gpu` 指定 GPU 序号
- `--debug` 保存调试图片
- `--debug-dir` 调试图片保存目录
