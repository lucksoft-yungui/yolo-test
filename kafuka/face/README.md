# Kafka 人脸校验

## 功能简介
- 订阅 `face_recognition_alarm` 主题，批量拉取消息并进行人脸识别。
- 人脸不在 `people` 人脸库内则触发报警。
- 报警消息推送到 `face_recognition_alarm_result` 主题。
- 传入 `--debug` 时保存带人脸框的图片到 `kafuka/face/debug`。
- 开启 `--gpu` 且使用 `cnn` 模型时启用批量检测优化。

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
    "topic": "face_recognition_alarm",
    "deviceId": "uuid",
    "areaId": "uuid",
    "photoPath": "/abs/path/to/image.jpg",
    "boxes": [
      {
        "className": "Alice",
        "conf": 0.59,
        "xyxy": [10, 50, 100, 200]
      },
      {
        "className": "unknown",
        "conf": 0.28,
        "xyxy": [20, 80, 120, 220]
      }
    ],
    "unknownCount": 1,
    "timestamp": "2025-01-01T12:00:00"
  }
]
```

## 依赖安装
```bash
uv add kafka-python face_recognition opencv-python numpy ultralytics
```

## 人脸库准备
将已知人脸图片放入 `people/` 目录，文件名会作为识别名字。

## 运行示例
```bash
uv run python kafuka/face/face_consumer.py \
  --bootstrap-servers localhost:9092 \
  --topic face_recognition_alarm \
  --alarm-topic face_recognition_alarm_result \
  --model cnn \
  --batch-size 10 \
  --tolerance 0.4 \
  --max-poll-interval-ms 60000 \
  --session-timeout-ms 60000 \
  --heartbeat-interval-ms 20000 \
  --auto-offset-reset latest \
  --yolo-model yolo11s.pt \
  --yolo-conf 0.6 \
  --person-class-id 0 \
  --debug

uv run python kafuka/face/face_consumer.py \
  --bootstrap-servers localhost:9092 \
  --topic face_recognition_alarm \
  --alarm-topic face_recognition_alarm_result \
  --model cnn \
  --gpu \
  --face-batch-size 128 \
  --debug

```

## 手动推送消息

用于向 Kafka 主题手动推送图片消息（只推送，不消费）。

```bash
uv run python kafuka/face/face_alarm_producer.py \
  --bootstrap-servers localhost:9092 \
  --topic face_recognition_alarm \
  --image-dir test_image \
  --limit 50

uv run python kafuka/face/face_alarm_producer.py \
  --bootstrap-servers localhost:9092 \
  --topic face_recognition_alarm \
  --limit 10
```

参数说明：
- `--image` 指定单张图片路径，可多次传入
- `--image-dir` 图片目录（递归收集 `.jpg`），默认 `/Users/peiyandong/Documents/code/ai/yolo-test/datasets/labcoat/images/train`
- `--limit` 限制发送数量，0 表示不限制
- `--device-id` 固定 deviceId（不填则每条随机）
- `--area-id` 固定 areaId（不填则不发送）
- `--area-no` 固定 areaNo（不填则不发送）
- `--zone-no` 固定 zoneNo（不填则不发送）
- `--timestamp` 毫秒时间戳（不填使用当前时间）

## 单图调试

用于单张图片调试人脸检测结果，会打印检测到的人脸数、最接近的人脸及距离，并在 `kafuka/face/debug/` 输出标注图。

```bash
uv run python kafuka/face/debug_face_check.py \
  --image "/Users/peiyandong/Documents/code/ai/yolo-test/datasets/labcoat/images/train/Neg -gloves with labcoat_20260116125612302_iql46q-compressed-20260116233753.jpg" \
  --model cnn \
  --yolo-model yolo11s.pt \
  --yolo-conf 0.7 \
  --person-class-id 0 \
  --tolerance 0.4 \
  --output kafuka/face/debug/face_debug.jpg

uv run python kafuka/face/debug_face_check.py \
  --image "/Users/peiyandong/Documents/code/ai/yolo-test/datasets/labcoat/images/train/g-Neg -gloves with labcoat_20260116125612424_0qsf3r-compressed-20260116233801.jpg" \
  --model cnn \
  --yolo-model yolo11s.pt \
  --yolo-conf 0.7 \
  --person-class-id 0 \
  --tolerance 0.4 \
  --output kafuka/face/debug/face_debug.jpg

```

参数说明：
- `--image` 待检测图片路径（必填）
- `--people-dir` 人脸库目录
- `--model` 检测模型（hog / cnn）
- `--no-yolo` 不使用 YOLO 预检测（默认启用）
- `--yolo-model` YOLO 模型路径
- `--yolo-conf` YOLO 置信度阈值
- `--person-class-id` 人员类别索引
- `--num-upsample` 上采样次数
- `--tolerance` 人脸匹配阈值，越小越严格
- `--output` 标注结果保存路径

## 参数说明
- `--bootstrap-servers` Kafka 地址，默认 `localhost:9092`
- `--topic` 主题名，默认 `face_recognition_alarm`
- `--alarm-topic` 报警消息推送主题，默认 `face_recognition_alarm_result`
- `--group-id` 消费者组 ID
- `--batch-size` 单次拉取数量
- `--max-wait-sec` 等待凑满批次的最长时间
- `--max-batches` 最大处理批次数，0 表示不限制
- `--max-poll-interval-ms` 最大拉取间隔（毫秒），默认 60000
- `--session-timeout-ms` 会话超时时间（毫秒），默认 30000
- `--heartbeat-interval-ms` 心跳间隔（毫秒），默认 10000
- `--auto-offset-reset` 起始偏移策略（latest / earliest / none）
- `--people-dir` 人脸库目录
- `--tolerance` 人脸匹配阈值（face distance），越小越严格，默认 0.6；一般 0.4-0.6 较稳健，阈值过大会把非人脸/相似物体误判为已知人脸，过小会导致同一人被判定为 unknown
- `--model` 检测模型（hog / cnn）
- `--no-yolo` 不使用 YOLO 预检测（默认启用）
- `--yolo-model` YOLO 模型路径
- `--yolo-conf` YOLO 置信度阈值
- `--person-class-id` 人员类别索引
- `--num-upsample` 上采样次数
- `--gpu` 开启 GPU 批量检测（建议配合 cnn）
- `--face-batch-size` GPU 批量检测大小
- `--debug` 保存调试图片
- `--debug-dir` 调试图片保存目录
