# kafuka/labcoat

防护服检测消费者。订阅 `ppe_alarm`，先批量检测“未穿实验服”，再裁剪目标检测手套，命中后逐条推送到 `ppe_alarm_result`。

## 使用

```bash
uv run python kafuka/labcoat/labcoat_consumer.py
```

生产者（从 `kafuka/labcoat/test/images` 推送）：

```bash
uv run python kafuka/labcoat/labcoat_alarm_producer.py
```

## 完整参数示例

```bash
uv run python kafuka/labcoat/labcoat_consumer.py \
  --bootstrap-servers localhost:9092 \
  --topic ppe_alarm \
  --alarm-topic ppe_alarm_result \
  --group-id ppe-alarm-consumer \
  --batch-size 10 \
  --max-wait-sec 2 \
  --max-batches 0 \
  --auto-offset-reset latest \
  --labcoat-model model/labcoat/best.pt \
  --glove-model model/glove/best.pt \
  --labcoat-yaml labcoat.yaml \
  --glove-yaml glove.yaml \
  --labcoat-class-name "no labcoat" \
  --glove-class-name "with glove" \
  --labcoat-class-id 1 \
  --glove-class-id 0 \
  --labcoat-conf 0.7 \
  --glove-conf 0.7 \
  --gpu -1 \
  --device mps \
  --debug \
  --debug-dir kafuka/labcoat/debug
```

生产者完整参数示例：

```bash
uv run python kafuka/labcoat/labcoat_alarm_producer.py \
  --bootstrap-servers localhost:9092 \
  --topic ppe_alarm \
  --image-dir kafuka/labcoat/test/images \
  --limit 0 \
  --device-id "" \
  --area-id ""
```

## 常用参数

- `--topic`：消费主题，默认 `ppe_alarm`
- `--alarm-topic`：结果主题，默认 `ppe_alarm_result`
- `--bootstrap-servers`：Kafka 地址，默认 `localhost:9092`
- `--batch-size`：批处理条数，默认 10
- `--debug`：开启调试图保存
- `--debug-dir`：调试图保存目录，默认 `kafuka/labcoat/debug`
- `--labcoat-model` / `--glove-model`：模型路径
- `--labcoat-conf` / `--glove-conf`：置信度阈值（默认 0.7）
