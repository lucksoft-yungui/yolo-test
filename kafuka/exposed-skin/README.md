# kafuka/exposed-skin

实验人员皮肤裸露检测消费者。订阅 `exposed-skin-alarm`，先批量检测“穿实验服”，再裁剪对应区域检测“手套 + 皮肤裸露”，同一实验服区域同时命中后逐条推送到 `exposed-skin-alarm-result`。

## 使用

```bash
uv run python kafuka/exposed-skin/exposed_skin_consumer.py
```

最小可用（三模型路径显式指定）：

```bash
uv run python kafuka/exposed-skin/exposed_skin_consumer.py \
  --topic exposed-skin-alarm \
  --alarm-topic exposed-skin-alarm-result \
  --labcoat-model model/labcoat/best.pt \
  --glove-model model/glove/best.pt \
  --exposed-skin-model model/exposed-skin/best.pt
```

生产者（从 `kafuka/exposed-skin/test/images` 推送）：

```bash
uv run python kafuka/exposed-skin/exposed_skin_alarm_producer.py
```

## 完整参数示例

```bash
uv run python kafuka/exposed-skin/exposed_skin_consumer.py \
  --bootstrap-servers localhost:9092 \
  --topic exposed-skin-alarm \
  --alarm-topic exposed-skin-alarm-result \
  --group-id exposed-skin-alarm-consumer \
  --batch-size 10 \
  --max-wait-sec 2 \
  --max-batches 0 \
  --auto-offset-reset latest \
  --labcoat-model model/labcoat/best.pt \
  --glove-model model/glove/best.pt \
  --exposed-skin-model model/exposed-skin/best.pt \
  --labcoat-yaml labcoat.yaml \
  --glove-yaml glove.yaml \
  --exposed-skin-yaml exposed-skin.yaml \
  --labcoat-class-name "with labcoat" \
  --glove-class-name "with glove" \
  --exposed-skin-class-name "exposed-skin" \
  --labcoat-class-id 0 \
  --glove-class-id 0 \
  --exposed-skin-class-id 0 \
  --labcoat-conf 0.7 \
  --glove-conf 0.7 \
  --exposed-skin-conf 0.7 \
  --device mps \
  --debug \
  --debug-dir kafuka/exposed-skin/debug
```

生产者完整参数示例：

```bash
uv run python kafuka/exposed-skin/exposed_skin_alarm_producer.py \
  --bootstrap-servers localhost:9092 \
  --topic exposed-skin-alarm \
  --image-dir kafuka/exposed-skin/test/images \
  --limit 0 \
  --device-id "device1" \
  --area-id "area1" \
  --area-no "B311" \
  --zone-no "Z001" \
  --timestamp 0
```

## 常用参数

- `--topic`：消费主题，默认 `exposed-skin-alarm`
- `--alarm-topic`：结果主题，默认 `exposed-skin-alarm-result`
- `--labcoat-model` / `--glove-model` / `--exposed-skin-model`：模型路径
- `--labcoat-class-name`：默认 `with labcoat`
- `--glove-class-name`：默认 `with glove`
- `--exposed-skin-class-name`：默认 `exposed-skin`
- `--labcoat-conf` / `--glove-conf` / `--exposed-skin-conf`：置信度阈值（默认 0.7）
- `--debug`：开启调试图保存
- `--debug-dir`：调试图保存目录，默认 `kafuka/exposed-skin/debug`

## 模型配置（必须三套）

- 实验服模型：`--labcoat-model` + `--labcoat-yaml` + `--labcoat-class-name` + `--labcoat-conf`
- 手套模型：`--glove-model` + `--glove-yaml` + `--glove-class-name` + `--glove-conf`
- 皮肤裸露模型：`--exposed-skin-model` + `--exposed-skin-yaml` + `--exposed-skin-class-name` + `--exposed-skin-conf`

例如你给的这组参数就是完整三模型配置，不是单模型：

```bash
--labcoat-model model/labcoat/best.pt --labcoat-yaml labcoat.yaml --labcoat-class-name "with labcoat" --labcoat-conf 0.7
--glove-model model/glove/best.pt --glove-yaml glove.yaml --glove-class-name "with glove" --glove-conf 0.7
--exposed-skin-model model/exposed-skin/best.pt --exposed-skin-yaml exposed-skin.yaml --exposed-skin-class-name "exposed-skin" --exposed-skin-conf 0.7
```
