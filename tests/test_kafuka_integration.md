# test_kafuka_integration.py 说明

这个测试是 Kafka + 模型推理的端到端集成测试，验证 `kafuka/base/alarm_consumer.py` 在消费 `fire-alarm` 主题消息时，能按批次正常处理并退出。

## 测试做了什么

- 确保模型文件存在：`model/fire-kaggle/weights/best.pt`。
- 连接本地 Kafka：`localhost:9092`。
- 创建主题 `fire-alarm`（已存在则跳过）。
- 选择固定图片 + `datasets/fire/images` 下的其余图片，总计 23 张。
- 启动 `kafuka/base/alarm_consumer.py`，设置：
  - `--batch-size 10`
  - `--max-wait-sec 1`
  - `--max-batches 3`
  - `--auto-offset-reset earliest`
  - `--device cpu`
- 将 23 条图片消息写入 Kafka。
- 断言消费者输出中出现 3 行 `处理批次:`。

## 运行方式

```bash
uv run python -m unittest tests.test_kafuka_integration
```

如果你使用系统 Python：

```bash
python -m unittest tests.test_kafuka_integration
```

## Kafka 配置

可通过环境变量自定义 Kafka 地址与主题名：

- `KAFKA_BOOTSTRAP_SERVERS`（默认 `localhost:9092`）
- `KAFKA_TOPIC`（默认 `fire-alarm`）

示例：

```bash
KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:19092 KAFKA_TOPIC=fire-test \
  uv run python -m unittest tests.test_kafuka_integration

KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 KAFKA_TOPIC=fire-test \
  uv run python -m unittest tests.test_kafuka_integration
```

## 运行前准备

- Kafka 已在本机 `localhost:9092` 运行。
- 模型文件已训练或已放置到 `model/fire-kaggle/weights/best.pt`。
- `datasets/fire/images` 下至少有 23 张 `.jpg` 图片。
- `datasets/fanghufu-clothes/images/train/20200818_37.jpg` 存在（测试会优先使用）。

## 常见跳过原因

测试会在以下情况下自动 `skip`：

- 模型文件缺失。
- Kafka 未启动或无法连接。
- 指定的固定图片不存在。
- `datasets/fire` 图片数量不足 23。

## 可调参数

如需修改 Kafka 地址、主题名、批次大小等，可直接改 `tests/test_kafuka_integration.py` 中对应的常量或命令行参数。
