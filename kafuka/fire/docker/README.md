# Fire Alarm Consumer Docker

该镜像用于运行 `kafuka/base/alarm_consumer.py`。依赖通过 `uv.lock` 同步，所有运行参数可通过环境变量配置。

## 构建镜像

在项目根目录执行：

```bash
docker build -t alarm-consumer -f kafuka/base/docker/Dockerfile .
```

CUDA 版本：

```bash
docker build -t alarm-consumer:cuda -f kafuka/base/docker/Dockerfile.cuda .
```

也可以用脚本指定架构和 Dockerfile（构建后直接加载到本地）：

```bash
kafuka/base/docker/build.sh -f kafuka/base/docker/Dockerfile -t alarm-consumer -p linux/amd64
kafuka/base/docker/build.sh -f kafuka/base/docker/Dockerfile.cuda -t alarm-consumer:v1-amd64 -p linux/amd64
```

## 运行示例

```bash
docker run --rm \
  -e KAFKA_BOOTSTRAP_SERVERS=host.docker.internal:9092 \
  -e KAFKA_TOPIC=fire-alarm \
  -e KAFKA_ALARM_TOPIC=fire-alarm-result \
  -e KAFKA_GROUP_ID=fire-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=0 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e MODEL_DEVICE=mps \
  -e MODEL_IMGSZ=1920 \
  -e TARGET_CLASS_NAME=fire,smoke \
  -e TARGET_CLASS_INDEX=0,1 \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/model:/app/model:ro \
  alarm-consumer
```

CUDA 运行示例：

```bash
docker run --restart=always --gpus all \
  --name fire-consumer \
  --network host \
  --restart=always \
  -e KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 \
  -e KAFKA_TOPIC=fire-alarm \
  -e KAFKA_ALARM_TOPIC=fire-alarm-result \
  -e KAFKA_GROUP_ID=fire-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e MODEL_DEVICE=cuda \
  -e MODEL_GPU=0 \
  -e MODEL_CONF=0.6 \
  -e TARGET_CLASS_NAME=fire,smoke \
  -e TARGET_CLASS_INDEX=0,1 \
  -e MODEL_IMGSZ=1280 \
  -e MODEL_PATH=/app/models/fire-lab.pt \
  -v /mnt/nfs/code/kafuka:/app/kafuka:ro \
  -v /mnt/nfs/models:/app/models:ro \
  -v /mnt/nfs/collector:/mnt/nfs/collector:ro \
  alarm-consumer:v1-amd64
```

如果 Kafka 在同一台宿主机：
- macOS/Windows 用 `host.docker.internal:9092`
- Linux 可改为 `--network host` 或填写宿主机实际地址

## 环境变量

- `KAFKA_BOOTSTRAP_SERVERS` Kafka 地址，默认 `localhost:9092`
- `KAFKA_TOPIC` 来源主题（消费主题），默认 `fire-alarm`
- `KAFKA_ALARM_TOPIC` 报警推送主题，默认 `fire-alarm-result`
- `KAFKA_GROUP_ID` 消费者组，默认 `fire-alarm-consumer`
- `KAFKA_BATCH_SIZE` 批次大小，默认 `10`
- `KAFKA_MAX_WAIT_SEC` 等待凑满批次时间（秒），默认 `2`
- `KAFKA_MAX_BATCHES` 最大批次（0 不限制），默认 `0`
- `KAFKA_AUTO_OFFSET_RESET` 起始偏移策略，默认 `latest`
  - `latest`：从最新消息开始（不补历史）
  - `earliest`：从最早消息开始（补历史）
  - `none`：没有已提交 offset 就报错
- `MODEL_PATH` 模型路径，默认 `/app/model/fire-kaggle/weights/best.pt`
- `MODEL_CONF` 置信度阈值，默认 `0.6`
- `MODEL_DEVICE` 设备（cpu / cuda / mps），默认空
- `MODEL_GPU` GPU 序号（仅当 `MODEL_DEVICE=cuda` 时生效），默认 `0`
- `MODEL_IMGSZ` 推理输入尺寸（正方形边长），默认 `0` 表示模型默认
- `TARGET_CLASS_NAME` 目标类别名称，支持逗号分隔多个（如 `fire,smoke`），默认 `fire`
- `TARGET_CLASS_INDEX` 报警触发的类别索引，支持逗号分隔多个（如 `0,1`），默认 `0`

## 说明

- 镜像不再内置脚本与模型，请在运行时挂载 `kafuka` 与 `model` 目录，并设置 `MODEL_PATH`。

示例：

```bash
docker run --rm \
  -e KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 \
  -e MODEL_DEVICE=cpu \
  alarm-consumer
```
