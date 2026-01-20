# headcount-consumer Docker

该镜像用于运行 `kafuka/headcount/headcount_consumer.py`。依赖通过 `uv.lock` 同步，所有运行参数可通过环境变量配置。

## 构建镜像

在项目根目录执行：

```bash
docker build -t headcount-consumer -f kafuka/headcount/docker/Dockerfile .
```

CUDA 版本：

```bash
docker build -t headcount-consumer:cuda -f kafuka/headcount/docker/Dockerfile.cuda .
```

也可以用脚本指定架构和 Dockerfile（构建后直接加载到本地）：

```bash
kafuka/headcount/docker/build.sh -f kafuka/headcount/docker/Dockerfile -t headcount-consumer -p linux/amd64
kafuka/headcount/docker/build.sh -f kafuka/headcount/docker/Dockerfile.cuda -t headcount-consumer:cuda -p linux/amd64
kafuka/headcount/docker/build.sh -f kafuka/headcount/docker/Dockerfile.cuda -n headcount-consumer --tag-name v1-amd64 -p linux/amd64
```

## 运行示例

```bash
docker run --name headcount \
  -e KAFKA_BOOTSTRAP_SERVERS=host.docker.internal:9092 \
  -e KAFKA_TOPIC=unattended_alarm \
  -e KAFKA_ALARM_TOPIC=unattended_alarm_result \
  -e KAFKA_GROUP_ID=unattended-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e HEADCOUNT_MODEL_PATH=/app/yolo11n.pt \
  -e HEADCOUNT_CONF=0.5 \
  -e HEADCOUNT_TIME_START=00:00 \
  -e HEADCOUNT_TIME_END=08:00 \
  -e MODEL_DEVICE=cpu \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/yolo11n.pt:/app/yolo11n.pt:ro \
  -v /Users/peiyandong/Documents/code/ai/yolo-test:/Users/peiyandong/Documents/code/ai/yolo-test:ro \
  headcount-consumer

  docker run --name headcount \
  --network host \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KAFKA_TOPIC=unattended_alarm \
  -e KAFKA_ALARM_TOPIC=unattended_alarm_result \
  -e KAFKA_GROUP_ID=unattended-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e HEADCOUNT_MODEL_PATH=/app/yolo11s.pt \
  -e HEADCOUNT_CONF=0.3 \
  -e HEADCOUNT_TIME_START=00:00 \
  -e HEADCOUNT_TIME_END=08:00 \
  -e MODEL_DEVICE=cpu \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/yolo11s.pt:/app/yolo11s.pt:ro \
  -v /Users/peiyandong/Documents/code/ai/yolo-test:/Users/peiyandong/Documents/code/ai/yolo-test:ro \
  headcount-consumer
```

CUDA 运行示例：

```bash
docker run --rm --gpus all \
  -e KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 \
  -e MODEL_DEVICE=cuda \
  -e MODEL_GPU=0 \
  -e HEADCOUNT_MODEL_PATH=/app/yolo11n.pt \
  -e HEADCOUNT_CONF=0.5 \
  -e HEADCOUNT_TIME_START=00:00 \
  -e HEADCOUNT_TIME_END=08:00 \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/yolo11n.pt:/app/yolo11n.pt:ro \
  -v /Users/peiyandong/Documents/code/ai/yolo-test:/Users/peiyandong/Documents/code/ai/yolo-test:ro \
  headcount-consumer:cuda
```

如果 Kafka 在同一台宿主机：
- macOS/Windows 用 `host.docker.internal:9092`
- Linux 可改为 `--network host` 或填写宿主机实际地址

## 参数说明

- `KAFKA_BOOTSTRAP_SERVERS` Kafka 地址，默认 `localhost:9092`
- `KAFKA_TOPIC` 来源主题（消费主题），默认 `unattended_alarm`
- `KAFKA_ALARM_TOPIC` 报警推送主题，默认 `unattended_alarm_result`
- `KAFKA_GROUP_ID` 消费者组，默认 `unattended-alarm-consumer`
- `KAFKA_BATCH_SIZE` 批次大小，默认 `10`
- `KAFKA_MAX_WAIT_SEC` 等待凑满批次时间（秒），默认 `2`
- `KAFKA_MAX_BATCHES` 最大批次（0 不限制），默认 `0`
- `KAFKA_AUTO_OFFSET_RESET` 起始偏移策略，默认 `latest`
  - `latest`：从最新消息开始（不补历史）
  - `earliest`：从最早消息开始（补历史）
  - `none`：没有已提交 offset 就报错
- `HEADCOUNT_MODEL_PATH` 模型路径，默认 `/app/yolo11n.pt`
- `HEADCOUNT_CONF` 置信度阈值，默认 `0.5`
- `HEADCOUNT_PERSON_CLASS_NAME` 人员类别名称，默认 `person`
- `HEADCOUNT_PERSON_CLASS_ID` 人员类别索引，默认 `0`
- `HEADCOUNT_TIME_START` 报警开始时间（HH:MM），默认 `00:00`
- `HEADCOUNT_TIME_END` 报警结束时间（HH:MM），默认 `08:00`
- `MODEL_DEVICE` 设备（cpu / cuda / mps），默认空
- `MODEL_GPU` GPU 序号（仅当 `MODEL_DEVICE=cuda` 时生效），默认 `0`
- `DEBUG_MODE` 开启调试图输出（有值则生效）
- `DEBUG_DIR` 调试图输出目录

## 说明

- 镜像不内置模型文件，可通过挂载 `yolo11n.pt` 或依赖自动下载。
- 镜像不再内置脚本，请在运行时挂载 `kafuka` 目录。

示例：

```bash
docker run --rm \
  -e KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 \
  -e MODEL_DEVICE=cpu \
  headcount-consumer
```
