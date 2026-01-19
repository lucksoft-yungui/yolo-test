# labcoat-consumer Docker

该镜像用于运行 `kafuka/labcoat/labcoat_consumer.py`。依赖通过 `uv.lock` 同步，所有运行参数可通过环境变量配置。

## 构建镜像

在项目根目录执行：

```bash
docker build -t labcoat-consumer -f kafuka/labcoat/docker/Dockerfile .
```

CUDA 版本：

```bash
docker build -t labcoat-consumer:cuda -f kafuka/labcoat/docker/Dockerfile.cuda .
```

也可以用脚本指定架构和 Dockerfile（构建后直接加载到本地）：

```bash
kafuka/labcoat/docker/build.sh -f kafuka/labcoat/docker/Dockerfile -t labcoat-consumer -p linux/amd64
kafuka/labcoat/docker/build.sh -f kafuka/labcoat/docker/Dockerfile.cuda -t labcoat-consumer:cuda -p linux/amd64
kafuka/labcoat/docker/build.sh -f kafuka/labcoat/docker/Dockerfile -n labcoat-consumer --tag-name v1-amd64 -p linux/amd64
```

## 运行示例

```bash
docker run --name labcoat \
  -e KAFKA_BOOTSTRAP_SERVERS=host.docker.internal:9092 \
  -e KAFKA_TOPIC=ppe_alarm \
  -e KAFKA_ALARM_TOPIC=ppe_alarm_result \
  -e KAFKA_GROUP_ID=ppe-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e MODEL_DEVICE=cpu \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/model:/app/model:ro \
  -v /Users/peiyandong/Documents/code/ai/yolo-test:/Users/peiyandong/Documents/code/ai/yolo-test:ro \
  labcoat-consumer

  docker run --name labcoat \
  --network host \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KAFKA_TOPIC=ppe_alarm \
  -e KAFKA_ALARM_TOPIC=ppe_alarm_result \
  -e KAFKA_GROUP_ID=ppe-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e MODEL_DEVICE=cpu \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/model:/app/model:ro \
  -v /Users/peiyandong/Documents/code/ai/yolo-test:/Users/peiyandong/Documents/code/ai/yolo-test:ro \
  labcoat-consumer
```

CUDA 运行示例：

```bash
docker run --rm --gpus all \
  -e KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 \
  -e MODEL_DEVICE=cuda \
  -e MODEL_GPU=0 \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/model:/app/model:ro \
  -v /Users/peiyandong/Documents/code/ai/yolo-test:/Users/peiyandong/Documents/code/ai/yolo-test:ro \
  labcoat-consumer:cuda
```

如果 Kafka 在同一台宿主机：
- macOS/Windows 用 `host.docker.internal:9092`
- Linux 可改为 `--network host` 或填写宿主机实际地址

## 参数说明

- `KAFKA_BOOTSTRAP_SERVERS` Kafka 地址，默认 `localhost:9092`
- `KAFKA_TOPIC` 来源主题（消费主题），默认 `ppe_alarm`
- `KAFKA_ALARM_TOPIC` 报警推送主题，默认 `ppe_alarm_result`
- `KAFKA_GROUP_ID` 消费者组，默认 `ppe-alarm-consumer`
- `KAFKA_BATCH_SIZE` 批次大小，默认 `10`
- `KAFKA_MAX_WAIT_SEC` 等待凑满批次时间（秒），默认 `2`
- `KAFKA_MAX_BATCHES` 最大批次（0 不限制），默认 `0`
- `KAFKA_AUTO_OFFSET_RESET` 起始偏移策略，默认 `latest`
  - `latest`：从最新消息开始（不补历史）
  - `earliest`：从最早消息开始（补历史）
  - `none`：没有已提交 offset 就报错
- `LABCOAT_MODEL_PATH` 实验服模型路径，默认 `/app/model/labcoat/best.pt`
- `GLOVE_MODEL_PATH` 手套模型路径，默认 `/app/model/glove/best.pt`
- `LABCOAT_YAML` 实验服数据集 yaml，默认 `/app/labcoat.yaml`
- `GLOVE_YAML` 手套数据集 yaml，默认 `/app/glove.yaml`
- `LABCOAT_CLASS_NAME` 未穿实验服类别名称，默认 `no labcoat`
- `GLOVE_CLASS_NAME` 戴手套类别名称，默认 `with glove`
- `LABCOAT_CLASS_ID` 未穿实验服类别索引（可选）
- `GLOVE_CLASS_ID` 戴手套类别索引（可选）
- `LABCOAT_CONF` 实验服模型置信度阈值，默认 `0.7`
- `GLOVE_CONF` 手套模型置信度阈值，默认 `0.7`
- `MODEL_DEVICE` 设备（cpu / cuda / mps），默认空
- `MODEL_GPU` GPU 序号（仅当 `MODEL_DEVICE=cuda` 时生效），默认 `0`
- `DEBUG_MODE` 开启调试图输出（有值则生效）
- `DEBUG_DIR` 调试图输出目录

## 说明

- 镜像内已包含 `labcoat.yaml` 与 `glove.yaml`，无需额外挂载。
- 镜像不再内置脚本与模型，请在运行时挂载 `kafuka` 与 `model` 目录，并设置相应路径。

示例：

```bash
docker run --rm \
  -e KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 \
  -e MODEL_DEVICE=cpu \
  labcoat-consumer
```
