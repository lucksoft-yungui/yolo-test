# exposed-skin-consumer Docker

该镜像用于运行 `kafuka/exposed-skin/exposed_skin_consumer.py`。依赖通过 `uv.lock` 同步，所有运行参数可通过环境变量配置。

## 构建镜像

在项目根目录执行：

```bash
docker build -t exposed-skin-consumer -f kafuka/exposed-skin/docker/Dockerfile .
```

CUDA 版本：

```bash
docker build -t exposed-skin-consumer:cuda -f kafuka/exposed-skin/docker/Dockerfile.cuda .
```

也可以用脚本指定架构和 Dockerfile（构建后直接加载到本地）：

```bash
kafuka/exposed-skin/docker/build.sh -f kafuka/exposed-skin/docker/Dockerfile -t exposed-skin-consumer -p linux/amd64
kafuka/exposed-skin/docker/build.sh -f kafuka/exposed-skin/docker/Dockerfile.cuda -t exposed-skin-consumer:cuda -p linux/amd64
kafuka/exposed-skin/docker/build.sh -f kafuka/exposed-skin/docker/Dockerfile.cuda -n exposed-skin-consumer --tag-name v1-amd64 -p linux/amd64
```

## 运行示例

```bash
docker run --name exposed-skin \
  --network host \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KAFKA_TOPIC=exposed-skin-alarm \
  -e KAFKA_ALARM_TOPIC=exposed-skin-alarm-result \
  -e KAFKA_GROUP_ID=exposed-skin-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e LABCOAT_MODEL_PATH=/app/model/labcoat/best.pt \
  -e GLOVE_MODEL_PATH=/app/model/glove/best.pt \
  -e EXPOSED_SKIN_MODEL_PATH=/app/model/exposed-skin/best.pt \
  -e MODEL_DEVICE=cpu \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/model:/app/model:ro \
  exposed-skin-consumer
```

CUDA 运行示例：

```bash
docker run --name exposed-skin-consumer --gpus '"device=1"' \
  --restart=always \
  --network host \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KAFKA_TOPIC=exposed-skin-alarm \
  -e KAFKA_ALARM_TOPIC=exposed-skin-alarm-result \
  -e KAFKA_GROUP_ID=exposed-skin-alarm-consumer \
  -e LABCOAT_MODEL_PATH=/app/model/labcoat/best.pt \
  -e GLOVE_MODEL_PATH=/app/model/glove/best.pt \
  -e EXPOSED_SKIN_MODEL_PATH=/app/model/exposed-skin/best.pt \
  -e LABCOAT_CLASS_NAME='with labcoat' \
  -e GLOVE_CLASS_NAME='with glove' \
  -e EXPOSED_SKIN_CLASS_NAME='exposed-skin' \
  -e LABCOAT_CONF=0.7 \
  -e GLOVE_CONF=0.7 \
  -e EXPOSED_SKIN_CONF=0.7 \
  -e MODEL_DEVICE=cuda \
  -e MODEL_GPU=0 \
  -v /mnt/nfs/code/kafuka:/app/kafuka:ro \
  -v /mnt/nfs/models:/app/model:ro \
  -v /mnt/nfs/collector:/mnt/nfs/collector:ro \
  exposed-skin-consumer:v1-amd64
```

## 参数说明

- `KAFKA_BOOTSTRAP_SERVERS` Kafka 地址，默认 `localhost:9092`
- `KAFKA_TOPIC` 来源主题，默认 `exposed-skin-alarm`
- `KAFKA_ALARM_TOPIC` 报警主题，默认 `exposed-skin-alarm-result`
- `KAFKA_GROUP_ID` 消费者组，默认 `exposed-skin-alarm-consumer`
- `LABCOAT_MODEL_PATH` 实验服模型路径，默认 `/app/model/labcoat/best.pt`
- `GLOVE_MODEL_PATH` 手套模型路径，默认 `/app/model/glove/best.pt`
- `EXPOSED_SKIN_MODEL_PATH` 皮肤裸露模型路径，默认 `/app/model/exposed-skin/best.pt`
- `LABCOAT_CLASS_NAME` 默认 `with labcoat`
- `GLOVE_CLASS_NAME` 默认 `with glove`
- `EXPOSED_SKIN_CLASS_NAME` 默认 `exposed-skin`
- `LABCOAT_CONF` / `GLOVE_CONF` / `EXPOSED_SKIN_CONF`：阈值（默认 0.7）
- `MODEL_DEVICE` 设备（cpu / cuda / mps）
- `MODEL_GPU` GPU 序号（默认 0）
- `DEBUG_MODE` 开启调试图输出（有值则生效）
- `DEBUG_DIR` 调试图输出目录
