# face-consumer Docker

该镜像用于运行 `face_consumer.py`。依赖通过 `uv.lock` 同步，所有运行参数可通过环境变量配置。

## 构建镜像

在项目根目录执行：

```bash
docker build -t face-consumer -f kafuka/face/docker/Dockerfile kafuka/face
```

CUDA 版本：

```bash
docker build -t face-consumer:cuda -f kafuka/face/docker/Dockerfile.cuda kafuka/face
```

也可以用脚本指定架构和 Dockerfile（构建后直接加载到本地）：

```bash

./kafuka/face/docker/build.sh -f kafuka/face/docker/Dockerfile -n face-consumer --tag-name v1-arm64 -p linux/arm64 -C kafuka/face \
  -a http_proxy=http://host.docker.internal:7890 \
  -a https_proxy=http://host.docker.internal:7890 \
  -a all_proxy=socks5://host.docker.internal:7890

./kafuka/face/docker/build.sh -f kafuka/face/docker/Dockerfile.cuda -n face-consumer --tag-name v1-amd64 -p linux/amd64 -C kafuka/face \
  -a http_proxy=http://host.docker.internal:7890 \
  -a https_proxy=http://host.docker.internal:7890 \
  -a all_proxy=socks5://host.docker.internal:7890
```

说明：
- 使用 buildx 的 docker-container 驱动时不支持 `--add-host ...:host-gateway`
- macOS/Windows 通常可直接使用 `host.docker.internal`
- Linux 可改为宿主机实际 IP 或切换到本地 docker driver 再使用 `--add-host`

## 运行示例

```bash
docker run --name face-consumer \
  -e KAFKA_BOOTSTRAP_SERVERS=host.docker.internal:9092 \
  -e KAFKA_TOPIC=face_recognition_alarm \
  -e KAFKA_ALARM_TOPIC=face_recognition_alarm_result \
  -e KAFKA_GROUP_ID=face-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e FACE_PEOPLE_DIR=/app/people \
  -e FACE_TOLERANCE=0.4 \
  -e FACE_MODEL=cnn \
  -e FACE_YOLO_MODEL=/app/yolo11s.pt \
  -e FACE_YOLO_CONF=0.7 \
  -e FACE_PERSON_CLASS_ID=0 \
  -e DEBUG_MODE=1 \
  -e DEBUG_DIR=/app/debug \
  -v $(pwd)/kafuka/face:/app/face \
  -v $(pwd)/kafuka/face/people:/app/people:ro \
  -v $(pwd)/kafuka/face/yolo11s.pt:/app/yolo11s.pt:ro \
  -v $(pwd)/kafuka/face/debug:/app/debug \
  -v /Users/peiyandong/Documents/code/ai/yolo-test/kafuka/face:/Users/peiyandong/Documents/code/ai/yolo-test/kafuka/face:ro \
  face-consumer:v1-arm64

docker run --name face-consumer \
  --network host \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KAFKA_TOPIC=face_recognition_alarm \
  -e KAFKA_ALARM_TOPIC=face_recognition_alarm_result \
  -e KAFKA_GROUP_ID=face-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e FACE_PEOPLE_DIR=/app/people \
  -e FACE_TOLERANCE=0.4 \
  -e FACE_MODEL=cnn \
  -e FACE_YOLO_MODEL=/app/yolo11s.pt \
  -e FACE_YOLO_CONF=0.7 \
  -e FACE_PERSON_CLASS_ID=0 \
  -e DEBUG_DIR=/app/debug \
  -v $(pwd)/kafuka/face:/app/face \
  -v $(pwd)/kafuka/face/people:/app/people:ro \
  -v $(pwd)/kafuka/face/yolo11s.pt:/app/yolo11s.pt:ro \
  -v $(pwd)/kafuka/face/debug:/app/debug \
  -v /Users/peiyandong/Documents/code/ai/yolo-test/kafuka/face:/Users/peiyandong/Documents/code/ai/yolo-test/kafuka/face:ro \
  face-consumer:v1-arm64
```

CUDA 运行示例：

```bash
docker run --rm --gpus all \
  -e KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 \
  -e FACE_MODEL=cnn \
  -e FACE_GPU=1 \
  -e FACE_YOLO_MODEL=/app/yolo11s.pt \
  -e FACE_YOLO_CONF=0.25 \
  -e DEBUG_DIR=/app/debug \
  -v $(pwd)/kafuka/face/people:/app/people:ro \
  -v $(pwd)/kafuka/face/yolo11s.pt:/app/yolo11s.pt:ro \
  -v $(pwd)/kafuka/face/debug:/app/debug \
  -v /Users/peiyandong/Documents/code/ai/yolo-test/kafuka/face:/Users/peiyandong/Documents/code/ai/yolo-test/kafuka/face:ro \
  face-consumer:cuda
```

如果 Kafka 在同一台宿主机：
- macOS/Windows 用 `host.docker.internal:9092`
- Linux 可改为 `--network host` 或填写宿主机实际地址

## 参数说明

- `KAFKA_BOOTSTRAP_SERVERS` Kafka 地址，默认 `localhost:9092`
- `KAFKA_TOPIC` 来源主题（消费主题），默认 `face_recognition_alarm`
- `KAFKA_ALARM_TOPIC` 报警推送主题，默认 `face_recognition_alarm_result`
- `KAFKA_GROUP_ID` 消费者组，默认 `face-alarm-consumer`
- `KAFKA_BATCH_SIZE` 批次大小，默认 `10`
- `KAFKA_MAX_WAIT_SEC` 等待凑满批次时间（秒），默认 `2`
- `KAFKA_MAX_BATCHES` 最大批次（0 不限制），默认 `0`
- `KAFKA_MAX_POLL_INTERVAL_MS` 最大拉取间隔（毫秒），默认 `60000`
- `KAFKA_SESSION_TIMEOUT_MS` 会话超时时间（毫秒），默认 `30000`
- `KAFKA_HEARTBEAT_INTERVAL_MS` 心跳间隔（毫秒），默认 `10000`
- `KAFKA_AUTO_OFFSET_RESET` 起始偏移策略，默认 `latest`
  - `latest`：从最新消息开始（不补历史）
  - `earliest`：从最早消息开始（补历史）
  - `none`：没有已提交 offset 就报错
- `FACE_PEOPLE_DIR` 人脸库目录，默认 `/app/people`
- `FACE_TOLERANCE` 人脸匹配阈值，默认 `0.6`
- `FACE_MODEL` 人脸检测模型（hog / cnn），默认 `hog`
- `FACE_NO_YOLO` 不使用 YOLO 预检测（为空则启用）
- `FACE_YOLO_MODEL` YOLO 模型路径，默认 `/app/yolo11s.pt`
- `FACE_YOLO_CONF` YOLO 置信度阈值，默认 `0.25`
- `FACE_PERSON_CLASS_ID` 人员类别索引，默认 `0`
- `FACE_NUM_UPSAMPLE` 上采样次数，默认 `0`
- `FACE_GPU` 启用 GPU 批量检测（有值则启用）
- `FACE_BATCH_SIZE` 人脸批量检测大小，默认 `128`
- `DEBUG_MODE` 开启调试图输出（有值则生效）
- `DEBUG_DIR` 调试图输出目录

## 说明

- 镜像不内置模型文件，可通过挂载 `yolo11s.pt` 或依赖自动下载。
- 镜像不内置人脸库，请在运行时挂载 `people/` 目录。
- 挂载代码目录 `-v $(pwd)/kafuka/face:/app/face` 可直接热更新代码，无需重新构建镜像。

示例：

```bash
docker run --rm \
  -e KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 \
  face-consumer
```
