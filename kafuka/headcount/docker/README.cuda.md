# headcount-consumer CUDA

该镜像用于在 NVIDIA GPU 环境中运行 `kafuka/headcount/headcount_consumer.py`。

## 构建镜像

```bash
docker build -t headcount-consumer:cuda -f kafuka/headcount/docker/Dockerfile.cuda .
```

## 运行示例

```bash
docker run --rm --gpus all \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/yolo11n.pt:/app/yolo11n.pt:ro \
  -v /Users/peiyandong/Documents/code/ai/yolo-test:/Users/peiyandong/Documents/code/ai/yolo-test:ro \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KAFKA_TOPIC=unattended_alarm \
  -e KAFKA_GROUP_ID=unattended-alarm-consumer \
  -e KAFKA_ALARM_TOPIC=unattended_alarm_result \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_MAX_BATCHES=0 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e HEADCOUNT_MODEL_PATH=/app/yolo11n.pt \
  -e HEADCOUNT_CONF=0.5 \
  -e HEADCOUNT_TIME_START=00:00 \
  -e HEADCOUNT_TIME_END=08:00 \
  -e MODEL_DEVICE=cuda \
  -e MODEL_GPU=0 \
  -e DEBUG_MODE= \
  -e DEBUG_DIR=/app/kafuka/headcount/debug \
  headcount-consumer:cuda
```

参数与 CPU 版本一致，见 `kafuka/headcount/docker/README.md`。
