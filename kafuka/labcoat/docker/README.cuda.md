# labcoat-consumer CUDA

该镜像用于在 NVIDIA GPU 环境中运行 `kafuka/labcoat/labcoat_consumer.py`。

## 构建镜像

```bash
docker build -t labcoat-consumer:cuda -f kafuka/labcoat/docker/Dockerfile.cuda .
```

## 运行示例

```bash
docker run --rm --gpus all \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/model:/app/model:ro \
  -v /Users/peiyandong/Documents/code/ai/yolo-test:/Users/peiyandong/Documents/code/ai/yolo-test:ro \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KAFKA_TOPIC=ppe_alarm \
  -e KAFKA_GROUP_ID=ppe-alarm-consumer \
  -e KAFKA_ALARM_TOPIC=ppe_alarm_result \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_MAX_BATCHES=0 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e LABCOAT_MODEL_PATH=/app/model/labcoat/best.pt \
  -e GLOVE_MODEL_PATH=/app/model/glove/best.pt \
  -e LABCOAT_YAML=/app/labcoat.yaml \
  -e GLOVE_YAML=/app/glove.yaml \
  -e LABCOAT_CLASS_NAME="no labcoat" \
  -e GLOVE_CLASS_NAME="with glove" \
  -e LABCOAT_CONF=0.7 \
  -e GLOVE_CONF=0.7 \
  -e MODEL_DEVICE=cuda \
  -e MODEL_GPU=0 \
  -e DEBUG_MODE= \
  -e DEBUG_DIR=/app/kafuka/labcoat/debug \
  labcoat-consumer:cuda
```

参数与 CPU 版本一致，见 `kafuka/labcoat/docker/README.md`。
