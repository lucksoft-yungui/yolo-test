# exposed-skin-consumer CUDA

该镜像用于在 NVIDIA GPU 环境中运行 `kafuka/exposed-skin/exposed_skin_consumer.py`。

## 构建镜像

```bash
docker build -t exposed-skin-consumer:cuda -f kafuka/exposed-skin/docker/Dockerfile.cuda .
```

## 运行示例

```bash
docker run --rm --gpus all \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/model:/app/model:ro \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KAFKA_TOPIC=exposed-skin-alarm \
  -e KAFKA_GROUP_ID=exposed-skin-alarm-consumer \
  -e KAFKA_ALARM_TOPIC=exposed-skin-alarm-result \
  -e LABCOAT_MODEL_PATH=/app/model/labcoat/best.pt \
  -e GLOVE_MODEL_PATH=/app/model/glove/best.pt \
  -e EXPOSED_SKIN_MODEL_PATH=/app/model/exposed-skin/best.pt \
  -e LABCOAT_YAML=/app/labcoat.yaml \
  -e GLOVE_YAML=/app/glove.yaml \
  -e EXPOSED_SKIN_YAML=/app/exposed-skin.yaml \
  -e LABCOAT_CLASS_NAME='with labcoat' \
  -e GLOVE_CLASS_NAME='with glove' \
  -e EXPOSED_SKIN_CLASS_NAME='exposed-skin' \
  -e LABCOAT_CONF=0.7 \
  -e GLOVE_CONF=0.7 \
  -e EXPOSED_SKIN_CONF=0.7 \
  -e MODEL_DEVICE=cuda \
  -e MODEL_GPU=0 \
  -e DEBUG_DIR=/app/kafuka/exposed-skin/debug \
  exposed-skin-consumer:cuda

docker run --rm --gpus all \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/model:/app/model:ro \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KAFKA_TOPIC=exposed-skin-alarm \
  -e KAFKA_GROUP_ID=exposed-skin-alarm-consumer \
  -e KAFKA_ALARM_TOPIC=exposed-skin-alarm-result \
  -e LABCOAT_MODEL_PATH=/app/model/labcoat/best.pt \
  -e GLOVE_MODEL_PATH=/app/model/glove/best.pt \
  -e EXPOSED_SKIN_MODEL_PATH=/app/model/exposed-skin/best.pt \
  -e LABCOAT_YAML=/app/labcoat.yaml \
  -e GLOVE_YAML=/app/glove.yaml \
  -e EXPOSED_SKIN_YAML=/app/exposed-skin.yaml \
  -e LABCOAT_CLASS_NAME='with labcoat' \
  -e GLOVE_CLASS_NAME='with glove' \
  -e EXPOSED_SKIN_CLASS_NAME='exposed-skin' \
  -e LABCOAT_CONF=0.7 \
  -e GLOVE_CONF=0.7 \
  -e EXPOSED_SKIN_CONF=0.7 \
  -e MODEL_DEVICE=cuda \
  -e MODEL_GPU=0 \
  -e DEBUG_DIR=/app/kafuka/exposed-skin/debug \
  -v /mnt/nfs/code/kafuka:/app/kafuka:ro \
  -v /mnt/nfs/models:/app/model:ro \
  -v /mnt/nfs/collector:/mnt/nfs/collector:ro \
  exposed-skin-consumer:cuda
```

参数与 CPU 版本一致，见 `kafuka/exposed-skin/docker/README.md`。
