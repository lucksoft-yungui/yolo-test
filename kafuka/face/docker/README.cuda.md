# face-consumer CUDA

该镜像用于在 NVIDIA GPU 环境中运行 `face_consumer.py`。

## 构建镜像

```bash
docker build -t face-consumer:cuda -f kafuka/face/docker/Dockerfile.cuda kafuka/face
```

## 运行示例

```bash
docker run --rm --gpus all \
  -v $(pwd):/app/face \
  -v $(pwd)/people:/app/people:ro \
  -v $(pwd)/yolo11s.pt:/app/yolo11s.pt:ro \
  -v $(pwd)/debug:/app/debug \
  -v /Users/peiyandong/Documents/code/ai/yolo-test:/Users/peiyandong/Documents/code/ai/yolo-test:ro \
  -e KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
  -e KAFKA_TOPIC=face_recognition_alarm \
  -e KAFKA_GROUP_ID=face-alarm-consumer \
  -e KAFKA_ALARM_TOPIC=face_recognition_alarm_result \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=2 \
  -e KAFKA_MAX_BATCHES=0 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e FACE_PEOPLE_DIR=/app/people \
  -e FACE_TOLERANCE=0.6 \
  -e FACE_MODEL=cnn \
  -e FACE_YOLO_MODEL=/app/yolo11s.pt \
  -e FACE_YOLO_CONF=0.25 \
  -e FACE_PERSON_CLASS_ID=0 \
  -e FACE_GPU=1 \
  -e FACE_BATCH_SIZE=128 \
  -e DEBUG_MODE= \
  -e DEBUG_DIR=/app/debug \
  face-consumer:cuda
```

参数与 CPU 版本一致，见 `docker/README.md`。
