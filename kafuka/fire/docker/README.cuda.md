# Fire Alarm Consumer Docker (CUDA)

该镜像用于在 NVIDIA GPU 环境中运行 `kafuka/base/alarm_consumer.py`。适用于需要 CUDA 推理的服务器。

## 前置条件

- NVIDIA GPU 驱动已安装
- 安装了 `nvidia-container-toolkit`

## 构建镜像

```bash
docker build -t alarm-consumer:cuda -f kafuka/base/docker/Dockerfile.cuda .
```

## 运行示例

```bash
docker run --restart=always --gpus all \
  --name firec \
  --network host \
  -e KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 \
  -e KAFKA_TOPIC=fire-alarm \
  -e KAFKA_ALARM_TOPIC=fire-alarm-result \
  -e KAFKA_GROUP_ID=fire-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=0 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e MODEL_PATH=/app/model/fire-kaggle/weights/best.pt \
  -e MODEL_DEVICE=cuda \
  -e MODEL_GPU=0 \
  -e MODEL_CONF=0.6 \
  -e MODEL_IMGSZ=1920 \
  -e TARGET_CLASS_NAME=fire,smoke \
  -e TARGET_CLASS_INDEX=0,1 \
  -v /mnt/nfs/datasets:/mnt/nfs/datasets:ro \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/model:/app/model:ro \
  alarm-consumer:cuda

docker run --restart=always --gpus all \
  --name luckyun-fire-c \
  --network host \
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
  -e MODEL_IMGSZ=1920 \
  -e TARGET_CLASS_NAME=fire,smoke \
  -e TARGET_CLASS_INDEX=0,1 \
  -e MODEL_PATH=/app/model/fire-kaggle/weights/best.pt \
  -v /mnt/nfs/datasets:/mnt/nfs/datasets:ro \
  -v $(pwd)/kafuka:/app/kafuka:ro \
  -v $(pwd)/model:/app/model:ro \
  alarm-consumer:cuda
```

## 环境变量

参数与 CPU 版本一致，见 `kafuka/base/docker/README.md`。
默认值：
- `KAFKA_TOPIC=fire-alarm`
- `MODEL_DEVICE=cuda`
- `MODEL_GPU=0`
- `MODEL_CONF=0.6`
- `MODEL_IMGSZ=0`
- `TARGET_CLASS_NAME=fire`
- `TARGET_CLASS_INDEX=0`
- `MODEL_PATH=/app/model/fire-kaggle/weights/best.pt`

## 说明

- CUDA 版本为 `12.3.2`，基于 `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`。
- 镜像不再内置脚本与模型，请在运行时挂载 `kafuka` 与 `model` 目录，并设置 `MODEL_PATH`。
