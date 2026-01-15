# Fire Alarm Consumer Docker (CUDA)

该镜像用于在 NVIDIA GPU 环境中运行 `kafuka/fire_alarm_consumer.py`。适用于需要 CUDA 推理的服务器。

## 前置条件

- NVIDIA GPU 驱动已安装
- 安装了 `nvidia-container-toolkit`

## 构建镜像

```bash
docker build -t fire-alarm-consumer:cuda -f docker/fire-alarm/Dockerfile.cuda .
```

## 运行示例

```bash
docker run --rm --gpus all \
  --name firec \
  --network host \
  -e KAFKA_BOOTSTRAP_SERVERS=10.10.6.13:9092 \
  -e KAFKA_TOPIC=fire-alarm \
  -e KAFKA_ALARM_TOPIC=alarm-queue \
  -e KAFKA_GROUP_ID=fire-alarm-consumer \
  -e KAFKA_BATCH_SIZE=10 \
  -e KAFKA_MAX_WAIT_SEC=0 \
  -e KAFKA_AUTO_OFFSET_RESET=latest \
  -e MODEL_DEVICE=cuda \
  -e MODEL_GPU=0 \
  -v /mnt/nfs/datasets:/mnt/nfs/datasets:ro \
  fire-alarm-consumer:cuda
```

## 环境变量

参数与 CPU 版本一致，见 `docker/fire-alarm/README.md`。
默认值：
- `MODEL_DEVICE=cuda`
- `MODEL_GPU=0`

## 说明

- CUDA 版本为 `12.3.2`，基于 `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`。
- 镜像包含模型文件 `model/fire-kaggle/weights/best.pt`。
