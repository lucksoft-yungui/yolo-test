# 验证脚本说明

本目录包含各类视频验证脚本，用于加载模型并对视频进行检测与可视化。

## 通用视频验证

```bash

uv run python val/video-val.py \
  --model model/fire-store/weights/best.pt \
  --video videos/shoe1.mp4 \
  --conf 0.7

uv run python val/video-val.py \
  --model model/fire-store/weights/best.pt \
  --video videos/helmet.mp4 \
  --conf 0.7
```

可选参数：
- `--conf` 置信度阈值，默认 0.6
- `--device` 指定设备（如 `cpu` / `cuda` / `mps`）
