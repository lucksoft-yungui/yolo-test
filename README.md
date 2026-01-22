# yolo-test

## 环境设置 (Environment Setup)

本项目使用 [uv](https://github.com/astral-sh/uv) 进行包管理。

### 1. 安装 uv
如果你的机器上还没有安装 uv，请运行：

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 恢复环境
克隆代码或 `git pull` 后，在项目根目录运行以下命令即可一键安装 Python 版本及所有依赖：

```bash
uv sync
```

该命令会自动：
- 下载并安装所需的 Python 版本（由 `.python-version` 指定）。
- 创建 `.venv` 虚拟环境。
- 安装所有锁定在 `uv.lock` 中的依赖。

### 3. 运行代码
确保使用虚拟环境中的 Python：

```bash
uv run python main.py
# 或者激活环境后运行
source .venv/bin/activate
python main.py
```

## 训练使用说明
- 数据集配置：`train.py --data <name_or_yaml>`，可传数据集名（如 `labcoat`）自动匹配 `datasets/labcoat/labcoat.yaml`，也可直接传 yaml 路径。
- 也可直接指定配置文件：`train.py --file play-phone-kaggle.yaml`，优先于 `--data`。
- 模型输出：默认写入 `model/<数据集名>/`，可用 `--project` 自定义根目录、`--name` 自定义 run 名。
- 基本示例：`uv run python train.py --data labcoat --epochs 5 --imgsz 640`
- 模型规模：`--model-size yolo11n|yolo11s|yolo11m|yolo11l|yolo11x`（不填默认 yolo11n）
- 断点恢复：`--resume` 自动寻找最近的 `last.pt`，或用 `--checkpoint path/to/last.pt` 指定。
- Early stopping：`--patience 10`（验证集指标 10 个 epoch 无提升则提前停止）。
- 设备选择：`--device cuda:0` / `--device 0` / `--device cpu` / `--device mps`（不填则自动检测）。
- 硬件检测：自动检测 MPS/CUDA/CPU 并提示当前训练设备。

YOLO11 模型规格参考（官方基准，imgsz=640）：

| 模型 | imgsz | mAP50-95 | CPU(ms) | T4(ms) | 参数量(M) | FLOPs(B) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO11n | 640 | 39.5 | 56.1 ± 0.8 | 1.5 ± 0.0 | 2.6 | 6.5 |
| YOLO11s | 640 | 47.0 | 90.0 ± 1.2 | 2.5 ± 0.0 | 9.4 | 21.5 |
| YOLO11m | 640 | 51.5 | 183.2 ± 2.0 | 4.7 ± 0.1 | 20.1 | 68.0 |
| YOLO11l | 640 | 53.4 | 238.6 ± 1.4 | 6.2 ± 0.1 | 25.3 | 86.9 |
| YOLO11x | 640 | 54.7 | 462.8 ± 6.7 | 11.3 ± 0.2 | 56.9 | 194.9 |

## 微调
```
yolo train \
  model=model/fanghufu-clothes-2cls-kaggle/weights/best.pt \
  data=datasets/labcoat-add/labcoat-add.yaml \
  epochs=200 imgsz=640 batch=8 \
  lr0=1e-3 weight_decay=1e-4 patience=0 \
  mosaic=0.0 mixup=0.0 copy_paste=0.0 \
  project=model name=labcoat-add-overfit
```

## 视频验证脚本

通用视频验证脚本：加载模型对视频进行检测并绘制目标框。

```bash
uv run python val/video-val.py \
  --model model/fire-store/weights/best.pt \
  --video videos/shoe.mp4
```

可选参数：
- `--conf` 置信度阈值，默认 0.6
- `--device` 指定设备（如 `cpu` / `cuda` / `mps`）

## RTSP 负样本录制

用于从 RTSP 流录制“无目标”的负样本视频，默认保存到 `videos/`。

```bash
uv run python rtsp_record_negative.py \
  --url "rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1"
```

常用参数：
- `--output videos/neg_xxx.mp4` 指定输出路径
- `--duration 60` 录制 60 秒自动停止（0 表示手动停止）
- `--no-show` 后台录制不弹窗
- `--codec mp4v` / `--fps 25` 设置编码和帧率

## RTSP 录制为 MP4

将 RTSP 流直接录制为 MP4 文件，默认保存到 `videos/`。

```bash
uv run python rtsp_record.py \
  --url "rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1" \
  --output-dir videos

uv run python rtsp_record.py \
  --url "rtsp://admin:abcd1234@10.0.114.31:554/cam/realmonitor?channel=1&subtype=1" \
  --output-dir videos
```

参数：
- `--url` RTSP 流地址
- `--output-dir` 输出目录（默认 `videos`）

## HTTP 流媒体播放

用于播放 HTTP/FLV 等流媒体地址（OpenCV + FFmpeg）。

```bash
uv run python http_play.py \
  --url "http://ai-tim.zju-qz.edu.cn/media0/proxy/ipc-hg7i64qn.live.flv"
```

参数：
- `--url` HTTP 流媒体地址
- `--window` 窗口名称
- `--reconnect` 读取失败时重连次数
- `--wait` 重连前等待秒数

## HTTP 流媒体录制

将 HTTP 流媒体录制为 MP4 文件，默认保存到 `videos/`。

```bash
uv run python http_record.py \
  --url "http://ai-tim.zju-qz.edu.cn/media0/proxy/ipc-hg7i64qn.live.flv" \
  --output-dir videos
```

参数：
- `--url` HTTP 流媒体地址
- `--output-dir` 输出目录（默认 `videos`）

## 训练

```
uv run python train.py --file fire-store.yaml --epochs 50 --imgsz 640 --patience 10

uv run python train.py --file lab.yaml --epochs 50 --imgsz 640 --patience 10
```

可指定基础模型规模，例如：

```
uv run python train.py --file fire-store.yaml --model-size yolo11x --epochs 50 --imgsz 640 --patience 10

uv run python train.py --file labcoat.yaml --model-size yolo11m --epochs 50 --imgsz 640 --patience 10 --device 0

uv run python train.py --file glove.yaml --model-size yolo11s --epochs 50 --imgsz 640 --patience 10 --device 1

uv run python train.py --file fire-lab.yaml --model-size yolo11s --epochs 50 --imgsz 640 --patience 10 --device 1
```
