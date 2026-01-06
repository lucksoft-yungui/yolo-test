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
- 基本示例：`python train.py --data labcoat --epochs 5 --imgsz 640`
- 断点恢复：`--resume` 自动寻找最近的 `last.pt`，或用 `--checkpoint path/to/last.pt` 指定。
- 硬件检测：自动检测 MPS/CUDA/CPU 并提示当前训练设备。

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
python val/video-val.py \
  --model model/fire-store/weights/best.pt \
  --video videos/shoe.mp4
```

可选参数：
- `--conf` 置信度阈值，默认 0.6
- `--device` 指定设备（如 `cpu` / `cuda` / `mps`）
