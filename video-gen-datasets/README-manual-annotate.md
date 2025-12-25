# 手工标注视频生成 YOLO 数据集

该脚本用于从视频中抽帧，手工框选目标并生成 YOLO 标注文件。适用于单类别视频（一个视频里只有一种类别）。

## 使用方式

```bash
python video-gen-datasets/ai-manual-annotate.py \
  --video videos/fire.mp4 \
  --dataset-name fire_manual \
  --output-dir datasets \
  --class-name fire

uv run python video-gen-datasets/ai-manual-annotate.py \
  --video videos/shoe.mp4 \
  --dataset-name fire-store \
  --output-dir datasets \
  --class-name shoes

uv run python video-gen-datasets/ai-manual-annotate.py \
  --video videos/shoe1.mp4 \
  --dataset-name fire-store \
  --output-dir datasets \
  --class-name shoes

uv run python video-gen-datasets/ai-manual-annotate.py \
  --video videos/helmet.mp4 \
  --dataset-name fire-store \
  --output-dir datasets \
  --class-name helmet

uv run python video-gen-datasets/ai-manual-annotate.py \
  --video videos/helmet1.mp4 \
  --dataset-name fire-store \
  --output-dir datasets \
  --class-name helmet

uv run python video-gen-datasets/ai-manual-annotate.py \
  --video videos/radio.mp4 \
  --dataset-name fire-store \
  --output-dir datasets \
  --class-name sos
```

## 主要参数

- `--video`：输入视频路径
- `--dataset-name`：数据集名称
- `--output-dir`：数据集输出目录
- `--class-name`：类别名称（单类别）
- `--interval-sec`：抽帧时间间隔（秒），默认1秒一帧
- `--frame-step`：抽帧步长（帧数），设置后优先于 `--interval-sec`
- `--start-frame`：起始帧序号
- `--max-frames`：最大处理帧数

## 操作说明

- 鼠标拖拽绘制框
- `s` 或 空格：保存当前帧
- `n`：跳过当前帧
- `r` / `c`：清空当前帧标注
- `q` / `ESC`：退出标注

## 输出结构

```
datasets/<dataset-name>/
  images/train/
  labels/train/
  <dataset-name>.yaml
```

如果 `datasets/<dataset-name>` 已存在，会继续写入同一数据集；当 `class-name` 在配置中不存在时，会自动追加到 `names` 并使用新的类别 ID。文件名会自动加上视频名作为前缀，减少多次标注冲突。
