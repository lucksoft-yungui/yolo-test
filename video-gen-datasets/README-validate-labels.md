# 检查缺失或空标注

该脚本用于扫描 YOLO 数据集，找出没有 label 文件或 label 为空的图片，并可逐张显示。

## 使用方式

```bash
uv run python video-gen-datasets/validate-empty-labels.py \
  --dataset-yaml datasets/fire-store/fire-store.yaml
```

默认会进入补充标注（每张图片画完框后按数字选择类别ID）：

```bash
uv run python video-gen-datasets/validate-empty-labels.py \
  --dataset-yaml datasets/fire-store/fire-store.yaml \
  --class-name helmet
```

## 参数说明

- `--dataset-yaml`：数据集 yaml 路径
- `--validate-only`：只验证并输出列表，不进入补充标注
- `--class-name`：指定默认类别名称（每张图可回车使用默认，或按数字选择类别ID）
- `--no-show`：只输出列表，不打开窗口显示

## 输出说明

脚本会输出缺失或空标注的图片路径，并在窗口中逐张展示：

- `missing`：没有 label 文件
- `empty`：label 文件存在但内容为空

按键说明：`q` / `ESC` 退出；其他任意键继续下一张。
