# 生成验证集（复制方式）

该脚本从训练集复制出一份验证集，保留原始数据集不做移动或删除。

## 使用方式

```bash
uv run python video-gen-datasets/split-val-dataset.py \
  --dataset-yaml datasets/fire-store/fire-store.yaml
```

## 参数说明

- `--dataset-yaml`：数据集 yaml 路径
- `--val-ratio`：验证集比例，默认 0.2
- `--seed`：随机种子，默认 42
- `--update-yaml`：自动更新 yaml 的 `val: images/val`
- 默认会备份原始数据集到 `datasets/bak`
- `--no-backup`：不备份原始数据集

## 输出结构

```
datasets/<dataset>/
  images/train/
  labels/train/
  images/val/
  labels/val/
```

## 说明

脚本会从训练集中随机抽取并复制图片与标签到 `images/val`、`labels/val`。如果标签不存在，会创建空标签文件。
