# Labcoat-3cls 验证说明

本目录包含实验服三分类模型的图片验证脚本与输出说明。

## 快速开始

```bash
uv run python val/labcoat-3cls/val.py
```

## 参数说明

- `--images` 待检测图片目录，默认 `val/labcoat-3cls/images`
- `--model` 模型权重路径，默认 `model/labcoat-3cls/best.pt`
- `--yaml` 类别配置 YAML，默认 `labcoat-3cls.yaml`
- `--conf` 置信度阈值，默认 `0.6`
- `--device` 指定设备，例如 `cpu` / `cuda` / `mps`
- `--output` 输出 JSON 路径，默认 `val/labcoat-3cls/predictions.json`
- `--save-vis` 保存可视化结果到 `val/labcoat-3cls/annotated`

## 输出说明

控制台会打印每张图片命中的类别列表；同时输出 JSON 文件，结构示例：

```json
{
  "model": "model/labcoat-3cls/best.pt",
  "images_dir": "val/labcoat-3cls/images",
  "class_names": ["with white labcoat", "no labcoat", "with blue labcoat"],
  "summary": {
    "with white labcoat": 12,
    "no labcoat": 3,
    "with blue labcoat": 9
  },
  "results": [
    {
      "image": "val/labcoat-3cls/images/xxx.jpg",
      "classes": ["with white labcoat"],
      "detections": [
        {
          "class_id": 0,
          "class_name": "with white labcoat",
          "confidence": 0.92,
          "box": [12.3, 45.6, 200.1, 320.8]
        }
      ]
    }
  ]
}
```

## 常见问题

- 目录中无图片会报错，请确认 `val/labcoat-3cls/images` 下有图片。
- 如果 `labcoat-3cls.yaml` 中类别名称调整，输出也会自动更新。
