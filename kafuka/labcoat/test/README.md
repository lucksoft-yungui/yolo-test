# kafuka/labcoat/test

用于从 `datasets/labcoat` 中筛选“未穿防护服且戴手套”的图片，并拷贝到 `kafuka/labcoat/test/images`。

## 使用

```bash
uv run python kafuka/labcoat/test/find_no_labcoat_with_glove.py
```

## 常用参数

- `--max-count`：最多拷贝多少张，默认 23
- `--glove-conf`：手套模型置信度阈值，默认 0.7（可单独设置）
- `--labcoat-conf`：防护服模型置信度阈值，默认 0.7（可单独设置）
- `--device`：指定推理设备，例如 `cpu` 或 `0`

## 调试模式

输出检测框与日志（分类 + 置信度），用于排查误检：

```bash
uv run python kafuka/labcoat/test/find_no_labcoat_with_glove.py \
  --debug-dir kafuka/labcoat/test/debug \
  --debug-target "Neg -no gloves with labcoat_20260116125612852_3em3qf-compressed-20260116233824.jpg" \
  --only-image "Neg -no gloves with labcoat_20260116125612852_3em3qf-compressed-20260116233824.jpg" \
  --max-count 1
```

输出文件：

- `kafuka/labcoat/test/debug/*_labcoat.jpg`：整图防护服检测框
- `kafuka/labcoat/test/debug/*_glove_*.jpg`：裁剪后手套检测框
- `kafuka/labcoat/test/image-with-tag/*`：命中的图片带“未穿实验服 + 戴手套”标注
