# kafuka/exposed-skin/test

用于从 `datasets/labcoat` 中筛选“穿实验服 + 戴手套 + 皮肤裸露”的图片，并拷贝到 `kafuka/exposed-skin/test/images`。

## 使用

```bash
uv run python kafuka/exposed-skin/test/find_labcoat_with_glove_and_exposed_skin.py
```

## 常用参数

- `--max-count`：最多拷贝多少张，默认 23
- `--labcoat-conf`：实验服模型阈值，默认 0.7
- `--glove-conf`：手套模型阈值，默认 0.7
- `--exposed-skin-conf`：皮肤裸露模型阈值，默认 0.7
- `--device`：指定推理设备，例如 `cpu` 或 `0`

## 调试模式

```bash
uv run python kafuka/exposed-skin/test/find_labcoat_with_glove_and_exposed_skin.py \
  --debug-dir kafuka/exposed-skin/test/debug \
  --only-image sample.jpg \
  --max-count 1
```

输出文件：

- `kafuka/exposed-skin/test/debug/*_labcoat.jpg`：整图实验服检测框
- `kafuka/exposed-skin/test/debug/*_glove_*.jpg`：裁剪后手套检测框
- `kafuka/exposed-skin/test/debug/*_exposed_skin_*.jpg`：裁剪后皮肤裸露检测框
- `kafuka/exposed-skin/test/image-with-tag/*`：命中的图片带标注
