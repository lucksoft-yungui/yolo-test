# RTSP 视频录制（GUI）

脚本路径：`record/rtsp_record.py`

这是一个图形界面工具，界面提供三个按钮：

- `开始`：开始录制视频（录制中会变成 `取消`）
- `保存`：将当前会话视频保存为 MP4
- `设置`：配置 RTSP 地址、录制时长、录制帧率、输出目录（仅开始前可改）

## uv 环境准备

在项目根目录执行：

```bash
uv sync
```

## 启动方式

```bash
uv run python record/rtsp_record.py
```

也支持启动时传默认参数（可在界面里再改）：

```bash
uv run python record/rtsp_record.py \
  --url "rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1" \
  --duration 30 \
  --record-fps 25
```

## 使用步骤

1. 点击 `设置`，填好 RTSP 地址和参数。
2. 点击 `开始`，显示实时视频并开始录制。
3. 录制中可点击 `取消` 手动停止；或到达录制时长后自动停止。
4. 点击 `保存`，选择 MP4 输出路径并保存。

## 默认值

- 录制时长：`30s`
- 录制帧率：`25 fps`（每秒写入 25 帧，业界常用）
- 输出目录：`record/records`

每次开始录制会在输出目录生成临时会话视频文件（`record_YYYYMMDD_HHMMSS.mp4`），点击 `保存` 后可导出到指定路径。

## 参数说明

- `--url`：默认 RTSP 地址
- `--duration`：默认录制时长（秒）
- `--record-fps`：默认录制帧率（每秒写入几帧）
- `--output-dir`：默认视频输出目录
- `--tick-ms`：界面刷新间隔（毫秒）
