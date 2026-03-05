# store/rtsp_play.py 运行说明（uv）

本程序是一个基于 `PySide6 + OpenCV + Ultralytics YOLO` 的仓储出入库识别 UI，输入 RTSP 视频流并在界面内进行识别、清单编辑与白名单配置。

## 1. 环境准备

在项目根目录执行：

```bash
uv sync
```

说明：
- 会按项目 `pyproject.toml` 和 `uv.lock` 安装依赖。
- 默认使用项目 `.python-version` 指定的 Python 版本。

## 2. 启动程序

在项目根目录执行：

```bash
uv run python store/rtsp_play.py \
  --url "rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1"
```

最小示例（使用默认模型 `yolo11n.pt`）：

```bash
uv run python store/rtsp_play.py --url "rtsp://admin:password@ip:554/Streaming/Channels/1"
```

## 3. 常用启动参数

```bash
uv run python store/rtsp_play.py \
  --model model/fire-store-cls11/weights/best.pt \
  --url "rtsp://admin:password@ip:554/Streaming/Channels/1" \
  --conf 0.5 \
  --reconnect 5 \
  --wait 2 \
  --clear-hold 1.5 \
  --display-width 960 \
  --ui-interval 120 \
  --whitelist-threshold 0.6 \
  --debug
```

参数说明：
- `--model`：模型权重路径或权重名（默认 `yolo11n.pt`）
- `--url`：RTSP 地址
- `--conf`：全局置信度阈值
- `--reconnect`：读流失败重连次数，`0` 表示不重试
- `--wait`：每次重连前等待秒数
- `--clear-hold`：点击“入库/出库”后，等待画面清空的时间（秒）
- `--display-width`：UI 视频显示宽度，`0` 表示不缩放
- `--ui-interval`：UI 刷新间隔（毫秒）
- `--whitelist-threshold`：白名单默认阈值
- `--debug`：打印调试日志

## 4. 界面使用要点

- `出入库` 页签：
  - 自动识别结果会形成清单。
  - 可手动“添加/更新”“删除”清单项。
  - 点击“入库/出库”后会进入锁定，直到画面清空一段时间再恢复。
- `配置` 页签：
  - 可切换模型文件与数据集 YAML（用于类别名）。
  - 可选择识别模式：
    - `实时渲染清单`（realtime）
    - `发现一次加入清单`（sticky）
  - 可配置白名单（类别、别名、阈值、启用状态）。

## 5. 配置文件保存位置

程序会自动读写：

```text
.config/warehouse_whitelist.json
```

其中保存：
- `model_path`
- `dataset_path`
- `detect_mode`
- `whitelist`

## 6. 常见问题

- 模型路径不存在：
  - 程序会提示并尝试把 `--model` 当作权重标识直接加载。
- RTSP 无法打开：
  - 检查地址、账号密码、网络连通性与摄像头通道。
  - 增大 `--reconnect` 与 `--wait` 以提高容错。
- 类别名显示不符合预期：
  - 在“配置”中设置正确的数据集 YAML（含 `names` 字段）并保存。
