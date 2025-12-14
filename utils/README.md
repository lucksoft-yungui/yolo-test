# AlertQueue 使用说明

提供流式检测的去重告警工具 `AlertQueue`，核心行为：在冷却时间内仅触发一次告警。

## 快速使用
```python
from utils.alert_queue import AlertQueue

queue = AlertQueue(
    model="model/fire/weights/best.pt",
    video="videos/fire.mp4",
    cooldown_seconds=5.0,   # 冷却时间，秒
)

# 检测循环中调用，返回值表示本次是否触发告警
triggered = queue.enqueue(confidence=0.82, threshold=0.5)
if triggered:
    print("触发告警！")
```

## 常用属性与方法
- `last_alert_at`: 上次触发告警的时间戳（`time.monotonic` 基准）。
- `records`: 历史记录元组，包含时间戳、置信度、是否触发。
- `in_cooldown()`: 判断当前是否处于冷却窗口。
- `reset()`: 清空历史并重置冷却状态。

## 可选参数
- `cooldown_seconds`: 冷却窗口长度，默认 5 秒。
- `time_provider`: 自定义时间函数（默认 `time.monotonic`），用于测试或自定义时钟。
- `max_history`: 历史记录最大长度，默认 1000。

---

# 告警推送（Alert Push）说明

`utils/alert_push.py` 提供通用方法 `push_alert`，用于在触发告警时：
- 截取“事件帧前后各 N 秒”的视频片段并保存为网页可播放的 H.264 MP4；
- 保存事件帧封面图（JPG）；
- 调用报警推送接口 `POST /alarm/receive`（JSON）。

默认异步处理，避免阻塞检测循环。

## 依赖
- 必需：系统已安装 `ffmpeg`（用于输出 H.264，网页 `video` 标签兼容性更好）
  - macOS：`brew install ffmpeg`
- Python 依赖：`opencv-python(-headless)` 用于读帧、写截图
- 可选：`ultralytics`（当开启“视频带框输出”时会用到 YOLO 推理）

## 输出位置
- 告警视频：`runs/alerts/alert_YYYYmmdd_HHMMSS.mp4`
- 告警封面：`runs/alerts/alert_YYYYmmdd_HHMMSS.jpg`

## 推送接口参数
推送 JSON 结构如下（字段名固定）：
- `fileVideoPath`: 告警视频绝对路径
- `fileCoverPath`: 封面绝对路径
- `zoneId`: 防区 ID
- `deviceId`: 设备 ID
- `zoneTypeNo`: 防区类型编码

## 快速使用（仅截取+推送）
```python
from utils.alert_push import push_alert

push_alert(
    video_path="videos/fire.mp4",
    frame=frame,                 # 当前帧（numpy.ndarray）
    event_msec=event_msec,       # 当前播放进度（毫秒）
    context_sec=10.0,            # 前后各 10 秒
    device_id="D0001",
    zone_id="Z0001",
    zone_type_no="fire_alarm",
)
```

## 视频带框输出（重新推理并绘制框）
在需要“告警视频包含识别框”时，传入模型路径与类别名：
```python
push_alert(
    video_path=args.video,
    frame=frame,
    event_msec=cap.get(cv2.CAP_PROP_POS_MSEC),
    context_sec=10.0,
    device_id="...",
    zone_id="...",
    zone_type_no="...",
    annotate_model_path=args.model,
    annotate_conf=args.conf,
    annotate_class_names=names,
    annotate_device=args.device,   # cpu/cuda/mps
)
```

### 处理流程说明
启用 `annotate_model_path` 后，告警视频的生成流程是：
1. 以事件时间点为中心，截取前后各 `context_sec` 秒的时间窗口；
2. 读取该时间窗口内的每一帧（或按 `annotate_every_n` 跳帧）；
3. 对需要推理的帧调用 YOLO 推理获取检测框；
4. 将检测框绘制到视频帧上（其余跳过推理的帧复用上一帧推理结果进行绘制）；
5. 将带框帧序列编码为 H.264 MP4 输出（`ffmpeg`）。

未传 `annotate_model_path` 时，不会在截取过程中调用模型，只做“截取 + 编码 + 推送”。

## 性能调优（推荐）
“带框输出”会对截取片段逐帧推理，主要耗时在 YOLO 推理（不是画框）。可用以下参数显著提速：
- `annotate_every_n`: 每 N 帧才做一次推理，其余帧复用上一帧结果画框（推荐先用 5 或 10）
- `annotate_imgsz`: 降低推理输入尺寸（如 640/480/416），速度更快但精度略降

## 测试用固定路径（不依赖本次截取结果）
当你只想验证接口入参，可覆写推送的文件路径：
```python
push_alert(
    video_path=args.video,
    frame=frame,
    event_msec=0,
    video_override="/abs/path/to/test.mp4",
    cover_override="/abs/path/to/test.jpg",
)
```

## 诊断耗时
开启 `debug_timing=True` 会打印一行 `[告警耗时] ...`，包含：
- `cut+encode`: 截取/编码耗时
- `infer/draw`: 推理与画框耗时
- `infer_frames/skipped_infer_frames/every_n`: 跳帧推理统计
