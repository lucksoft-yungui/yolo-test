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
- `max_history`: 历史记录最大长度，默认 1000。***
