from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque


@dataclass(frozen=True)
class DetectionRecord:
    """单次检测的记录，用于回溯告警触发情况。"""

    timestamp: float
    confidence: float
    triggered: bool


class AlertQueue:
    """
    用于流式检测场景的去重告警队列。

    - 满足阈值的检测会尝试触发一次告警；
    - 在冷却时间窗口内的后续检测会被抑制，避免重复告警。
    """

    def __init__(
        self,
        model: str | Path,
        video: str | Path,
        cooldown_seconds: float = 5.0,
        time_provider: Callable[[], float] | None = None,
        max_history: int = 1000,
    ) -> None:
        self.model_path = Path(model)
        self.video_source = Path(video)
        self.cooldown_seconds = float(cooldown_seconds)
        self._time = time_provider or time.monotonic
        self._records: Deque[DetectionRecord] = deque(maxlen=max_history)
        self._last_alert_at: float | None = None

    @property
    def last_alert_at(self) -> float | None:
        return self._last_alert_at

    @property
    def records(self) -> tuple[DetectionRecord, ...]:
        return tuple(self._records)

    def enqueue(self, confidence: float, threshold: float) -> bool:
        """
        将一次检测结果放入队列，返回本次是否真正触发告警。
        """
        now = self._time()
        triggered = False
        if confidence >= threshold:
            if self._last_alert_at is None or now - self._last_alert_at >= self.cooldown_seconds:
                triggered = True
                self._last_alert_at = now
        self._records.append(DetectionRecord(timestamp=now, confidence=confidence, triggered=triggered))
        return triggered

    def in_cooldown(self) -> bool:
        """检查当前是否仍处于冷却窗口内。"""
        if self._last_alert_at is None:
            return False
        return self._time() - self._last_alert_at < self.cooldown_seconds

    def reset(self) -> None:
        """重置告警状态与历史记录。"""
        self._records.clear()
        self._last_alert_at = None
