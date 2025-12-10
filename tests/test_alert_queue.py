from pathlib import Path
import unittest

from utils.alert_queue import AlertQueue


class FakeTime:
    def __init__(self, start: float = 0.0) -> None:
        self.current = start

    def now(self) -> float:
        # 模拟时间源，替代 time.monotonic
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


class AlertQueueTests(unittest.TestCase):
    def test_first_high_confidence_triggers(self) -> None:
        clock = FakeTime()
        queue = AlertQueue("model.pt", "video.mp4", cooldown_seconds=3.0, time_provider=clock.now)

        # 第一次达到阈值直接触发
        triggered = queue.enqueue(confidence=0.9, threshold=0.5)

        self.assertTrue(triggered)
        self.assertEqual(queue.last_alert_at, 0.0)
        self.assertEqual(queue.records[-1].confidence, 0.9)
        self.assertTrue(queue.records[-1].triggered)
        self.assertEqual(queue.model_path, Path("model.pt"))
        self.assertEqual(queue.video_source, Path("video.mp4"))

    def test_cooldown_suppresses_repeated_alerts(self) -> None:
        clock = FakeTime()
        queue = AlertQueue("m.pt", "v.mp4", cooldown_seconds=10.0, time_provider=clock.now)

        # 冷却窗口内的高置信度不会重复触发
        self.assertTrue(queue.enqueue(confidence=0.8, threshold=0.5))
        clock.advance(3.0)
        self.assertFalse(queue.enqueue(confidence=0.9, threshold=0.5))

        triggered_count = len([r for r in queue.records if r.triggered])
        self.assertEqual(triggered_count, 1)
        self.assertTrue(queue.in_cooldown())

    def test_retrigger_after_cooldown(self) -> None:
        clock = FakeTime()
        queue = AlertQueue("m.pt", "v.mp4", cooldown_seconds=5.0, time_provider=clock.now)

        # 冷却过后再次触发
        queue.enqueue(confidence=0.7, threshold=0.5)
        clock.advance(5.1)
        triggered = queue.enqueue(confidence=0.95, threshold=0.5)

        self.assertTrue(triggered)
        self.assertAlmostEqual(queue.last_alert_at, clock.current)
        triggered_count = len([r for r in queue.records if r.triggered])
        self.assertEqual(triggered_count, 2)
        self.assertTrue(queue.in_cooldown())
        clock.advance(5.0)
        self.assertFalse(queue.in_cooldown())

    def test_low_confidence_never_triggers(self) -> None:
        clock = FakeTime()
        queue = AlertQueue("m.pt", "v.mp4", cooldown_seconds=2.0, time_provider=clock.now)

        # 未达阈值的检测不会触发且不进入冷却
        queue.enqueue(confidence=0.2, threshold=0.5)
        clock.advance(1.0)
        queue.enqueue(confidence=0.4, threshold=0.5)

        self.assertTrue(all(not r.triggered for r in queue.records))
        self.assertIsNone(queue.last_alert_at)
        self.assertFalse(queue.in_cooldown())


if __name__ == "__main__":
    unittest.main()
