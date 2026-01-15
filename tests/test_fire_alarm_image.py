import json
import sys
import unittest
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kafuka.fire_alarm_consumer import decode_messages, load_model


IMAGE_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fire"
    / "32a97bd5-22c2-4719-90a0-d3f88500a8c4_7d7765a8-9330-4bfb-b94c-fbd35e2aff30.jpg"
)
MODEL_PATH = PROJECT_ROOT / "model" / "fire-kaggle" / "weights" / "best.pt"
CONFIDENCE = 0.6
OVERLAY_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fire"
    / "32a97bd5-22c2-4719-90a0-d3f88500a8c4_7d7765a8-9330-4bfb-b94c-fbd35e2aff30_overlay.jpg"
)


class FireAlarmImageTests(unittest.TestCase):
    def test_alarm_image_triggers_fire_box(self) -> None:
        if not IMAGE_PATH.exists():
            self.skipTest(f"图片不存在: {IMAGE_PATH}")
        if not MODEL_PATH.exists():
            self.skipTest(f"模型不存在: {MODEL_PATH}")

        message = json.dumps(
            [
                {
                    "topic": "fire-alarm",
                    "deviceId": "C110-2",
                    "areaId": "C110",
                    "photoPath": str(IMAGE_PATH),
                }
            ]
        )
        entries = decode_messages(message)
        self.assertEqual(len(entries), 1)
        self.assertEqual(Path(entries[0].photo_path), IMAGE_PATH)

        image = cv2.imread(entries[0].photo_path)
        self.assertIsNotNone(image, f"读取图片失败: {IMAGE_PATH}")

        model = load_model(MODEL_PATH, "cpu", ["fire"])
        results = model([image], conf=CONFIDENCE, verbose=False)
        self.assertEqual(len(results), 1)

        fire_boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue
            conf = float(box.conf[0]) if hasattr(box, "conf") else None
            coords = [float(v) for v in box.xyxy[0]]
            fire_boxes.append({"conf": conf, "xyxy": coords})

        print(f"fire boxes count: {len(fire_boxes)}")
        if fire_boxes:
            best = max(fire_boxes, key=lambda item: item["conf"] or 0.0)
            print(f"best fire box conf: {best['conf']}, xyxy: {best['xyxy']}")
            x1, y1, x2, y2 = [int(round(v)) for v in best["xyxy"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"fire {best['conf']:.3f}" if best["conf"] is not None else "fire"
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(OVERLAY_PATH), image)
            print(f"overlay saved: {OVERLAY_PATH}")
        else:
            print("no fire boxes detected")

        self.assertTrue(
            fire_boxes,
            f"未检测到烟火框，阈值={CONFIDENCE}，请检查模型或阈值设置。",
        )


if __name__ == "__main__":
    unittest.main()
