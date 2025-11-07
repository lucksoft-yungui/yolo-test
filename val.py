import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 YOLO 对单张图片进行推理并显示检测结果。")
    parser.add_argument(
        "--model",
        default="model/best.pt",
        help="模型文件路径，默认使用训练得到的 best.pt。",
    )
    parser.add_argument(
        "--source",
        default="/Users/peiyandong/Documents/code/ai/yolo-test/train-img/20251013_095449_494527.jpg",
        help="需要检测的图片路径，默认演示 sample。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    image_path = Path(args.source).expanduser()

    if not model_path.is_file():
        raise FileNotFoundError(f"未找到模型文件：{model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"未找到图片文件：{image_path}")

    model = YOLO(str(model_path))
    start_time = time.perf_counter()
    results = model(str(image_path))
    elapsed = time.perf_counter() - start_time

    for idx, result in enumerate(results):
        num_boxes = len(result.boxes) if result.boxes is not None else 0
        print(f"检测结果 #{idx + 1}: {num_boxes} 个目标")

        if num_boxes:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls.int(), result.boxes.conf):
                class_name = result.names[int(cls)]
                x1, y1, x2, y2 = box.tolist()
                print(f" - {class_name}: 左上 ({x1:.1f}, {y1:.1f}) -> 右下 ({x2:.1f}, {y2:.1f}), 置信度 {conf:.2f}")
        else:
            print(" - 未检测到目标。")

        annotated = result.plot()  # 绘制检测框后的图像（BGR）
        window_name = f"result_{idx + 1}"
        cv2.imshow(window_name, annotated)

    print(f"模型推理耗时：{elapsed * 1000:.2f} ms")
    print("按任意键关闭窗口。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
