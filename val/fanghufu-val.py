import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from ultralytics import YOLO
from utils.alert_queue import AlertQueue
from utils.alert_push import push_alert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="播放视频并用防护服模型绘制目标框，对未穿防护服触发告警")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("videos") / "fanghufu.mp4",
        help="源视频路径，默认 videos/fanghufu.mp4",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model/labcoat-add-overfit/weights/best.pt"),
        help="训练好的模型权重，默认 model/labcoat-add-ft/weights/best.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.7,
        help="置信度阈值，默认0.01（可调高降低误检）",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=60.0,
        help="告警冷却时间（秒），在该时间窗口内不重复触发报警，默认5秒",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备，例如 cpu / cuda / mps，不填则自动选择",
    )
    return parser.parse_args()


def load_model(weight_path: Path, device: str | None, class_names: list[str]) -> YOLO:
    if not weight_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {weight_path}")
    model = YOLO(str(weight_path))
    # 覆盖类别名，避免权重中缺少名称时显示数字
    if class_names:
        model.model.names = {i: name for i, name in enumerate(class_names)}
    if device:
        model.to(device)
    return model


def main() -> None:
    args = parse_args()
    names = ["clothes", "no_clothes"]
    model = load_model(args.model, args.device, names)

    alert_queue = AlertQueue(model=args.model, video=args.video, cooldown_seconds=args.alert_cooldown)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件: {args.video}")

    window_name = "Fanghufu Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("按 q 或 ESC 退出播放。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=args.conf, verbose=False)
            annotated = frame.copy()
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{names[cls_id] if cls_id < len(names) else cls_id} {conf:.2f}"
                    color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    # 仅对未穿防护服的检测记录告警
                    if cls_id == 1 and alert_queue.enqueue(confidence=conf, threshold=args.conf):
                        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        print(
                            f"[{ts}] 触发告警（未穿防护服），模型={alert_queue.model_path}, 视频={alert_queue.video_source}, 置信度={conf:.2f}"
                        )
                        try:
                            saved = push_alert(
                                video_path=alert_queue.video_source,
                                frame=frame,
                                event_msec=cap.get(cv2.CAP_PROP_POS_MSEC),
                                context_sec=10.0,
                                device_id="fa6922a824e94a7e9af59c470e99951d",
                                zone_id="7b2b894b-8dcc-4e65-9f8b-210426eafb03",
                                zone_type_no="fanghufu_alarm",
                                annotate_model_path=args.model,
                                annotate_conf=args.conf,
                                annotate_every_n=5,
                                annotate_imgsz=640,
                                annotate_class_names=names,
                                annotate_color_map={0: (0, 255, 0), 1: (0, 0, 255)},
                                annotate_device=args.device,
                                debug_timing=True,
                            )
                            print(f"告警素材保存: 视频={saved['video']}, 帧图={saved['frame']}")
                        except Exception as exc:  # 避免告警推送异常影响检测循环
                            print(f"告警推送失败: {exc}")

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
