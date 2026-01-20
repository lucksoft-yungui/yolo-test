import argparse
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单图人脸检测调试")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="待检测图片路径",
    )
    parser.add_argument(
        "--people-dir",
        type=Path,
        default=Path("people"),
        help="人脸库目录，默认 people",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hog",
        choices=["hog", "cnn"],
        help="人脸检测模型，默认 hog",
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="不使用 YOLO 预检测（默认启用）",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("yolo11s.pt"),
        help="YOLO 模型路径，默认 yolo11s.pt",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.25,
        help="YOLO 置信度阈值，默认 0.25",
    )
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=0,
        help="人员类别索引，默认 0",
    )
    parser.add_argument(
        "--num-upsample",
        type=int,
        default=0,
        help="上采样次数，默认 0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kafuka/face/debug/face_debug.jpg"),
        help="输出标注图路径",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.6,
        help="人脸匹配阈值，默认 0.6",
    )
    return parser.parse_args()


def load_known_faces(people_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    encodings: list[np.ndarray] = []
    names: list[str] = []
    if not people_dir.exists():
        return encodings, names
    for image_path in sorted(people_dir.glob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = face_recognition.load_image_file(str(image_path))
        face_encodings = face_recognition.face_encodings(image)
        if not face_encodings:
            continue
        encodings.append(face_encodings[0])
        names.append(image_path.stem)
    return encodings, names


def load_yolo_model(weight_path: Path) -> YOLO:
    if not weight_path.exists():
        print(f"YOLO 模型不存在，尝试自动下载: {weight_path}")
    return YOLO(str(weight_path))


def collect_person_crops(
    result: object,
    image_rgb: np.ndarray,
    person_class_id: int,
) -> tuple[list[np.ndarray], list[tuple[int, int, int, float]]]:
    crops: list[np.ndarray] = []
    meta: list[tuple[int, int, int, float]] = []
    person_confs: list[float] = []
    if result.boxes is None:
        return crops, meta
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id != person_class_id:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_rgb.shape[1], x2)
        y2 = min(image_rgb.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_rgb[y1:y2, x1:x2]
        crops.append(np.ascontiguousarray(crop))
        meta.append((x1, y1, 0, conf))
        person_confs.append(conf)
    if person_confs:
        conf_text = ", ".join(f"{value:.3f}" for value in person_confs)
        print(f"人员检测 conf={conf_text}")
    else:
        print("未检测到人员目标")
    return crops, meta


def draw_landmarks(image_bgr: np.ndarray, landmarks: list[dict[str, list[tuple[int, int]]]]) -> None:
    for face_landmarks in landmarks:
        for points in face_landmarks.values():
            for x, y in points:
                cv2.circle(image_bgr, (x, y), 1, (255, 200, 0), -1)


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        print(f"图片不存在: {args.image}")
        return

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        print(f"读取图片失败: {args.image}")
        return
    image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])

    known_encodings, known_names = load_known_faces(args.people_dir)

    use_yolo = not args.no_yolo
    locations: list[tuple[int, int, int, int]] = []
    encodings: list[np.ndarray] = []
    landmarks: list[dict[str, list[tuple[int, int]]]] = []

    if use_yolo:
        yolo_model = load_yolo_model(args.yolo_model)
        yolo_results = yolo_model([image_bgr], conf=args.yolo_conf, verbose=False)
        crops, meta = collect_person_crops(
            yolo_results[0], image_rgb, args.person_class_id
        )
        for crop_idx, crop in enumerate(crops):
            crop_locations = face_recognition.face_locations(
                crop, number_of_times_to_upsample=args.num_upsample, model=args.model
            )
            crop_encodings = face_recognition.face_encodings(
                crop, known_face_locations=crop_locations
            )
            crop_landmarks = face_recognition.face_landmarks(
                crop, face_locations=crop_locations
            )
            offset_x, offset_y, _, _ = meta[crop_idx]
            for face_encoding, location in zip(crop_encodings, crop_locations):
                top, right, bottom, left = location
                locations.append(
                    (
                        top + offset_y,
                        right + offset_x,
                        bottom + offset_y,
                        left + offset_x,
                    )
                )
                encodings.append(face_encoding)
            for lm in crop_landmarks:
                mapped: dict[str, list[tuple[int, int]]] = {}
                for key, points in lm.items():
                    mapped[key] = [(x + offset_x, y + offset_y) for x, y in points]
                landmarks.append(mapped)
    else:
        locations = face_recognition.face_locations(
            image_rgb, number_of_times_to_upsample=args.num_upsample, model=args.model
        )
        encodings = face_recognition.face_encodings(
            image_rgb, known_face_locations=locations
        )
        landmarks = face_recognition.face_landmarks(image_rgb, face_locations=locations)

    print(f"检测到 {len(locations)} 张人脸")
    if not locations:
        print("未检测到人脸，可能需要降低 yolo 阈值或改用 hog/cnn 调整参数。")

    for idx, (location, encoding) in enumerate(zip(locations, encodings), start=1):
        top, right, bottom, left = location
        label = f"face-{idx}"
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_idx = int(np.argmin(distances))
            label = f"{known_names[best_idx]}:{distances[best_idx]:.2f}"
            print(
                f"人脸 {idx}: 最接近 {known_names[best_idx]}，距离 {distances[best_idx]:.4f}"
            )
            print(f"人脸 distance={distances[best_idx]:.4f}")
            if distances[best_idx] > args.tolerance:
                label = f"unknown:{distances[best_idx]:.2f}"
        else:
            print(f"人脸 {idx}: box=({top},{right},{bottom},{left})")

        color = (0, 0, 255)
        cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
        cv2.putText(
            image_bgr,
            label,
            (left, max(0, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    draw_landmarks(image_bgr, landmarks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), image_bgr)
    print(f"已保存标注图: {args.output}")


if __name__ == "__main__":
    main()
