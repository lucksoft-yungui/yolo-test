from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence
import threading
import shutil
import subprocess
import json
from urllib import error, request

import cv2


def _ensure_output_dir() -> Path:
    """确保告警输出目录存在并返回绝对路径。"""
    output_dir = (Path("runs") / "alerts").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _cut_video_segment(
    source_video: Path,
    event_msec: float | None,
    context_sec: float,
    output_path: Path,
    annotate_frame: Callable[[Any], Any] | None = None,
) -> Path:
    """截取事件帧前后各 context_sec 秒的片段并保存，返回保存路径。"""
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频文件用于截取: {source_video}")

    event_msec = 0.0 if event_msec is None else max(0.0, float(event_msec))
    context_msec = max(0.0, context_sec) * 1000.0
    start_msec = max(0.0, event_msec - context_msec)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_msec)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = fps if fps > 1e-3 else 25.0  # 兜底帧率
    end_msec = event_msec + context_msec

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("截取片段失败：无法读取起始帧")

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        writer.write(annotate_frame(frame) if annotate_frame else frame)
        while cap.get(cv2.CAP_PROP_POS_MSEC) < end_msec:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(annotate_frame(frame) if annotate_frame else frame)
    finally:
        writer.release()
        cap.release()

    return output_path


def _save_frame_image(frame: Any, output_path: Path) -> Path:
    """保存当前帧图像，返回保存路径。"""
    if frame is None:
        raise ValueError("帧数据为空，无法保存截图")
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"保存帧图像失败: {output_path}")
    return output_path


def _transcode_to_h264_mp4(input_path: Path, output_path: Path) -> Path:
    """
    使用 ffmpeg 将视频转为 H.264（avc1）编码的 mp4，提升网页 video 标签兼容性。

    说明：OpenCV 写入的 mp4（如 mp4v）在部分浏览器下不可播放，因此统一转码。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法将告警视频转码为 H.264（请先安装 ffmpeg）")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg 转码失败: "
            f"code={proc.returncode}, stdout={proc.stdout.strip()}, stderr={proc.stderr.strip()}"
        )
    return output_path


def _draw_yolo_boxes(
    frame: Any,
    results: Any,
    class_names: Sequence[str] | None,
    color_map: dict[int, tuple[int, int, int]] | None,
) -> Any:
    annotated = frame.copy() if hasattr(frame, "copy") else frame
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if class_names and 0 <= cls_id < len(class_names):
                name = class_names[cls_id]
            else:
                name = str(cls_id)
            label = f"{name} {conf:.2f}"
            color = (0, 0, 255)
            if color_map and cls_id in color_map:
                color = color_map[cls_id]
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
    return annotated


def _mock_push_api(
    model_id: str, device_id: str, zone_id: str, video_path: Path, frame_image: Path
) -> None:
    """模拟报警推送接口。"""
    print(
        f"[模拟推送] 模型={model_id} 设备={device_id} 防区={zone_id} 视频={video_path} 帧图={frame_image}"
    )


def _send_push_api(
    endpoint: str,
    device_id: str,
    zone_id: str,
    zone_type_no: str,
    video_path: Path,
    frame_image: Path,
    video_override: str | Path | None = None,
    cover_override: str | Path | None = None,
) -> None:
    """调用真实报警推送接口。"""
    file_video = Path(video_override) if video_override is not None else video_path
    file_cover = Path(cover_override) if cover_override is not None else frame_image
    payload = {
        "fileVideoPath": str(file_video),
        "fileCoverPath": str(file_cover),
        "zoneId": zone_id,
        "deviceId": device_id,
        "zoneTypeNo": zone_type_no,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"[推送请求] payload={payload}")
    try:
        with request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            print(f"[推送成功] status={resp.status} body={body}")
    except error.HTTPError as exc:
        print(f"[推送失败] HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')}")
    except Exception as exc:  # 捕获网络/解析等其它异常
        print(f"[推送异常] {exc}")


def push_alert(
    video_path: Path,
    frame: Any,
    event_msec: float | None,
    context_sec: float = 10.0,
    device_id: str = "D0001",
    zone_id: str = "Z0001",
    zone_type_no: str = "fire_alarm",
    push_endpoint: str = "http://127.0.0.1:38080/alarm/receive",
    video_override: str | Path | None = None,
    cover_override: str | Path | None = None,
    annotate_model_path: str | Path | None = None,
    annotate_conf: float = 0.25,
    annotate_class_names: Sequence[str] | None = None,
    annotate_color_map: dict[int, tuple[int, int, int]] | None = None,
    annotate_device: str | None = None,
    async_mode: bool = True,
) -> dict[str, Path | threading.Thread]:
    """
    通用告警处理：截取当前事件帧前后各 context_sec 秒的视频片段（默认10秒前+10秒后）、保存帧截图，并调用推送接口。
    默认异步发送，避免阻塞检测循环。

    返回保存的文件路径字典，便于调用方记录。
    """
    output_dir = _ensure_output_dir()
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    segment_raw_path = (output_dir / f"alert_{ts}_raw.mp4").resolve()
    segment_path = (output_dir / f"alert_{ts}.mp4").resolve()
    frame_path = (output_dir / f"alert_{ts}.jpg").resolve()

    frame_copy = frame.copy() if hasattr(frame, "copy") else frame

    def _worker() -> None:
        try:
            annotate_frame = None
            if annotate_model_path is not None:
                from ultralytics import YOLO  # 延迟导入，避免影响其它脚本启动速度

                yolo = YOLO(str(annotate_model_path))
                if annotate_class_names:
                    yolo.model.names = {i: name for i, name in enumerate(annotate_class_names)}
                if annotate_device:
                    yolo.to(annotate_device)

                def annotate_frame(frame_to_annotate: Any) -> Any:
                    results = yolo(frame_to_annotate, conf=annotate_conf, verbose=False)
                    return _draw_yolo_boxes(
                        frame_to_annotate,
                        results,
                        class_names=annotate_class_names,
                        color_map=annotate_color_map,
                    )

            segment_raw = _cut_video_segment(
                video_path,
                event_msec,
                context_sec,
                segment_raw_path,
                annotate_frame=annotate_frame,
            )
            segment = _transcode_to_h264_mp4(segment_raw, segment_path)
            try:
                segment_raw.unlink(missing_ok=True)
            except Exception:
                pass
            image = _save_frame_image(frame_copy, frame_path)

            _send_push_api(
                endpoint=push_endpoint,
                device_id=device_id,
                zone_id=zone_id,
                zone_type_no=zone_type_no,
                video_path=segment,
                frame_image=image,
                video_override=video_override,
                cover_override=cover_override,
            )
        except Exception as exc:  # 捕获并打印，避免线程静默失败
            print(f"[告警推送异常] {exc}")

    if async_mode:
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return {"video": segment_path, "frame": frame_path, "thread": thread}

    _worker()
    return {"video": segment_path, "frame": frame_path}
