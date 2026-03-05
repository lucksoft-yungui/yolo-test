from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class RecorderConfig:
    url: str
    duration: float = 30.0
    record_fps: float = 25.0
    output_dir: Path = Path(__file__).resolve().parent / "records"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTSP 视频录制 GUI")
    parser.add_argument(
        "--url",
        default="rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1",
        help="RTSP 流地址",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="录制时长（秒），默认 30",
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=25.0,
        help="录制帧率（每秒写入几帧），默认 25",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "records"),
        help="视频输出目录",
    )
    parser.add_argument(
        "--tick-ms",
        type=int,
        default=30,
        help="界面刷新间隔（毫秒）",
    )
    return parser.parse_args()


def open_capture(url: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, config: RecorderConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("录制设置")
        self.setModal(True)
        self.resize(600, 180)

        self.url_input = QtWidgets.QLineEdit(config.url)
        self.duration_input = QtWidgets.QDoubleSpinBox()
        self.duration_input.setRange(1.0, 24 * 3600.0)
        self.duration_input.setValue(config.duration)
        self.duration_input.setDecimals(1)
        self.duration_input.setSuffix(" s")

        self.fps_input = QtWidgets.QDoubleSpinBox()
        self.fps_input.setRange(0.1, 120.0)
        self.fps_input.setValue(config.record_fps)
        self.fps_input.setDecimals(2)
        self.fps_input.setSuffix(" fps")

        self.output_dir_input = QtWidgets.QLineEdit(str(config.output_dir))
        browse_btn = QtWidgets.QPushButton("浏览...")
        browse_btn.clicked.connect(self._select_output_dir)

        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output_dir_input)
        output_row.addWidget(browse_btn)

        form = QtWidgets.QFormLayout()
        form.addRow("RTSP 地址", self.url_input)
        form.addRow("录制时长", self.duration_input)
        form.addRow("录制帧率", self.fps_input)
        form.addRow("输出目录", output_row)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(btn_box)
        self.setLayout(layout)

    def _select_output_dir(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            self.output_dir_input.text().strip() or str(Path.cwd()),
        )
        if selected:
            self.output_dir_input.setText(selected)

    def build_config(self) -> RecorderConfig:
        return RecorderConfig(
            url=self.url_input.text().strip(),
            duration=float(self.duration_input.value()),
            record_fps=float(self.fps_input.value()),
            output_dir=Path(self.output_dir_input.text().strip()).expanduser(),
        )


class RTSPRecorderWindow(QtWidgets.QWidget):
    def __init__(self, config: RecorderConfig, tick_ms: int = 30) -> None:
        super().__init__()
        self.setWindowTitle("RTSP 视频录制")
        self.resize(1024, 720)

        self.config = config
        self.tick_ms = max(10, tick_ms)

        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self.is_recording = False
        self.start_ts = 0.0
        self.next_write_ts = 0.0
        self.written_frames = 0
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.current_video_path: Optional[Path] = None
        self.last_video_path: Optional[Path] = None
        self.last_frame: Optional[cv2.Mat] = None

        self.video_label = QtWidgets.QLabel("点击“开始”开始录制视频")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setStyleSheet("background:#111;color:#eee;font-size:18px;")

        self.status_label = QtWidgets.QLabel("状态：待开始")
        self.summary_label = QtWidgets.QLabel("已写入：0 帧")

        self.start_btn = QtWidgets.QPushButton("开始")
        self.cancel_btn = QtWidgets.QPushButton("取消")
        self.save_btn = QtWidgets.QPushButton("保存")
        self.settings_btn = QtWidgets.QPushButton("设置")

        self.start_btn.clicked.connect(self.start_recording)
        self.cancel_btn.clicked.connect(self.cancel_recording)
        self.save_btn.clicked.connect(self.save_mp4)
        self.settings_btn.clicked.connect(self.open_settings)
        self.cancel_btn.setVisible(False)
        self.save_btn.setEnabled(False)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.settings_btn)
        btn_row.addStretch()

        info_row = QtWidgets.QHBoxLayout()
        info_row.addWidget(self.status_label)
        info_row.addStretch()
        info_row.addWidget(self.summary_label)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.video_label, stretch=1)
        main_layout.addLayout(info_row)
        main_layout.addLayout(btn_row)
        self.setLayout(main_layout)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(f"状态：{text}")

    def cancel_recording(self) -> None:
        if self.is_recording:
            self._stop_recording("已取消录制")

    def open_settings(self) -> None:
        if self.is_recording:
            QtWidgets.QMessageBox.warning(self, "提示", "录制中不能修改设置。")
            return
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            new_cfg = dialog.build_config()
            if not new_cfg.url:
                QtWidgets.QMessageBox.warning(self, "参数错误", "RTSP 地址不能为空。")
                return
            self.config = new_cfg
            self._set_status("设置已更新")

    def start_recording(self) -> None:
        if self.is_recording:
            return
        if not self.config.url.strip():
            QtWidgets.QMessageBox.warning(self, "参数错误", "RTSP 地址不能为空。")
            return

        self.cap = open_capture(self.config.url)
        if self.cap is None:
            QtWidgets.QMessageBox.critical(self, "连接失败", f"无法连接 RTSP:\n{self.config.url}")
            return

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        video_name = datetime.now().strftime("record_%Y%m%d_%H%M%S.mp4")
        self.current_video_path = self.config.output_dir / video_name
        self.last_video_path = self.current_video_path
        self.video_writer = None
        self.written_frames = 0
        self.summary_label.setText("已写入：0 帧")
        self.save_btn.setEnabled(False)
        self.is_recording = True
        self.start_ts = time.time()
        self.next_write_ts = self.start_ts
        self.start_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.settings_btn.setEnabled(False)
        self._set_status(
            f"录制中（时长 {self.config.duration:.1f}s，帧率 {self.config.record_fps:.2f} fps）"
        )
        self.timer.start(self.tick_ms)

    def _stop_recording(self, reason: str) -> None:
        self.timer.stop()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        was_recording = self.is_recording
        self.is_recording = False
        self.start_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.settings_btn.setEnabled(True)
        if was_recording:
            self._set_status(reason)
            self.summary_label.setText(f"已写入：{self.written_frames} 帧")
            self.save_btn.setEnabled(
                self.written_frames > 0
                and self.last_video_path is not None
                and self.last_video_path.exists()
            )

    def _on_tick(self) -> None:
        if self.cap is None:
            self._stop_recording("未连接到视频流")
            return

        ok, frame = self.cap.read()
        if not ok:
            self._stop_recording("读取帧失败，录制已停止")
            return

        self.last_frame = frame
        now = time.time()
        elapsed = now - self.start_ts
        remain = max(self.config.duration - elapsed, 0.0)

        if now >= self.next_write_ts and self.current_video_path is not None:
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                size = (frame.shape[1], frame.shape[0])
                self.video_writer = cv2.VideoWriter(
                    str(self.current_video_path), fourcc, self.config.record_fps, size
                )
                if not self.video_writer.isOpened():
                    self.video_writer = None
                    self._stop_recording("无法创建 MP4 文件，录制已停止")
                    QtWidgets.QMessageBox.critical(
                        self,
                        "写入失败",
                        f"无法创建 MP4 文件：\n{self.current_video_path}",
                    )
                    return

            self.video_writer.write(frame)
            self.written_frames += 1
            interval = 1.0 / self.config.record_fps
            self.next_write_ts += interval
            while self.next_write_ts <= now:
                self.next_write_ts += interval

        drawn = frame.copy()
        overlay = (
            f"Written: {self.written_frames} | FPS: {self.config.record_fps:.2f} | Remain: {remain:.1f}s"
        )
        cv2.putText(
            drawn,
            overlay,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        self._show_frame(drawn)
        self.summary_label.setText(f"已写入：{self.written_frames} 帧")

        if elapsed >= self.config.duration:
            self._stop_recording("录制时长到达，已自动停止")

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(image)
        scaled = pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def save_mp4(self) -> None:
        if self.is_recording:
            QtWidgets.QMessageBox.warning(self, "提示", "请先等待录制停止后再保存。")
            return
        if self.last_video_path is None or not self.last_video_path.exists():
            QtWidgets.QMessageBox.warning(self, "提示", "没有可保存的录制视频。")
            return
        if self.written_frames <= 0:
            QtWidgets.QMessageBox.warning(self, "提示", "当前会话没有有效视频帧。")
            return

        default_mp4 = self.last_video_path
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存 MP4",
            str(default_mp4),
            "MP4 Files (*.mp4)",
        )
        if not file_path:
            return
        mp4_path = Path(file_path)
        if mp4_path.suffix.lower() != ".mp4":
            mp4_path = mp4_path.with_suffix(".mp4")

        shutil.copy2(self.last_video_path, mp4_path)
        self._set_status(f"已保存 MP4：{mp4_path}")
        QtWidgets.QMessageBox.information(self, "完成", f"MP4 已保存：\n{mp4_path}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._stop_recording("程序已退出")
        super().closeEvent(event)


def main() -> None:
    args = parse_args()
    cfg = RecorderConfig(
        url=args.url,
        duration=max(1.0, float(args.duration)),
        record_fps=max(0.1, float(args.record_fps)),
        output_dir=Path(args.output_dir).expanduser(),
    )
    app = QtWidgets.QApplication(sys.argv)
    win = RTSPRecorderWindow(cfg, tick_ms=args.tick_ms)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
