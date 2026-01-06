from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Dict, Optional, Tuple

import cv2
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO
import yaml

DEFAULT_MODEL = "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="仓储出入库识别应用")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO 模型权重，默认使用官方 COCO 80 类别模型。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="最小置信度阈值，低于该值的检测将被过滤，且不会低于 0.6。",
    )
    parser.add_argument(
        "--url",
        default="rtsp://admin:luck2024@162.1.1.102:554/Streaming/Channels/1",
        help="RTSP 流地址",
    )
    parser.add_argument(
        "--reconnect",
        type=int,
        default=5,
        help="读取失败时重新连接的尝试次数，0 表示不重试",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=2.0,
        help="重新连接前的等待秒数",
    )
    parser.add_argument(
        "--clear-hold",
        type=float,
        default=1.5,
        help="点击出库/入库后，画面清空持续多少秒才恢复识别",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="UI 展示画面宽度，0 表示不缩放",
    )
    parser.add_argument(
        "--ui-interval",
        type=int,
        default=120,
        help="UI 刷新间隔（毫秒）",
    )
    return parser.parse_args()


def open_capture(url: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


class AsyncAnnotator:
    def __init__(
        self,
        model: YOLO,
        conf_threshold: float,
        whitelist_provider,
    ) -> None:
        self.model = model
        self.conf_threshold = conf_threshold
        self.whitelist_provider = whitelist_provider
        self.frames = Queue(maxsize=1)
        self.latest = None
        self.latest_counts: Dict[str, int] = {}
        self.latest_has_objects = False
        self.running = True
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, frame):
        if not self.running:
            return
        try:
            self.frames.get_nowait()
        except Empty:
            pass
        try:
            self.frames.put_nowait(frame.copy())
        except Full:
            pass

    def get_latest(self) -> Tuple[Optional[cv2.Mat], Dict[str, int], bool]:
        with self.lock:
            annotated = None if self.latest is None else self.latest.copy()
            return annotated, dict(self.latest_counts), self.latest_has_objects

    def stop(self) -> None:
        self.running = False
        try:
            self.frames.put_nowait(None)
        except Full:
            pass
        self.worker.join(timeout=1)

    def _extract_counts(self, results) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if not results:
            return counts
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return counts
        names = result.names or {}
        whitelist = self.whitelist_provider()
        whitelist_enabled = bool(whitelist)
        confs = boxes.conf.tolist()
        classes = boxes.cls.tolist()
        for cls_id, conf in zip(classes, confs):
            class_name = names.get(int(cls_id), str(int(cls_id)))
            entry = whitelist.get(class_name)
            if whitelist_enabled:
                if entry is None or not entry.enabled:
                    continue
            threshold = entry.threshold if entry and entry.threshold is not None else self.conf_threshold
            if conf < threshold:
                continue
            display_name = entry.alias if entry and entry.alias else class_name
            counts[display_name] = counts.get(display_name, 0) + 1
        return counts

    def _loop(self) -> None:
        while self.running:
            try:
                frame = self.frames.get(timeout=0.5)
            except Empty:
                continue
            if frame is None:
                continue
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            annotated = results[0].plot() if results else frame
            counts = self._extract_counts(results)
            with self.lock:
                self.latest = annotated
                self.latest_counts = counts
                self.latest_has_objects = bool(counts)


@dataclass
class SharedState:
    auto_items: Dict[str, int] = field(default_factory=dict)
    manual_items: Dict[str, int] = field(default_factory=dict)
    excluded_items: set[str] = field(default_factory=set)
    whitelist: Dict[str, "WhitelistEntry"] = field(default_factory=dict)
    model_path: str = ""
    dataset_path: str = ""
    class_names: list[str] = field(default_factory=list)
    class_version: int = 0
    reload_token: int = 0
    locked: bool = False
    manual_override: bool = False
    last_action: Optional[str] = None
    no_object_since: Optional[float] = None
    latest_frame: Optional[cv2.Mat] = None
    running: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)
    config_path: Optional[Path] = None

    def whitelist_snapshot(self) -> Dict[str, "WhitelistEntry"]:
        with self.lock:
            return {name: entry.copy() for name, entry in self.whitelist.items()}

    def update_class_names(self, names: list[str]) -> None:
        with self.lock:
            if names == self.class_names:
                return
            self.class_names = list(names)
            self.class_version += 1

    def request_reload(self) -> None:
        with self.lock:
            self.reload_token += 1

    def merged_items(self) -> Dict[str, int]:
        merged = dict(self.auto_items)
        for name in self.excluded_items:
            merged.pop(name, None)
        for name, qty in self.manual_items.items():
            merged[name] = qty
        return merged

    def current_items(self) -> Dict[str, int]:
        with self.lock:
            return self.merged_items()

    def save_config(self) -> None:
        if self.config_path is None:
            return
        with self.lock:
            payload = {
                "model_path": self.model_path,
                "dataset_path": self.dataset_path,
                "whitelist": {
                    name: {
                        "enabled": entry.enabled,
                        "alias": entry.alias,
                        "threshold": entry.threshold,
                    }
                    for name, entry in self.whitelist.items()
                }
            }
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        except OSError:
            pass

    def load_config(self) -> None:
        if self.config_path is None or not self.config_path.exists():
            return
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        model_path = payload.get("model_path", "")
        dataset_path = payload.get("dataset_path", "")
        whitelist = {}
        for name, data in (payload.get("whitelist") or {}).items():
            if not isinstance(data, dict):
                continue
            whitelist[name] = WhitelistEntry(
                enabled=bool(data.get("enabled", True)),
                alias=str(data.get("alias", "")),
                threshold=data.get("threshold", None),
            )
        with self.lock:
            self.whitelist = whitelist
            self.model_path = str(model_path or "")
            self.dataset_path = str(dataset_path or "")


@dataclass
class WhitelistEntry:
    enabled: bool = True
    alias: str = ""
    threshold: Optional[float] = None

    def copy(self) -> "WhitelistEntry":
        return WhitelistEntry(enabled=self.enabled, alias=self.alias, threshold=self.threshold)


def update_inventory_state(
    state: SharedState,
    counts: Dict[str, int],
    has_objects: bool,
    clear_hold_s: float,
) -> None:
    now = time.time()
    with state.lock:
        if state.locked:
            if has_objects:
                state.no_object_since = None
            else:
                if state.no_object_since is None:
                    state.no_object_since = now
                elif now - state.no_object_since >= clear_hold_s:
                    state.locked = False
                    state.manual_override = bool(state.manual_items or state.excluded_items)
                    state.auto_items = {}
                    state.last_action = None
                    state.no_object_since = None
            return

        state.no_object_since = None
        state.auto_items = dict(counts) if counts else {}


def video_loop(
    url: str,
    reconnect: int,
    wait_seconds: float,
    model: YOLO,
    conf_threshold: float,
    clear_hold_s: float,
    state: SharedState,
) -> None:
    attempts = 0
    cap = open_capture(url)
    annotator = AsyncAnnotator(model, conf_threshold, state.whitelist_snapshot)
    with state.lock:
        last_reload_token = state.reload_token

    try:
        while state.running and attempts <= reconnect:
            with state.lock:
                reload_token = state.reload_token
                model_path = state.model_path
                dataset_path = state.dataset_path
            if reload_token != last_reload_token:
                last_reload_token = reload_token
                model_input = model_path or DEFAULT_MODEL
                model_candidate = Path(model_input).expanduser()
                if model_candidate.is_file():
                    model_source = str(model_candidate)
                else:
                    model_source = model_input
                try:
                    new_model = YOLO(model_source)
                except Exception as exc:
                    print(f"模型加载失败：{exc}")
                else:
                    annotator.stop()
                    annotator = AsyncAnnotator(new_model, conf_threshold, state.whitelist_snapshot)
                    class_names = load_dataset_names(dataset_path)
                    if not class_names and hasattr(new_model, "names") and isinstance(new_model.names, dict):
                        class_names = sorted({str(name) for name in new_model.names.values()})
                    state.update_class_names(class_names)

            if cap is None:
                attempts += 1
                if attempts > reconnect:
                    break
                time.sleep(wait_seconds)
                cap = open_capture(url)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.release()
                cap = None
                continue

            annotator.submit(frame)
            annotated, counts, has_objects = annotator.get_latest()
            if annotated is None:
                continue

            with state.lock:
                state.latest_frame = annotated

            update_inventory_state(state, counts, has_objects, clear_hold_s)
    finally:
        if cap is not None:
            cap.release()
        annotator.stop()


def frame_to_qimage(frame: cv2.Mat, target_width: int) -> Optional[QtGui.QImage]:
    if target_width > 0 and frame.shape[1] != target_width:
        scale = target_width / frame.shape[1]
        target_height = max(1, int(frame.shape[0] * scale))
        frame = cv2.resize(frame, (target_width, target_height))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    image = QtGui.QImage(rgb.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
    return image.copy()


def load_dataset_names(dataset_path: str) -> list[str]:
    if not dataset_path:
        return []
    path = Path(dataset_path).expanduser()
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(data, dict):
        return []
    names = data.get("names")
    if isinstance(names, dict):
        return sorted({str(name) for name in names.values()})
    if isinstance(names, list):
        return [str(name) for name in names if name is not None]
    return []


class ItemTable(QtWidgets.QTableWidget):
    def __init__(self, parent_window: "WarehouseWindow"):
        super().__init__(0, 2, parent_window)
        self.parent_window = parent_window

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        text = event.text()
        if text.isdigit():
            self.parent_window.apply_quantity_input(text)
            return
        super().keyPressEvent(event)


class WarehouseWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        state: SharedState,
        ui_interval_ms: int,
        display_width: int,
        class_names: list[str],
    ):
        super().__init__()
        self.state = state
        self.ui_interval_ms = ui_interval_ms
        self.display_width = display_width
        self.rendered_items: Dict[str, int] = {}
        self.rendered_whitelist: Dict[str, Tuple[bool, str, Optional[float]]] = {}
        self.qty_input_buffer = ""
        self.qty_input_ts = 0.0
        self.class_version = -1

        self.setWindowTitle("仓储出入库识别")
        self.setMinimumSize(1200, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        self.video_label = QtWidgets.QLabel("等待视频流...")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background:#111;color:#ddd;")
        layout.addWidget(self.video_label, stretch=3)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(10, 0, 0, 0)
        layout.addWidget(side, stretch=2)

        self.tabs = QtWidgets.QTabWidget()
        side_layout.addWidget(self.tabs, stretch=1)

        main_tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs.addTab(main_tab, "出入库")

        self.status_label = QtWidgets.QLabel("识别状态：等待画面")
        main_layout.addWidget(self.status_label)

        self.table = ItemTable(self)
        self.table.setHorizontalHeaderLabels(["物品", "数量"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.sync_inputs_from_selection)
        main_layout.addWidget(self.table, stretch=1)

        form = QtWidgets.QWidget()
        form_layout = QtWidgets.QGridLayout(form)
        form_layout.setContentsMargins(0, 10, 0, 0)
        main_layout.addWidget(form)

        form_layout.addWidget(QtWidgets.QLabel("物品"), 0, 0)
        self.item_name = QtWidgets.QLineEdit()
        self.item_name.returnPressed.connect(self.add_or_update)
        form_layout.addWidget(self.item_name, 0, 1)

        form_layout.addWidget(QtWidgets.QLabel("数量"), 1, 0)
        self.qty_spin = QtWidgets.QSpinBox()
        self.qty_spin.setRange(1, 999)
        self.qty_spin.setValue(1)
        self.qty_spin.lineEdit().returnPressed.connect(self.add_or_update)
        form_layout.addWidget(self.qty_spin, 1, 1)

        btn_row = QtWidgets.QHBoxLayout()
        form_layout.addLayout(btn_row, 2, 0, 1, 2)
        add_btn = QtWidgets.QPushButton("添加/更新")
        add_btn.clicked.connect(self.add_or_update)
        btn_row.addWidget(add_btn)
        del_btn = QtWidgets.QPushButton("删除")
        del_btn.clicked.connect(self.delete_selected)
        btn_row.addWidget(del_btn)
        resume_btn = QtWidgets.QPushButton("恢复识别")
        resume_btn.clicked.connect(self.resume_auto)
        btn_row.addWidget(resume_btn)

        action_row = QtWidgets.QHBoxLayout()
        form_layout.addLayout(action_row, 3, 0, 1, 2)
        in_btn = QtWidgets.QPushButton("入库")
        in_btn.clicked.connect(lambda: self.apply_action("入库"))
        action_row.addWidget(in_btn)
        out_btn = QtWidgets.QPushButton("出库")
        out_btn.clicked.connect(lambda: self.apply_action("出库"))
        action_row.addWidget(out_btn)

        self.notice_label = QtWidgets.QLabel("")
        self.notice_label.setStyleSheet("color:#444;")
        form_layout.addWidget(self.notice_label, 4, 0, 1, 2)

        config_tab = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(config_tab)
        config_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs.addTab(config_tab, "配置")

        model_group = QtWidgets.QGroupBox("模型配置")
        model_layout = QtWidgets.QGridLayout(model_group)
        config_layout.addWidget(model_group)

        model_layout.addWidget(QtWidgets.QLabel("模型路径"), 0, 0)
        self.model_path_input = QtWidgets.QLineEdit()
        with self.state.lock:
            self.model_path_input.setText(self.state.model_path)
        model_layout.addWidget(self.model_path_input, 0, 1)
        browse_btn = QtWidgets.QPushButton("浏览")
        browse_btn.clicked.connect(self.browse_model_path)
        model_layout.addWidget(browse_btn, 0, 2)

        model_layout.addWidget(QtWidgets.QLabel("数据集配置"), 1, 0)
        self.dataset_path_input = QtWidgets.QLineEdit()
        with self.state.lock:
            self.dataset_path_input.setText(self.state.dataset_path)
        model_layout.addWidget(self.dataset_path_input, 1, 1)
        browse_dataset_btn = QtWidgets.QPushButton("浏览")
        browse_dataset_btn.clicked.connect(self.browse_dataset_path)
        model_layout.addWidget(browse_dataset_btn, 1, 2)

        save_model_btn = QtWidgets.QPushButton("保存模型配置")
        save_model_btn.clicked.connect(self.save_model_path)
        model_layout.addWidget(save_model_btn, 2, 1, 1, 2)

        whitelist_title = QtWidgets.QLabel("类别白名单")
        whitelist_title.setStyleSheet("font-weight:600;")
        config_layout.addWidget(whitelist_title)

        self.whitelist_table = QtWidgets.QTableWidget(0, 4)
        self.whitelist_table.setHorizontalHeaderLabels(["类别", "别名", "阈值", "启用"])
        self.whitelist_table.horizontalHeader().setStretchLastSection(True)
        self.whitelist_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.whitelist_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.whitelist_table.itemSelectionChanged.connect(self.sync_whitelist_inputs)
        config_layout.addWidget(self.whitelist_table, stretch=1)

        whitelist_form = QtWidgets.QWidget()
        whitelist_layout = QtWidgets.QGridLayout(whitelist_form)
        whitelist_layout.setContentsMargins(0, 6, 0, 0)
        config_layout.addWidget(whitelist_form)

        whitelist_layout.addWidget(QtWidgets.QLabel("类别"), 0, 0)
        self.class_combo = QtWidgets.QComboBox()
        self.class_combo.setEditable(True)
        self.class_combo.addItems(class_names)
        whitelist_layout.addWidget(self.class_combo, 0, 1)

        whitelist_layout.addWidget(QtWidgets.QLabel("别名"), 1, 0)
        self.alias_input = QtWidgets.QLineEdit()
        whitelist_layout.addWidget(self.alias_input, 1, 1)

        self.enable_checkbox = QtWidgets.QCheckBox("启用")
        self.enable_checkbox.setChecked(True)
        whitelist_layout.addWidget(self.enable_checkbox, 2, 0, 1, 2)

        self.threshold_checkbox = QtWidgets.QCheckBox("自定义阈值")
        whitelist_layout.addWidget(self.threshold_checkbox, 3, 0)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.6)
        self.threshold_spin.setEnabled(False)
        whitelist_layout.addWidget(self.threshold_spin, 3, 1)
        self.threshold_checkbox.toggled.connect(self.threshold_spin.setEnabled)

        whitelist_btns = QtWidgets.QHBoxLayout()
        whitelist_layout.addLayout(whitelist_btns, 4, 0, 1, 2)
        wl_add_btn = QtWidgets.QPushButton("添加/更新白名单")
        wl_add_btn.clicked.connect(self.add_or_update_whitelist)
        whitelist_btns.addWidget(wl_add_btn)
        wl_del_btn = QtWidgets.QPushButton("删除白名单")
        wl_del_btn.clicked.connect(self.delete_whitelist)
        whitelist_btns.addWidget(wl_del_btn)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.ui_interval_ms)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        with self.state.lock:
            self.state.running = False
        event.accept()

    def add_or_update(self) -> None:
        name = self.item_name.text().strip()
        if not name:
            self.notice_label.setText("请输入物品名称。")
            return
        qty = int(self.qty_spin.value())
        with self.state.lock:
            self.state.manual_items[name] = qty
            self.state.excluded_items.discard(name)
            self.state.manual_override = True
        self.notice_label.setText("已更新清单。")
        self.qty_input_buffer = ""

    def browse_model_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            "模型文件 (*.pt *.onnx *.engine *.bin);;所有文件 (*.*)",
        )
        if path:
            self.model_path_input.setText(path)

    def browse_dataset_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择数据集配置文件",
            "",
            "数据集配置 (*.yaml *.yml);;所有文件 (*.*)",
        )
        if path:
            self.dataset_path_input.setText(path)

    def save_model_path(self) -> None:
        path = self.model_path_input.text().strip()
        dataset_path = self.dataset_path_input.text().strip()
        with self.state.lock:
            self.state.model_path = path
            self.state.dataset_path = dataset_path
        self.state.save_config()
        self.state.request_reload()
        self.notice_label.setText("已保存模型配置，正在重新加载。")

    def sync_inputs_from_selection(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        name_item = self.table.item(row, 0)
        qty_item = self.table.item(row, 1)
        if name_item is None or qty_item is None:
            return
        self.item_name.setText(name_item.text())
        try:
            qty = int(qty_item.text())
        except ValueError:
            qty = 1
        self.qty_spin.setValue(max(1, qty))

    def apply_quantity_input(self, digit_text: str) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        now = time.time()
        if now - self.qty_input_ts > 1.0:
            self.qty_input_buffer = ""
        self.qty_input_ts = now
        self.qty_input_buffer = (self.qty_input_buffer + digit_text).lstrip("0") or "0"
        qty = int(self.qty_input_buffer)
        qty = max(1, qty)
        row = selected[0].row()
        name_item = self.table.item(row, 0)
        if name_item is None:
            return
        name = name_item.text()
        with self.state.lock:
            self.state.manual_items[name] = qty
            self.state.excluded_items.discard(name)
            self.state.manual_override = True
        self.qty_spin.setValue(qty)
        self.notice_label.setText(f"已更新 {name} 数量为 {qty}")
        self.table.setFocus()

    def sync_whitelist_inputs(self) -> None:
        selected = self.whitelist_table.selectionModel().selectedRows()
        if not selected:
            return
        row = selected[0].row()
        name_item = self.whitelist_table.item(row, 0)
        alias_item = self.whitelist_table.item(row, 1)
        threshold_item = self.whitelist_table.item(row, 2)
        enabled_item = self.whitelist_table.item(row, 3)
        if name_item is None:
            return
        self.class_combo.setCurrentText(name_item.text())
        self.alias_input.setText("" if alias_item is None or alias_item.text() == "-" else alias_item.text())
        threshold_text = "" if threshold_item is None else threshold_item.text()
        if threshold_text in ("", "-"):
            self.threshold_checkbox.setChecked(False)
        else:
            self.threshold_checkbox.setChecked(True)
            try:
                self.threshold_spin.setValue(float(threshold_text))
            except ValueError:
                pass
        if enabled_item is not None:
            self.enable_checkbox.setChecked(enabled_item.text() != "否")

    def add_or_update_whitelist(self) -> None:
        class_name = self.class_combo.currentText().strip()
        if not class_name:
            self.notice_label.setText("请输入类别名称。")
            return
        alias = self.alias_input.text().strip()
        threshold = self.threshold_spin.value() if self.threshold_checkbox.isChecked() else None
        entry = WhitelistEntry(
            enabled=self.enable_checkbox.isChecked(),
            alias=alias,
            threshold=threshold,
        )
        with self.state.lock:
            self.state.whitelist[class_name] = entry
        self.state.save_config()
        self.notice_label.setText("已更新白名单配置。")

    def delete_selected(self) -> None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.notice_label.setText("请选择要删除的条目。")
            return
        names = [self.table.item(idx.row(), 0).text() for idx in selected]
        with self.state.lock:
            for name in names:
                self.state.manual_items.pop(name, None)
                self.state.excluded_items.add(name)
            self.state.manual_override = bool(self.state.manual_items or self.state.excluded_items)
        self.notice_label.setText("已删除条目。")

    def delete_whitelist(self) -> None:
        selected = self.whitelist_table.selectionModel().selectedRows()
        if not selected:
            self.notice_label.setText("请选择要删除的白名单条目。")
            return
        class_names = [self.whitelist_table.item(idx.row(), 0).text() for idx in selected]
        with self.state.lock:
            for name in class_names:
                self.state.whitelist.pop(name, None)
        self.state.save_config()
        self.notice_label.setText("已删除白名单条目。")

    def resume_auto(self) -> None:
        with self.state.lock:
            if self.state.locked:
                self.notice_label.setText("已锁定，等待画面清空。")
                return
            self.state.manual_items = {}
            self.state.excluded_items = set()
            self.state.manual_override = False
        self.notice_label.setText("已恢复自动识别。")

    def apply_action(self, action: str) -> None:
        with self.state.lock:
            if not self.state.merged_items():
                self.notice_label.setText("当前无清单可操作。")
                return
            self.state.auto_items = {}
            self.state.manual_items = {}
            self.state.excluded_items = set()
            self.state.locked = True
            self.state.manual_override = True
            self.state.last_action = action
            self.state.no_object_since = None
        self.notice_label.setText(f"已确认{action}，等待画面清空。")

    def update_ui(self) -> None:
        with self.state.lock:
            frame = None if self.state.latest_frame is None else self.state.latest_frame.copy()
            items = self.state.merged_items()
            locked = self.state.locked
            manual_override = self.state.manual_override
            last_action = self.state.last_action
            whitelist = {name: entry.copy() for name, entry in self.state.whitelist.items()}
            class_names = list(self.state.class_names)
            class_version = self.state.class_version

        if frame is not None:
            image = frame_to_qimage(frame, self.display_width)
            if image is not None:
                pixmap = QtGui.QPixmap.fromImage(image)
                self.video_label.setPixmap(pixmap)

        if items != self.rendered_items:
            selected_name = None
            selected_rows = self.table.selectionModel().selectedRows()
            if selected_rows:
                row = selected_rows[0].row()
                name_item = self.table.item(row, 0)
                if name_item is not None:
                    selected_name = name_item.text()
            self.table.setRowCount(0)
            for name, qty in sorted(items.items()):
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
                self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(qty)))
            if selected_name:
                for row in range(self.table.rowCount()):
                    name_item = self.table.item(row, 0)
                    if name_item is not None and name_item.text() == selected_name:
                        self.table.setCurrentCell(row, 0)
                        self.table.selectRow(row)
                        self.table.setFocus()
                        break
            self.rendered_items = items

        whitelist_render = {
            name: (entry.enabled, entry.alias, entry.threshold) for name, entry in whitelist.items()
        }
        if whitelist_render != self.rendered_whitelist:
            selected_name = None
            selected_rows = self.whitelist_table.selectionModel().selectedRows()
            if selected_rows:
                row = selected_rows[0].row()
                name_item = self.whitelist_table.item(row, 0)
                if name_item is not None:
                    selected_name = name_item.text()
            self.whitelist_table.setRowCount(0)
            for name, entry in sorted(whitelist.items()):
                row = self.whitelist_table.rowCount()
                self.whitelist_table.insertRow(row)
                self.whitelist_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
                alias_text = entry.alias if entry.alias else "-"
                self.whitelist_table.setItem(row, 1, QtWidgets.QTableWidgetItem(alias_text))
                threshold_text = "-" if entry.threshold is None else f"{entry.threshold:.2f}"
                self.whitelist_table.setItem(row, 2, QtWidgets.QTableWidgetItem(threshold_text))
                enabled_text = "是" if entry.enabled else "否"
                self.whitelist_table.setItem(row, 3, QtWidgets.QTableWidgetItem(enabled_text))
            if selected_name:
                for row in range(self.whitelist_table.rowCount()):
                    name_item = self.whitelist_table.item(row, 0)
                    if name_item is not None and name_item.text() == selected_name:
                        self.whitelist_table.setCurrentCell(row, 0)
                        self.whitelist_table.selectRow(row)
                        self.whitelist_table.setFocus()
                        break
            self.rendered_whitelist = whitelist_render

        if class_version != self.class_version:
            current_text = self.class_combo.currentText()
            self.class_combo.blockSignals(True)
            self.class_combo.clear()
            self.class_combo.addItems(class_names)
            if current_text:
                self.class_combo.setCurrentText(current_text)
            self.class_combo.blockSignals(False)
            self.class_version = class_version

        if locked:
            action_text = last_action or "入库/出库"
            self.status_label.setText(f"识别状态：已锁定（{action_text}），等待画面清空")
        elif manual_override:
            self.status_label.setText("识别状态：手动编辑中")
        else:
            self.status_label.setText("识别状态：自动识别中")


def main() -> None:
    args = parse_args()

    config_path = Path(".config") / "warehouse_whitelist.json"
    state = SharedState(config_path=config_path)
    state.load_config()

    model_input = args.model
    if model_input == DEFAULT_MODEL and state.model_path:
        model_input = state.model_path

    model_path = Path(model_input).expanduser()
    if model_path.is_file():
        model_source = str(model_path)
    else:
        model_source = model_input
        if len(model_path.parts) > 1:
            print(f"提示：未找到 {model_path}，尝试直接使用权重标识 {model_input}。")
    conf_threshold = max(args.conf, 0.6)
    model = YOLO(model_source)
    class_names = load_dataset_names(state.dataset_path)
    if not class_names and hasattr(model, "names") and isinstance(model.names, dict):
        class_names = sorted({str(name) for name in model.names.values()})
    state.update_class_names(class_names)

    worker = threading.Thread(
        target=video_loop,
        args=(args.url, args.reconnect, args.wait, model, conf_threshold, args.clear_hold, state),
        daemon=True,
    )
    worker.start()

    app = QtWidgets.QApplication(sys.argv)
    window = WarehouseWindow(state, args.ui_interval, args.display_width, class_names)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到中断信号，已退出。")
        sys.exit(0)
