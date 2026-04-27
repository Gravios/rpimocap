#!/usr/bin/env python3
"""
arena_aligner.py — Interactive arena alignment + distortion annotator
=====================================================================
Two-mode GUI tool (PyQt6 >= 6.8) for:

  Tab 1 — Corner annotation
      Click known physical landmarks in both cameras.  Kabsch rigid transform
      maps reconstructed world coordinates → arena space.
      Output: align_points.csv  →  rpimocap-run --align-points

  Tab 2 — Edge tracing
      Click points along straight 3-D box edges in each camera independently.
      Plumb-line optimiser fits k1, k2, k3 radial distortion coefficients.
      Output: edges.csv + calibration_refined.npz

  Bundle adjustment button (bottom bar)
      Joint refinement of K, dist, R, T using corner pixel clicks + edge
      line constraints.  Requires ≥ 4 corners annotated in this session
      (older CSVs without stored pixel coordinates cannot be used).

Usage
-----
    rpimocap-align \\
        --cam0   data/cam0.mp4 \\
        --cam1   data/cam1.mp4 \\
        --calib  calibration.npz \\
        --out    data/align_points.csv

    # Resume a previous session
    rpimocap-align ... --load align_points.csv --load-edges edges.csv

Keyboard shortcuts
------------------
Left / Right          step 1 frame
Shift+Left/Right      step 10 frames
Delete                remove selected table row
Escape                cancel current annotation
Enter                 confirm current item
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Qt import: prefer PyQt6, fall back to PySide6 ───────────────────────────
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QTabWidget,
        QLabel, QPushButton, QSlider, QLineEdit, QRadioButton,
        QButtonGroup, QGroupBox, QTableWidget, QTableWidgetItem,
        QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy,
        QSplitter, QMessageBox, QHeaderView, QAbstractItemView,
        QFrame, QStatusBar,
    )
    from PyQt6.QtCore import Qt, QSize, pyqtSignal as Signal
    from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont
    PYQT = True
except ImportError:
    from PySide6.QtWidgets import (                          # type: ignore
        QApplication, QMainWindow, QWidget, QTabWidget,
        QLabel, QPushButton, QSlider, QLineEdit, QRadioButton,
        QButtonGroup, QGroupBox, QTableWidget, QTableWidgetItem,
        QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy,
        QSplitter, QMessageBox, QHeaderView, QAbstractItemView,
        QFrame, QStatusBar,
    )
    from PySide6.QtCore import Qt, QSize, Signal                # type: ignore
    from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont  # type: ignore
    PYQT = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rpimocap.reconstruction.triangulate import triangulate_dlt
from rpimocap.reconstruction.align import (
    AlignPoint, save_align_csv, load_align_csv, kabsch_align,
    TracedEdge, fit_distortion_plumb_line,
    save_edges_csv, load_edges_csv,
    patch_calibration_distortion,
    refine_calibration_from_arena,
    _edge_line_rmse,
)


# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

PANEL_W = 640
PANEL_H = 480
CROSS_R  = 10
DOT_R    = 4

COL_CAM0  = QColor("#e03030")
COL_CAM1  = QColor("#3060e0")
EDGE_PALETTE = [
    "#e8a020", "#20c080", "#c040c0", "#20b8e0",
    "#e06020", "#80e020", "#e02080", "#4080ff",
    "#ff8040", "#40ffc0", "#c0c020", "#ff40c0",
]


# --------------------------------------------------------------------------- #
#  Frame → QPixmap with overlays                                               #
# --------------------------------------------------------------------------- #

def _bgr_to_qimage(frame: np.ndarray, w: int, h: int) -> QImage:
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    small = np.ascontiguousarray(small)
    return QImage(small.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def _draw_cross(painter: QPainter, x: int, y: int,
                colour: QColor, r: int = CROSS_R) -> None:
    pen = QPen(colour, 2)
    painter.setPen(pen)
    painter.drawLine(x - r, y, x + r, y)
    painter.drawLine(x, y - r, x, y + r)
    painter.drawEllipse(x - 4, y - 4, 8, 8)


def _draw_dot(painter: QPainter, x: int, y: int,
              colour: QColor, r: int = DOT_R) -> None:
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(colour))
    painter.drawEllipse(x - r, y - r, r * 2, r * 2)


# --------------------------------------------------------------------------- #
#  Camera canvas widget                                                        #
# --------------------------------------------------------------------------- #

class CameraCanvas(QLabel):
    """QLabel that displays a video frame and emits pixel-level click signals."""

    clicked = Signal(float, float)   # emits (video_x, video_y) in video pixels

    def __init__(self, cam_id: int, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.setFixedSize(PANEL_W, PANEL_H)
        self.setStyleSheet("background: #1a1a1a;")
        self.setCursor(Qt.CursorShape.CrossCursor)

        # Overlays stored as lists of (x, y, colour_hex) in *display* pixels
        self._crosses: list[tuple[int, int, QColor]] = []
        self._dots:    list[tuple[int, int, QColor]] = []
        self._lines:   list[tuple[int, int, int, int, QColor]] = []

        self._base_pixmap: Optional[QPixmap] = None

        # Scale factors (set when a frame is loaded)
        self.sx = 1.0
        self.sy = 1.0

    # -- public API -------------------------------------------------------

    def set_frame(self, frame: np.ndarray) -> None:
        qi = _bgr_to_qimage(frame, PANEL_W, PANEL_H)
        self._base_pixmap = QPixmap.fromImage(qi)
        self._repaint()

    def clear_overlay(self) -> None:
        self._crosses.clear()
        self._dots.clear()
        self._lines.clear()
        self._repaint()

    def add_cross(self, vx: float, vy: float, colour: QColor) -> None:
        cx, cy = int(vx * self.sx), int(vy * self.sy)
        self._crosses.append((cx, cy, colour))
        self._repaint()

    def add_dot(self, vx: float, vy: float, colour: QColor) -> None:
        cx, cy = int(vx * self.sx), int(vy * self.sy)
        if self._dots:
            px, py, _ = self._dots[-1]
            self._lines.append((px, py, cx, cy, colour))
        self._dots.append((cx, cy, colour))
        self._repaint()

    def clear_dots(self) -> None:
        self._dots.clear()
        self._lines.clear()
        self._repaint()

    # -- internal ---------------------------------------------------------

    def _repaint(self) -> None:
        if self._base_pixmap is None:
            return
        pm = self._base_pixmap.copy()
        p  = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        for x0, y0, x1, y1, col in self._lines:
            p.setPen(QPen(col, 1))
            p.drawLine(x0, y0, x1, y1)
        for cx, cy, col in self._dots:
            _draw_dot(p, cx, cy, col)
        for cx, cy, col in self._crosses:
            _draw_cross(p, cx, cy, col)

        p.end()
        self.setPixmap(pm)

    # -- events -----------------------------------------------------------

    def mousePressEvent(self, event):
        if self._base_pixmap is None:
            return
        vx = event.position().x() / self.sx
        vy = event.position().y() / self.sy
        self.clicked.emit(vx, vy)


# --------------------------------------------------------------------------- #
#  Box preset group                                                            #
# --------------------------------------------------------------------------- #

PRESET_CORNERS = [
    ("BFL", (-1, -1,  0)), ("BFR", (+1, -1,  0)),
    ("BBR", (+1, +1,  0)), ("BBL", (-1, +1,  0)),
    ("TFL", (-1, -1, +1)), ("TFR", (+1, -1, +1)),
    ("TBR", (+1, +1, +1)), ("TBL", (-1, +1, +1)),
]


class BoxPresetWidget(QGroupBox):
    corners_ready = Signal(list)   # list of (label, ax, ay, az)

    def __init__(self, parent=None):
        super().__init__("Box preset", parent)
        lay = QHBoxLayout(self)

        for attr, lbl, default in [
            ("_w", "Width X (mm)", "600"),
            ("_d", "Depth Y (mm)", "400"),
            ("_h", "Height Z (mm)", "350"),
        ]:
            lay.addWidget(QLabel(lbl))
            edit = QLineEdit(default)
            edit.setFixedWidth(60)
            setattr(self, attr, edit)
            lay.addWidget(edit)

        btn = QPushButton("Fill 8 corners →")
        btn.clicked.connect(self._fill)
        lay.addWidget(btn)
        lay.addStretch()

    def _fill(self):
        try:
            W = float(self._w.text())
            D = float(self._d.text())
            H = float(self._h.text())
        except ValueError:
            QMessageBox.critical(self, "Box preset", "Dimensions must be numeric.")
            return
        corners = [(lbl, sx * W / 2, sy * D / 2, sz * H)
                   for lbl, (sx, sy, sz) in PRESET_CORNERS]
        self.corners_ready.emit(corners)


# --------------------------------------------------------------------------- #
#  Corner annotation tab                                                       #
# --------------------------------------------------------------------------- #

class CornerTab(QWidget):
    def __init__(self, app: "ArenaAligner", parent=None):
        super().__init__(parent)
        self._app = app
        self.points: list[AlignPoint] = []
        self._click0: Optional[tuple[float, float]] = None
        self._click1: Optional[tuple[float, float]] = None
        self._rec_xyz: Optional[np.ndarray] = None
        self._preset_queue: list = []
        self._build()

    # -- build UI ---------------------------------------------------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(4)

        # Box preset
        self._preset = BoxPresetWidget()
        self._preset.corners_ready.connect(self._fill_preset)
        root.addWidget(self._preset)

        # Status bar
        self._status = QLabel("Step 1: click the landmark in Camera 0")
        self._status.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self._status.setStyleSheet("color: #303030; padding: 2px 6px;")
        root.addWidget(self._status)

        # Triangulated result
        tri_box = QGroupBox("Triangulated position (mm)")
        tri_lay = QHBoxLayout(tri_box)
        self._tri_lbl = QLabel("—")
        self._tri_lbl.setFont(QFont("Courier", 11))
        tri_lay.addWidget(self._tri_lbl)
        tri_lay.addStretch()
        cancel_btn = QPushButton("Cancel  [Esc]")
        cancel_btn.clicked.connect(self.cancel)
        tri_lay.addWidget(cancel_btn)
        root.addWidget(tri_box)

        # Arena coordinate entry
        entry_box = QGroupBox("Known arena coordinate (mm)")
        entry_lay = QGridLayout(entry_box)
        self._ax = QLineEdit("0"); self._ax.setFixedWidth(70)
        self._ay = QLineEdit("0"); self._ay.setFixedWidth(70)
        self._az = QLineEdit("0"); self._az.setFixedWidth(70)
        self._lbl_edit = QLineEdit(); self._lbl_edit.setFixedWidth(90)
        for col, (lbl, w) in enumerate(
            [("X:", self._ax), ("Y:", self._ay), ("Z:", self._az), ("Label:", self._lbl_edit)]
        ):
            entry_lay.addWidget(QLabel(lbl), 0, col * 2)
            entry_lay.addWidget(w, 0, col * 2 + 1)
        self._add_btn = QPushButton("Add point  [Enter]")
        self._add_btn.setEnabled(False)
        self._add_btn.setStyleSheet(
            "background:#1a5c1a; color:white; font-weight:bold; padding:4px 12px;")
        self._add_btn.clicked.connect(self.add_point)
        entry_lay.addWidget(self._add_btn, 0, 8)
        root.addWidget(entry_box)

        # Table
        tbl_box = QGroupBox("Correspondences")
        tbl_lay = QVBoxLayout(tbl_box)
        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["Label", "Rec X", "Rec Y", "Rec Z", "Arena X", "Arena Y", "Arena Z"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl_lay.addWidget(self._table)
        root.addWidget(tbl_box, stretch=1)

        # Bottom bar
        bot = QHBoxLayout()
        self._rmse_lbl = QLabel("Add ≥ 3 points for RMSE preview")
        self._rmse_lbl.setStyleSheet("color:#206020; font-weight:bold;")
        bot.addWidget(self._rmse_lbl)
        bot.addStretch()
        rem_btn = QPushButton("Remove selected  [Del]")
        rem_btn.clicked.connect(self.remove_selected)
        bot.addWidget(rem_btn)
        root.addLayout(bot)

    # -- preset -----------------------------------------------------------

    def _fill_preset(self, corners: list):
        self._preset_queue = list(corners)
        self._load_next_preset()

    def _load_next_preset(self):
        if not self._preset_queue:
            return
        label, ax, ay, az = self._preset_queue.pop(0)
        self._ax.setText(f"{ax:.1f}")
        self._ay.setText(f"{ay:.1f}")
        self._az.setText(f"{az:.1f}")
        self._lbl_edit.setText(label)
        rem = len(self._preset_queue)
        self._status.setText(
            f"[Preset: {label}]  Step 1: click in Camera 0  ({rem} corners remaining)")

    # -- click handlers ---------------------------------------------------

    def on_click_cam0(self, vx: float, vy: float):
        if self._click1 is not None:
            return
        self._click0 = (vx, vy)
        self._click1 = None
        self._rec_xyz = None
        self._tri_lbl.setText("—")
        self._add_btn.setEnabled(False)
        self._app.canvas0.add_cross(vx, vy, COL_CAM0)
        self._status.setText("Step 2: click the same landmark in Camera 1")

    def on_click_cam1(self, vx: float, vy: float):
        if self._click0 is None:
            self._status.setText("Click Camera 0 first!")
            return
        self._click1 = (vx, vy)
        self._app.canvas1.add_cross(vx, vy, COL_CAM1)
        xyz = triangulate_dlt(self._app.P0, self._app.P1,
                               self._click0, self._click1)
        self._rec_xyz = xyz[:3]
        self._tri_lbl.setText(
            f"X={self._rec_xyz[0]:+9.2f}   Y={self._rec_xyz[1]:+9.2f}   "
            f"Z={self._rec_xyz[2]:+9.2f}")
        self._add_btn.setEnabled(True)
        self._status.setText("Step 3: enter arena coords and click 'Add point'")

    # -- point management -------------------------------------------------

    def add_point(self):
        if self._rec_xyz is None:
            self._status.setText("Click both cameras first.")
            return
        try:
            ax, ay, az = (float(self._ax.text()),
                          float(self._ay.text()),
                          float(self._az.text()))
        except ValueError:
            QMessageBox.critical(self, "Input error", "Arena X / Y / Z must be numeric.")
            return
        pt = AlignPoint(
            rec_xyz=self._rec_xyz.copy(),
            arena_xyz=np.array([ax, ay, az]),
            label=self._lbl_edit.text().strip(),
            px0=np.array(self._click0) if self._click0 else None,
            px1=np.array(self._click1) if self._click1 else None,
        )
        self.points.append(pt)
        self._refresh_table()
        self._update_rmse()
        self.cancel()
        self._load_next_preset()
        self._status.setText(
            f"Point added ({len(self.points)} total). "
            "Step 1: click next landmark in Camera 0")

    def cancel(self):
        self._click0 = self._click1 = self._rec_xyz = None
        self._tri_lbl.setText("—")
        self._add_btn.setEnabled(False)
        self._app.canvas0._crosses.clear()
        self._app.canvas0._repaint()
        self._app.canvas1._crosses.clear()
        self._app.canvas1._repaint()
        self._status.setText("Step 1: click the landmark in Camera 0")

    def remove_selected(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        self._table.removeRow(idx)
        del self.points[idx]
        self._update_rmse()

    def _refresh_table(self):
        self._table.setRowCount(0)
        for pt in self.points:
            r = self._table.rowCount()
            self._table.insertRow(r)
            for c, val in enumerate([
                pt.label,
                f"{pt.rec_xyz[0]:.2f}", f"{pt.rec_xyz[1]:.2f}",
                f"{pt.rec_xyz[2]:.2f}",
                f"{pt.arena_xyz[0]:.2f}", f"{pt.arena_xyz[1]:.2f}",
                f"{pt.arena_xyz[2]:.2f}",
            ]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(r, c, item)

    def _update_rmse(self):
        n = len(self.points)
        if n >= 3:
            try:
                r = kabsch_align(self.points)
                self._rmse_lbl.setText(
                    f"Kabsch RMSE: {r.rmse_mm:.2f} mm  ({r.n_points} points)")
            except Exception as e:
                self._rmse_lbl.setText(f"RMSE error: {e}")
        else:
            self._rmse_lbl.setText(
                f"Add {3 - n} more point{'s' if 3-n > 1 else ''} for RMSE preview")


# --------------------------------------------------------------------------- #
#  Edge tracing tab                                                            #
# --------------------------------------------------------------------------- #

class EdgeTab(QWidget):
    def __init__(self, app: "ArenaAligner", parent=None):
        super().__init__(parent)
        self._app = app
        self.edges: list[TracedEdge] = []
        self._active: Optional[dict] = None
        self._fit_result = None
        self._build()

    # -- build UI ---------------------------------------------------------

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(4)

        instr = QLabel(
            "Click points along a straight box edge in ONE camera.  "
            "Minimum 4 points per edge.  'Finish edge' when done.\n"
            "Trace ≥ 3 edges per camera, spanning different image regions.")
        instr.setWordWrap(True)
        instr.setStyleSheet("color:#404040; padding:4px;")
        root.addWidget(instr)

        # Edge controls
        ctrl_box = QGroupBox("Current edge")
        ctrl_lay = QHBoxLayout(ctrl_box)

        ctrl_lay.addWidget(QLabel("Label:"))
        self._lbl = QLineEdit()
        self._lbl.setFixedWidth(100)
        ctrl_lay.addWidget(self._lbl)

        ctrl_lay.addWidget(QLabel("Camera:"))
        self._cam_grp = QButtonGroup(self)
        for i, txt in enumerate(["0", "1"]):
            rb = QRadioButton(txt)
            if i == 0:
                rb.setChecked(True)
            self._cam_grp.addButton(rb, i)
            ctrl_lay.addWidget(rb)

        self._start_btn = QPushButton("Start edge")
        self._start_btn.setStyleSheet("background:#1a4a7a; color:white;")
        self._start_btn.clicked.connect(self._start)
        ctrl_lay.addWidget(self._start_btn)

        self._finish_btn = QPushButton("Finish edge  [Enter]")
        self._finish_btn.setStyleSheet("background:#1a5c1a; color:white;")
        self._finish_btn.setEnabled(False)
        self._finish_btn.clicked.connect(self._finish)
        ctrl_lay.addWidget(self._finish_btn)

        cancel_btn = QPushButton("Cancel  [Esc]")
        cancel_btn.clicked.connect(self._cancel)
        ctrl_lay.addWidget(cancel_btn)
        ctrl_lay.addStretch()
        root.addWidget(ctrl_box)

        # Status + counters
        self._status = QLabel("Enter a label and camera, then click 'Start edge'")
        self._status.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self._status.setStyleSheet("color:#303030; padding:2px 6px;")
        root.addWidget(self._status)

        badge_row = QHBoxLayout()
        self._n0_lbl = QLabel("Camera 0: 0 edges")
        self._n0_lbl.setStyleSheet(f"color:{COL_CAM0.name()}; font-weight:bold;")
        self._n1_lbl = QLabel("Camera 1: 0 edges")
        self._n1_lbl.setStyleSheet(f"color:{COL_CAM1.name()}; font-weight:bold;")
        badge_row.addWidget(self._n0_lbl)
        badge_row.addWidget(self._n1_lbl)
        badge_row.addStretch()
        root.addLayout(badge_row)

        # Table
        tbl_box = QGroupBox("Traced edges")
        tbl_lay = QVBoxLayout(tbl_box)
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Camera", "Label", "Points", "Raw line err (px)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl_lay.addWidget(self._table)
        root.addWidget(tbl_box, stretch=1)

        # Bottom bar
        bot = QHBoxLayout()
        rem_btn = QPushButton("Remove selected  [Del]")
        rem_btn.clicked.connect(self._remove_selected)
        bot.addWidget(rem_btn)
        bot.addStretch()
        self._fit_lbl = QLabel("")
        self._fit_lbl.setStyleSheet("color:#5a1a5a; font-weight:bold;")
        bot.addWidget(self._fit_lbl)
        self._fit_btn = QPushButton("Fit distortion")
        self._fit_btn.setStyleSheet(
            "background:#5a1a5a; color:white; font-weight:bold; padding:4px 14px;")
        self._fit_btn.setEnabled(False)
        self._fit_btn.clicked.connect(self._run_fit)
        bot.addWidget(self._fit_btn)
        root.addLayout(bot)

    # -- edge lifecycle ---------------------------------------------------

    def _start(self):
        lbl = self._lbl.text().strip()
        if not lbl:
            QMessageBox.critical(self, "Edge label", "Enter a label before starting.")
            return
        cam = self._cam_grp.checkedId()
        self._active = {"camera": cam, "pts_px": [], "label": lbl}
        self._start_btn.setEnabled(False)
        self._finish_btn.setEnabled(False)
        self._status.setText(
            f"[{lbl} — cam{cam}]  Click ≥ 4 points along the edge")

    def _finish(self):
        if self._active is None:
            return
        pts = np.array(self._active["pts_px"], dtype=float)
        if len(pts) < 4:
            QMessageBox.critical(self, "Too few points",
                                 "Click at least 4 points along the edge.")
            return
        self.edges.append(TracedEdge(
            pts_px=pts, camera=self._active["camera"],
            label=self._active["label"]))
        self._cancel()
        self._refresh_table()
        self._update_counters()
        self._check_fit_ready()

    def _cancel(self):
        self._active = None
        self._start_btn.setEnabled(True)
        self._finish_btn.setEnabled(False)
        self._app.canvas0.clear_dots()
        self._app.canvas1.clear_dots()
        self._app._redraw_edge_overlays()
        self._status.setText("Enter a label and camera, then click 'Start edge'")

    # -- click handler ----------------------------------------------------

    def on_click(self, cam: int, vx: float, vy: float):
        if self._active is None or self._active["camera"] != cam:
            return
        self._active["pts_px"].append([vx, vy])
        colour = QColor(EDGE_PALETTE[len(self.edges) % len(EDGE_PALETTE)])
        canvas = self._app.canvas0 if cam == 0 else self._app.canvas1
        canvas.add_dot(vx, vy, colour)
        n = len(self._active["pts_px"])
        self._status.setText(
            f"[{self._active['label']} — cam{cam}]  {n} pts  "
            f"{'— finish when ready' if n >= 4 else f'need {4 - n} more'}")
        if n >= 4:
            self._finish_btn.setEnabled(True)

    # -- table + counters -------------------------------------------------

    def _refresh_table(self):
        self._table.setRowCount(0)
        for i, edge in enumerate(self.edges):
            colour = QColor(EDGE_PALETTE[i % len(EDGE_PALETTE)])
            err = _edge_line_rmse(edge.pts_px)
            r = self._table.rowCount()
            self._table.insertRow(r)
            for c, val in enumerate([
                f"cam{edge.camera}", edge.label,
                str(len(edge.pts_px)), f"{err:.2f}",
            ]):
                item = QTableWidgetItem(val)
                item.setForeground(colour)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(r, c, item)

    def _update_counters(self):
        n0 = sum(1 for e in self.edges if e.camera == 0)
        n1 = sum(1 for e in self.edges if e.camera == 1)
        self._n0_lbl.setText(f"Camera 0: {n0} edge{'s' if n0 != 1 else ''}")
        self._n1_lbl.setText(f"Camera 1: {n1} edge{'s' if n1 != 1 else ''}")

    def _check_fit_ready(self):
        n0 = sum(1 for e in self.edges if e.camera == 0)
        n1 = sum(1 for e in self.edges if e.camera == 1)
        ready = n0 >= 3 and n1 >= 3
        self._fit_btn.setEnabled(ready)
        if not ready:
            parts = []
            if n0 < 3: parts.append(f"cam0 needs {3-n0} more")
            if n1 < 3: parts.append(f"cam1 needs {3-n1} more")
            self._fit_lbl.setText("  ".join(parts))
        else:
            self._fit_lbl.setText("Ready to fit")

    def _remove_selected(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        self._table.removeRow(idx)
        del self.edges[idx]
        self._update_counters()
        self._check_fit_ready()
        self._app._redraw_edge_overlays()

    # -- fit --------------------------------------------------------------

    def _run_fit(self):
        self._fit_lbl.setText("Fitting…")
        QApplication.processEvents()
        try:
            k_init = (float(self._app.dist0_init[0]),
                      float(self._app.dist0_init[1]),
                      float(self._app.dist0_init[4])
                      if len(self._app.dist0_init) > 4 else 0.0)
            result = fit_distortion_plumb_line(
                self.edges, self._app.K0, self._app.K1, k_init=k_init)
        except Exception as e:
            QMessageBox.critical(self, "Fit failed", str(e))
            self._fit_lbl.setText("Fit failed")
            return

        self._fit_result = result
        conv = "converged" if result.converged else "WARNING: not converged"
        self._fit_lbl.setText(
            f"cam0 RMSE={result.rmse0:.2f}px  cam1 RMSE={result.rmse1:.2f}px  {conv}")

        msg = (
            f"Plumb-line distortion fit\n"
            f"{'─'*40}\n"
            f"Camera 0  ({result.n_edges0} edges)\n"
            f"  k1 = {result.dist0[0]:+.6f}\n"
            f"  k2 = {result.dist0[1]:+.6f}\n"
            f"  k3 = {result.dist0[4]:+.6f}\n"
            f"  RMSE = {result.rmse0:.3f} px\n\n"
            f"Camera 1  ({result.n_edges1} edges)\n"
            f"  k1 = {result.dist1[0]:+.6f}\n"
            f"  k2 = {result.dist1[1]:+.6f}\n"
            f"  k3 = {result.dist1[4]:+.6f}\n"
            f"  RMSE = {result.rmse1:.3f} px\n\n"
            f"Status: {conv}\n\n"
            f"Save a refined calibration.npz?"
        )
        reply = QMessageBox.question(self, "Fit result", msg,
                                     QMessageBox.StandardButton.Ok |
                                     QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Ok:
            out = Path(self._app.calib_path).with_name(
                Path(self._app.calib_path).stem + "_refined.npz")
            try:
                patch_calibration_distortion(
                    self._app.calib_path, result, out)
                QMessageBox.information(
                    self, "Saved",
                    f"Refined calibration written to:\n{out}\n\n"
                    f"Use with:\n  rpimocap-run --calib {out}")
            except Exception as e:
                QMessageBox.critical(self, "Save failed", str(e))


# --------------------------------------------------------------------------- #
#  Main window                                                                 #
# --------------------------------------------------------------------------- #

class ArenaAligner(QMainWindow):

    def __init__(self, cam0_path: str, cam1_path: str,
                 calib_path: str, out_path: str,
                 load_existing: Optional[str] = None,
                 load_edges: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("rpimocap — Arena Aligner")

        # ── Video ─────────────────────────────────────────────────────────
        self.cap0 = cv2.VideoCapture(cam0_path)
        self.cap1 = cv2.VideoCapture(cam1_path)
        if not self.cap0.isOpened():
            raise IOError(f"Cannot open cam0: {cam0_path}")
        if not self.cap1.isOpened():
            raise IOError(f"Cannot open cam1: {cam1_path}")
        self.n_frames = int(min(
            self.cap0.get(cv2.CAP_PROP_FRAME_COUNT),
            self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._vid_w = int(self.cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._vid_h = int(self.cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fidx  = 0

        # ── Calibration ───────────────────────────────────────────────────
        self.calib_path = calib_path
        cal = np.load(calib_path)
        self.K0, self.K1 = cal["K0"], cal["K1"]
        d0 = np.ravel(cal.get("dist0", np.zeros(5)))
        d1 = np.ravel(cal.get("dist1", np.zeros(5)))
        self.dist0_init = np.pad(d0, (0, max(0, 5 - len(d0))))
        self.dist1_init = np.pad(d1, (0, max(0, 5 - len(d1))))
        R, T = cal["R"], cal["T"]
        self.P0 = self.K0 @ np.hstack([np.eye(3),   np.zeros((3, 1))])
        self.P1 = self.K1 @ np.hstack([R, T.reshape(3, 1)])

        # ── Paths ─────────────────────────────────────────────────────────
        self.out_path   = Path(out_path)
        self._edges_out = self.out_path.with_name(
            self.out_path.stem + "_edges.csv")

        self._build_ui()
        self._setup_shortcuts()

        # Load existing data
        if load_existing and Path(load_existing).exists():
            self._ct.points = load_align_csv(load_existing)
            self._ct._refresh_table()
            self._ct._update_rmse()
        if load_edges and Path(load_edges).exists():
            self._et.edges = load_edges_csv(load_edges)
            self._et._refresh_table()
            self._et._update_counters()
            self._et._check_fit_ready()

        self._seek(0)
        self.show()

    # -- UI construction --------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        root.setContentsMargins(6, 6, 6, 6)

        # Camera panels
        cam_row = QHBoxLayout()
        self.canvas0 = CameraCanvas(0)
        self.canvas0.sx = PANEL_W / self._vid_w
        self.canvas0.sy = PANEL_H / self._vid_h
        self.canvas0.clicked.connect(self._on_click_cam0)
        self.canvas1 = CameraCanvas(1)
        self.canvas1.sx = PANEL_W / self._vid_w
        self.canvas1.sy = PANEL_H / self._vid_h
        self.canvas1.clicked.connect(self._on_click_cam1)

        for canvas, lbl_text in [
            (self.canvas0, "Camera 0  (click first — red)"),
            (self.canvas1, "Camera 1  (click second — blue)"),
        ]:
            col = QVBoxLayout()
            lbl = QLabel(lbl_text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(lbl)
            col.addWidget(canvas)
            cam_row.addLayout(col)

        root.addLayout(cam_row)

        # Scrubber
        scrub_row = QHBoxLayout()
        scrub_row.addWidget(QLabel("Frame:"))
        self._frame_lbl = QLabel("0")
        self._frame_lbl.setFixedWidth(55)
        scrub_row.addWidget(self._frame_lbl)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, max(0, self.n_frames - 1))
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._seek)
        scrub_row.addWidget(self._slider, stretch=1)
        for sym, delta in [("◀", -1), ("▶", +1)]:
            btn = QPushButton(sym)
            btn.setFixedWidth(30)
            btn.clicked.connect(lambda _, d=delta: self._step(d))
            scrub_row.addWidget(btn)
        root.addLayout(scrub_row)

        # Notebook
        self._nb = QTabWidget()
        self._ct = CornerTab(self)
        self._et = EdgeTab(self)
        self._nb.addTab(self._ct, " Corner annotation ")
        self._nb.addTab(self._et, " Edge tracing ")
        root.addWidget(self._nb, stretch=1)

        # Save / action bar
        bot_row = QHBoxLayout()
        save_corner_btn = QPushButton("Save alignment CSV")
        save_corner_btn.setStyleSheet(
            "background:#204080; color:white; font-weight:bold; padding:5px 16px;")
        save_corner_btn.clicked.connect(self._save_corners)
        bot_row.addWidget(save_corner_btn)

        save_edge_btn = QPushButton("Save edges CSV + fit distortion")
        save_edge_btn.setStyleSheet(
            "background:#5a1a5a; color:white; font-weight:bold; padding:5px 16px;")
        save_edge_btn.clicked.connect(self._save_edges)
        bot_row.addWidget(save_edge_btn)

        refine_btn = QPushButton("Refine calibration (bundle adjust)")
        refine_btn.setStyleSheet(
            "background:#6a3000; color:white; font-weight:bold; padding:5px 16px;")
        refine_btn.clicked.connect(self._refine_calibration)
        bot_row.addWidget(refine_btn)

        bot_row.addStretch()
        root.addLayout(bot_row)

        # Status bar
        self.setStatusBar(QStatusBar())

    # -- keyboard shortcuts -----------------------------------------------

    def _setup_shortcuts(self):
        from PyQt6.QtGui import QKeySequence, QShortcut
        bindings = [
            (QKeySequence(Qt.Key.Key_Left),                   lambda: self._step(-1)),
            (QKeySequence(Qt.Key.Key_Right),                  lambda: self._step(+1)),
            (QKeySequence("Shift+Left"),                      lambda: self._step(-10)),
            (QKeySequence("Shift+Right"),                     lambda: self._step(+10)),
            (QKeySequence(Qt.Key.Key_Return),                 self._on_enter),
            (QKeySequence(Qt.Key.Key_Enter),                  self._on_enter),
            (QKeySequence(Qt.Key.Key_Escape),                 self._on_escape),
            (QKeySequence(Qt.Key.Key_Delete),                 self._on_delete),
        ]
        for seq, fn in bindings:
            sc = QShortcut(seq, self)
            sc.activated.connect(fn)

    # -- navigation -------------------------------------------------------

    def _seek(self, idx: int):
        idx = max(0, min(idx, self.n_frames - 1))
        self._fidx = idx
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)
        self._frame_lbl.setText(str(idx))

        for cap in (self.cap0, self.cap1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret0, f0 = self.cap0.read()
        ret1, f1 = self.cap1.read()
        if ret0 and ret1:
            self._f0, self._f1 = f0, f1
            self.canvas0.set_frame(f0)
            self.canvas1.set_frame(f1)
            self._redraw_edge_overlays()
            self._ct.cancel()

    def _step(self, delta: int):
        self._seek(self._fidx + delta)

    # -- edge overlays ----------------------------------------------------

    def _redraw_edge_overlays(self):
        """Redraw all completed edge dot chains on the current frame."""
        # Rebuild canvases from frame, then draw edges
        for canvas, frame in [(self.canvas0, self._f0), (self.canvas1, self._f1)]:
            if hasattr(self, '_f0'):
                canvas._dots.clear()
                canvas._lines.clear()

        for i, edge in enumerate(self._et.edges):
            colour = QColor(EDGE_PALETTE[i % len(EDGE_PALETTE)])
            canvas = self.canvas0 if edge.camera == 0 else self.canvas1
            for j, pt in enumerate(edge.pts_px):
                prev_dots = len(canvas._dots)
                canvas._dots.append((
                    int(pt[0] * canvas.sx),
                    int(pt[1] * canvas.sy),
                    colour))
                if j > 0:
                    prev = canvas._dots[-2]
                    curr = canvas._dots[-1]
                    canvas._lines.append((
                        prev[0], prev[1], curr[0], curr[1], colour))

        self.canvas0._repaint()
        self.canvas1._repaint()

    # -- click dispatch ---------------------------------------------------

    @property
    def _active_tab(self) -> str:
        return "corner" if self._nb.currentIndex() == 0 else "edge"

    def _on_click_cam0(self, vx: float, vy: float):
        if self._active_tab == "corner":
            self._ct.on_click_cam0(vx, vy)
        else:
            self._et.on_click(0, vx, vy)

    def _on_click_cam1(self, vx: float, vy: float):
        if self._active_tab == "corner":
            self._ct.on_click_cam1(vx, vy)
        else:
            self._et.on_click(1, vx, vy)

    # -- key events -------------------------------------------------------

    def _on_enter(self):
        if self._active_tab == "corner":
            self._ct.add_point()
        else:
            self._et._finish()

    def _on_escape(self):
        if self._active_tab == "corner":
            self._ct.cancel()
        else:
            self._et._cancel()

    def _on_delete(self):
        if self._active_tab == "corner":
            self._ct.remove_selected()
        else:
            self._et._remove_selected()

    # -- save / refine ----------------------------------------------------

    def _save_corners(self):
        pts = self._ct.points
        if len(pts) < 3:
            QMessageBox.critical(self, "Too few points",
                                 f"Need ≥ 3 correspondences (have {len(pts)}).")
            return
        try:
            r = kabsch_align(pts)
        except ValueError as e:
            QMessageBox.critical(self, "Alignment error", str(e))
            return
        save_align_csv(self.out_path, pts)
        QMessageBox.information(
            self, "Saved",
            f"Saved {len(pts)} correspondences → {self.out_path}\n"
            f"Kabsch RMSE: {r.rmse_mm:.2f} mm\n\n"
            f"Use with:\n  rpimocap-run ... --align-points {self.out_path}")

    def _save_edges(self):
        edges = self._et.edges
        if not edges:
            QMessageBox.critical(self, "No edges", "Trace at least one edge first.")
            return
        save_edges_csv(self._edges_out, edges)
        n0 = sum(1 for e in edges if e.camera == 0)
        n1 = sum(1 for e in edges if e.camera == 1)
        msg = f"Saved {len(edges)} edges → {self._edges_out}\n"
        if n0 >= 3 and n1 >= 3:
            msg += "\nClick 'Fit distortion' in the Edge tracing tab."
        else:
            if n0 < 3: msg += f"\nCamera 0 needs {3-n0} more edge(s)."
            if n1 < 3: msg += f"\nCamera 1 needs {3-n1} more edge(s)."
        QMessageBox.information(self, "Saved", msg)

    def _refine_calibration(self):
        pts    = self._ct.points
        usable = [p for p in pts if p.px0 is not None and p.px1 is not None]
        if len(usable) < 4:
            n_old = len(pts) - len(usable)
            msg = (f"Bundle adjustment needs ≥ 4 corners with pixel coordinates.\n\n"
                   f"You have {len(usable)} usable corners")
            if n_old:
                msg += (f" ({n_old} loaded from an older CSV without pixel "
                        f"coordinates — re-annotate them).")
            QMessageBox.critical(self, "Too few corners", msg)
            return

        edges = self._et.edges
        out   = Path(self.calib_path).with_name(
            Path(self.calib_path).stem + "_refined.npz")
        try:
            import io, contextlib
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                result = refine_calibration_from_arena(
                    pts, edges, self.calib_path, out, verbose=True)
        except Exception as e:
            QMessageBox.critical(self, "Bundle adjustment failed", str(e))
            return

        conv = "converged" if result["converged"] else "WARNING: not converged"
        msg = (
            f"Bundle adjustment complete\n"
            f"{'─'*44}\n"
            f"Corners used:    {len(usable)}\n"
            f"Edges used:      {len(edges)}\n\n"
            f"RMSE before:     {result['cost_before']:.3f} px\n"
            f"RMSE after:      {result['cost_after']:.3f} px\n"
            f"Corner RMSE:     {result['rmse_corners_px']:.3f} px\n"
            f"Edge RMSE:       {result['rmse_edges_px']:.3f} px\n"
            f"Status:          {conv}\n\n"
            f"Camera 0:\n"
            f"  fx={result['K0'][0,0]:.1f}  fy={result['K0'][1,1]:.1f}  "
            f"cx={result['K0'][0,2]:.1f}  cy={result['K0'][1,2]:.1f}\n"
            f"  k1={result['dist0'][0]:+.4f}  k2={result['dist0'][1]:+.4f}  "
            f"k3={result['dist0'][4]:+.4f}\n\n"
            f"Camera 1:\n"
            f"  fx={result['K1'][0,0]:.1f}  fy={result['K1'][1,1]:.1f}  "
            f"cx={result['K1'][0,2]:.1f}  cy={result['K1'][1,2]:.1f}\n"
            f"  k1={result['dist1'][0]:+.4f}  k2={result['dist1'][1]:+.4f}  "
            f"k3={result['dist1'][4]:+.4f}\n\n"
            f"Saved to:\n{out}"
        )
        QMessageBox.information(self, "Refined calibration", msg)

    # -- cleanup ----------------------------------------------------------

    def closeEvent(self, event):
        self.cap0.release()
        self.cap1.release()
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

def main():
    """Entry point for the rpimocap-align command."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0",       required=True)
    ap.add_argument("--cam1",       required=True)
    ap.add_argument("--calib",      required=True)
    ap.add_argument("--out",        default="align_points.csv")
    ap.add_argument("--load",       default=None, metavar="CSV",
                    help="Resume from existing alignment CSV")
    ap.add_argument("--load-edges", default=None, metavar="CSV",
                    help="Resume from existing edges CSV")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = ArenaAligner(
        cam0_path=args.cam0, cam1_path=args.cam1,
        calib_path=args.calib, out_path=args.out,
        load_existing=args.load, load_edges=args.load_edges)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
