#!/usr/bin/env python3
"""
arena_aligner.py - Interactive arena alignment + distortion annotator (PyQt6 >= 6.8)

Zoom/pan : wheel zooms around cursor, middle-drag pans, double-click or Z resets.
Drag     : left-click near any annotation point to grab and reposition it.
           Corner drags re-triangulate live; edge drags update the trace.
Extrapolate: zoom OUT past the image edge to reveal black margins.
           Click or drag into the margin to place points outside the camera FOV.
           Out-of-image points are drawn with a dashed ring so they are visually
           distinct from in-image points.  Useful when a maze edge disappears for
           a short segment -- extrapolate the line through the gap.
Ctrl+Z   : undo last added corner point.
"""
from __future__ import annotations
import argparse, sys
from enum import Enum, auto
from pathlib import Path
from typing import Optional
import cv2, numpy as np

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QTabWidget, QLabel, QPushButton,
        QSlider, QLineEdit, QRadioButton, QButtonGroup, QGroupBox,
        QTableWidget, QTableWidgetItem, QHBoxLayout, QVBoxLayout, QGridLayout,
        QMessageBox, QHeaderView, QAbstractItemView, QFrame, QStatusBar)
    from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal as Signal
    from PyQt6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QBrush,
                              QFont, QKeySequence, QShortcut)
except ImportError:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QTabWidget, QLabel, QPushButton,
        QSlider, QLineEdit, QRadioButton, QButtonGroup, QGroupBox,
        QTableWidget, QTableWidgetItem, QHBoxLayout, QVBoxLayout, QGridLayout,
        QMessageBox, QHeaderView, QAbstractItemView, QFrame, QStatusBar)
    from PySide6.QtCore import Qt, QPointF, QRectF, Signal
    from PySide6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QBrush,
                                QFont, QKeySequence, QShortcut)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rpimocap.reconstruction.triangulate import triangulate_dlt
from rpimocap.reconstruction.align import (
    AlignPoint, save_align_csv, load_align_csv, kabsch_align,
    TracedEdge, fit_distortion_plumb_line, save_edges_csv, load_edges_csv,
    patch_calibration_distortion, refine_calibration_from_arena, _edge_line_rmse)

# ---------------------------------------------------------------------------
PANEL_W, PANEL_H     = 640, 480
CROSS_R, DOT_R       = 10, 5
HIT_R                = 14
ZOOM_STEP            = 1.15
ZOOM_MIN, ZOOM_MAX   = 0.25, 16.0   # allow 0.25x so image is much smaller than panel
COL_CAM0  = QColor("#e03030")
COL_CAM1  = QColor("#3060e0")
COL_SEL   = QColor("#ffdd00")
COL_OOB   = QColor("#ffffff")       # out-of-image point tint overlay
EDGE_PALETTE = [
    "#e8a020","#20c080","#c040c0","#20b8e0","#e06020","#80e020",
    "#e02080","#4080ff","#ff8040","#40ffc0","#c0c020","#ff40c0"]


class _MS(Enum):
    IDLE=auto(); PANNING=auto(); DRAG_CROSS=auto(); DRAG_DOT=auto()


# ---------------------------------------------------------------------------
def _bgr_to_qimage(frame, w, h):
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = np.ascontiguousarray(cv2.resize(rgb, (w, h), cv2.INTER_AREA))
    return QImage(small.data, w, h, w*3, QImage.Format.Format_RGB888).copy()


def _draw_cross(p, x, y, col, r=CROSS_R, sel=False):
    if sel:
        p.setPen(QPen(COL_SEL, 3)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(x-r-4, y-r-4, (r+4)*2, (r+4)*2))
    p.setPen(QPen(col, 2))
    p.drawLine(QPointF(x-r, y), QPointF(x+r, y))
    p.drawLine(QPointF(x, y-r), QPointF(x, y+r))
    p.drawEllipse(QRectF(x-4, y-4, 8, 8))


def _draw_dot(p, x, y, col, r=DOT_R, sel=False, oob=False):
    """Draw an annotation dot.  Out-of-image points get a dashed white ring."""
    if sel:
        p.setPen(QPen(COL_SEL, 2)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(x-r-3, y-r-3, (r+3)*2, (r+3)*2))
    p.setPen(Qt.PenStyle.NoPen); p.setBrush(QBrush(col))
    p.drawEllipse(QRectF(x-r, y-r, r*2, r*2))
    if oob:
        pen = QPen(COL_OOB, 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(x-r-3, y-r-3, (r+3)*2, (r+3)*2))


# ---------------------------------------------------------------------------
class CameraCanvas(QLabel):
    """
    Camera view with zoom/pan, draggable overlays, and out-of-image point support.

    Coordinate systems
    ------------------
    video space   : pixel coords in the original video (may be negative or
                    exceed frame dimensions for extrapolated points)
    display space : pixel coords in this widget (0..PANEL_W, 0..PANEL_H)

    Viewport transform  video -> display:
        dx = vx * sx * vp_s + vp_ox
        dy = vy * sy * vp_s + vp_oy

    Out-of-image points
    -------------------
    Zoom out (wheel down) until the image is smaller than the panel.  The
    surrounding black margin accepts clicks just like the image area.  The
    resulting video-space coordinates will be outside [0, vid_w] x [0, vid_h].
    These are drawn with a dashed white ring to distinguish them visually.
    They are fully valid for plumb-line fitting -- you are extrapolating the
    straight edge beyond the visible portion of the frame.

    Signals
    -------
    clicked(vx, vy)
    cross_moved(idx, vx, vy)
    cross_released(idx, vx, vy)
    dot_moved(edge_idx, pt_idx, vx, vy)
    dot_released(edge_idx, pt_idx, vx, vy)
    """
    clicked        = Signal(float, float)
    cross_moved    = Signal(int, float, float)
    cross_released = Signal(int, float, float)
    dot_moved      = Signal(int, int, float, float)
    dot_released   = Signal(int, int, float, float)

    def __init__(self, cam_id, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.setFixedSize(PANEL_W, PANEL_H)
        self.setStyleSheet("background:#1a1a1a;")
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)
        self.sx = self.sy = 1.0          # base video->panel scale
        self.vid_w = PANEL_W             # actual video width (set by ArenaAligner)
        self.vid_h = PANEL_H             # actual video height
        self._vp_s  = 1.0               # zoom
        self._vp_ox = self._vp_oy = 0.0 # pan offset
        # overlays in video space
        self._crosses: list[list] = []             # [[vx,vy,col], ...]
        self._dots:    list[list[list]] = []       # [[[vx,vy,col],...], ...]
        self._base: Optional[QPixmap] = None
        self._state    = _MS.IDLE
        self._pan_last: Optional[QPointF] = None
        self._drag_ci = self._drag_ei = self._drag_pi = -1
        self._hov_ci  = self._hov_ei  = self._hov_pi  = -1

    # -- viewport ---------------------------------------------------------
    def _v2d(self, vx, vy):
        return (vx * self.sx * self._vp_s + self._vp_ox,
                vy * self.sy * self._vp_s + self._vp_oy)

    def _d2v(self, dx, dy):
        sx = self.sx * self._vp_s or 1
        sy = self.sy * self._vp_s or 1
        return ((dx - self._vp_ox) / sx,
                (dy - self._vp_oy) / sy)

    def _is_oob(self, vx, vy):
        """True if video coord is outside the actual frame dimensions."""
        return vx < 0 or vx >= self.vid_w or vy < 0 or vy >= self.vid_h

    def reset_view(self):
        self._vp_s = 1.0; self._vp_ox = self._vp_oy = 0.0
        self._repaint()

    def _zoom_at(self, dx, dy, f):
        ns = max(ZOOM_MIN, min(ZOOM_MAX, self._vp_s * f))
        r  = ns / self._vp_s
        self._vp_ox = dx - r * (dx - self._vp_ox)
        self._vp_oy = dy - r * (dy - self._vp_oy)
        self._vp_s  = ns
        self._repaint()

    # -- overlay API (video-space coords) ---------------------------------
    def set_frame(self, frame):
        qi = _bgr_to_qimage(frame, PANEL_W, PANEL_H)
        self._base = QPixmap.fromImage(qi)
        self._repaint()

    def clear_overlay(self):
        self._crosses.clear(); self._dots.clear(); self._repaint()

    def add_cross(self, vx, vy, col):
        self._crosses.append([vx, vy, col])
        self._repaint()
        return len(self._crosses) - 1

    def update_cross(self, i, vx, vy):
        if 0 <= i < len(self._crosses):
            self._crosses[i][0] = vx; self._crosses[i][1] = vy
            self._repaint()

    def clear_crosses(self):
        self._crosses.clear(); self._repaint()

    def start_edge(self):
        self._dots.append([])
        return len(self._dots) - 1

    def add_dot(self, ei, vx, vy, col):
        while len(self._dots) <= ei:
            self._dots.append([])
        self._dots[ei].append([vx, vy, col])
        self._repaint()
        return len(self._dots[ei]) - 1

    def update_dot(self, ei, pi, vx, vy):
        if 0 <= ei < len(self._dots) and 0 <= pi < len(self._dots[ei]):
            self._dots[ei][pi][0] = vx; self._dots[ei][pi][1] = vy
            self._repaint()

    def remove_edge(self, ei):
        if 0 <= ei < len(self._dots):
            del self._dots[ei]; self._repaint()

    def clear_dots(self):
        self._dots.clear(); self._repaint()

    # -- render -----------------------------------------------------------
    def _repaint(self):
        if self._base is None: return
        pm = QPixmap(PANEL_W, PANEL_H)
        pm.fill(QColor("#1a1a1a"))
        p  = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # base frame with viewport
        p.drawPixmap(QRectF(self._vp_ox, self._vp_oy,
                            PANEL_W * self._vp_s, PANEL_H * self._vp_s),
                     self._base, QRectF(0, 0, PANEL_W, PANEL_H))

        # draw a subtle border around the image extent so the user knows
        # where the actual frame ends when zoomed out
        if self._vp_s < 0.98:
            p.setPen(QPen(QColor("#404040"), 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(self._vp_ox, self._vp_oy,
                              PANEL_W * self._vp_s, PANEL_H * self._vp_s))

        # edge dot chains
        for ei, edge in enumerate(self._dots):
            prev = None
            for pi, (vx, vy, col) in enumerate(edge):
                dx, dy = self._v2d(vx, vy)
                sel = (self._hov_ei == ei and self._hov_pi == pi)
                oob = self._is_oob(vx, vy)
                if prev:
                    pen = QPen(col, 1,
                               Qt.PenStyle.DashLine if oob else Qt.PenStyle.SolidLine)
                    p.setPen(pen)
                    p.drawLine(QPointF(*prev), QPointF(dx, dy))
                _draw_dot(p, dx, dy, col, DOT_R, sel, oob)
                prev = (dx, dy)

        # crosses
        for ci, (vx, vy, col) in enumerate(self._crosses):
            dx, dy = self._v2d(vx, vy)
            _draw_cross(p, dx, dy, col, CROSS_R, self._hov_ci == ci)

        p.end()
        self.setPixmap(pm)

    # -- hit test ---------------------------------------------------------
    def _hit_cross(self, dx, dy):
        best, bd = -1, HIT_R
        for i, (vx, vy, _) in enumerate(self._crosses):
            cdx, cdy = self._v2d(vx, vy)
            d = ((dx-cdx)**2 + (dy-cdy)**2) ** 0.5
            if d < bd: best, bd = i, d
        return best

    def _hit_dot(self, dx, dy):
        be, bp, bd = -1, -1, HIT_R
        for ei, edge in enumerate(self._dots):
            for pi, (vx, vy, _) in enumerate(edge):
                cdx, cdy = self._v2d(vx, vy)
                d = ((dx-cdx)**2 + (dy-cdy)**2) ** 0.5
                if d < bd: be, bp, bd = ei, pi, d
        return be, bp

    # -- mouse events -----------------------------------------------------
    def wheelEvent(self, e):
        f = ZOOM_STEP if e.angleDelta().y() > 0 else 1/ZOOM_STEP
        pos = e.position()
        self._zoom_at(pos.x(), pos.y(), f)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.reset_view()

    def mousePressEvent(self, e):
        pos = e.position(); dx, dy = pos.x(), pos.y()
        if e.button() == Qt.MouseButton.MiddleButton:
            self._state = _MS.PANNING; self._pan_last = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor); return
        if e.button() != Qt.MouseButton.LeftButton: return
        ci = self._hit_cross(dx, dy)
        if ci >= 0:
            self._state = _MS.DRAG_CROSS; self._drag_ci = ci
            self.setCursor(Qt.CursorShape.SizeAllCursor); return
        ei, pi = self._hit_dot(dx, dy)
        if ei >= 0:
            self._state = _MS.DRAG_DOT; self._drag_ei = ei; self._drag_pi = pi
            self.setCursor(Qt.CursorShape.SizeAllCursor); return
        # new annotation click -- allowed anywhere in the widget
        vx, vy = self._d2v(dx, dy)
        self.clicked.emit(vx, vy)

    def mouseMoveEvent(self, e):
        pos = e.position(); dx, dy = pos.x(), pos.y()
        if self._state == _MS.PANNING and self._pan_last:
            self._vp_ox += dx - self._pan_last.x()
            self._vp_oy += dy - self._pan_last.y()
            self._pan_last = pos; self._repaint(); return
        if self._state == _MS.DRAG_CROSS:
            vx, vy = self._d2v(dx, dy)
            self.update_cross(self._drag_ci, vx, vy)
            self.cross_moved.emit(self._drag_ci, vx, vy); return
        if self._state == _MS.DRAG_DOT:
            vx, vy = self._d2v(dx, dy)
            self.update_dot(self._drag_ei, self._drag_pi, vx, vy)
            self.dot_moved.emit(self._drag_ei, self._drag_pi, vx, vy); return
        # hover
        oh = (self._hov_ci, self._hov_ei, self._hov_pi)
        self._hov_ci = self._hit_cross(dx, dy)
        self._hov_ei, self._hov_pi = self._hit_dot(dx, dy)
        if (self._hov_ci, self._hov_ei, self._hov_pi) != oh:
            cur = (Qt.CursorShape.SizeAllCursor
                   if self._hov_ci >= 0 or self._hov_ei >= 0
                   else Qt.CursorShape.CrossCursor)
            self.setCursor(cur); self._repaint()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self._state = _MS.IDLE; self.setCursor(Qt.CursorShape.CrossCursor); return
        if e.button() != Qt.MouseButton.LeftButton: return
        pos = e.position(); vx, vy = self._d2v(pos.x(), pos.y())
        if self._state == _MS.DRAG_CROSS:
            self.cross_released.emit(self._drag_ci, vx, vy); self._drag_ci = -1
        elif self._state == _MS.DRAG_DOT:
            self.dot_released.emit(self._drag_ei, self._drag_pi, vx, vy)
            self._drag_ei = self._drag_pi = -1
        self._state = _MS.IDLE; self.setCursor(Qt.CursorShape.CrossCursor)


# ---------------------------------------------------------------------------
PRESET_CORNERS = [
    ("BFL",(-1,-1, 0)),("BFR",(+1,-1, 0)),("BBR",(+1,+1, 0)),("BBL",(-1,+1, 0)),
    ("TFL",(-1,-1,+1)),("TFR",(+1,-1,+1)),("TBR",(+1,+1,+1)),("TBL",(-1,+1,+1))]


class BoxPresetWidget(QGroupBox):
    corners_ready = Signal(list)
    def __init__(self, parent=None):
        super().__init__("Box preset", parent)
        lay = QHBoxLayout(self)
        for attr, lbl, dflt in [("_w","Width X (mm)","600"),
                                  ("_d","Depth Y (mm)","400"),
                                  ("_h","Height Z (mm)","350")]:
            lay.addWidget(QLabel(lbl))
            e = QLineEdit(dflt); e.setFixedWidth(60)
            setattr(self, attr, e); lay.addWidget(e)
        btn = QPushButton("Fill 8 corners ->")
        btn.clicked.connect(self._fill); lay.addWidget(btn); lay.addStretch()
    def _fill(self):
        try: W,D,H = float(self._w.text()),float(self._d.text()),float(self._h.text())
        except ValueError:
            QMessageBox.critical(self,"Box preset","Dimensions must be numeric."); return
        self.corners_ready.emit(
            [(l, sx*W/2, sy*D/2, sz*H) for l,(sx,sy,sz) in PRESET_CORNERS])


# ---------------------------------------------------------------------------
class CornerTab(QWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self._app = app; self.points = []
        self._click0 = self._click1 = self._rec_xyz = None
        self._cross0_idx = self._cross1_idx = -1
        self._preset_queue = []
        self._build()
        app.canvas0.cross_moved.connect(
            lambda i,vx,vy: self._on_drag(0,i,vx,vy))
        app.canvas0.cross_released.connect(
            lambda i,vx,vy: self._on_drag(0,i,vx,vy))
        app.canvas1.cross_moved.connect(
            lambda i,vx,vy: self._on_drag(1,i,vx,vy))
        app.canvas1.cross_released.connect(
            lambda i,vx,vy: self._on_drag(1,i,vx,vy))

    def _on_drag(self, cam, idx, vx, vy):
        if cam==0 and idx==self._cross0_idx: self._click0=(vx,vy)
        elif cam==1 and idx==self._cross1_idx: self._click1=(vx,vy)
        else: return
        if self._click0 and self._click1:
            xyz = triangulate_dlt(self._app.P0, self._app.P1,
                                   self._click0, self._click1)
            self._rec_xyz = xyz[:3]
            self._tri_lbl.setText(
                f"X={xyz[0]:+9.2f}   Y={xyz[1]:+9.2f}   Z={xyz[2]:+9.2f}")
            self._add_btn.setEnabled(True)

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(4)
        pre = BoxPresetWidget(); pre.corners_ready.connect(self._fill_preset)
        root.addWidget(pre)
        self._status = QLabel("Step 1: click landmark in Camera 0")
        self._status.setFrameStyle(QFrame.Shape.Panel|QFrame.Shadow.Sunken)
        self._status.setStyleSheet("color:#303030;padding:2px 6px;")
        root.addWidget(self._status)
        tri = QGroupBox("Triangulated position (mm)"); tl = QHBoxLayout(tri)
        self._tri_lbl = QLabel("--"); self._tri_lbl.setFont(QFont("Courier",11))
        tl.addWidget(self._tri_lbl); tl.addStretch()
        cb = QPushButton("Cancel [Esc]"); cb.clicked.connect(self.cancel)
        tl.addWidget(cb); root.addWidget(tri)
        ef = QGroupBox("Known arena coordinate (mm)"); el = QGridLayout(ef)
        self._ax = QLineEdit("0"); self._ax.setFixedWidth(70)
        self._ay = QLineEdit("0"); self._ay.setFixedWidth(70)
        self._az = QLineEdit("0"); self._az.setFixedWidth(70)
        self._lbl_e = QLineEdit(); self._lbl_e.setFixedWidth(90)
        for c,(l,w) in enumerate([("X:",self._ax),("Y:",self._ay),
                                    ("Z:",self._az),("Label:",self._lbl_e)]):
            el.addWidget(QLabel(l),0,c*2); el.addWidget(w,0,c*2+1)
        self._add_btn = QPushButton("Add point [Enter]")
        self._add_btn.setEnabled(False)
        self._add_btn.setStyleSheet(
            "background:#1a5c1a;color:white;font-weight:bold;padding:4px 12px;")
        self._add_btn.clicked.connect(self.add_point)
        el.addWidget(self._add_btn,0,8); root.addWidget(ef)
        tb = QGroupBox("Correspondences"); tbl_l = QVBoxLayout(tb)
        self._table = QTableWidget(0,7)
        self._table.setHorizontalHeaderLabels(
            ["Label","Rec X","Rec Y","Rec Z","Arena X","Arena Y","Arena Z"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl_l.addWidget(self._table); root.addWidget(tb, stretch=1)
        bot = QHBoxLayout()
        self._rmse_lbl = QLabel("Add >= 3 points for RMSE preview")
        self._rmse_lbl.setStyleSheet("color:#206020;font-weight:bold;")
        bot.addWidget(self._rmse_lbl); bot.addStretch()
        rb = QPushButton("Remove selected [Del]")
        rb.clicked.connect(self.remove_selected); bot.addWidget(rb)
        root.addLayout(bot)
        br = QHBoxLayout(); br.addWidget(QLabel("--bounds:"))
        self._bounds_e = QLineEdit(); self._bounds_e.setReadOnly(True)
        self._bounds_e.setPlaceholderText("computed from corner annotations")
        self._bounds_e.setStyleSheet(
            "font-family:monospace;color:#204080;background:#f0f4ff;")
        br.addWidget(self._bounds_e, stretch=1)
        cb2 = QPushButton("Copy"); cb2.setFixedWidth(54)
        cb2.clicked.connect(self._copy_bounds); br.addWidget(cb2)
        root.addLayout(br)

    def _fill_preset(self, corners):
        self._preset_queue = list(corners); self._load_next_preset()

    def _load_next_preset(self):
        if not self._preset_queue: return
        l,ax,ay,az = self._preset_queue.pop(0)
        self._ax.setText(f"{ax:.1f}"); self._ay.setText(f"{ay:.1f}")
        self._az.setText(f"{az:.1f}"); self._lbl_e.setText(l)
        self._status.setText(
            f"[Preset: {l}]  Click Camera 0  ({len(self._preset_queue)} remaining)")

    def on_click_cam0(self, vx, vy):
        if self._click1 is not None: return
        self._click0=(vx,vy); self._click1=self._rec_xyz=None
        self._tri_lbl.setText("--"); self._add_btn.setEnabled(False)
        self._app.canvas0.clear_crosses()
        self._cross0_idx = self._app.canvas0.add_cross(vx,vy,COL_CAM0)
        self._cross1_idx = -1
        self._status.setText("Step 2: click same landmark in Camera 1")

    def on_click_cam1(self, vx, vy):
        if self._click0 is None:
            self._status.setText("Click Camera 0 first!"); return
        self._click1=(vx,vy)
        self._app.canvas1.clear_crosses()
        self._cross1_idx = self._app.canvas1.add_cross(vx,vy,COL_CAM1)
        self._on_drag(1, self._cross1_idx, vx, vy)
        self._status.setText(
            "Step 3: enter arena coords and Add  (drag crosses to adjust)")

    def add_point(self):
        if self._rec_xyz is None:
            self._status.setText("Click both cameras first."); return
        try: ax,ay,az = (float(self._ax.text()),
                          float(self._ay.text()),float(self._az.text()))
        except ValueError:
            QMessageBox.critical(self,"Input error","X/Y/Z must be numeric."); return
        self.points.append(AlignPoint(
            rec_xyz=self._rec_xyz.copy(), arena_xyz=np.array([ax,ay,az]),
            label=self._lbl_e.text().strip(),
            px0=np.array(self._click0) if self._click0 else None,
            px1=np.array(self._click1) if self._click1 else None))
        self._refresh_table(); self._update_rmse()
        self.cancel(); self._load_next_preset()
        self._status.setText(
            f"Point added ({len(self.points)} total).  Click next in Camera 0")

    def cancel(self):
        self._click0=self._click1=self._rec_xyz=None
        self._cross0_idx=self._cross1_idx=-1
        self._tri_lbl.setText("--"); self._add_btn.setEnabled(False)
        self._app.canvas0.clear_crosses(); self._app.canvas1.clear_crosses()
        self._status.setText("Step 1: click the landmark in Camera 0")

    def remove_selected(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows: return
        idx = rows[0].row(); self._table.removeRow(idx)
        del self.points[idx]; self._update_rmse()

    def undo_last(self):
        if not self.points: return
        self.points.pop(); self._table.removeRow(self._table.rowCount()-1)
        self._update_rmse()

    def _refresh_table(self):
        self._table.setRowCount(0)
        for pt in self.points:
            r = self._table.rowCount(); self._table.insertRow(r)
            for c,v in enumerate([pt.label,
                f"{pt.rec_xyz[0]:.2f}",f"{pt.rec_xyz[1]:.2f}",f"{pt.rec_xyz[2]:.2f}",
                f"{pt.arena_xyz[0]:.2f}",f"{pt.arena_xyz[1]:.2f}",
                f"{pt.arena_xyz[2]:.2f}"]):
                item = QTableWidgetItem(v)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(r,c,item)

    def _update_rmse(self):
        n = len(self.points)
        if n >= 3:
            try:
                r = kabsch_align(self.points)
                self._rmse_lbl.setText(
                    f"Kabsch RMSE: {r.rmse_mm:.2f} mm  ({r.n_points} pts)")
            except Exception as e: self._rmse_lbl.setText(f"RMSE error: {e}")
        else:
            self._rmse_lbl.setText(
                f"Add {3-n} more point{'s' if 3-n>1 else ''} for RMSE")
        self._update_bounds()

    def _update_bounds(self):
        if not self.points: self._bounds_e.setText(""); return
        pts = np.stack([p.arena_xyz for p in self.points])
        self._bounds_e.setText(
            f"{pts[:,0].min():.0f},{pts[:,0].max():.0f},"
            f"{pts[:,1].min():.0f},{pts[:,1].max():.0f},"
            f"{pts[:,2].min():.0f},{pts[:,2].max():.0f}")

    def _copy_bounds(self):
        txt = self._bounds_e.text()
        if not txt: return
        QApplication.clipboard().setText(txt)
        self._bounds_e.setStyleSheet(
            "font-family:monospace;color:white;background:#206020;")
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(600, lambda: self._bounds_e.setStyleSheet(
            "font-family:monospace;color:#204080;background:#f0f4ff;"))


# ---------------------------------------------------------------------------
class EdgeTab(QWidget):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self._app = app; self.edges = []; self._active = None
        self._active_cei = -1; self._fit_result = None
        self._build()
        app.canvas0.dot_moved.connect(
            lambda ei,pi,vx,vy: self._dot_upd(0,ei,pi,vx,vy,False))
        app.canvas0.dot_released.connect(
            lambda ei,pi,vx,vy: self._dot_upd(0,ei,pi,vx,vy,True))
        app.canvas1.dot_moved.connect(
            lambda ei,pi,vx,vy: self._dot_upd(1,ei,pi,vx,vy,False))
        app.canvas1.dot_released.connect(
            lambda ei,pi,vx,vy: self._dot_upd(1,ei,pi,vx,vy,True))

    def _dot_upd(self, cam, cei, pi, vx, vy, final):
        e = self._canvas_to_edge(cam, cei)
        if e and 0 <= pi < len(e.pts_px):
            e.pts_px[pi] = [vx, vy]
            if final: self._refresh_table()

    def _canvas_to_edge(self, cam, canvas_ei):
        ci = 0
        for edge in self.edges:
            if edge.camera == cam:
                if ci == canvas_ei: return edge
                ci += 1
        return None

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(4)
        hint = QLabel(
            "Click points along a straight box edge in ONE camera.  Min 4 points.\n"
            "Zoom OUT (wheel) to reveal black margins and place extrapolated points\n"
            "where the edge disappears -- shown with a dashed ring.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#404040;padding:4px;")
        root.addWidget(hint)
        ctrl = QGroupBox("Current edge"); cl = QHBoxLayout(ctrl)
        cl.addWidget(QLabel("Label:"))
        self._lbl = QLineEdit(); self._lbl.setFixedWidth(100); cl.addWidget(self._lbl)
        cl.addWidget(QLabel("Camera:"))
        self._cam_grp = QButtonGroup(self)
        for i,txt in enumerate(["0","1"]):
            rb = QRadioButton(txt)
            if i==0: rb.setChecked(True)
            self._cam_grp.addButton(rb,i); cl.addWidget(rb)
        self._start_btn = QPushButton("Start edge")
        self._start_btn.setStyleSheet("background:#1a4a7a;color:white;")
        self._start_btn.clicked.connect(self._start); cl.addWidget(self._start_btn)
        self._finish_btn = QPushButton("Finish edge [Enter]")
        self._finish_btn.setStyleSheet("background:#1a5c1a;color:white;")
        self._finish_btn.setEnabled(False)
        self._finish_btn.clicked.connect(self._finish); cl.addWidget(self._finish_btn)
        can_btn = QPushButton("Cancel [Esc]")
        can_btn.clicked.connect(self._cancel); cl.addWidget(can_btn)
        cl.addStretch(); root.addWidget(ctrl)
        self._status = QLabel("Enter a label and camera, then Start edge")
        self._status.setFrameStyle(QFrame.Shape.Panel|QFrame.Shadow.Sunken)
        self._status.setStyleSheet("color:#303030;padding:2px 6px;")
        root.addWidget(self._status)
        badge = QHBoxLayout()
        self._n0_lbl = QLabel("Camera 0: 0 edges")
        self._n0_lbl.setStyleSheet(f"color:{COL_CAM0.name()};font-weight:bold;")
        self._n1_lbl = QLabel("Camera 1: 0 edges")
        self._n1_lbl.setStyleSheet(f"color:{COL_CAM1.name()};font-weight:bold;")
        badge.addWidget(self._n0_lbl); badge.addWidget(self._n1_lbl)
        badge.addStretch(); root.addLayout(badge)
        tb = QGroupBox("Traced edges"); tl = QVBoxLayout(tb)
        self._table = QTableWidget(0,5)
        self._table.setHorizontalHeaderLabels(
            ["Camera","Label","Points","OOB pts","Raw err (px)"])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tl.addWidget(self._table); root.addWidget(tb,stretch=1)
        bot = QHBoxLayout()
        rb2 = QPushButton("Remove selected [Del]")
        rb2.clicked.connect(self._remove_selected); bot.addWidget(rb2)
        bot.addStretch()
        self._fit_lbl = QLabel("")
        self._fit_lbl.setStyleSheet("color:#5a1a5a;font-weight:bold;")
        bot.addWidget(self._fit_lbl)
        self._fit_btn = QPushButton("Fit distortion")
        self._fit_btn.setStyleSheet(
            "background:#5a1a5a;color:white;font-weight:bold;padding:4px 14px;")
        self._fit_btn.setEnabled(False)
        self._fit_btn.clicked.connect(self._run_fit)
        bot.addWidget(self._fit_btn); root.addLayout(bot)

    def _start(self):
        lbl = self._lbl.text().strip()
        if not lbl:
            QMessageBox.critical(self,"Edge label","Enter a label first."); return
        cam = self._cam_grp.checkedId()
        canvas = self._app.canvas0 if cam==0 else self._app.canvas1
        self._active_cei = canvas.start_edge()
        self._active = {"camera":cam,"pts_px":[],"label":lbl}
        self._start_btn.setEnabled(False); self._finish_btn.setEnabled(False)
        self._status.setText(
            f"[{lbl} cam{cam}]  Click >= 4 pts  "
            "(zoom out to extrapolate beyond image edge)")

    def _finish(self):
        if self._active is None: return
        pts = np.array(self._active["pts_px"],dtype=float)
        if len(pts) < 4:
            QMessageBox.critical(self,"Too few points","Click at least 4 points."); return
        self.edges.append(TracedEdge(
            pts_px=pts, camera=self._active["camera"],
            label=self._active["label"]))
        self._active = None; self._active_cei = -1
        self._start_btn.setEnabled(True); self._finish_btn.setEnabled(False)
        self._refresh_table(); self._update_counters(); self._check_fit_ready()
        self._status.setText("Edge added.  Enter label for next edge.")

    def _cancel(self):
        if self._active is not None:
            cam = self._active["camera"]
            canvas = self._app.canvas0 if cam==0 else self._app.canvas1
            if self._active_cei >= 0: canvas.remove_edge(self._active_cei)
        self._active = None; self._active_cei = -1
        self._start_btn.setEnabled(True); self._finish_btn.setEnabled(False)
        self._status.setText("Enter a label and camera, then Start edge")

    def on_click(self, cam, vx, vy):
        if self._active is None or self._active["camera"] != cam: return
        self._active["pts_px"].append([vx,vy])
        col = QColor(EDGE_PALETTE[len(self.edges) % len(EDGE_PALETTE)])
        canvas = self._app.canvas0 if cam==0 else self._app.canvas1
        canvas.add_dot(self._active_cei, vx, vy, col)
        n = len(self._active["pts_px"])
        oob = canvas._is_oob(vx,vy)
        oob_note = "  [extrapolated]" if oob else ""
        self._status.setText(
            f"[{self._active['label']} cam{cam}]  {n} pts{oob_note}  "
            f"{'-- finish when ready' if n>=4 else f'need {4-n} more'}")
        if n >= 4: self._finish_btn.setEnabled(True)

    def _refresh_table(self):
        self._table.setRowCount(0)
        for i,edge in enumerate(self.edges):
            col = QColor(EDGE_PALETTE[i % len(EDGE_PALETTE)])
            err = _edge_line_rmse(edge.pts_px)
            canvas = self._app.canvas0 if edge.camera==0 else self._app.canvas1
            n_oob = sum(1 for pt in edge.pts_px if canvas._is_oob(pt[0],pt[1]))
            r = self._table.rowCount(); self._table.insertRow(r)
            for c,v in enumerate([
                f"cam{edge.camera}", edge.label,
                str(len(edge.pts_px)), str(n_oob), f"{err:.2f}"]):
                item = QTableWidgetItem(v)
                item.setForeground(col)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(r,c,item)

    def _update_counters(self):
        n0 = sum(1 for e in self.edges if e.camera==0)
        n1 = sum(1 for e in self.edges if e.camera==1)
        self._n0_lbl.setText(f"Camera 0: {n0} edge{'s' if n0!=1 else ''}")
        self._n1_lbl.setText(f"Camera 1: {n1} edge{'s' if n1!=1 else ''}")

    def _check_fit_ready(self):
        n0 = sum(1 for e in self.edges if e.camera==0)
        n1 = sum(1 for e in self.edges if e.camera==1)
        ready = n0>=3 and n1>=3
        self._fit_btn.setEnabled(ready)
        if not ready:
            parts=[]
            if n0<3: parts.append(f"cam0 needs {3-n0} more")
            if n1<3: parts.append(f"cam1 needs {3-n1} more")
            self._fit_lbl.setText("  ".join(parts))
        else: self._fit_lbl.setText("Ready to fit")

    def _remove_selected(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows: return
        idx = rows[0].row(); edge = self.edges[idx]
        cam = edge.camera
        canvas = self._app.canvas0 if cam==0 else self._app.canvas1
        ci = sum(1 for e in self.edges[:idx] if e.camera==cam)
        canvas.remove_edge(ci)
        self._table.removeRow(idx); del self.edges[idx]
        self._update_counters(); self._check_fit_ready()

    def _run_fit(self):
        self._fit_lbl.setText("Fitting..."); QApplication.processEvents()
        try:
            k = (float(self._app.dist0_init[0]),
                 float(self._app.dist0_init[1]),
                 float(self._app.dist0_init[4]) if len(self._app.dist0_init)>4 else 0.0)
            result = fit_distortion_plumb_line(
                self.edges, self._app.K0, self._app.K1, k_init=k)
        except Exception as e:
            QMessageBox.critical(self,"Fit failed",str(e))
            self._fit_lbl.setText("Fit failed"); return
        self._fit_result = result
        conv = "converged" if result.converged else "WARNING: not converged"
        self._fit_lbl.setText(
            f"cam0 RMSE={result.rmse0:.2f}px  cam1 RMSE={result.rmse1:.2f}px  {conv}")
        msg = (f"Plumb-line fit\n{'─'*36}\n"
               f"Camera 0 ({result.n_edges0} edges)\n"
               f"  k1={result.dist0[0]:+.6f}  k2={result.dist0[1]:+.6f}"
               f"  k3={result.dist0[4]:+.6f}\n  RMSE={result.rmse0:.3f}px\n\n"
               f"Camera 1 ({result.n_edges1} edges)\n"
               f"  k1={result.dist1[0]:+.6f}  k2={result.dist1[1]:+.6f}"
               f"  k3={result.dist1[4]:+.6f}\n  RMSE={result.rmse1:.3f}px\n\n"
               f"Status: {conv}\n\nSave refined calibration.npz?")
        if (QMessageBox.question(self,"Fit result",msg,
                                  QMessageBox.StandardButton.Ok|
                                  QMessageBox.StandardButton.Cancel)
                == QMessageBox.StandardButton.Ok):
            out = Path(self._app.calib_path).with_name(
                Path(self._app.calib_path).stem+"_refined.npz")
            try:
                patch_calibration_distortion(self._app.calib_path,result,out)
                QMessageBox.information(
                    self,"Saved",
                    f"Written to:\n{out}\n\nUse with:\n  rpimocap-run --calib {out}")
            except Exception as e:
                QMessageBox.critical(self,"Save failed",str(e))


# ---------------------------------------------------------------------------
class ArenaAligner(QMainWindow):
    def __init__(self, cam0_path, cam1_path, calib_path, out_path,
                 load_existing=None, load_edges=None):
        super().__init__()
        self.setWindowTitle("rpimocap -- Arena Aligner")

        def _open(path):
            if Path(path).suffix.lower() in (".tif",".tiff"):
                from rpimocap.io.export import TiffCapture
                return TiffCapture(path)
            return cv2.VideoCapture(path)

        self.cap0 = _open(cam0_path); self.cap1 = _open(cam1_path)
        if not self.cap0.isOpened(): raise IOError(f"Cannot open cam0: {cam0_path}")
        if not self.cap1.isOpened(): raise IOError(f"Cannot open cam1: {cam1_path}")
        self.n_frames = int(min(
            self.cap0.get(cv2.CAP_PROP_FRAME_COUNT),
            self.cap1.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._vid_w = int(self.cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._vid_h = int(self.cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fidx  = 0; self._f0 = self._f1 = None

        self.calib_path = calib_path
        cal = np.load(calib_path)
        self.K0, self.K1 = cal["K0"], cal["K1"]
        d0 = np.ravel(cal.get("dist0",np.zeros(5)))
        d1 = np.ravel(cal.get("dist1",np.zeros(5)))
        self.dist0_init = np.pad(d0,(0,max(0,5-len(d0))))
        self.dist1_init = np.pad(d1,(0,max(0,5-len(d1))))
        R,T = cal["R"],cal["T"]
        self.P0 = self.K0 @ np.hstack([np.eye(3),  np.zeros((3,1))])
        self.P1 = self.K1 @ np.hstack([R, T.reshape(3,1)])

        self.out_path   = Path(out_path)
        self._edges_out = self.out_path.with_name(self.out_path.stem+"_edges.csv")

        # build canvases before tabs (tabs wire signals in __init__)
        self.canvas0 = CameraCanvas(0)
        self.canvas0.sx    = PANEL_W / self._vid_w
        self.canvas0.sy    = PANEL_H / self._vid_h
        self.canvas0.vid_w = self._vid_w
        self.canvas0.vid_h = self._vid_h
        self.canvas0.clicked.connect(self._on_click_cam0)

        self.canvas1 = CameraCanvas(1)
        self.canvas1.sx    = PANEL_W / self._vid_w
        self.canvas1.sy    = PANEL_H / self._vid_h
        self.canvas1.vid_w = self._vid_w
        self.canvas1.vid_h = self._vid_h
        self.canvas1.clicked.connect(self._on_click_cam1)

        self._build_ui(); self._setup_shortcuts()

        if load_existing and Path(load_existing).exists():
            self._ct.points = load_align_csv(load_existing)
            self._ct._refresh_table(); self._ct._update_rmse()
        if load_edges and Path(load_edges).exists():
            self._et.edges = load_edges_csv(load_edges)
            self._et._refresh_table()
            self._et._update_counters(); self._et._check_fit_ready()

        self._seek(0); self.show()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setSpacing(4)
        root.setContentsMargins(6,6,6,6)
        cam_row = QHBoxLayout()
        hints = [
            "Camera 0  | wheel: zoom  | MMB: pan  | dbl-click: reset  | "
            "zoom OUT to extrapolate past image edge",
            "Camera 1"]
        for canvas,hint in [(self.canvas0,hints[0]),(self.canvas1,hints[1])]:
            col = QVBoxLayout()
            lbl = QLabel(hint); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color:#505050;font-size:11px;")
            col.addWidget(lbl); col.addWidget(canvas)
            cam_row.addLayout(col)
        root.addLayout(cam_row)
        scrub = QHBoxLayout(); scrub.addWidget(QLabel("Frame:"))
        self._frame_lbl = QLabel("0"); self._frame_lbl.setFixedWidth(55)
        scrub.addWidget(self._frame_lbl)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0,max(0,self.n_frames-1))
        self._slider.valueChanged.connect(self._seek)
        scrub.addWidget(self._slider,stretch=1)
        for sym,d in [("◀",-1),("▶",+1)]:
            btn=QPushButton(sym); btn.setFixedWidth(30)
            btn.clicked.connect(lambda _,dd=d: self._step(dd))
            scrub.addWidget(btn)
        root.addLayout(scrub)
        self._nb = QTabWidget()
        self._ct = CornerTab(self); self._et = EdgeTab(self)
        self._nb.addTab(self._ct," Corner annotation ")
        self._nb.addTab(self._et," Edge tracing ")
        root.addWidget(self._nb,stretch=1)
        bot = QHBoxLayout()
        for lbl,col,fn in [
            ("Save alignment CSV","#204080",self._save_corners),
            ("Save edges CSV + fit distortion","#5a1a5a",self._save_edges),
            ("Refine calibration (bundle adjust)","#6a3000",self._refine_calibration)]:
            btn=QPushButton(lbl)
            btn.setStyleSheet(
                f"background:{col};color:white;font-weight:bold;padding:5px 16px;")
            btn.clicked.connect(fn); bot.addWidget(btn)
        bot.addStretch(); root.addLayout(bot)
        self.setStatusBar(QStatusBar())

    def _setup_shortcuts(self):
        for key,fn in [
            ("Left",    lambda: self._step(-1)),
            ("Right",   lambda: self._step(+1)),
            ("Shift+Left",  lambda: self._step(-10)),
            ("Shift+Right", lambda: self._step(+10)),
            ("Return",  self._on_enter), ("Enter", self._on_enter),
            ("Escape",  self._on_escape), ("Delete", self._on_delete),
            ("Z",       self._reset_views), ("Ctrl+Z", self._undo)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(fn)

    def _seek(self, idx):
        idx = max(0,min(idx,self.n_frames-1)); self._fidx=idx
        self._slider.blockSignals(True); self._slider.setValue(idx)
        self._slider.blockSignals(False); self._frame_lbl.setText(str(idx))
        for cap in (self.cap0,self.cap1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ret0,f0=self.cap0.read(); ret1,f1=self.cap1.read()
        if ret0 and ret1:
            self._f0,self._f1=f0,f1
            self.canvas0.set_frame(f0); self.canvas1.set_frame(f1)
            self._ct.cancel()

    def _step(self,d): self._seek(self._fidx+d)
    def _reset_views(self): self.canvas0.reset_view(); self.canvas1.reset_view()

    @property
    def _active_tab(self):
        return "corner" if self._nb.currentIndex()==0 else "edge"

    def _on_click_cam0(self,vx,vy):
        if self._active_tab=="corner": self._ct.on_click_cam0(vx,vy)
        else: self._et.on_click(0,vx,vy)

    def _on_click_cam1(self,vx,vy):
        if self._active_tab=="corner": self._ct.on_click_cam1(vx,vy)
        else: self._et.on_click(1,vx,vy)

    def _on_enter(self):
        if self._active_tab=="corner": self._ct.add_point()
        else: self._et._finish()

    def _on_escape(self):
        if self._active_tab=="corner": self._ct.cancel()
        else: self._et._cancel()

    def _on_delete(self):
        if self._active_tab=="corner": self._ct.remove_selected()
        else: self._et._remove_selected()

    def _undo(self):
        if self._active_tab=="corner": self._ct.undo_last()

    def _save_corners(self):
        pts=self._ct.points
        if len(pts)<3:
            QMessageBox.critical(self,"Too few points",
                                  f"Need >= 3 (have {len(pts)})."); return
        try: r=kabsch_align(pts)
        except ValueError as e:
            QMessageBox.critical(self,"Alignment error",str(e)); return
        save_align_csv(self.out_path,pts)
        QMessageBox.information(self,"Saved",
            f"Saved {len(pts)} correspondences -> {self.out_path}\n"
            f"Kabsch RMSE: {r.rmse_mm:.2f} mm\n\n"
            f"Use with:\n  rpimocap-run ... --align-points {self.out_path}")

    def _save_edges(self):
        edges=self._et.edges
        if not edges:
            QMessageBox.critical(self,"No edges","Trace at least one edge first."); return
        save_edges_csv(self._edges_out,edges)
        n0=sum(1 for e in edges if e.camera==0)
        n1=sum(1 for e in edges if e.camera==1)
        msg=f"Saved {len(edges)} edges -> {self._edges_out}\n"
        if n0>=3 and n1>=3: msg+="\nClick 'Fit distortion' to run the fit."
        else:
            if n0<3: msg+=f"\nCamera 0 needs {3-n0} more edge(s)."
            if n1<3: msg+=f"\nCamera 1 needs {3-n1} more edge(s)."
        QMessageBox.information(self,"Saved",msg)

    def _refine_calibration(self):
        pts=self._ct.points
        usable=[p for p in pts if p.px0 is not None and p.px1 is not None]
        if len(usable)<4:
            n_old=len(pts)-len(usable)
            msg=(f"Need >= 4 corners with pixel coordinates.\n\n"
                 f"Have {len(usable)} usable corners")
            if n_old: msg+=f" ({n_old} from older CSV -- re-annotate them)."
            QMessageBox.critical(self,"Too few corners",msg); return
        out=Path(self.calib_path).with_name(
            Path(self.calib_path).stem+"_refined.npz")
        try:
            import io,contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                result=refine_calibration_from_arena(
                    pts,self._et.edges,self.calib_path,out,verbose=True)
        except Exception as e:
            QMessageBox.critical(self,"Bundle adjustment failed",str(e)); return
        conv="converged" if result["converged"] else "WARNING: not converged"
        QMessageBox.information(self,"Refined calibration",(
            f"Bundle adjustment\n{'─'*40}\n"
            f"Corners: {len(usable)}  Edges: {len(self._et.edges)}\n\n"
            f"RMSE  {result['cost_before']:.3f} px  ->  {result['cost_after']:.3f} px\n"
            f"Corner RMSE: {result['rmse_corners_px']:.3f} px\n"
            f"Edge RMSE:   {result['rmse_edges_px']:.3f} px\n"
            f"Status: {conv}\n\n"
            f"Camera 0:\n"
            f"  fx={result['K0'][0,0]:.1f}  fy={result['K0'][1,1]:.1f}"
            f"  cx={result['K0'][0,2]:.1f}  cy={result['K0'][1,2]:.1f}\n"
            f"  k1={result['dist0'][0]:+.4f}  k2={result['dist0'][1]:+.4f}"
            f"  k3={result['dist0'][4]:+.4f}\n\n"
            f"Camera 1:\n"
            f"  fx={result['K1'][0,0]:.1f}  fy={result['K1'][1,1]:.1f}"
            f"  cx={result['K1'][0,2]:.1f}  cy={result['K1'][1,2]:.1f}\n"
            f"  k1={result['dist1'][0]:+.4f}  k2={result['dist1'][1]:+.4f}"
            f"  k3={result['dist1'][4]:+.4f}\n\n"
            f"Saved to:\n{out}"))

    def closeEvent(self,e):
        self.cap0.release(); self.cap1.release(); super().closeEvent(e)


# ---------------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam0",required=True)
    ap.add_argument("--cam1",required=True)
    ap.add_argument("--calib",required=True)
    ap.add_argument("--out",default="align_points.csv")
    ap.add_argument("--load",default=None,metavar="CSV")
    ap.add_argument("--load-edges",default=None,metavar="CSV")
    args=ap.parse_args()
    app=QApplication(sys.argv); app.setStyle("Fusion")
    win=ArenaAligner(cam0_path=args.cam0,cam1_path=args.cam1,
                     calib_path=args.calib,out_path=args.out,
                     load_existing=args.load,load_edges=args.load_edges)
    sys.exit(app.exec())

if __name__=="__main__":
    main()
