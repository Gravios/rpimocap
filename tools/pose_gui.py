#!/usr/bin/env python3
"""
tools/pose_gui.py — interactive manual 3D pose fitting for rpimocap.

Set a rat body pose by hand against the two calibrated camera views, save it as
a keyframe, then let the auto-fit refine neighbouring frames within a
restricted neighbourhood of that pose.

Usage
-----
    python tools/pose_gui.py --frames-dir SESSION/raw --calib calib_from_corners.npz \
        [--model procedural|capsule|/path/to/RAT_MODEL.obj] [--poses keyframes.json]

--frames-dir : directory containing cam0_*.png and cam1_*.png (paired by sort
               order). Or pass --cam0-glob / --cam1-glob for other names.
--model      : 'procedural' (built-in skinned mesh, default), 'capsule'
               (tapered-capsule model, fastest), or a path to an artist OBJ.
--poses      : JSON keyframe file to load on start and save to.

Controls: sliders set position / orientation / scale / spine bend / limb tuck;
the model (green) overlays each view with the detector mask (orange) for
reference. Buttons: Fit (local) bounds the search around the current pose;
Fit (detect) fits freely from the detection; Save keyframe stores the pose for
this frame; Prev/Next carry the pose forward as a warm start.
"""
import argparse
import glob
import os
import sys

import numpy as np

try:                                   # allow running directly from a clone
    import rpimocap  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_render_fn(model: str):
    from rpimocap.model.body_model import render_silhouette
    from rpimocap.model.rat_skeleton import forward_kinematics
    if model == "capsule":
        return lambda pose, P, shp: render_silhouette(
            forward_kinematics(pose), P, image_shape=shp)
    from rpimocap.model.mesh_model import (build_rat_mesh, load_obj_mesh,
                                           render_mesh_pose_silhouette)
    if model == "procedural":
        mesh = build_rat_mesh()
    else:                                    # an OBJ path
        mesh = load_obj_mesh(model, trim_tail=True, decimate=0.85)
    return lambda pose, P, shp: render_mesh_pose_silhouette(mesh, pose, P, shp)


def _pair_frames(frames_dir, cam0_glob, cam1_glob):
    c0 = sorted(glob.glob(os.path.join(frames_dir, cam0_glob)))
    c1 = sorted(glob.glob(os.path.join(frames_dir, cam1_glob)))
    if not c0 or not c1:
        sys.exit(f"No frames matched {cam0_glob!r} / {cam1_glob!r} in {frames_dir}")
    n = min(len(c0), len(c1))
    return list(zip(c0[:n], c1[:n]))


def main():
    ap = argparse.ArgumentParser(description="Manual 3D pose fitting GUI")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--model", default="procedural")
    ap.add_argument("--poses", default=None)
    ap.add_argument("--cam0-glob", default="cam0_*.png")
    ap.add_argument("--cam1-glob", default="cam1_*.png")
    ap.add_argument("--max-width", type=int, default=680)
    ap.add_argument("--self-test", action="store_true",
                    help="construct, render, and exit (no event loop) — for CI")
    ap.add_argument("--screenshot", default=None,
                    help="fit the model on the rat, save a window PNG, and exit")
    args = ap.parse_args()

    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import (QApplication, QFileDialog, QGridLayout,
                                       QGroupBox, QHBoxLayout, QLabel,
                                       QMainWindow, QPushButton, QSlider,
                                       QVBoxLayout, QWidget)
    except ImportError:
        sys.exit("PySide6 is required:  pip install PySide6")

    from rpimocap.gui.pose_state import PoseFitterState
    from rpimocap.model.fit import TUCKED_ANGLES
    from rpimocap.model.rat_skeleton import RatPose

    cal = np.load(args.calib)
    Ps = [cal["dlt_P0"], cal["dlt_P1"]]
    frames = _pair_frames(args.frames_dir, args.cam0_glob, args.cam1_glob)
    render_fn = _build_render_fn(args.model)
    state = PoseFitterState(frames, Ps, render_fn)
    if args.poses and os.path.exists(args.poses):
        state.read_poses(args.poses)

    class FloatSlider(QWidget):
        def __init__(self, label, lo, hi, init, cb, fmt="{:.1f}", steps=1000):
            super().__init__()
            self.lo, self.hi, self.steps, self.cb, self.fmt = lo, hi, steps, cb, fmt
            self.sl = QSlider(Qt.Horizontal); self.sl.setRange(0, steps)
            self.name = QLabel(label); self.name.setMinimumWidth(70)
            self.val = QLabel(); self.val.setMinimumWidth(52)
            self.set(init)
            self.sl.valueChanged.connect(self._on)
            lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.name); lay.addWidget(self.sl); lay.addWidget(self.val)

        def value(self):
            return self.lo + (self.sl.value() / self.steps) * (self.hi - self.lo)

        def set(self, v):
            v = float(np.clip(v, self.lo, self.hi))
            self.sl.blockSignals(True)
            self.sl.setValue(int((v - self.lo) / (self.hi - self.lo) * self.steps))
            self.sl.blockSignals(False)
            self.val.setText(self.fmt.format(v))

        def _on(self):
            self.val.setText(self.fmt.format(self.value()))
            self.cb()

    class GUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("rpimocap — manual pose fitting")
            self.img0 = QLabel(); self.img1 = QLabel()
            for im in (self.img0, self.img1):
                im.setAlignment(Qt.AlignCenter)
            imgs = QVBoxLayout(); imgs.addWidget(self.img0); imgs.addWidget(self.img1)

            self.s = {}
            grid = QGridLayout(); r = 0
            def add(key, label, lo, hi, init, fmt="{:.1f}"):
                nonlocal r
                self.s[key] = FloatSlider(label, lo, hi, init, self._changed, fmt)
                grid.addWidget(self.s[key], r, 0, 1, 2); r += 1
            add("x", "X mm", -140, 140, 0); add("y", "Y mm", -215, 215, 0)
            add("z", "Z mm", 0, 220, 60)
            add("yaw", "Yaw°", -180, 180, 0); add("pitch", "Pitch°", -90, 90, 0)
            add("roll", "Roll°", -90, 90, 0)
            add("scale", "Scale", 0.5, 2.0, 1.0, "{:.2f}")
            add("spineF", "Spine F°", -60, 60, 0)
            add("spineL", "Spine L°", -60, 60, 0)
            add("tuck", "Limb tuck", 0.0, 1.0, 0.0, "{:.2f}")
            ctl = QGroupBox("Pose"); ctl.setLayout(grid)

            self.info = QLabel("—")
            nav = QHBoxLayout()
            for txt, fn in [("◀ Prev", self.prev), ("Next ▶", self.next),
                            ("Fit (local)", self.fit_local),
                            ("Fit (detect)", self.fit_detect),
                            ("Save keyframe", self.save_kf),
                            ("Write file…", self.write_file)]:
                b = QPushButton(txt); b.clicked.connect(fn); nav.addWidget(b)

            right = QVBoxLayout(); right.addWidget(ctl)
            right.addWidget(self.info); right.addLayout(nav); right.addStretch(1)
            root = QHBoxLayout(); root.addLayout(imgs, 3); root.addLayout(right, 2)
            w = QWidget(); w.setLayout(root); self.setCentralWidget(w)

            self._sync_sliders()
            self.refresh()

        # pose <-> sliders
        def _apply_sliders(self):
            ja = {k: tuple(self.s["tuck"].value() * a for a in v)
                  for k, v in TUCKED_ANGLES.items()}
            ja["SpineF"] = (0.0, np.radians(self.s["spineF"].value()), 0.0)
            ja["SpineL"] = (0.0, np.radians(self.s["spineL"].value()), 0.0)
            state.pose = RatPose(
                root_pos=np.array([self.s["x"].value(), self.s["y"].value(),
                                   self.s["z"].value()]),
                root_rot=np.radians([self.s["roll"].value(),
                                     self.s["pitch"].value(),
                                     self.s["yaw"].value()]),
                scale=self.s["scale"].value(), joint_angles=ja)

        def _sync_sliders(self):
            p = state.pose
            self.s["x"].set(p.root_pos[0]); self.s["y"].set(p.root_pos[1])
            self.s["z"].set(p.root_pos[2])
            self.s["roll"].set(np.degrees(p.root_rot[0]))
            self.s["pitch"].set(np.degrees(p.root_rot[1]))
            self.s["yaw"].set(np.degrees(p.root_rot[2]))
            self.s["scale"].set(p.scale)
            self.s["spineF"].set(np.degrees(p.joint_angles.get("SpineF", (0, 0, 0))[1]))
            self.s["spineL"].set(np.degrees(p.joint_angles.get("SpineL", (0, 0, 0))[1]))
            ref = TUCKED_ANGLES["ElbowL"][1]
            el = p.joint_angles.get("ElbowL", (0, 0, 0))[1]
            self.s["tuck"].set(np.clip(el / ref, 0, 1) if ref else 0)

        def _pix(self, rgb):
            h, w, _ = rgb.shape
            rgb = np.ascontiguousarray(rgb)
            img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(img)
            return (pix.scaledToWidth(args.max_width, Qt.SmoothTransformation)
                    if w > args.max_width else pix)

        def refresh(self):
            self.img0.setPixmap(self._pix(state.overlay(0)))
            self.img1.setPixmap(self._pix(state.overlay(1)))
            try:
                iou = state.current_iou()
            except Exception:
                iou = float("nan")
            self.info.setText(f"frame {state.idx + 1}/{len(frames)}  "
                              f"{state.frame_name()}   IoU={iou:.2f}   "
                              f"keyframes saved: {len(state.saved)}")

        def _changed(self):
            self._apply_sliders(); self.refresh()

        def _status(self, msg):
            self.info.setText(msg); QApplication.processEvents()

        def prev(self):
            state.load_frame(state.idx - 1); self._sync_sliders(); self.refresh()

        def next(self):
            state.load_frame(state.idx + 1); self._sync_sliders(); self.refresh()

        def fit_local(self):
            self._status("Fitting (local, bounded)…")
            try:
                state.fit_local(downscale=6, maxiter=80)
            except Exception as e:
                self._status(f"fit failed: {e}"); return
            self._sync_sliders(); self.refresh()

        def fit_detect(self):
            self._status("Fitting (free, from detection)…")
            try:
                state.fit_from_detection(headings=4, downscale=6, maxiter=90)
            except Exception as e:
                self._status(f"fit failed: {e}"); return
            self._sync_sliders(); self.refresh()

        def save_kf(self):
            state.save_current_pose()
            if args.poses:
                state.write_poses(args.poses)
            self.refresh()

        def write_file(self):
            path, _ = QFileDialog.getSaveFileName(self, "Write keyframes",
                                                  args.poses or "keyframes.json",
                                                  "JSON (*.json)")
            if path:
                state.write_poses(path)

    app = QApplication(sys.argv[:1])
    gui = GUI()
    gui.resize(1180, 760)
    if args.self_test:
        gui.refresh(); gui._changed(); gui.next(); gui.save_kf()
        print("pose_gui self-test OK — render, slider, nav, save all ran")
        return
    if args.screenshot:
        gui.fit_detect()                    # place the model on the rat
        gui.grab().save(args.screenshot)
        print(f"saved screenshot {args.screenshot}")
        return
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
