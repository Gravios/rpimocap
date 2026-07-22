"""
export.py — Reconstruction output serialisation
================================================
Writes results to three complementary formats:

  PLY          Per-frame point clouds and meshes, compatible with MeshLab,
               ParaView, CloudCompare, Blender.  Two sub-formats:
                 - point cloud  (.ply, vertices only)
                 - mesh         (.ply, vertices + triangular faces from Marching Cubes)

  HDF5         Single archive containing the complete skeleton trajectory
               (one dataset per landmark) and optional voxel frames.
               Convenient for downstream Python / MATLAB analysis.

  Viewer JSON  Compact JSON consumed by the bundled Three.js viewer.
               Contains skeleton keypoints and edges across all frames.
               Optionally embeds a downsampled point cloud per frame for
               volume rendering in the browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

# --------------------------------------------------------------------------- #
#  PLY                                                                         #
# --------------------------------------------------------------------------- #

def _ply_header(n_vertices: int, n_faces: int = 0,
                has_color: bool = False) -> str:
    lines = [
        "ply",
        "format ascii 1.0",
        "comment reconstruct3d",
        f"element vertex {n_vertices}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    if n_faces > 0:
        lines += [
            f"element face {n_faces}",
            "property list uchar int vertex_indices",
        ]
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def write_ply_pointcloud(
    path: Union[str, Path],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
):
    """
    Write a PLY point cloud.

    Parameters
    ----------
    path   : output file path
    points : (N, 3) float
    colors : (N, 3) uint8 RGB, optional
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    if n == 0:
        return

    has_color = colors is not None
    with open(path, "w") as f:
        f.write(_ply_header(n, has_color=has_color))
        for i in range(n):
            row = f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f}"
            if has_color:
                row += f" {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}"
            f.write(row + "\n")


def write_ply_mesh(
    path: Union[str, Path],
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray] = None,
):
    """
    Write a PLY mesh.

    Parameters
    ----------
    vertices      : (V, 3) float
    faces         : (F, 3) int
    vertex_colors : (V, 3) uint8, optional
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nv, nf = len(vertices), len(faces)
    has_color = vertex_colors is not None

    with open(path, "w") as f:
        f.write(_ply_header(nv, nf, has_color=has_color))
        for i in range(nv):
            row = f"{vertices[i,0]:.4f} {vertices[i,1]:.4f} {vertices[i,2]:.4f}"
            if has_color:
                row += (f" {int(vertex_colors[i,0])}"
                        f" {int(vertex_colors[i,1])}"
                        f" {int(vertex_colors[i,2])}")
            f.write(row + "\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def write_ply_skeleton_frame(
    path: Union[str, Path],
    points_3d: list,
    color: tuple[int, int, int] = (0, 200, 255),
):
    """
    Write a frame's skeleton keypoints as a PLY point cloud with uniform colour.
    """
    pts = np.array([p.xyz for p in points_3d], dtype=np.float32) if points_3d else np.zeros((0, 3))
    colors = np.tile(np.array(color, dtype=np.uint8), (len(pts), 1)) if len(pts) else None
    write_ply_pointcloud(path, pts, colors)


# --------------------------------------------------------------------------- #
#  HDF5                                                                        #
# --------------------------------------------------------------------------- #

def write_hdf5(
    path: Union[str, Path],
    skeleton_frames: list[list],
    voxel_frames: Optional[list[Optional[np.ndarray]]] = None,
    fps: float = 30.0,
    metadata: Optional[dict] = None,
    detected_masks: Optional[dict] = None,
):
    """
    Write the full reconstruction to a single HDF5 archive.

    Structure
    ---------
    /skeleton/
        attrs: keypoint_names, fps, n_frames
        <name>/
            xyz                (n_frames, 3) float32, NaN = missing
            confidence         (n_frames,)   float32
            reprojection_error (n_frames,)   float32
            detected           (n_frames,)   bool
              True  = genuine triangulated detection in this frame
              False = gap-filled (linear interpolation, Kalman
                      prediction, or Kalman outlier corrected to
                      filter estimate)
    /voxels/
        frame_NNNNNN          (N, 3) float32 point cloud per frame

    Parameters
    ----------
    path            : output .h5 file path
    skeleton_frames : per-frame list of Point3D
    voxel_frames    : per-frame point clouds (Nx3 arrays), optional
    fps             : recording frame rate (stored as metadata)
    metadata        : additional key→value pairs written as root attributes
    detected_masks  : dict mapping keypoint name → bool array (n_frames,).
                      True = real detection, False = gap/Kalman prediction.
                      If None, derived from confidence + kalman_outlier flag.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("pip install h5py")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(skeleton_frames)

    all_names = sorted({p.name for frame in skeleton_frames for p in frame})

    with h5py.File(path, "w") as f:
        f.attrs["n_frames"] = n_frames
        f.attrs["fps"] = fps
        f.attrs["keypoint_names"] = all_names
        if metadata:
            for k, v in metadata.items():
                try:
                    f.attrs[k] = v
                except Exception:
                    f.attrs[k] = str(v)

        skel = f.create_group("skeleton")
        skel.attrs["keypoint_names"] = all_names

        for name in all_names:
            xyz = np.full((n_frames, 3), np.nan, dtype=np.float32)
            conf = np.zeros(n_frames, dtype=np.float32)
            err = np.zeros(n_frames, dtype=np.float32)
            for fi, frame in enumerate(skeleton_frames):
                for pt in frame:
                    if pt.name == name:
                        xyz[fi] = pt.xyz.astype(np.float32)
                        conf[fi] = pt.confidence
                        err[fi] = pt.reprojection_error
            # Build the 'detected' mask:
            # priority 1 — caller-supplied pre-post-processing mask
            # priority 2 — kalman_outlier attribute from Kalman filter
            # priority 3 — confidence > 0 (catches non-Kalman gap-fill)
            if detected_masks and name in detected_masks:
                det = np.asarray(detected_masks[name], dtype=bool)
                if det.shape != (n_frames,):
                    raise ValueError(
                        f"detected_masks[{name!r}] has shape {det.shape}; "
                        f"expected ({n_frames},)")
            else:
                det = np.zeros(n_frames, dtype=bool)
                for fi, frame in enumerate(skeleton_frames):
                    for pt in frame:
                        if pt.name == name:
                            outlier = getattr(pt, "kalman_outlier", False)
                            det[fi] = (pt.confidence > 0) and not outlier
            g = skel.create_group(name)
            g.create_dataset("xyz", data=xyz,
                             compression="gzip", compression_opts=4)
            g.create_dataset("confidence",         data=conf)
            g.create_dataset("reprojection_error", data=err)
            g.create_dataset("detected",           data=det)

        if voxel_frames:
            vox = f.create_group("voxels")
            for fi, pc in enumerate(voxel_frames):
                if pc is not None and len(pc) > 0:
                    vox.create_dataset(
                        f"frame_{fi:06d}",
                        data=pc.astype(np.float32),
                        compression="gzip",
                        compression_opts=4,
                    )

    print(f"  Saved HDF5 → {path}  "
          f"({n_frames} frames, {len(all_names)} landmarks)")


# --------------------------------------------------------------------------- #
#  Viewer JSON                                                                 #
# --------------------------------------------------------------------------- #

def write_viewer_json(
    path: Union[str, Path],
    skeleton_frames: list[list],
    keypoint_names: list[str],
    skeleton_edges: list[tuple[str, str]],
    fps: float = 30.0,
    voxel_frames: Optional[list[Optional[np.ndarray]]] = None,
    voxel_downsample: int = 8,
    bounds: Optional[dict] = None,
):
    """
    Write a JSON file for the bundled Three.js viewer.

    The format is designed for fast streaming loading:
      {
        "fps": 30,
        "keypoint_names": [...],
        "edges": [[a, b], ...],
        "bounds": {"x": [...], "y": [...], "z": [...]},
        "frames": [
          {
            "kp": {"nose": [x,y,z,conf], ...},
            "vol": [[x,y,z], ...]   // optional, downsampled voxels
          },
          ...
        ]
      }

    Parameters
    ----------
    voxel_downsample : keep every Nth voxel for the viewer (reduce file size)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine world bounds from skeleton
    if bounds is None:
        all_xyz = [
            pt.xyz for frame in skeleton_frames for pt in frame
            if not np.isnan(pt.xyz).any()
        ]
        if all_xyz:
            arr = np.array(all_xyz)
            pad = 50.0
            bounds = {
                "x": [float(arr[:,0].min()-pad), float(arr[:,0].max()+pad)],
                "y": [float(arr[:,1].min()-pad), float(arr[:,1].max()+pad)],
                "z": [float(arr[:,2].min()-pad), float(arr[:,2].max()+pad)],
            }
        else:
            bounds = {"x": [-500, 500], "y": [-500, 500], "z": [0, 500]}

    # Edges as index pairs for compact storage
    name_to_idx = {n: i for i, n in enumerate(keypoint_names)}
    edge_indices = []
    for a, b in skeleton_edges:
        if a in name_to_idx and b in name_to_idx:
            edge_indices.append([name_to_idx[a], name_to_idx[b]])

    frames_data = []
    for fi, frame in enumerate(skeleton_frames):
        kp_dict = {}
        for pt in frame:
            kp_dict[pt.name] = [
                round(float(pt.xyz[0]), 2),
                round(float(pt.xyz[1]), 2),
                round(float(pt.xyz[2]), 2),
                round(float(pt.confidence), 3),
            ]
        entry: dict = {"kp": kp_dict}

        if voxel_frames and fi < len(voxel_frames):
            pc = voxel_frames[fi]
            if pc is not None and len(pc) > 0:
                pc_ds = pc[::voxel_downsample]
                entry["vol"] = [
                    [round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 1)]
                    for p in pc_ds
                ]
        frames_data.append(entry)

    doc = {
        "fps": fps,
        "keypoint_names": keypoint_names,
        "edges": edge_indices,
        "bounds": bounds,
        "frames": frames_data,
    }

    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))

    size_kb = path.stat().st_size / 1024
    print(f"  Saved viewer JSON → {path}  ({size_kb:.1f} KB)")


# --------------------------------------------------------------------------- #
#  Utility: detection summary CSV                                              #
# --------------------------------------------------------------------------- #

def write_stats_csv(
    path: Union[str, Path],
    stats: dict,
):
    """Write triangulate.trajectory_stats() output to CSV."""
    import csv as _csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["landmark", "detection_rate", "n_detected",
                         "mean_repr_err_px", "max_repr_err_px"])
        for name, s in stats.items():
            writer.writerow([
                name,
                f"{s['detection_rate']:.4f}",
                s["n_detected"],
                f"{s['mean_repr_err']:.4f}",
                f"{s['max_repr_err']:.4f}",
            ])
    print(f"  Saved stats → {path}")



# =========================================================================== #
#  TIFF stack reader — streaming, memory-friendly for 50 GB+ files            #
# =========================================================================== #

# Sensor Bayer pattern -> OpenCV conversion code. SINGLE SOURCE OF TRUTH:
# every demosaic in the project must go through this map.
#
# OpenCV's Bayer constants are named for the components in the SECOND row,
# second and third columns — not the top-left 2x2 — so the code letters look
# "opposite" to the sensor pattern. For a sensor laid out
#
#     R G R G
#     G B G B      <- second row is G B G B; cols 1,2 are B,G
#     R G R G
#
# the correct call is COLOR_BayerBG2BGR. Getting this backwards silently swaps
# the R and B channels: luminance and every green-channel feature are unchanged
# (so the detector never notices), but any colour ratio inverts.
BAYER_CODES = {
    "RGGB": "COLOR_BayerBG2BGR",
    "BGGR": "COLOR_BayerRG2BGR",
    "GRBG": "COLOR_BayerGB2BGR",
    "GBRG": "COLOR_BayerGR2BGR",
}


def bayer_to_bgr(raw, bayer_pattern: str = "RGGB", lo: float = None,
                 hi: float = None):
    """Demosaic a raw Bayer frame to BGR uint8 using :data:`BAYER_CODES`.

    Non-uint8 input is percentile-normalised (0.1 / 99.9 by default) rather
    than bit-shifted. A fixed shift such as ``raw >> 2`` assumes 10-bit data in
    a uint16 container and silently WRAPS for anything wider — on 12-bit-in-16
    data (range ~4k-65k) it corrupts every pixel.
    """
    import cv2
    import numpy as _np
    raw = _np.asarray(raw)
    if raw.dtype != _np.uint8:
        if lo is None:
            lo = float(_np.percentile(raw, 0.1))
        if hi is None:
            hi = float(_np.percentile(raw, 99.9))
        raw = _np.clip((raw.astype(_np.float32) - lo) / max(hi - lo, 1.0)
                       * 255.0, 0, 255).astype(_np.uint8)
    code = getattr(cv2, BAYER_CODES.get(bayer_pattern.upper(),
                                        "COLOR_BayerBG2BGR"))
    return cv2.cvtColor(raw, code)


class TiffCapture:
    """Stream a multi-frame TIFF stack one frame at a time (cv2.VideoCapture-compatible).

    Uses ``tifffile.TiffFile`` page access so only one decompressed frame is
    ever in RAM regardless of file size.  uint16 dtype is normalised using a
    percentile range sampled from the first and last pages only.

    Parameters
    ----------
    path : str or Path — path to a .tif / .tiff file
    """

    def __init__(self, path: str, bayer_pattern: str = "RGGB") -> None:
        import cv2
        import tifffile
        self._path  = str(path)
        self._cv2   = cv2
        self._tf    = tifffile.TiffFile(str(path))
        self._pages = self._tf.pages
        self._series = self._tf.series[0] if self._tf.series else None

        page0 = self._pages[0]
        if self._series is not None:
            shape = self._series.shape
            axes  = self._series.axes.upper()
        else:
            shape = (len(self._pages), page0.shape[0], page0.shape[1])
            axes  = "ZHW"

        def _ax(cands):
            for c in cands:
                if c in axes: return axes.index(c)
            return None

        i_n = _ax("ZQTF"); i_h = _ax("YH"); i_w = _ax("XW")
        if i_n is None or i_h is None or i_w is None:
            if len(shape) == 2:
                self._n, self._h, self._w = 1, shape[0], shape[1]
            elif len(shape) == 3:
                self._n, self._h, self._w = shape[0], shape[1], shape[2]
            else:
                self._n, self._h, self._w = shape[0], shape[1], shape[2]
        else:
            self._n = shape[i_n]; self._h = shape[i_h]; self._w = shape[i_w]

        self._pages_per_frame = max(1, len(self._pages) // self._n)
        self._dtype = page0.dtype

        if self._dtype == np.uint8:
            self._lo, self._hi = 0.0, 255.0
        elif np.issubdtype(self._dtype, np.unsignedinteger):
            s = np.concatenate([self._pages[0].asarray().ravel(),
                                 self._pages[-1].asarray().ravel()])
            self._lo = float(np.percentile(s, 0.1))
            hi = float(np.percentile(s, 99.9))
            self._hi = hi if hi > self._lo else float(np.iinfo(self._dtype).max)
        else:
            self._lo, self._hi = 0.0, 1.0

        self._pos = 0; self._fps = 25.0; self._opened = True
        # Bayer demosaic pattern — set to match your sensor/libcamera config
        # IMX477 common values: RGGB, BGGR, GRBG, GBRG
        self._bayer_pattern = bayer_pattern.upper()

    def _read_raw(self, idx: int) -> np.ndarray:
        if self._series is not None:
            try:
                import zarr as _zarr
                z = _zarr.open(self._series.aszarr(), mode="r")
                return np.asarray(z[idx])
            except Exception:
                pass
        start = idx * self._pages_per_frame
        pages = [self._pages[i].asarray()
                 for i in range(start, min(start + self._pages_per_frame,
                                           len(self._pages)))]
        return pages[0] if len(pages) == 1 else np.stack(pages, axis=-1)

    # Bayer pattern → OpenCV conversion code. Module-level BAYER_CODES is the
    # single source of truth; this alias keeps the historical attribute name.
    _BAYER_CODES = BAYER_CODES

    def _to_bgr(self, raw: np.ndarray) -> np.ndarray:
        """Convert a raw frame array to BGR uint8, demosaicing if single-channel."""
        cv2 = self._cv2

        # Normalise dtype → uint8 (preserve 16-bit range for demosaic quality)
        if raw.dtype == np.uint16:
            # Demosaic in 16-bit then scale, for better colour accuracy
            if raw.ndim == 2:
                code = getattr(cv2, self._BAYER_CODES.get(
                    self._bayer_pattern, "COLOR_BayerBG2BGR"))
                bgr16 = cv2.cvtColor(raw, code)
                lo, hi = self._lo, self._hi
                bgr = np.clip((bgr16.astype(np.float32) - lo)
                              / max(hi - lo, 1.0) * 255.0,
                              0, 255).astype(np.uint8)
                return bgr
            # Multi-channel uint16 — scale then convert
            raw = np.clip((raw.astype(np.float32) - self._lo)
                          / max(self._hi - self._lo, 1.0) * 255.0,
                          0, 255).astype(np.uint8)
        elif raw.dtype != np.uint8:
            raw = np.clip((raw.astype(np.float32) - self._lo)
                          / max(self._hi - self._lo, 1.0) * 255.0,
                          0, 255).astype(np.uint8)

        # Single-channel → Bayer demosaic
        if raw.ndim == 2:
            code = getattr(cv2, self._BAYER_CODES.get(
                self._bayer_pattern, "COLOR_BayerBG2BGR"))
            return cv2.cvtColor(raw, code)
        c = raw.shape[2]
        if c == 1:
            code = getattr(cv2, self._BAYER_CODES.get(
                self._bayer_pattern, "COLOR_BayerBG2BGR"))
            return cv2.cvtColor(raw[:, :, 0], code)
        if c == 3: return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        if c == 4: return cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(raw[:, :, :3], cv2.COLOR_RGB2BGR)

    def isOpened(self) -> bool: return self._opened

    def get(self, prop_id: int) -> float:
        cv2 = self._cv2
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:  return float(self._n)
        if prop_id == cv2.CAP_PROP_FPS:          return self._fps
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:  return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._h)
        if prop_id == cv2.CAP_PROP_POS_FRAMES:   return float(self._pos)
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        cv2 = self._cv2
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._pos = max(0, min(int(value), self._n - 1)); return True
        if prop_id == cv2.CAP_PROP_FPS:
            self._fps = float(value); return True
        return False

    def read(self):
        if self._pos >= self._n: return False, None
        try:
            bgr = self._to_bgr(self._read_raw(self._pos))
        except Exception as e:
            print(f"  TiffCapture frame {self._pos} error: {e}")
            return False, None
        self._pos += 1
        return True, bgr

    def release(self) -> None:
        try: self._tf.close()
        except Exception: pass
        self._opened = False
