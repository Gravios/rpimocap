"""
autocalib_report.py — Self-calibration validation report (standalone HTML)
==========================================================================
Generates a single-file HTML report with embedded SVG charts:

  1. Focal length convergence curve (cost vs f)
  2. Per-frame F quality histogram
  3. Inlier counts / Sampson distances scatter
  4. Essential matrix residual before vs after refinement
  5. Quality gate checklist
  6. Final parameter table
  7. Recommended next steps

No external dependencies beyond numpy; all charts are pure SVG.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
#  Tiny SVG charting helpers                                                   #
# --------------------------------------------------------------------------- #

def _svg_line_chart(
    xs: list[float], ys: list[float],
    width: int = 480, height: int = 180,
    color: str = "#00c8ff",
    fill_color: str = "rgba(0,200,255,0.08)",
    x_label: str = "", y_label: str = "",
    vline: Optional[float] = None,
    vline_color: str = "#ff4e8a",
    title: str = "",
) -> str:
    """Generate an inline SVG line chart."""
    pad = dict(l=52, r=20, t=30, b=36)
    W = width  - pad['l'] - pad['r']
    H = height - pad['t'] - pad['b']

    xs_a = np.array(xs, dtype=float)
    ys_a = np.array(ys, dtype=float)

    xmin, xmax = xs_a.min(), xs_a.max()
    ymin, ymax = ys_a.min(), ys_a.max()
    if xmax == xmin: xmax = xmin + 1
    if ymax == ymin: ymax = ymin + 1
    ypad = (ymax - ymin) * 0.08
    ymin -= ypad; ymax += ypad

    def tx(v): return pad['l'] + (v - xmin) / (xmax - xmin) * W
    def ty(v): return pad['t'] + H - (v - ymin) / (ymax - ymin) * H

    pts = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in zip(xs_a, ys_a))

    # Filled area
    area_pts = (f"{tx(xs_a[0]):.1f},{ty(ymin):.1f} " +
                pts + f" {tx(xs_a[-1]):.1f},{ty(ymin):.1f}")

    # Axis ticks (5 on each)
    xticks = np.linspace(xmin, xmax, 5)
    yticks = np.linspace(ymin, ymax, 5)

    def fmt(v, rng):
        if rng > 1000: return f"{v:.0f}"
        if rng > 10: return f"{v:.1f}"
        return f"{v:.3f}"

    xrng = xmax - xmin; yrng = ymax - ymin

    xt_svg = "".join(
        f'<text x="{tx(v):.1f}" y="{pad["t"]+H+16}" text-anchor="middle" '
        f'font-size="9" fill="#4a6080">{fmt(v,xrng)}</text>'
        for v in xticks
    )
    yt_svg = "".join(
        f'<text x="{pad["l"]-6}" y="{ty(v)+3:.1f}" text-anchor="end" '
        f'font-size="9" fill="#4a6080">{fmt(v,yrng)}</text>'
        for v in yticks
    )
    grid_svg = "".join(
        f'<line x1="{pad["l"]}" x2="{pad["l"]+W}" y1="{ty(v):.1f}" y2="{ty(v):.1f}" '
        f'stroke="#1a2540" stroke-width="1"/>'
        for v in yticks
    )

    vline_svg = ""
    if vline is not None and xmin <= vline <= xmax:
        vx = tx(vline)
        vline_svg = (
            f'<line x1="{vx:.1f}" x2="{vx:.1f}" '
            f'y1="{pad["t"]}" y2="{pad["t"]+H}" '
            f'stroke="{vline_color}" stroke-width="1.5" stroke-dasharray="4 2"/>'
            f'<text x="{vx+4:.1f}" y="{pad["t"]+14}" '
            f'font-size="9" fill="{vline_color}">{vline:.1f}</text>'
        )

    title_svg = (f'<text x="{width//2}" y="16" text-anchor="middle" '
                 f'font-size="11" fill="#c8d8f0" font-family="Space Mono, monospace">'
                 f'{title}</text>') if title else ""

    xlabel_svg = (f'<text x="{pad["l"]+W//2}" y="{height-4}" '
                  f'text-anchor="middle" font-size="10" fill="#4a6080">{x_label}</text>'
                  ) if x_label else ""
    ylabel_svg = (f'<text x="12" y="{pad["t"]+H//2}" '
                  f'text-anchor="middle" font-size="10" fill="#4a6080" '
                  f'transform="rotate(-90 12 {pad["t"]+H//2})">{y_label}</text>'
                  ) if y_label else ""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
      style="display:block;max-width:100%;background:#0d1420;border-radius:6px">
  {title_svg}{xlabel_svg}{ylabel_svg}
  <rect x="{pad['l']}" y="{pad['t']}" width="{W}" height="{H}" fill="none" stroke="#1a2540"/>
  {grid_svg}
  <polygon points="{area_pts}" fill="{fill_color}"/>
  <polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/>
  {vline_svg}{xt_svg}{yt_svg}
</svg>"""


def _svg_scatter(
    xs: list[float], ys: list[float],
    colors: Optional[list[str]] = None,
    width: int = 400, height: int = 180,
    x_label: str = "", y_label: str = "",
    title: str = "",
) -> str:
    """Tiny scatter plot SVG."""
    pad = dict(l=52, r=20, t=30, b=36)
    W = width - pad['l'] - pad['r']
    H = height - pad['t'] - pad['b']
    xs_a, ys_a = np.array(xs, dtype=float), np.array(ys, dtype=float)
    xmin, xmax = xs_a.min(), xs_a.max()
    ymin, ymax = ys_a.min(), ys_a.max()
    if xmax == xmin: xmax += 1
    if ymax == ymin: ymax += 1
    def tx(v): return pad['l'] + (v - xmin)/(xmax - xmin)*W
    def ty(v): return pad['t'] + H - (v - ymin)/(ymax - ymin)*H

    dots = ""
    for i, (x, y) in enumerate(zip(xs_a, ys_a)):
        c = colors[i] if colors else "#00c8ff"
        dots += f'<circle cx="{tx(x):.1f}" cy="{ty(y):.1f}" r="3" fill="{c}" opacity="0.7"/>'

    xticks = np.linspace(xmin, xmax, 5)
    yticks = np.linspace(ymin, ymax, 5)
    def fmt(v, r): return f"{v:.0f}" if r > 100 else (f"{v:.1f}" if r > 1 else f"{v:.3f}")
    xrng = xmax - xmin; yrng = ymax - ymin
    xt_svg = "".join(f'<text x="{tx(v):.1f}" y="{pad["t"]+H+16}" text-anchor="middle" font-size="9" fill="#4a6080">{fmt(v,xrng)}</text>' for v in xticks)
    yt_svg = "".join(f'<text x="{pad["l"]-6}" y="{ty(v)+3:.1f}" text-anchor="end" font-size="9" fill="#4a6080">{fmt(v,yrng)}</text>' for v in yticks)
    grid_svg = "".join(f'<line x1="{pad["l"]}" x2="{pad["l"]+W}" y1="{ty(v):.1f}" y2="{ty(v):.1f}" stroke="#1a2540" stroke-width="1"/>' for v in yticks)
    title_svg = f'<text x="{width//2}" y="16" text-anchor="middle" font-size="11" fill="#c8d8f0" font-family="Space Mono, monospace">{title}</text>' if title else ""
    xlabel_svg = f'<text x="{pad["l"]+W//2}" y="{height-4}" text-anchor="middle" font-size="10" fill="#4a6080">{x_label}</text>' if x_label else ""
    ylabel_svg = f'<text x="12" y="{pad["t"]+H//2}" text-anchor="middle" font-size="10" fill="#4a6080" transform="rotate(-90 12 {pad["t"]+H//2})">{y_label}</text>' if y_label else ""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
      style="display:block;max-width:100%;background:#0d1420;border-radius:6px">
  {title_svg}{xlabel_svg}{ylabel_svg}
  <rect x="{pad['l']}" y="{pad['t']}" width="{W}" height="{H}" fill="none" stroke="#1a2540"/>
  {grid_svg}{dots}{xt_svg}{yt_svg}
</svg>"""


# --------------------------------------------------------------------------- #
#  Quality gates                                                               #
# --------------------------------------------------------------------------- #

def _quality_gate(ok: bool, label: str, detail: str = "") -> str:
    icon  = "✓" if ok else "✗"
    color = "#39ff82" if ok else "#ff4e8a"
    return (f'<div class="gate" style="border-left:3px solid {color}">'
            f'<span style="color:{color};font-family:monospace">{icon}</span>'
            f'<div><strong>{label}</strong>'
            f'{"<br><span class=dim>" + detail + "</span>" if detail else ""}'
            f'</div></div>')


# --------------------------------------------------------------------------- #
#  Main report generator                                                       #
# --------------------------------------------------------------------------- #

def generate_report(
    result,                        # KruppaResult
    estimates,                     # list[FundamentalEstimate]
    image_size: tuple[int, int],
    arena_bounds: Optional[tuple],
    output_path: str,
    video_paths: tuple[str, str] = ("", ""),
    calib_rms: Optional[float] = None,
):
    """
    Generate the HTML validation report.

    Parameters
    ----------
    result       : KruppaResult from autocalib_kruppa.run_focal_estimation
    estimates    : list of FundamentalEstimate (all, not filtered)
    image_size   : (width, height) in pixels
    arena_bounds : None or ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    output_path  : path to write the .html file
    calib_rms    : if provided, compare to a known checkerboard calibration
    """
    w, h = image_size
    diag = math.hypot(w, h)

    # ── Charts ──────────────────────────────────────────────────────────────
    chart_conv = _svg_line_chart(
        result.f_search_vals.tolist(),
        result.f_search_costs.tolist(),
        width=540, height=200,
        x_label="Focal length f (px)",
        y_label="E constraint cost",
        vline=result.f_kruppa,
        title="Focal length cost landscape (Kruppa)",
    )

    qualities = [e.quality for e in estimates]
    frame_ids = [e.frame_idx for e in estimates]
    chart_quality = _svg_scatter(
        frame_ids, qualities,
        colors=["#39ff82" if q >= 0.4 else "#ff4e8a" for q in qualities],
        width=540, height=160,
        x_label="Frame index",
        y_label="Quality score",
        title="Per-frame F estimate quality",
    )

    inliers = [e.n_inliers for e in estimates]
    sampsons = [e.mean_sampson for e in estimates]
    chart_sampson = _svg_scatter(
        inliers, sampsons,
        colors=["#00c8ff" if s < 0.5 else ("#ffa050" if s < 1.5 else "#ff4e8a")
                for s in sampsons],
        width=540, height=160,
        x_label="Inlier count",
        y_label="Mean Sampson dist (px)",
        title="Inliers vs Sampson distance",
    )

    # ── Quality gates ────────────────────────────────────────────────────────
    n_good = sum(1 for e in estimates if e.quality >= 0.4)
    n_total = len(estimates)
    f_fov_deg = math.degrees(2 * math.atan(diag / 2 / result.f_px))
    f_ratio = result.f_px / diag

    gate_n   = _quality_gate(n_good >= 20,
                              f"≥20 quality estimates ({n_good}/{n_total})",
                              "Need at least 20 high-quality F estimates for stable solve")
    gate_fov = _quality_gate(20 < f_fov_deg < 160,
                              f"Focal length plausible (FOV ≈ {f_fov_deg:.0f}°)",
                              f"f = {result.f_px:.1f}px = {f_ratio:.2f}× diagonal")
    gate_res = _quality_gate(result.mean_essential_residual < 0.05,
                              f"Essential residual {result.mean_essential_residual:.4f} < 0.05",
                              "Mean ‖2EEᵀE − tr(EEᵀ)E‖ / ‖E‖³")
    gate_met = _quality_gate(result.scale_factor is not None,
                              "Metric scale refinement applied" if result.scale_factor else
                              "Metric refinement skipped (no arena bounds)",
                              f"Scale factor {result.scale_factor:.4f}" if result.scale_factor else "")
    gate_rms = ("" if calib_rms is None else
                _quality_gate(calib_rms < 1.0,
                              f"Checkerboard RMS {calib_rms:.3f}px < 1.0px",
                              "Cross-validation against checkerboard calibration"))

    # ── Next steps ───────────────────────────────────────────────────────────
    caveats = []
    if n_good < 30:
        caveats.append("Collect more footage with strong texture contrast.")
    if result.mean_essential_residual > 0.05:
        caveats.append("High essential residual — radial distortion may be significant; "
                       "run calibrate.py with rational model to verify.")
    if result.f_metric is not None and abs(result.f_metric - result.f_kruppa) / result.f_kruppa > 0.15:
        caveats.append("Large metric correction (>15%): ensure subject visits all arena extremes "
                       "and arena bounds are accurately measured.")
    if not caveats:
        caveats.append("Calibration looks solid. Use autocalib.npz directly or as an initial guess "
                       "for checkerboard refinement.")

    steps_html = "".join(f'<li>{c}</li>' for c in caveats)

    # ── Parameter table ──────────────────────────────────────────────────────
    arena_str = "—"
    if arena_bounds is not None:
        (x0,x1),(y0,y1),(z0,z1) = arena_bounds
        arena_str = f"X [{x0},{x1}]  Y [{y0},{y1}]  Z [{z0},{z1}] mm"

    rows = [
        ("Image size",        f"{w} × {h} px"),
        ("Principal point",   f"({result.cx_px:.1f}, {result.cy_px:.1f}) px  [assumed centre]"),
        ("Focal length",      f"<strong style='color:#39ff82'>{result.f_px:.2f} px</strong>"),
        ("Focal / diagonal",  f"{result.f_px / diag:.3f}"),
        ("Diagonal FOV",      f"≈ {f_fov_deg:.1f}°"),
        ("f Kruppa",          f"{result.f_kruppa:.2f} px"),
        ("f metric",          f"{result.f_metric:.2f} px" if result.f_metric else "—"),
        ("Scale factor",      f"{result.scale_factor:.5f}" if result.scale_factor else "—"),
        ("F estimates used",  str(result.n_estimates_used)),
        ("Essential residual",f"{result.mean_essential_residual:.5f}"),
        ("Arena bounds",      arena_str),
    ]
    table_rows = "".join(
        f'<tr><td class="td-label">{k}</td><td class="td-val">{v}</td></tr>'
        for k, v in rows
    )

    # ── Build K repr ─────────────────────────────────────────────────────────
    K = result.K
    k_html = (f"<pre class='kmat'>"
              f"⎡ {K[0,0]:8.2f}  {K[0,1]:6.2f}  {K[0,2]:8.2f} ⎤\n"
              f"⎢ {K[1,0]:8.2f}  {K[1,1]:6.2f}  {K[1,2]:8.2f} ⎥\n"
              f"⎣ {K[2,0]:8.2f}  {K[2,1]:6.2f}  {K[2,2]:8.2f} ⎦"
              f"</pre>")

    # ── Full HTML ────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Self-Calibration Report</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
:root {{
  --bg:#080c12; --panel:#0d1420; --border:#1a2540;
  --accent:#00c8ff; --accent2:#ff4e8a; --accent3:#39ff82;
  --text:#c8d8f0; --dim:#4a6080;
  --mono:'Space Mono',monospace; --sans:'DM Sans',sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--sans);
     line-height:1.6;padding:32px;max-width:900px;margin:auto}}
h1{{font-family:var(--mono);font-size:20px;color:var(--accent);letter-spacing:.05em;margin-bottom:4px}}
.subtitle{{font-size:12px;color:var(--dim);font-family:var(--mono);margin-bottom:32px}}
h2{{font-family:var(--mono);font-size:13px;color:var(--accent);letter-spacing:.08em;
   text-transform:uppercase;margin:28px 0 12px;border-bottom:1px solid var(--border);padding-bottom:6px}}
.card{{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:20px;margin-bottom:16px}}
.dim{{color:var(--dim);font-size:12px}}
.gate{{display:flex;gap:12px;padding:8px 12px;background:var(--bg);border-radius:4px;
      margin-bottom:8px;align-items:flex-start;font-size:13px}}
table{{width:100%;border-collapse:collapse}}
.td-label{{color:var(--dim);font-family:var(--mono);font-size:11px;padding:5px 12px 5px 0;
          white-space:nowrap;vertical-align:top}}
.td-val{{font-family:var(--mono);font-size:12px;color:var(--text)}}
tr{{border-bottom:1px solid var(--border)}}
.kmat{{font-family:var(--mono);font-size:13px;color:var(--accent3);
      background:#060a10;padding:16px;border-radius:6px;line-height:1.8}}
.steps li{{margin-bottom:6px;font-size:13px;color:var(--text)}}
.steps{{padding-left:20px}}
.chart-pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:700px){{.chart-pair{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>reconstruct3d — self-calibration report</h1>
<div class="subtitle">
  cam0: {video_paths[0] or "—"} &nbsp;|&nbsp;
  cam1: {video_paths[1] or "—"}
</div>

<h2>Focal Length Cost Landscape</h2>
<div class="card">{chart_conv}</div>

<h2>Intrinsic Matrix K</h2>
<div class="card">{k_html}</div>

<h2>Parameter Summary</h2>
<div class="card">
  <table>{table_rows}</table>
</div>

<h2>Quality Gates</h2>
<div class="card">
  {gate_n}{gate_fov}{gate_res}{gate_met}{gate_rms}
</div>

<h2>F Estimate Quality</h2>
<div class="card">
  {chart_quality}
  <p class="dim" style="margin-top:8px">Green = quality ≥ 0.4 (used in solve), red = below threshold (excluded)</p>
</div>

<h2>Inlier Count vs Sampson Distance</h2>
<div class="card">
  {chart_sampson}
  <p class="dim" style="margin-top:8px">
    Blue = Sampson &lt; 0.5px (excellent) · Orange = 0.5–1.5px · Red = &gt;1.5px (noisy)
  </p>
</div>

<h2>Recommended Next Steps</h2>
<div class="card">
  <ul class="steps">{steps_html}</ul>
</div>

<p class="dim" style="margin-top:24px;font-family:monospace;font-size:11px">
  Generated by reconstruct3d autocalib.py
</p>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"  Report → {output_path}")
