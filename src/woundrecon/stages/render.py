"""Stage 7: rendered views (screenshots) of the 3D outputs.

The mesh uses perspective-correct triangle rasterization with a depth buffer.
This preserves real holes without introducing sample-spacing gaps. No OpenGL
context is needed.

Outputs (``outputs/renders/``):
    01_textured_top.png        textured mesh seen from above (phantom surface view)
    02_geometry_oblique.png    shaded geometry without colour, raking light (relief)
    03_wound_depth.png         wound depth map (scale bar only when the scale is metric)
    04_sparse_and_cameras.png  sparse cloud and camera frusta
    05_textured_oblique.png    textured mesh, oblique view
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from .. import io_utils
from .. import render as rd
from ..config import stage_dir

log = logging.getLogger(__name__)


def _save(path, rgb: np.ndarray) -> None:
    """Save rendered RGB pixels through OpenCV's BGR file-writing convention."""
    cv2.imwrite(str(path), rgb[..., ::-1])


def _scale_bar(
    cam: rd.OrbitCamera, depth_buffer: np.ndarray, metric: bool, mm: float
) -> tuple[float, str] | None:
    """Scale bar of ``mm`` millimetres at the median rendered depth (top views).

    Returns None when the scale is unknown: a bar in arbitrary units would only
    suggest a precision that does not exist.
    """
    if not metric:
        return None
    z = np.median(depth_buffer[np.isfinite(depth_buffer)])
    return cam.K()[0, 0] * mm / z, f"{mm:g} mm"


def _frusta(cams: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Polylines of the camera frusta stored as 5-vertex pyramids (apex + 4 corners)."""
    lines = []
    v = cams["points"]
    for k in range(0, len(v), 5):
        apex, c = v[k], v[k + 1 : k + 5]
        lines.append(np.vstack([c, c[:1]]))
        lines += [np.vstack([apex, ci]) for ci in c]
    return lines


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Render inspection images of the saved geometry without changing its measurements.

    The stage fails when required geometry or masks are missing, inconsistent
    or unreadable.

    Args:
        cfg: Validated configuration with resolved paths; reads aligned mesh/sparse/camera
            PLYs, scale metadata and optional wound-depth artifacts.

    Returns:
        Writes renders/ inspection PNGs; returns the generated-image inventory. Output
        coordinates are display pixels; capture calibration and numerical measurement values
        are not modified.

    Raises:
        OSError: Required inputs or output paths cannot be accessed.
    """
    out = stage_dir(cfg, "render")
    o = cfg["paths"]["output"]
    scale = io_utils.load_json(o / "scale" / "scale.json")
    metric = scale["metric"]
    tag = "" if metric else " (scale unknown)"
    mesh_path = o / "scale" / "mesh_aligned.ply"
    mesh = io_utils.read_ply_vertices(mesh_path)
    faces = io_utils.read_ply_faces(mesh_path)
    v, c, n = mesh["points"], mesh["colors"], mesh["normals"]
    files = []
    top = dict(view_dir=np.array([0, 0, -1.0]), up=np.array([0, -1.0, 0]))

    # Camera directions, light vectors and ambient levels affect presentation only;
    # they do not change saved 3D geometry or any measurement.
    # 1. Textured, top view (looking down -z, forearm axis horizontal).
    cam = rd.frame_points(v, margin=1.05, **top)
    img, zbuf = rd.rasterize_mesh(
        v, faces, rd.shade(n, c, np.array([0, 0, 1.0]), ambient=0.85), cam
    )
    _save(
        out / "01_textured_top.png",
        rd.add_caption(img, "Textured mesh, top view" + tag, _scale_bar(cam, zbuf, metric, 20.0)),
    )
    files.append("01_textured_top.png")

    # 2. Geometry only, oblique view with raking light (reveals relief and noise).
    grey = np.full_like(c, 205)
    cam = rd.frame_points(
        v, view_dir=np.array([0.35, 0.6, -0.72]), up=np.array([0, 0, 1.0]), margin=1.0
    )
    img, _ = rd.rasterize_mesh(
        v, faces, rd.shade(n, grey, np.array([-0.6, -0.3, 0.75]), ambient=0.2), cam
    )
    _save(
        out / "02_geometry_oblique.png",
        rd.add_caption(img, "Geometry only, oblique view, raking light" + tag),
    )
    files.append("02_geometry_oblique.png")

    # 3. Wound depth (top view, cropped to the wound and its periwound ring).
    wpath = o / "measure" / "wound_depth.ply"
    if wpath.exists():
        wd = io_utils.read_ply_vertices(wpath)
        masks = np.load(o / "measure" / "wound_masks.npz")
        meas = io_utils.load_json(o / "measure" / "measurements.json")
        roi = masks["wound"] | masks["ring"]
        wv, wc, wn = wd["points"], wd["colors"], n
        cam = rd.frame_points(wd["points"][roi], margin=1.2, **top)
        img, zbuf = rd.rasterize_mesh(
            wv, faces, rd.shade(wn, wc, np.array([0, 0, 1.0]), ambient=0.8), cam
        )
        rel = meas["scale_free"]["depth_p99"]
        if metric:
            label = (
                f"Experimental reference depth: 0 (purple) to {meas['absolute']['depth_p99']:.1f} mm "
                f"(yellow, 99th percentile)"
            )
        else:
            label = (
                f"Experimental reference depth: 0 (purple) to {rel:.1%} of wound length "
                f"(yellow, 99th percentile)"
            )
        img = rd.add_caption(img, label, _scale_bar(cam, zbuf, metric, 10.0))
        _save(out / "03_wound_depth.png", img)
        files.append("03_wound_depth.png")

    # 4. Sparse cloud + camera frusta.
    sp = io_utils.read_ply_vertices(o / "scale" / "sparse_aligned.ply")
    cams = io_utils.read_ply_vertices(o / "scale" / "cameras_aligned.ply")
    pts = np.vstack([sp["points"], cams["points"]])
    centre = 0.5 * (np.median(sp["points"], axis=0) + cams["points"].mean(axis=0))
    cam = rd.frame_points(
        pts,
        view_dir=np.array([0.2, 0.75, -0.6]),
        up=np.array([0, 0, 1.0]),
        margin=1.3,
        centre=centre,
    )
    img, _ = rd.splat(sp["points"], sp["colors"], cam, point_px=3)
    img = rd.draw_polylines(img, _frusta(cams), cam)
    _save(
        out / "04_sparse_and_cameras.png",
        rd.add_caption(
            img,
            f"Sparse SfM points ({len(sp['points']):,}, track length >= 3) and {len(cams['points']) // 5} cameras",
        ),
    )
    files.append("04_sparse_and_cameras.png")

    # 5. Textured, oblique.
    cam = rd.frame_points(
        v, view_dir=np.array([-0.3, 0.65, -0.7]), up=np.array([0, 0, 1.0]), margin=1.0
    )
    img, _ = rd.rasterize_mesh(
        v, faces, rd.shade(n, c, np.array([0.2, -0.4, 0.9]), ambient=0.75), cam
    )
    _save(out / "05_textured_oblique.png", rd.add_caption(img, "Textured mesh, oblique view" + tag))
    files.append("05_textured_oblique.png")

    log.info("Rendered %d views", len(files))
    return {"renders": files}
