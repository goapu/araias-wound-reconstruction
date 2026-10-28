"""Stage 5: anatomical frame and (if a reference exists) metric scale.

All 3D outputs are expressed in an anatomical frame (x along the forearm,
z away from the table). The wound images are searched for fiducial markers;
a marker of known size, or a known distance between two reconstructed points,
makes the outputs metric (millimetres). Without either, the scale is reported
as unknown and the outputs keep the arbitrary reconstruction unit.

Outputs (``outputs/scale/``):
    scale.json                       units, scale source, marker search, frame transform
    sparse_aligned.ply, dense_aligned_thinned.ply,
    mesh_aligned.ply, cameras_aligned.ply   outputs in the anatomical frame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .. import io_utils
from .. import scale as sc
from ..camera import camera_centre
from ..config import stage_dir
from ..dense import voxel_downsample

log = logging.getLogger(__name__)

UNKNOWN_UNIT = "reconstruction units"


def _transform(points: np.ndarray, s: float, R: np.ndarray, c: np.ndarray) -> np.ndarray:
    """SfM coordinates -> anatomical frame, multiplied by the unit factor s: X' = s * R (X - c)."""
    return s * (points - c) @ R.T


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Align all reconstructions anatomically and apply a verified physical scale when available.

    The stage fails when required geometry/images are unreadable. Invalid
    reference sources are recorded and skipped; missing scale retains arbitrary units.

    Args:
        cfg: Validated configuration with resolved paths; reads SfM poses,
            dense/sparse/surface PLYs and configured reference evidence.

    Returns:
        Writes scale/aligned PLYs and scale.json; returns unit/frame/reference evidence.
        Aligned vertices use the anatomical frame, in millimetres only with verified scale
        evidence; otherwise units remain arbitrary.

    Raises:
        OSError: Required inputs or output paths cannot be accessed.
    """
    out = stage_dir(cfg, "scale")
    o = cfg["paths"]["output"]
    c = cfg["scale"]
    sfm = dict(np.load(o / "sfm" / "sfm.npz", allow_pickle=True))
    dense = io_utils.read_ply_vertices(o / "dense" / "dense.ply")
    centres = np.array(
        [
            camera_centre(cv2.Rodrigues(sfm["rvecs"][i])[0], sfm["tvecs"][i])
            for i in np.flatnonzero(sfm["registered"])
        ]
    )
    R, origin = sc.canonical_frame(dense["points"], centres)

    # Scale sources, in order of preference. A source that fails its checks is
    # logged with the reason and the next one is tried; nothing is guessed.
    detections = sc.detect_markers(io_utils.list_images(cfg["paths"]["images"]))
    n_hits = sum(len(v) for v in detections.values())
    log.info(
        "Fiducial marker search: %d detections in %d images (%d dictionaries)",
        n_hits,
        len(detections),
        len(sc.ARUCO_DICTS),
    )
    result, rejected = None, {}
    sources = []
    if n_hits and c.get("marker_side_mm") is not None:
        sources.append(
            ("marker", lambda: sc.scale_from_markers(detections, sfm, c["marker_side_mm"]))
        )
    if c.get("reference_annotation"):
        ann_path = Path(c["reference_annotation"])
        ann_path = ann_path if ann_path.is_absolute() else cfg["paths"]["data_root"] / ann_path
        sources.append(
            ("annotation", lambda: sc.scale_from_annotation(io_utils.load_json(ann_path), sfm))
        )
    if c.get("known_length_mm") is not None or c.get("reference_points") is not None:
        sources.append(
            (
                "reference_points",
                lambda: sc.scale_from_reference_points(
                    c.get("reference_points"), c.get("known_length_mm")
                ),
            )
        )
    for name, source in sources:
        try:
            result = source()
        except sc.ScaleError as e:
            rejected[name] = str(e)
            log.error("Scale source '%s' rejected: %s", name, e)
            continue
        if result is not None:
            break

    if result is None:
        metric, s, units, method = (
            False,
            1.0,
            UNKNOWN_UNIT,
            "none (no valid reference of known size)",
        )
        log.warning("Metric scale UNKNOWN: outputs keep the reconstruction unit")
    else:
        metric, s, units, method = True, result["mm_per_unit"], "mm", result["method"]
        log.info("Metric scale: %.4f mm per reconstruction unit (%s)", s, method)

    # Aligned (and, if possible, metric) copies of all outputs ---------------------------
    comment = [f"anatomical frame (x along forearm, z away from table); units: {units}"]
    sp = io_utils.read_ply_vertices(o / "sfm" / "sparse.ply")
    io_utils.write_ply(
        out / "sparse_aligned.ply",
        _transform(sp["points"], s, R, origin),
        sp["colors"],
        comments=comment,
    )
    # Dense cloud thinned to the mesh voxel: the same detail as the mesh at a fraction
    # of the size (the full cloud stays in outputs/dense/dense.ply, SfM frame).
    voxel = io_utils.load_json(o / "mesh" / "mesh_stats.json")["voxel"]
    p_, c_, n_, _ = voxel_downsample(dense["points"], dense["colors"], dense["normals"], voxel)
    io_utils.write_ply(
        out / "dense_aligned_thinned.ply",
        _transform(p_, s, R, origin),
        c_,
        normals=n_ @ R.T,
        comments=comment + ["dense cloud thinned to the mesh voxel size"],
    )
    mesh_path = o / "mesh" / "mesh.ply"
    if mesh_path.exists():
        m = io_utils.read_ply_vertices(mesh_path)
        io_utils.write_ply(
            out / "mesh_aligned.ply",
            _transform(m["points"], s, R, origin),
            m["colors"],
            normals=m["normals"] @ R.T,
            faces=io_utils.read_ply_faces(mesh_path),
            comments=comment,
        )
    cams_path = o / "sfm" / "cameras.ply"
    cams = io_utils.read_ply_vertices(cams_path)
    io_utils.write_ply(
        out / "cameras_aligned.ply",
        _transform(cams["points"], s, R, origin),
        cams["colors"],
        faces=io_utils.read_ply_faces(cams_path),
        comments=comment,
    )

    record = {
        "metric": metric,
        "units": units,
        "unit_factor": s,
        "method": method,
        "scale_result": result,
        "rejected_sources": rejected,
        "marker_search": {
            "dictionaries": sc.ARUCO_DICTS,
            "detections": n_hits,
            "images_searched": len(detections),
        },
        "frame_rotation": R,
        "frame_origin_sfm": origin,
    }
    io_utils.save_json(out / "scale.json", record)
    return record
