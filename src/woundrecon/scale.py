"""Metric scale and a canonical anatomical coordinate frame.

Monocular SfM recovers geometry only up to an unknown similarity (rotation,
translation, scale). Scale must come from an object of known size; three
measured sources are supported, in order of preference:

1. A fiducial marker (ArUco / AprilTag) of known side length, detected
   automatically and triangulated from all views in which it is seen.
2. A known length marked in the photographs: the two end points of a measured
   segment (e.g. a ruler graduation, or the pad length measured with a calliper)
   clicked in two or more photos are triangulated.
3. A known distance between two reconstructed points picked in a point-cloud viewer.

Every source is checked before it is used (finite positive length, separated
end points, enough viewing angle, small reprojection error, points in front of
the cameras). A source that fails a check is rejected with the reason; the
scale is then reported as unknown. It is never guessed.

The canonical frame: origin at the cloud centre, x along the forearm,
z pointing away from the table (towards the cameras), y completing a
right-handed frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from . import io_utils
from .camera import triangulate_dlt

log = logging.getLogger(__name__)

ARUCO_DICTS = [
    "DICT_4X4_250",
    "DICT_5X5_250",
    "DICT_6X6_250",
    "DICT_7X7_250",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h11",
]


# --------------------------------------------------------------------------- #
# 1. Fiducial markers
# --------------------------------------------------------------------------- #
def detect_markers(image_paths: list[Path], scale: float = 0.5) -> dict[str, list[dict]]:
    """Search every image for fiducial markers of the common dictionaries.

    Returns ``{image name: [{"dict", "id", "corners" (4x2, full-res displayed px)}]}``.
    """
    found: dict[str, list[dict]] = {}
    params = cv2.aruco.DetectorParameters()
    detectors = {
        name: cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)), params
        )
        for name in ARUCO_DICTS
    }
    for path in image_paths:
        gray = io_utils.read_image(path, grayscale=True)
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        hits = []
        for name, det in detectors.items():
            corners, ids, _ = det.detectMarkers(small)
            if ids is None:
                continue
            for c, i in zip(corners, ids.ravel(), strict=True):
                hits.append(
                    {"dict": name, "id": int(i), "corners": (c.reshape(4, 2) + 0.5) / scale - 0.5}
                )
        found[path.name] = hits
    return found


class ScaleError(ValueError):
    """A scale reference failed a validity check; the message says which."""


def check_scale(mm_per_unit: float) -> float:
    """Validate a physical scale factor without altering its sign or magnitude.

    Args:
        mm_per_unit: Millimetres per arbitrary SfM reconstruction length unit.

    Returns:
        The finite positive scale factor as a Python float.

    Raises:
        ScaleError: The supplied factor is non-finite, zero or negative.
    """
    if not np.isfinite(mm_per_unit) or mm_per_unit <= 0:
        raise ScaleError(f"invalid scale factor {mm_per_unit!r}")
    return float(mm_per_unit)


def _check_length(length_mm) -> float:
    """Parse a finite positive independently measured reference length in millimetres."""
    try:
        length = float(length_mm)
    except (TypeError, ValueError):
        raise ScaleError(f"known length must be a number, got {length_mm!r}") from None
    if not np.isfinite(length) or length <= 0:
        raise ScaleError(f"known length must be finite and positive, got {length_mm!r}")
    return length


def triangulate_checked(
    rays: np.ndarray,
    poses: list[tuple[np.ndarray, np.ndarray]],
    pixels: np.ndarray,
    cam,
    max_rms_px: float,
    min_angle_deg: float,
) -> tuple[np.ndarray, float]:
    """Triangulate one point from >= 2 views and check angle, depth sign and reprojection error.

    Returns the point and its reprojection RMS (native pixels); raises ScaleError on failure.
    """
    if len(rays) < 2:
        raise ScaleError("a reference point must be seen in at least two registered images")
    X = triangulate_dlt(rays, poses)
    if not np.all(np.isfinite(X)):
        raise ScaleError("triangulation failed (degenerate geometry)")
    centres = np.array([-R.T @ t for R, t in poses])
    if any((R @ X + t)[2] <= 0 for R, t in poses):
        raise ScaleError("reference point triangulates behind a camera")
    d = (X - centres) / np.linalg.norm(X - centres, axis=1, keepdims=True)
    angle = np.degrees(np.arccos(np.clip((d @ d.T).min(), -1, 1)))
    if angle < min_angle_deg:
        raise ScaleError(
            f"viewing angle {angle:.1f} deg is below {min_angle_deg} deg (too little baseline)"
        )
    proj = np.array([cam.project((R @ X + t)[None])[0] for R, t in poses])
    rms = float(np.sqrt(np.mean(np.sum((proj - pixels) ** 2, axis=1))))
    if rms > max_rms_px:
        raise ScaleError(
            f"reprojection RMS {rms:.1f} px exceeds {max_rms_px} px (inconsistent clicks or matches)"
        )
    return X, rms


def _camera_and_poses(sfm: dict):
    """Recover the native-pixel camera and registered world-to-camera poses from saved SfM arrays."""
    from .camera import Camera

    cam = Camera(sfm["K"][0, 0], sfm["K"][1, 1], sfm["K"][0, 2], sfm["K"][1, 2], sfm["dist"])
    names = [str(n) for n in sfm["names"]]
    poses = {
        n: (cv2.Rodrigues(sfm["rvecs"][i])[0], np.asarray(sfm["tvecs"][i], float).ravel())
        for i, n in enumerate(names)
        if sfm["registered"][i]
    }
    return cam, names, poses


def _displayed_to_native(points: np.ndarray, sfm: dict, i: int) -> np.ndarray:
    """Map (N, 2) full-resolution displayed pixels to the saved native sensor convention."""
    rule = str(sfm["rules"][i])
    w, h = (int(v) for v in sfm["image_size"])  # native (width, height)
    displayed_shape = (w, h) if rule in ("cw", "ccw") else (h, w)  # (rows, cols) as displayed
    return io_utils.to_native_points(points, displayed_shape, rule)


def scale_from_markers(
    detections: dict[str, list[dict]],
    sfm: dict,
    side_mm: float,
    max_rms_px: float = 3.0,
    min_angle_deg: float = 3.0,
) -> dict | None:
    """Triangulate the corners of each detected marker and compare with its known side length.

    Each marker gives one scale estimate (from its four sides and two diagonals,
    which share corners and are therefore not independent). With several markers
    the spread between them is reported; with one marker no independent repeat exists.
    """
    side = _check_length(side_mm)
    cam, names, poses = _camera_and_poses(sfm)
    obs: dict[tuple[str, int], list[tuple[np.ndarray, str]]] = {}
    for img, hits in detections.items():
        if img not in poses:
            continue
        i = names.index(img)
        for h in hits:
            obs.setdefault((h["dict"], h["id"]), []).append(
                (_displayed_to_native(h["corners"], sfm, i), img)
            )
    estimates, rejected = [], {}
    for key, views in obs.items():
        if len(views) < 2:
            continue
        try:
            C, rms = [], []
            for c in range(4):
                pix = np.array([v[0][c] for v in views])
                X, r = triangulate_checked(
                    np.array([cam.normalise(p[None])[0] for p in pix]),
                    [poses[v[1]] for v in views],
                    pix,
                    cam,
                    max_rms_px,
                    min_angle_deg,
                )
                C.append(X)
                rms.append(r)
            C = np.array(C)
            sides = [np.linalg.norm(C[k] - C[(k + 1) % 4]) for k in range(4)]
            diags = [
                np.linalg.norm(C[0] - C[2]) / np.sqrt(2),
                np.linalg.norm(C[1] - C[3]) / np.sqrt(2),
            ]
            lengths = np.array(sides + diags)
            estimates.append(
                {
                    "marker": f"{key[0]}:{key[1]}",
                    "views": len(views),
                    "mm_per_unit": check_scale(side / lengths.mean()),
                    "shape_residual": float(lengths.std() / lengths.mean()),
                    "corner_rms_px": float(max(rms)),
                }
            )
        except ScaleError as e:
            rejected[f"{key[0]}:{key[1]}"] = str(e)
    if not estimates:
        if rejected:
            log.warning("All markers rejected: %s", rejected)
        return None
    s = np.array([e["mm_per_unit"] for e in estimates])
    return {
        "method": "fiducial marker",
        "mm_per_unit": float(np.median(s)),
        "relative_sd_between_markers": float(s.std(ddof=1) / s.mean()) if len(s) > 1 else None,
        "markers": estimates,
        "rejected_markers": rejected,
    }


def scale_from_annotation(
    annotation: dict, sfm: dict, max_rms_px: float = 4.0, min_angle_deg: float = 3.0
) -> dict:
    """Scale from a known length whose two end points were clicked in >= 2 photographs.

    ``annotation`` = {"length_mm": L, "description": "...", "observations": [
        {"image": "IMG_2951.png", "endpoint_a": [u, v], "endpoint_b": [u, v]}, ...]}
    with pixel coordinates in the photograph as displayed (as opened in any image viewer).
    """
    length = _check_length(annotation.get("length_mm"))
    cam, names, poses = _camera_and_poses(sfm)
    rows = []
    for o in annotation.get("observations", []):
        if o["image"] not in poses:
            raise ScaleError(f"{o['image']} is not a registered image")
        i = names.index(o["image"])
        pts = np.array([o["endpoint_a"], o["endpoint_b"]], float)
        if pts.shape != (2, 2) or not np.all(np.isfinite(pts)):
            raise ScaleError(f"{o['image']}: end points must be two finite (u, v) pairs")
        rows.append((_displayed_to_native(pts, sfm, i), o["image"]))
    ends, rms = [], []
    for e in range(2):
        pix = np.array([r[0][e] for r in rows])
        X, r = triangulate_checked(
            np.array([cam.normalise(p[None])[0] for p in pix]) if len(pix) else pix,
            [poses[r[1]] for r in rows],
            pix,
            cam,
            max_rms_px,
            min_angle_deg,
        )
        ends.append(X)
        rms.append(r)
    dist = float(np.linalg.norm(ends[0] - ends[1]))
    if dist <= 0 or not np.isfinite(dist):
        raise ScaleError("the two end points coincide")
    return {
        "method": "known length marked in photographs",
        "description": annotation.get("description", ""),
        "mm_per_unit": check_scale(length / dist),
        "views": len(rows),
        "endpoint_rms_px": rms,
        "relative_sd": None,
    }


def scale_from_reference_points(points, length_mm, min_separation: float = 1e-9) -> dict:
    """Scale from a known distance between two reconstructed points (SfM coordinates)."""
    length = _check_length(length_mm)
    pts = np.asarray(points, float)
    if pts.shape != (2, 3) or not np.all(np.isfinite(pts)):
        raise ScaleError("reference_points must be two finite (x, y, z) points")
    dist = float(np.linalg.norm(pts[0] - pts[1]))
    if dist <= min_separation:
        raise ScaleError("the two reference points coincide")
    return {
        "method": "known distance between two reconstructed points",
        "mm_per_unit": check_scale(length / dist),
        "relative_sd": None,
    }


# --------------------------------------------------------------------------- #
# Canonical anatomical frame
# --------------------------------------------------------------------------- #
def canonical_frame(
    points: np.ndarray, camera_centres: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R (rows = new axes) and origin c of the anatomical frame.

    x: principal (long) axis of the arm; z: towards the cameras (away from the
    table); y = z x x.
    """
    c = np.median(points, axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    x = vt[0]
    up = np.mean(camera_centres - c, axis=0)
    z = up - (up @ x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z]), c
