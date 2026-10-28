"""Intrinsic camera calibration from checkerboard images.

Method overview
---------------
1. Corner detection with the sector-based detector (``findChessboardCornersSB``)
   on a downscaled copy, refined with ``cornerSubPix`` at full resolution.
2. All corners are mapped into the native sensor frame (see ``io_utils``).
   Because the landscape/portrait relation is unknown (EXIF removed), the
   landscape views are tested under both possible rotations.
3. Several distortion models are fitted, each with a rigid planar board and
   with the release-object method of Strobl & Hirzinger (2011), which also
   refines shared board points to relax the planar-target assumption. A lower
   residual does not establish the physical cause of that improvement.
   The model is chosen on *held-out* (leave-one-image-out) reprojection error,
   not on the training error, which always decreases when parameters are added.
4. Parameter uncertainty is reported twice: from the covariance of the final
   least-squares fit (OpenCV) and from a bootstrap over images, which also
   captures view-selection effects.
5. Diagnostics that answer "are these images sufficient?": sensor coverage,
   pose diversity, per-view errors, distortion uncertainty outside the covered
   region, and focal length versus distance (focus breathing).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import io_utils

log = logging.getLogger(__name__)

# Distortion models compared during model selection (OpenCV flag sets).
MODEL_FLAGS: dict[str, int] = {
    "k1": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
    "k1k2": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    "k1k2k3": cv2.CALIB_ZERO_TANGENT_DIST,
    "opencv5": 0,
    "rational": cv2.CALIB_RATIONAL_MODEL,
}
MODEL_DESCRIPTION = {
    "k1": "radial k1",
    "k1k2": "radial k1, k2",
    "k1k2k3": "radial k1, k2, k3",
    "opencv5": "radial k1-k3 + tangential p1, p2",
    "rational": "rational k1-k6 + tangential p1, p2",
}

_TERM = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10)
DIST_NAMES = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Detections:
    """Checkerboard corners of all usable views, in the *displayed* image frame."""

    names: list[str] = field(default_factory=list)
    corners: list[np.ndarray] = field(default_factory=list)  # (N, 2) per view
    shapes: list[tuple[int, int]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    def native(self, portrait_rule: str, landscape_rule: str) -> list[np.ndarray]:
        """Corners mapped into the native sensor frame under an orientation hypothesis."""
        return [
            io_utils.to_native_points(
                c, s, io_utils.native_transform(s, portrait_rule, landscape_rule)
            ).astype(np.float32)
            for c, s in zip(self.corners, self.shapes, strict=True)
        ]

    def portrait_mask(self) -> np.ndarray:
        """Return a (V,) boolean mask for displayed images taller than they are wide."""
        return np.array([io_utils.is_portrait(s) for s in self.shapes])


@dataclass
class CalibResult:
    """Output of one calibration run."""

    model: str
    release_object: bool
    K: np.ndarray
    dist: np.ndarray
    rms: float
    per_view_rms: np.ndarray
    std_intrinsics: np.ndarray  # covariance-based 1-sigma, OpenCV ordering
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    board: np.ndarray  # (N, 3) board points used (refined if release_object)

    @property
    def dist_names(self) -> list[str]:
        """Return the OpenCV coefficient names matching the stored distortion vector."""
        return DIST_NAMES[: len(self.dist)]

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-friendly summary."""
        std = self.std_intrinsics.ravel()
        n = len(self.dist)
        return {
            "model": self.model,
            "release_object": self.release_object,
            "K": self.K,
            "dist": dict(zip(self.dist_names, self.dist, strict=True)),
            "rms_px": self.rms,
            "covariance_std": {
                "fx": std[0],
                "fy": std[1],
                "cx": std[2],
                "cy": std[3],
                **{f"{k}": s for k, s in zip(self.dist_names, std[4 : 4 + n], strict=True)},
            },
        }


# --------------------------------------------------------------------------- #
# Corner detection
# --------------------------------------------------------------------------- #
def board_points(pattern: tuple[int, int], square: float) -> np.ndarray:
    """3D board coordinates (Z = 0) in OpenCV corner order (row-major, cols first)."""
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return obj


def detect_corners(
    gray: np.ndarray, pattern: tuple[int, int], detect_scale: float, subpix_window: int
) -> np.ndarray | None:
    """Detect inner checkerboard corners with sub-pixel accuracy.

    Detection runs on a downscaled image for speed and robustness; corners are
    then mapped back (pixel-centre convention) and refined on the full image.

    Returns:
        (N, 2) float32 corners in full-resolution pixel coordinates, or None.
    """
    small = cv2.resize(gray, None, fx=detect_scale, fy=detect_scale, interpolation=cv2.INTER_AREA)
    # Detector cascade. Note: CALIB_CB_NORMALIZE_IMAGE (histogram equalisation)
    # was found to *break* the sector-based detector on these bright, low-contrast
    # table scenes, so it is deliberately not used with it.
    found, corners = False, None
    for flags in (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY, 0):
        found, corners = cv2.findChessboardCornersSB(small, pattern, flags=flags)
        if found:
            break
    if not found:
        found, corners = cv2.findChessboardCorners(
            small, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    if not found:
        return None
    corners = (corners.reshape(-1, 2) + 0.5) / detect_scale - 0.5
    corners = corners.astype(np.float32).reshape(-1, 1, 2)
    win = (subpix_window, subpix_window)
    cv2.cornerSubPix(
        gray, corners, win, (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-4)
    )
    return corners.reshape(-1, 2)


def detect_all(
    image_paths: list[Path],
    pattern: tuple[int, int],
    detect_scale: float,
    subpix_window: int,
    overlay_dir: Path | None = None,
) -> Detections:
    """Run corner detection on every checkerboard image; optionally save overlays."""
    det = Detections()
    for path in image_paths:
        img = io_utils.read_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = detect_corners(gray, pattern, detect_scale, subpix_window)
        if corners is None:
            log.warning("Corners not found: %s", path.name)
            det.rejected.append(path.name)
            continue
        det.names.append(path.name)
        det.corners.append(corners)
        det.shapes.append(gray.shape[:2])
        if overlay_dir is not None:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern, corners.reshape(-1, 1, 2), True)
            vis = cv2.resize(vis, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            cv2.imwrite(
                str(overlay_dir / f"{path.stem}_corners.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
        log.info("Corners found: %s", path.name)
    return det


# --------------------------------------------------------------------------- #
# Core calibration
# --------------------------------------------------------------------------- #
def calibrate(
    obj: np.ndarray,
    img_pts: list[np.ndarray],
    image_size: tuple[int, int],
    model: str,
    release_object: bool = False,
    extra_flags: int = 0,
    K0: np.ndarray | None = None,
    dist0: np.ndarray | None = None,
) -> CalibResult:
    """Calibrate one distortion model on a set of views.

    Args:
        obj: (N, 3) nominal board points shared by all views.
        img_pts: list of (N, 2) corners in the native frame.
        image_size: (width, height) of the native frame.
        model: key of ``MODEL_FLAGS``.
        release_object: refine the board points as well (Strobl & Hirzinger);
            OpenCV keeps three points fixed (first corner, top-right corner,
            last corner) to remove the gauge freedom.
        extra_flags: additional OpenCV flags (e.g. fixed principal point).
        K0, dist0: initial guess, used together with ``CALIB_USE_INTRINSIC_GUESS``.
    """
    flags = MODEL_FLAGS[model] | extra_flags
    n_dist = 8 if model == "rational" else 5
    K_init, d_init = None, None
    if K0 is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        K_init = K0.copy()
        d_init = (
            np.zeros(n_dist)
            if dist0 is None
            else np.resize(np.ravel(dist0), n_dist).astype(np.float64)
        )
    objs = [obj] * len(img_pts)

    if release_object:
        fixed_point = int(np.argmax(obj[:, 0] - obj[:, 1]))  # top-right inner corner
        (rms, K, dist, rvecs, tvecs, new_obj, std_in, _, _, per_view) = (
            cv2.calibrateCameraROExtended(
                objs, img_pts, image_size, fixed_point, K_init, d_init, flags=flags, criteria=_TERM
            )
        )
        board = np.asarray(new_obj, np.float32).reshape(-1, 3)
    else:
        rms, K, dist, rvecs, tvecs, std_in, _, per_view = cv2.calibrateCameraExtended(
            objs, img_pts, image_size, K_init, d_init, flags=flags, criteria=_TERM
        )
        board = obj.astype(np.float32)

    return CalibResult(
        model=model,
        release_object=release_object,
        K=K,
        dist=dist.ravel()[:n_dist],
        rms=float(rms),
        per_view_rms=np.ravel(per_view),
        std_intrinsics=np.ravel(std_in),
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        board=board,
    )


def heldout_rms(board: np.ndarray, pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> float:
    """RMS reprojection error of a held-out view with the intrinsics fixed.

    The view's pose is estimated from the even-numbered corners and the error is
    measured on the odd-numbered corners only. Scoring the same corners that
    fixed the pose would let the pose absorb part of the error and flatter the
    model. With 9 corners per row the two sets alternate like a checkerboard and
    both cover the whole board.
    """
    fit = np.arange(len(pts)) % 2 == 0
    _, rvec, tvec = cv2.solvePnP(board[fit], pts[fit], K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(board[~fit], rvec, tvec, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts[~fit]) ** 2, axis=1))))


def leave_one_out(
    obj: np.ndarray,
    img_pts: list[np.ndarray],
    image_size: tuple[int, int],
    model: str,
    release_object: bool = False,
) -> np.ndarray:
    """Leave-one-image-out cross-validation; returns the held-out RMS per view.

    With release_object, the held-out view is evaluated against the board
    geometry estimated from the *training* views only.
    """
    errors = []
    for i in range(len(img_pts)):
        train = img_pts[:i] + img_pts[i + 1 :]
        res = calibrate(obj, train, image_size, model, release_object)
        errors.append(heldout_rms(res.board, img_pts[i], res.K, res.dist))
    return np.array(errors)


def bootstrap(
    obj: np.ndarray,
    img_pts: list[np.ndarray],
    image_size: tuple[int, int],
    model: str,
    release_object: bool,
    n_samples: int,
    seed: int,
    probes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Bootstrap over views: resample images with replacement and recalibrate.

    Args:
        probes: (P, 2) pixel positions at which the distortion displacement is
            recorded for every sample. Probes outside the region covered by
            the checkerboard reveal how poorly distortion is extrapolated there.

    Returns:
        ``params`` (S, 4 + n_dist) with fx, fy, cx, cy, distortion; and
        ``probe_displacement_px`` (S, P).
    """
    rng = np.random.default_rng(seed)
    params, disp = [], []
    for _ in range(n_samples):
        idx = rng.integers(0, len(img_pts), len(img_pts))
        if len(set(idx.tolist())) < 6:  # degenerate resample, skip
            continue
        res = calibrate(obj, [img_pts[i] for i in idx], image_size, model, release_object)
        params.append(
            np.concatenate([[res.K[0, 0], res.K[1, 1], res.K[0, 2], res.K[1, 2]], res.dist])
        )
        disp.append(distortion_displacement(probes, res.K, res.dist))
    return {"params": np.array(params), "probe_displacement_px": np.array(disp)}


# --------------------------------------------------------------------------- #
# Orientation hypothesis test
# --------------------------------------------------------------------------- #
def test_landscape_rotation(
    det: Detections, obj: np.ndarray, image_size: tuple[int, int], model: str
) -> dict[str, Any]:
    """Decide how landscape views relate to portrait views in the native frame.

    The camera is calibrated on portrait views only (portrait -> native by a
    clockwise rotation). Each landscape view is then evaluated, with intrinsics
    fixed, under both remaining possibilities ("none" or "180"). Only the
    correct relation is consistent with the principal point and distortion
    centre found on the portrait views. (Rotating portrait views the other way
    is equivalent to swapping these two answers, so this single test is complete.)
    """
    portrait = det.portrait_mask()
    pts_p = [p for p, m in zip(det.native("cw", "none"), portrait, strict=True) if m]
    ref = calibrate(obj, pts_p, image_size, model)
    result = {"portrait_only_rms_px": ref.rms, "views": {}}
    for rule in ("none", "180"):
        pts_l = [p for p, m in zip(det.native("cw", rule), portrait, strict=True) if not m]
        names = [n for n, m in zip(det.names, portrait, strict=True) if not m]
        for name, p in zip(names, pts_l, strict=True):
            result["views"].setdefault(name, {})[rule] = heldout_rms(obj, p, ref.K, ref.dist)
    best = {name: min(v, key=v.get) for name, v in result["views"].items()}
    result["best_rule_per_view"] = best
    return result


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def distortion_displacement(pixels: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Magnitude (px) of the lens-distortion shift at given (distorted) pixel positions."""
    pts = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
    ideal = cv2.undistortImagePoints(pts, K, dist, _TERM).reshape(-1, 2)
    return np.linalg.norm(pts.reshape(-1, 2) - ideal, axis=1)


def coverage(
    img_pts: list[np.ndarray], image_size: tuple[int, int], grid: tuple[int, int] = (16, 12)
) -> dict[str, Any]:
    """Fraction of sensor cells that contain at least one corner, plus radial reach.

    Radial reach is the corner distance from the image centre divided by the
    half-diagonal: 1.0 would mean corners reach the extreme image corners.
    """
    w, h = image_size
    all_pts = np.concatenate(img_pts)
    gx = np.clip((all_pts[:, 0] / w * grid[0]).astype(int), 0, grid[0] - 1)
    gy = np.clip((all_pts[:, 1] / h * grid[1]).astype(int), 0, grid[1] - 1)
    counts = np.zeros(grid[::-1], int)
    np.add.at(counts, (gy, gx), 1)
    centre = np.array([w / 2, h / 2])
    radius = np.linalg.norm(all_pts - centre, axis=1) / np.linalg.norm(centre)
    return {
        "cell_counts": counts,
        "fraction_cells_covered": float((counts > 0).mean()),
        "max_radial_reach": float(radius.max()),
        "p95_radial_reach": float(np.percentile(radius, 95)),
    }


def pose_statistics(res: CalibResult) -> dict[str, np.ndarray]:
    """Board tilt (deg) relative to the optical axis and distance (board units) per view."""
    tilts, dists = [], []
    for rvec, tvec in zip(res.rvecs, res.tvecs, strict=True):
        R, _ = cv2.Rodrigues(rvec)
        tilts.append(np.degrees(np.arccos(abs(R[2, 2]))))
        dists.append(float(np.linalg.norm(tvec)))
    return {"tilt_deg": np.array(tilts), "distance_board_units": np.array(dists)}


def per_view_focal(
    img_pts: list[np.ndarray], image_size: tuple[int, int], ref: CalibResult
) -> np.ndarray:
    """Re-estimate the focal length of each view alone (principal point,
    distortion and board fixed to the global solution, square pixels).

    A trend against board distance can reflect focus-related changes or other
    coupled effects. This conditional one-view fit does not isolate their cause.
    """
    flags = (
        cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_TANGENT_DIST
    )
    if ref.model == "rational":
        flags |= cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
    K0 = ref.K.copy()
    K0[0, 0] = K0[1, 1] = 0.5 * (ref.K[0, 0] + ref.K[1, 1])
    focals = []
    for pts in img_pts:
        res = calibrate(
            ref.board, [pts], image_size, ref.model, extra_flags=flags, K0=K0, dist0=ref.dist
        )
        focals.append(res.K[0, 0])
    return np.array(focals)


def board_deformation(nominal: np.ndarray, refined: np.ndarray) -> dict[str, float]:
    """Describe how the refined (release-object) board differs from the nominal grid.

    Returns the out-of-plane peak-to-peak deviation, the anisotropic print-scale
    error (x versus y) and the non-affine in-plane residual, all in board units.
    """
    A, *_ = np.linalg.lstsq(
        np.c_[nominal[:, :2], np.ones(len(nominal))], refined[:, :2], rcond=None
    )
    sx, sy = np.linalg.norm(A[0]), np.linalg.norm(A[1])
    in_plane = refined[:, :2] - np.c_[nominal[:, :2], np.ones(len(nominal))] @ A
    centred = refined - refined.mean(axis=0)
    normal = np.linalg.svd(centred)[2][-1]
    out_of_plane = centred @ normal
    return {
        "out_of_plane_ptp": float(np.ptp(out_of_plane)),
        "out_of_plane_rms": float(np.sqrt(np.mean(out_of_plane**2))),
        "print_scale_anisotropy_pct": float((sx / sy - 1.0) * 100.0),
        "in_plane_nonaffine_rms": float(np.sqrt(np.mean(np.sum(in_plane**2, axis=1)))),
    }


def residual_vectors(img_pts: list[np.ndarray], res: CalibResult) -> np.ndarray:
    """Per-view, per-corner reprojection residuals (observed - projected), shape (V, N, 2)."""
    out = []
    for pts, rvec, tvec in zip(img_pts, res.rvecs, res.tvecs, strict=True):
        proj, _ = cv2.projectPoints(res.board, rvec, tvec, res.K, res.dist)
        out.append(pts - proj.reshape(-1, 2))
    return np.array(out)


def physical_focal_prior(focal_mm: float, pixel_pitch_um: float) -> float:
    """Focal length in pixels implied by the lens focal length and pixel pitch (infinity focus)."""
    return focal_mm / (pixel_pitch_um * 1e-3)
