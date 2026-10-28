"""Dense multi-view stereo on the CPU with OpenCV.

Approach: "pairwise rectified stereo + multi-view consistency fusion".

For every registered reference image:
  1. choose neighbour views with a useful parallax angle and many shared
     sparse points;
  2. rectify each (reference, neighbour) pair with the SfM poses and the
     calibrated distortion; the disparity search range is derived from the
     sparse points the two views share, so the matcher never searches blindly;
  3. Semi-Global Block Matching (Hirschmuller 2008) in both directions,
     with a left-right consistency check;
  4. convert disparities to depth in the reference camera and resample them
     into the reference's undistorted image grid (z-buffer);
  5. keep a pixel only where at least two distinct stereo pairs agree on
     its depth (relative tolerance), i.e. it is seen consistently by three or
     more images.

Background (white table), specular highlights and pixels outside the
calibrated field of view are masked before matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import io_utils

log = logging.getLogger(__name__)


@dataclass
class View:
    """One registered image at the dense working scale (native frame)."""

    name: str
    image: np.ndarray  # BGR, distorted, native frame, working scale
    mask: np.ndarray  # uint8 0/255: pixels worth matching
    R: np.ndarray  # world -> camera rotation
    t: np.ndarray  # world -> camera translation
    K: np.ndarray  # intrinsics at working scale
    dist: np.ndarray  # distortion (scale-independent)

    @property
    def centre(self) -> np.ndarray:
        """Return this camera's (3,) centre in the SfM world frame and length units."""
        return -self.R.T @ self.t

    @property
    def size(self) -> tuple[int, int]:
        """Return working-image size as (width, height) in resized native-frame pixels."""
        return self.image.shape[1], self.image.shape[0]


def scale_intrinsics(K: np.ndarray, s: float) -> np.ndarray:
    """Intrinsics of an image resized by ``s`` (pixel-centre convention)."""
    Ks = K.copy().astype(float)
    Ks[0, 0] *= s
    Ks[1, 1] *= s
    Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return Ks


def foreground_mask(bgr: np.ndarray, valid_radius: float | None) -> np.ndarray:
    """Pixels worth matching: not the white table, not a specular highlight,
    textured, and inside the calibrated field of view.

    The table is bright and nearly colourless; skin, the silicone pad and the
    wound all carry colour. Highlights are bright and colourless too, so they
    are removed by the same rule.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32) / 255
    v = hsv[..., 2].astype(np.float32) / 255
    fg = (s > 0.12) | (v < 0.55)
    fg = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    # Keep the largest connected component (the arm), drop speckles.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg = np.where(labels == largest, 255, 0).astype(np.uint8)
    if valid_radius is not None:
        h, w = fg.shape
        yy, xx = np.mgrid[0:h, 0:w]
        fg[np.hypot(xx - w / 2, yy - h / 2) > valid_radius] = 0
    return fg


def select_neighbours(
    views: list[View],
    shared: np.ndarray,
    k: int,
    min_angle: float,
    max_angle: float,
    scene_centre: np.ndarray,
) -> dict[int, list[int]]:
    """Neighbour views per reference: parallax angle in range, ranked by shared points."""
    out = {}
    for r, vr in enumerate(views):
        cands = []
        for n, vn in enumerate(views):
            if n == r:
                continue
            a, b = vr.centre - scene_centre, vn.centre - scene_centre
            angle = np.degrees(
                np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
            )
            if min_angle <= angle <= max_angle and shared[r, n] > 0:
                cands.append((shared[r, n] * np.sin(np.radians(angle)), n))
        out[r] = [n for _, n in sorted(cands, reverse=True)[:k]]
    return out


@dataclass
class StereoParams:
    """Stereo settings: disparities/window in working pixels and depth tolerances as fractions."""

    block_size: int = 7
    max_disparities: int = 384
    uniqueness: int = 5
    clahe_clip: float = 3.0  # local contrast enhancement before matching (0 = off)
    lr_tolerance: float = 1.0
    depth_rel_tolerance: float = 0.01
    min_agreeing_pairs: int = 2


def _sgbm(
    left: np.ndarray, right: np.ndarray, min_disp: int, num_disp: int, p: StereoParams
) -> np.ndarray:
    """Estimate (H, W) floating disparities in working pixels from rectified grayscale images."""
    bs = p.block_size
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=bs,
        P1=8 * 3 * bs * bs,
        P2=32 * 3 * bs * bs,
        disp12MaxDiff=-1,
        uniquenessRatio=p.uniqueness,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(left, right).astype(np.float32) / 16.0


def left_right_check(
    disp_l: np.ndarray,
    disp_r: np.ndarray,
    invalid_value: float,
    tolerance: float,
    mask_l: np.ndarray | None = None,
    mask_r: np.ndarray | None = None,
) -> np.ndarray:
    """Pixels of the left disparity map that pass the left-right consistency test.

    A left pixel x with disparity d matches the right pixel x - d. It is kept only if
      * its own disparity is valid (SGBM marks invalid pixels with ``invalid_value``),
      * the matching column lies inside the right image (0 <= x - d <= W - 1),
      * the right-view disparity at that column is valid as well, and
      * both disparities agree within ``tolerance`` pixels
    (and, if given, both foreground masks are set). ``disp_r`` is the right view's
    disparity expressed with the same sign convention as ``disp_l`` (flip trick).

    Column indices are clipped only to read the arrays safely; validity is decided
    on the unclipped column, so out-of-image matches cannot borrow an edge value.
    """
    _, w = disp_l.shape
    x_right = np.round(np.arange(w)[None, :] - disp_l).astype(int)
    inside = (x_right >= 0) & (x_right <= w - 1)
    col = np.clip(x_right, 0, w - 1)  # safe read index only
    d_back = np.take_along_axis(disp_r, col, axis=1)
    valid = (disp_l > invalid_value) & inside & (d_back > invalid_value)
    valid &= np.abs(d_back - disp_l) <= tolerance
    if mask_l is not None:
        valid &= mask_l > 0
    if mask_r is not None:
        valid &= np.take_along_axis(mask_r, col, axis=1) > 0
    return valid


def pair_depth(
    ref: View, nb: View, sparse_world: np.ndarray, p: StereoParams
) -> tuple[np.ndarray, dict]:
    """Depth map of ``ref`` (undistorted grid, working scale) from one stereo pair.

    Returns:
        depth (H, W) float32 with NaN where no valid estimate, and pair statistics.
    """
    size = ref.size
    R_rel = nb.R @ ref.R.T
    t_rel = (nb.t - R_rel @ ref.t).reshape(3, 1)
    # No CALIB_ZERO_DISPARITY: the cameras converge strongly (and are rolled by up to
    # ~170 deg relative to each other), so letting the principal points differ keeps
    # the disparity range small; alpha=-1 keeps the original focal length (alpha=0
    # zoomed in by up to 25x for strongly converging pairs).
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        ref.K, ref.dist, nb.K, nb.dist, size, R_rel, t_rel, flags=0, alpha=-1
    )
    # stereoRectify chooses horizontal or vertical epipolar lines depending on the
    # dominant baseline direction. Vertical pairs are transposed so that the
    # matcher always searches along rows, and the disparity is transposed back.
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    axis = 1 if vertical else 0

    m1 = cv2.initUndistortRectifyMap(ref.K, ref.dist, R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(nb.K, nb.dist, R2, P2, size, cv2.CV_32FC1)
    left = cv2.cvtColor(cv2.remap(ref.image, *m1, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(cv2.remap(nb.image, *m2, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
    mask_l = cv2.remap(ref.mask, *m1, cv2.INTER_NEAREST)
    mask_r = cv2.remap(nb.mask, *m2, cv2.INTER_NEAREST)
    if vertical:
        left, right, mask_l, mask_r = (
            np.ascontiguousarray(a.T) for a in (left, right, mask_l, mask_r)
        )

    # Disparity range from shared sparse points: project them with P1 and P2 (both act
    # on coordinates of the first rectified camera) and take the coordinate difference.
    X_rect1 = (sparse_world @ ref.R.T + ref.t) @ R1.T
    X_rect1 = np.c_[X_rect1[X_rect1[:, 2] > 0], np.ones(int((X_rect1[:, 2] > 0).sum()))]
    x1 = X_rect1 @ P1.T
    x2 = X_rect1 @ P2.T
    d_sparse = x1[:, axis] / x1[:, 2] - x2[:, axis] / x2[:, 2]
    d_sparse = d_sparse[np.isfinite(d_sparse)]
    if len(d_sparse) < 20:
        return np.full(size[::-1], np.nan, np.float32), {"skipped": "too few shared points"}
    lo, hi = np.percentile(d_sparse, [1, 99])
    margin = 0.15 * (hi - lo) + 16
    min_disp = int(np.floor((lo - margin) / 16) * 16)
    num_disp = int(np.ceil((hi + margin - min_disp) / 16) * 16)
    if num_disp > p.max_disparities or abs(min_disp) + num_disp > left.shape[1] // 2:
        # Degenerate rectification (baseline close to the viewing direction).
        return np.full(size[::-1], np.nan, np.float32), {
            "skipped": f"disparity range {num_disp} too large"
        }
    num_disp = max(num_disp, 32)

    # Local contrast enhancement: the silicone pad and skin carry only faint texture;
    # CLAHE doubled the fraction of pixels that pass all consistency checks.
    if p.clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=p.clahe_clip, tileGridSize=(16, 16))
        left, right = clahe.apply(left), clahe.apply(right)

    disp_l = _sgbm(left, right, min_disp, num_disp, p)
    # Right-view disparity via the horizontal-flip trick (same disparity sign and range).
    disp_r = _sgbm(right[:, ::-1].copy(), left[:, ::-1].copy(), min_disp, num_disp, p)[:, ::-1]

    # Left-right consistency and masks (in the row-search orientation).
    valid = left_right_check(
        disp_l,
        disp_r,
        invalid_value=min_disp - 1,
        tolerance=p.lr_tolerance,
        mask_l=mask_l,
        mask_r=mask_r,
    )
    if vertical:
        disp_l, valid = np.ascontiguousarray(disp_l.T), np.ascontiguousarray(valid.T)
    h, w = disp_l.shape

    # Disparity -> 3D in the rectified reference frame -> reference camera frame.
    X_rect = cv2.reprojectImageTo3D(disp_l, Q)[valid]
    X_ref = X_rect @ R1  # R1^T applied row-wise
    z = X_ref[:, 2]
    ok = z > 0
    X_ref, z = X_ref[ok], z[ok]

    # Resample into the reference's undistorted grid (z-buffer keeps the nearest surface).
    u = ref.K[0, 0] * X_ref[:, 0] / z + ref.K[0, 2]
    v = ref.K[1, 1] * X_ref[:, 1] / z + ref.K[1, 2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    depth = np.full(h * w, np.inf, np.float32)
    np.minimum.at(depth, vi[inside] * w + ui[inside], z[inside].astype(np.float32))
    depth[np.isinf(depth)] = np.nan
    stats = {
        "min_disp": min_disp,
        "num_disp": num_disp,
        "valid_px": int(np.isfinite(depth).sum()),
        "baseline": float(np.linalg.norm(t_rel)),
    }
    return depth.reshape(h, w), stats


def fuse_reference(depths: list[np.ndarray], p: StereoParams) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel agreement of several pair depth maps of the same reference.

    A pixel survives when at least ``min_agreeing_pairs`` maps agree with the
    per-pixel median within ``depth_rel_tolerance``; its depth is the mean of
    the agreeing values.
    """
    stack = np.stack(depths)
    med = np.nanmedian(stack, axis=0)
    agree = np.abs(stack - med[None]) <= p.depth_rel_tolerance * med[None]
    agree &= np.isfinite(stack)
    count = agree.sum(axis=0)
    fused = np.where(agree, stack, 0).sum(axis=0) / np.maximum(count, 1)
    fused[count < p.min_agreeing_pairs] = np.nan
    return fused.astype(np.float32), count.astype(np.uint8)


def undistorted_image(view: View) -> np.ndarray:
    """Reference image resampled onto the undistorted grid used by the depth maps."""
    maps = cv2.initUndistortRectifyMap(view.K, view.dist, None, view.K, view.size, cv2.CV_32FC1)
    return cv2.remap(view.image, *maps, cv2.INTER_LINEAR)


def backproject(depth: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) and their pixel indices from a depth map on the undistorted grid."""
    h, w = depth.shape
    vv, uu = np.nonzero(np.isfinite(depth))
    z = depth[vv, uu]
    Xc = np.c_[(uu - view.K[0, 2]) / view.K[0, 0] * z, (vv - view.K[1, 2]) / view.K[1, 1] * z, z]
    Xw = (Xc - view.t) @ view.R  # R^T (Xc - t)
    return Xw, vv * w + uu


def depth_normals(depth: np.ndarray, view: View) -> np.ndarray:
    """Per-pixel camera-facing normals (world frame) from depth-map gradients."""
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    X = (xx - view.K[0, 2]) / view.K[0, 0] * depth
    Y = (yy - view.K[1, 2]) / view.K[1, 1] * depth
    P = np.dstack([X, Y, depth])
    dx = np.gradient(P, axis=1)
    dy = np.gradient(P, axis=0)
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
    n[n[..., 2] > 0] *= -1  # face the camera
    return n @ view.R  # camera -> world


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, normals: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse observed points; invalid gradient normals do not contaminate a voxel.

    Missing normals are estimated from local observed geometry, without adding
    points or bridging holes in the reconstructed surface.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = len(counts)

    def mean(a: np.ndarray) -> np.ndarray:
        """Average an attribute over the same occupied voxels used for point fusion."""
        out = np.zeros((n, a.shape[1]))
        np.add.at(out, inv, a)
        return out / counts[:, None]

    valid = np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 1e-10)
    nrm = np.zeros((n, 3))
    np.add.at(nrm, inv[valid], normals[valid])
    fused_points = mean(points)
    nrm = repair_normals(fused_points, nrm)
    linear = io_utils._srgb_eotf(colors.astype(float) / 255.0)
    encoded = 255 * io_utils._srgb_oetf(np.clip(mean(linear), 0, 1))
    return fused_points, np.rint(np.clip(encoded, 0, 255)).astype(np.uint8), nrm, counts


def repair_normals(points: np.ndarray, normals: np.ndarray, neighbours: int = 24) -> np.ndarray:
    """Normalise finite votes and estimate missing votes by local PCA.

    PCA supplies an unsigned plane normal. Its sign follows the nearest valid
    camera-facing vote. A rank-deficient neighbourhood is rejected, rather than
    exporting a fabricated direction. These normals are for inspection and
    shading; depth samples and the TSDF geometry remain the observations.
    """
    result = np.asarray(normals, float).copy()
    lengths = np.linalg.norm(result, axis=1)
    valid = np.isfinite(result).all(axis=1) & (lengths > 1e-10)
    result[valid] /= lengths[valid, None]
    missing = np.flatnonzero(~valid)
    if not len(missing):
        return result
    if len(points) < 3 or not valid.any():
        raise ValueError("Missing normals require local surface support and oriented normal votes")
    tree = cKDTree(points)
    valid_indices = np.flatnonzero(valid)
    oriented = cKDTree(points[valid])
    for start in range(0, len(missing), 10000):
        index = missing[start : start + 10000]
        _, neighbours_idx = tree.query(points[index], k=min(neighbours, len(points)))
        local = points[neighbours_idx]
        local = local - local.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", local, local)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if np.any(eigenvalues[:, 1] <= np.maximum(eigenvalues[:, 2], 1e-30) * 1e-10):
            raise ValueError("Cannot infer a normal from coincident or collinear points")
        normal = eigenvectors[:, :, 0]
        _, nearest = oriented.query(points[index])
        reference = result[valid_indices[nearest]]
        normal[np.einsum("ij,ij->i", normal, reference) < 0] *= -1
        result[index] = normal
    log.info("Estimated %d missing fused normals from local observed points", len(missing))
    return result


def save_depth_preview(depth: np.ndarray, path: Path) -> None:
    """Colour-coded depth map (near = dark) for visual inspection."""
    d = depth.copy()
    finite = np.isfinite(d)
    if finite.any():
        lo, hi = np.percentile(d[finite], [2, 98])
        d = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    vis = cv2.applyColorMap(
        (255 * (1 - np.nan_to_num(d, nan=1.0))).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    )
    vis[~finite] = 255
    cv2.imwrite(str(path), vis)


def load_view(
    path: Path,
    rule: str,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
    K_full: np.ndarray,
    dist: np.ndarray,
    valid_radius_full: float | None,
) -> View:
    """Load an image, rotate it into the native frame, resize, and build its mask."""
    img = io_utils.to_native_image(io_utils.read_image(path), rule)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    radius = None if not valid_radius_full else valid_radius_full * scale
    return View(
        path.name, img, foreground_mask(img, radius), R, t, scale_intrinsics(K_full, scale), dist
    )
