"""Sparse bundle adjustment with SciPy.

Minimises the reprojection error of all observations over camera poses, 3D
points and (optionally) a subset of the shared intrinsics:

    r_k = project(K, dist, R_c X_p + t_c) - x_k      (pixels, native frame)

A robust soft-L1 loss reduces the influence of large individual pixel residuals.
Finite differences use a known sparsity pattern: each observation depends on
one camera, one point and the shared intrinsics. Fixing the first camera
anchors the world pose; global scale remains arbitrary until a verified
reference is available.

Method: Triggs et al., "Bundle Adjustment -- A Modern Synthesis", LNCS 1883,
2000, pp. 298-372. Solver options follow SciPy's least_squares documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

Intrinsic parameters that may be refined (self-calibration):
    "f"   common focal scale (fx and fy are scaled together; aspect stays fixed)
    "cx", "cy", "k1", "k2", "k3"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from .camera import Camera, project_points, rotate

log = logging.getLogger(__name__)

INTRINSIC_NAMES = ("f", "cx", "cy", "k1", "k2", "k3")


@dataclass
class BAProblem:
    """Inputs of bundle adjustment. Arrays are modified in place by :func:`solve`."""

    rvecs: np.ndarray  # (C, 3) world-to-camera rotations (axis-angle)
    tvecs: np.ndarray  # (C, 3) world-to-camera translations
    points: np.ndarray  # (P, 3) world points
    cam_idx: np.ndarray  # (M,) camera of each observation
    pt_idx: np.ndarray  # (M,) point of each observation
    obs: np.ndarray  # (M, 2) observed (distorted) pixels
    camera: Camera  # shared intrinsics
    fixed_cameras: tuple[int, ...] = (0,)


def _intrinsics_vector(cam: Camera, free: tuple[str, ...]) -> np.ndarray:
    """Pack only the requested shared intrinsic parameters in solver order."""
    values = {
        "f": 1.0,
        "cx": cam.cx,
        "cy": cam.cy,
        "k1": cam.dist[0],
        "k2": cam.dist[1],
        "k3": cam.dist[4],
    }
    return np.array([values[n] for n in free], float)


def _apply_intrinsics(
    cam: Camera, free: tuple[str, ...], x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fx, fy, cx, cy) and dist with the free parameters substituted."""
    v = dict(zip(free, x, strict=True))
    s = v.get("f", 1.0)
    f4 = np.array([cam.fx * s, cam.fy * s, v.get("cx", cam.cx), v.get("cy", cam.cy)])
    dist = cam.dist.copy()
    for name, k in (("k1", 0), ("k2", 1), ("k3", 4)):
        if name in v:
            dist[k] = v[name]
    return f4, dist


def residuals(problem: BAProblem, free: tuple[str, ...], x: np.ndarray | None = None) -> np.ndarray:
    """Evaluate distorted-image reprojection residuals.

    Args:
        problem: World points (P, 3), world-to-camera poses and M native-pixel
            observations. Translations and points share arbitrary SfM length units.
        free: Names of shared intrinsics represented in the solver vector.
        x: Packed solver parameters, or None for the current problem state.

    Returns:
        (M, 2) predicted-minus-observed residuals in full-resolution native pixels.
    """
    if x is None:
        f4, dist = (
            np.array([problem.camera.fx, problem.camera.fy, problem.camera.cx, problem.camera.cy]),
            problem.camera.dist,
        )
        rv, tv, pts = problem.rvecs, problem.tvecs, problem.points
    else:
        rv, tv, pts, f4, dist = _unpack(problem, free, x)
    Xc = rotate(pts[problem.pt_idx], rv[problem.cam_idx]) + tv[problem.cam_idx]
    return project_points(Xc, f4, dist) - problem.obs


def _layout(problem: BAProblem, free: tuple[str, ...]) -> tuple[np.ndarray, int, int]:
    """Indices of the optimised cameras and the sizes of each parameter block."""
    n_cam = len(problem.rvecs)
    var_cams = np.array([c for c in range(n_cam) if c not in problem.fixed_cameras], int)
    return var_cams, 6 * len(var_cams), 3 * len(problem.points)


def _pack(problem: BAProblem, free: tuple[str, ...]) -> np.ndarray:
    """Flatten variable camera poses, world points and free intrinsics into solver order."""
    var_cams, _, _ = _layout(problem, free)
    cams = np.c_[problem.rvecs[var_cams], problem.tvecs[var_cams]].ravel()
    return np.concatenate([cams, problem.points.ravel(), _intrinsics_vector(problem.camera, free)])


def _unpack(problem: BAProblem, free: tuple[str, ...], x: np.ndarray):
    """Decode a solver vector while retaining the fixed world-to-camera poses."""
    var_cams, n_c, n_p = _layout(problem, free)
    rv, tv = problem.rvecs.copy(), problem.tvecs.copy()
    cams = x[:n_c].reshape(-1, 6)
    rv[var_cams], tv[var_cams] = cams[:, :3], cams[:, 3:]
    pts = x[n_c : n_c + n_p].reshape(-1, 3)
    f4, dist = _apply_intrinsics(problem.camera, free, x[n_c + n_p :])
    return rv, tv, pts, f4, dist


def _sparsity(problem: BAProblem, free: tuple[str, ...]) -> coo_matrix:
    """Jacobian sparsity: each residual pair touches one camera, one point, all free intrinsics."""
    var_cams, n_c, n_p = _layout(problem, free)
    m = len(problem.obs)
    obs = np.arange(m)
    cam_col = np.full(len(problem.rvecs), -1)
    cam_col[var_cams] = np.arange(len(var_cams))
    has_cam = cam_col[problem.cam_idx] >= 0

    rows, cols = [], []
    for axis in (0, 1):  # u and v residual rows
        r = 2 * obs + axis
        for s in range(6):  # camera block
            rows.append(r[has_cam])
            cols.append(6 * cam_col[problem.cam_idx[has_cam]] + s)
        for s in range(3):  # point block
            rows.append(r)
            cols.append(n_c + 3 * problem.pt_idx + s)
        for s in range(len(free)):  # shared intrinsics (dense columns)
            rows.append(r)
            cols.append(np.full(m, n_c + n_p + s))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return coo_matrix(
        (np.ones(len(rows), np.int8), (rows, cols)), shape=(2 * m, n_c + n_p + len(free))
    ).tocsr()


def solve(
    problem: BAProblem,
    free: tuple[str, ...] = (),
    loss: str = "soft_l1",
    f_scale_px: float = 2.0,
    max_nfev: int | None = 100,
) -> dict[str, float]:
    """Jointly refine camera poses, world points and optional shared intrinsics.

    Args:
        problem: Mutable BAProblem in the SfM world frame; length units are shared
            by points/translations and image observations use native pixels.
        free: Intrinsic parameter names to optimize; empty keeps calibration fixed.
        loss: SciPy loss name, applied to individual u/v pixel residuals.
        f_scale_px: Robust-loss transition scale in native sensor pixels.
        max_nfev: Evaluation budget; None delegates the budget to SciPy.

    Returns:
        Solver status, acceptance decision, before/after RMS in pixels and final
        intrinsics. Accepted results update problem in place; rejected results
        restore its initial parameters.

    Raises:
        ValueError: A free intrinsic name is unsupported or solver inputs are invalid.
    """
    for name in free:
        if name not in INTRINSIC_NAMES:
            raise ValueError(f"Unknown intrinsic parameter '{name}'")
    x0 = _pack(problem, free)
    r0 = residuals(problem, free)
    rms0 = float(np.sqrt(np.mean(np.sum(r0**2, axis=1))))

    def fun(x: np.ndarray) -> np.ndarray:
        """Return the interleaved (2M,) u/v residual vector in native sensor pixels."""
        return residuals(problem, free, x).ravel()

    # Rotations, world lengths, pixel intrinsics and distortion have different
    # scales. Jacobian-column scaling balances their local solver steps. TRF
    # with LSMR solves sparse least-squares subproblems without a dense Jacobian
    # (Triggs et al., 2000; SciPy least_squares). The remaining scale gauge is
    # not physical information. Soft-L1 changes smoothly from quadratic to
    # linear penalty around f_scale_px per u/v residual (default: 2 native px).
    result = least_squares(
        fun,
        x0,
        jac_sparsity=_sparsity(problem, free),
        x_scale="jac",
        loss=loss,
        f_scale=f_scale_px,
        method="trf",
        tr_solver="lsmr",
        max_nfev=max_nfev,
        verbose=0,
    )
    f_before = problem.camera.fx
    _write_back(problem, free, result.x)
    r1 = residuals(problem, free)
    rms1 = float(np.sqrt(np.mean(np.sum(r1**2, axis=1))))
    problems = acceptance_problems(problem, result.x, rms0, rms1, problem.camera.fx / f_before)
    if problems:  # keep the state from before this adjustment
        _write_back(problem, free, x0)
        log.warning("BA result rejected (%s); previous state kept", "; ".join(problems))
    # SciPy status: 0 = evaluation cap reached, 1-4 = a convergence tolerance was met.
    stopped = "evaluation cap reached" if result.status == 0 else "converged (tolerance met)"
    log.info(
        "BA: %d obs, %d points, %d cameras | RMS %.3f -> %.3f px | nfev %d/%s | %s",
        len(problem.obs),
        len(problem.points),
        len(problem.rvecs),
        rms0,
        rms1,
        result.nfev,
        max_nfev,
        stopped,
    )
    cam = problem.camera
    return {
        "rms_before_px": rms0,
        "rms_after_px": rms1,
        "nfev": int(result.nfev),
        "max_nfev": max_nfev,
        "status": int(result.status),
        "stopped": stopped,
        "message": str(result.message),
        "accepted": not problems,
        "rejection_reasons": problems,
        "fx": float(cam.fx),
        "fy": float(cam.fy),
        "cx": float(cam.cx),
        "cy": float(cam.cy),
        "dist": np.asarray(cam.dist).tolist(),
    }


def _write_back(problem: BAProblem, free: tuple[str, ...], x: np.ndarray) -> None:
    """Copy decoded poses, world points and shared intrinsics into the mutable problem."""
    rv, tv, pts, f4, dist = _unpack(problem, free, x)
    problem.rvecs[:], problem.tvecs[:], problem.points[:] = rv, tv, pts
    problem.camera.fx, problem.camera.fy, problem.camera.cx, problem.camera.cy = f4
    problem.camera.dist = dist


def acceptance_problems(
    problem: BAProblem,
    x: np.ndarray,
    rms_before: float,
    rms_after: float,
    focal_ratio: float,
    max_rms_increase: float = 1.05,
    focal_bounds: tuple[float, float] = (0.8, 1.25),
    min_front: float = 0.99,
) -> list[str]:
    """Reasons to reject an adjustment result (empty list = accept).

    Numerical drift guards reject non-finite values, an RMS increase above 5%,
    a focal ratio outside [0.8, 1.25], or a result that puts
    more than 1 % of the observed points behind their cameras.
    """
    reasons = []
    if not np.all(np.isfinite(x)):
        return ["non-finite parameters"]
    if not np.isfinite(rms_after) or rms_after > max_rms_increase * rms_before:
        reasons.append(f"RMS increased from {rms_before:.3f} to {rms_after:.3f} px")
    if not focal_bounds[0] <= focal_ratio <= focal_bounds[1]:
        reasons.append(f"focal length changed by a factor {focal_ratio:.3f}")
    z = (
        rotate(problem.points[problem.pt_idx], problem.rvecs[problem.cam_idx])[:, 2]
        + problem.tvecs[problem.cam_idx, 2]
    )
    if np.mean(z > 0) < min_front:
        reasons.append(f"only {np.mean(z > 0):.1%} of observations in front of their camera")
    return reasons
