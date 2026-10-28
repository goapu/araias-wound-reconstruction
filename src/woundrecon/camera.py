"""Pinhole camera with OpenCV radial-tangential distortion.

All functions are vectorised over points. Pixel coordinates follow the OpenCV
convention (pixel centres at integer coordinates) in the native sensor frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

_TERM = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-12)


@dataclass
class Camera:
    """Intrinsics in the native frame: fx, fy, cx, cy and k1, k2, p1, p2, k3."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    width: int = 4032
    height: int = 3024
    valid_radius_px: float | None = None  # radius where distortion is constrained

    @classmethod
    def from_calibration(cls, calib: dict[str, Any], valid_radius_scale: float = 1.0) -> Camera:
        """Construct a native-sensor camera from a saved calibration.

        Args:
            calib: Calibration dictionary with (3, 3) K in native pixels, distortion,
                image_size (width, height) and optional covered_radius_px.
            valid_radius_scale: Dimensionless multiplier of the observed corner radius.

        Returns:
            Camera with an independent (5,) OpenCV distortion vector and pixel radius.

        Raises:
            ValueError: The selected model needs more than five distortion coefficients;
                rational and higher-order camera transfer is not implemented.
        """
        K = np.asarray(calib["K"], float)
        dist = np.zeros(5)
        d = np.asarray(calib["dist"], float).ravel()
        if len(d) > 5:
            raise ValueError(
                "Reconstruction supports at most five OpenCV distortion coefficients; "
                "rational and higher-order camera transfer is not implemented"
            )
        dist[: len(d)] = d
        w, h = calib["image_size"]
        radius = calib.get("covered_radius_px")
        if radius is not None:
            radius = float(radius) * valid_radius_scale
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], dist, int(w), int(h), radius)

    @property
    def K(self) -> np.ndarray:
        """Return the (3, 3) intrinsic matrix in full-resolution native sensor pixels."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def copy(self) -> Camera:
        """Return an independent camera copy, including its mutable distortion array."""
        return Camera(
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.dist.copy(),
            self.width,
            self.height,
            self.valid_radius_px,
        )

    # ------------------------------------------------------------------ #
    def in_valid_region(self, pixels: np.ndarray) -> np.ndarray:
        """True where a (distorted) pixel lies inside the calibrated field of view."""
        if self.valid_radius_px is None:
            return np.ones(len(pixels), bool)
        r = np.hypot(pixels[:, 0] - self.width / 2, pixels[:, 1] - self.height / 2)
        return r <= self.valid_radius_px

    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Distorted pixel coordinates -> ideal (undistorted) pixel coordinates."""
        if len(pixels) == 0:
            return np.zeros((0, 2))
        pts = np.asarray(pixels, np.float64).reshape(-1, 1, 2)
        return cv2.undistortImagePoints(pts, self.K, self.dist, _TERM).reshape(-1, 2)

    def normalise(self, pixels: np.ndarray) -> np.ndarray:
        """Distorted pixel coordinates -> normalised ray coordinates (x/z, y/z)."""
        und = self.undistort_pixels(pixels)
        return np.c_[(und[:, 0] - self.cx) / self.fx, (und[:, 1] - self.cy) / self.fy]

    def project(self, X_cam: np.ndarray) -> np.ndarray:
        """Camera-frame 3D points (N, 3) -> distorted pixel coordinates (N, 2)."""
        return project_points(X_cam, np.array([self.fx, self.fy, self.cx, self.cy]), self.dist)


def project_points(X_cam: np.ndarray, fxfycxcy: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """OpenCV projection model, vectorised (used inside bundle adjustment).

    Args:
        X_cam: (N, 3) points in camera coordinates.
        fxfycxcy: (4,) or (N, 4) intrinsics.
        dist: (5,) or (N, 5) coefficients k1, k2, p1, p2, k3.
    """
    z = X_cam[:, 2]
    x = X_cam[:, 0] / z
    y = X_cam[:, 1] / z
    d = np.broadcast_to(dist, (len(X_cam), 5)) if np.ndim(dist) == 1 else dist
    k1, k2, p1, p2, k3 = d.T
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    f = np.broadcast_to(fxfycxcy, (len(X_cam), 4)) if np.ndim(fxfycxcy) == 1 else fxfycxcy
    return np.c_[f[:, 0] * xd + f[:, 2], f[:, 1] * yd + f[:, 3]]


def rotate(points: np.ndarray, rvecs: np.ndarray) -> np.ndarray:
    """Rotate points (N, 3) by axis-angle vectors (N, 3) with Rodrigues' formula."""
    theta = np.linalg.norm(rvecs, axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(theta > 1e-12, rvecs / theta, 0.0)
    cos, sin = np.cos(theta), np.sin(theta)
    dot = np.sum(points * v, axis=1)[:, None]
    return cos * points + sin * np.cross(v, points) + dot * (1 - cos) * v


def camera_centre(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World position of a camera with world-to-camera pose X_c = R X + t."""
    return -R.T @ t


def triangulate_dlt(rays: np.ndarray, poses: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Linear triangulation of one point from >= 2 views.

    Args:
        rays: (V, 2) normalised image coordinates.
        poses: V world-to-camera poses (R, t).
    """
    A = []
    for (x, y), (R, t) in zip(rays, poses, strict=True):
        P = np.c_[R, t]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    return X[:3] / X[3]
