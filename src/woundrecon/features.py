"""Local features, matching and two-view geometric verification.

Design choices
--------------
* SIFT on a half-resolution copy, converted to RootSIFT (Arandjelovic &
  Zisserman 2012) by L1 normalization and an elementwise square root.
* Keypoints are stored in full-resolution *displayed* coordinates; mapping to
  the native sensor frame happens later, because the orientation of some
  landscape images is only decided during registration.
* Matching: FLANN k-NN in both directions, Lowe ratio test and a mutual
  nearest-neighbour check.
* Verification: essential matrix with MAGSAC++ on undistorted pixel
  coordinates. The number of homography inliers is also stored, to avoid
  degenerate (planar / pure rotation) initial pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import io_utils

log = logging.getLogger(__name__)


@dataclass
class ImageFeatures:
    """Keypoints and descriptors of one image."""

    name: str
    displayed_shape: tuple[int, int]
    keypoints: np.ndarray  # (N, 2) full-resolution displayed-frame coordinates
    descriptors: np.ndarray  # (N, 128) float32 RootSIFT
    colors: np.ndarray  # (N, 3) uint8 sRGB colour at each keypoint


@dataclass
class PairMatches:
    """Verified correspondences between two images."""

    i: int
    j: int
    idx_i: np.ndarray  # keypoint indices in image i
    idx_j: np.ndarray  # keypoint indices in image j
    n_putative: int
    n_homography_inliers: int


def extract(path: Path, work_scale: float, n_features: int) -> ImageFeatures:
    """Detect SIFT keypoints on a downscaled grey image and compute RootSIFT descriptors."""
    bgr = io_utils.read_image(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.01)
    kps, desc = sift.detectAndCompute(small, None)
    if desc is None:
        return ImageFeatures(
            path.name,
            gray.shape[:2],
            np.zeros((0, 2)),
            np.zeros((0, 128), np.float32),
            np.zeros((0, 3), np.uint8),
        )
    pts = np.array([k.pt for k in kps], np.float64)
    pts = (pts + 0.5) / work_scale - 0.5  # back to full resolution
    desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-12)
    desc = np.sqrt(desc).astype(np.float32)  # RootSIFT
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, gray.shape[1] - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, gray.shape[0] - 1)
    colors = io_utils.p3_to_srgb(bgr[yi, xi, ::-1])
    return ImageFeatures(path.name, gray.shape[:2], pts, desc, colors)


def seed_opencv(seed: int = 0) -> None:
    """Seed OpenCV's random generator before a randomised call.

    FLANN's randomized k-d trees and RANSAC/USAC estimators draw from it.
    Seeding controls their OpenCV RNG state; native threading, floating-point
    arithmetic and threshold decisions can still affect end-to-end results.
    """
    cv2.setRNGSeed(seed)


def match_descriptors(
    d1: np.ndarray, d2: np.ndarray, ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest neighbours that also pass Lowe's ratio test."""
    if len(d1) < 2 or len(d2) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    seed_opencv()
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})

    def one_way(a: np.ndarray, b: np.ndarray) -> dict[int, int]:
        """Keep nearest descriptor matches separated from the runner-up by the ratio test."""
        out = {}
        for m in flann.knnMatch(a, b, k=2):
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                out[m[0].queryIdx] = m[0].trainIdx
        return out

    fwd, bwd = one_way(d1, d2), one_way(d2, d1)
    pairs = [(q, t) for q, t in fwd.items() if bwd.get(t) == q]
    if not pairs:
        return np.zeros(0, int), np.zeros(0, int)
    a = np.array(pairs, int)
    return a[:, 0], a[:, 1]


def verify(
    pts_i: np.ndarray, pts_j: np.ndarray, K: np.ndarray, threshold_px: float
) -> tuple[np.ndarray, int]:
    """Geometric verification with the essential matrix.

    Args:
        pts_i, pts_j: (M, 2) *undistorted* pixel coordinates of putative matches.
        K: camera matrix of the undistorted coordinates.
        threshold_px: Sampson-distance threshold in pixels.

    Returns:
        Boolean inlier mask and the number of homography inliers (degeneracy cue).
    """
    if len(pts_i) < 8:
        return np.zeros(len(pts_i), bool), 0
    seed_opencv()
    E, mask = cv2.findEssentialMat(
        pts_i, pts_j, K, method=cv2.USAC_MAGSAC, prob=0.9999, threshold=threshold_px
    )
    if E is None or mask is None:
        return np.zeros(len(pts_i), bool), 0
    inl = mask.ravel().astype(bool)
    seed_opencv()
    _, hmask = cv2.findHomography(pts_i, pts_j, cv2.USAC_MAGSAC, 3 * threshold_px)
    n_h = int(hmask.sum()) if hmask is not None else 0
    return inl, n_h
