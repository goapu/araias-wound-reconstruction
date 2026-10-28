"""Unit tests for the geometric building blocks.

Run with `python -m pytest tests` or, without pytest, `python tests/test_core.py`.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from woundrecon import bundle_adjustment as ba  # noqa: E402
from woundrecon import io_utils  # noqa: E402
from woundrecon.camera import Camera, rotate, triangulate_dlt  # noqa: E402


def test_orientation_points_match_image_rotation():
    """Mapping a pixel coordinate must agree with rotating the image itself."""
    img = np.zeros((40, 30), np.uint8)  # portrait, displayed frame
    img[7, 11] = 255
    for rule in ("cw", "ccw", "180", "none"):
        native = io_utils.to_native_image(img, rule)
        y, x = np.argwhere(native == 255)[0]
        mapped = io_utils.to_native_points(np.array([[11, 7]]), img.shape, rule)[0]
        assert np.allclose(mapped, [x, y]), rule


def test_projection_matches_opencv():
    cam = Camera(3000, 2990, 2000, 1500, np.array([0.2, -1.1, 0.0, 0.0, 2.0]))
    rng = np.random.default_rng(1)
    X = rng.uniform([-0.3, -0.2, 1.0], [0.3, 0.2, 2.0], size=(200, 3))
    ours = cam.project(X)
    ref, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), cam.K, cam.dist)
    assert np.allclose(ours, ref.reshape(-1, 2), atol=1e-6)


def test_rotate_matches_rodrigues():
    rng = np.random.default_rng(2)
    r = rng.normal(size=3) * 0.7
    X = rng.normal(size=(50, 3))
    R, _ = cv2.Rodrigues(r)
    assert np.allclose(rotate(X, np.repeat(r[None], 50, 0)), X @ R.T)


def test_triangulation_recovers_point():
    X = np.array([0.1, -0.05, 2.0])
    poses = [
        (np.eye(3), np.zeros(3)),
        (cv2.Rodrigues(np.array([0, 0.2, 0]))[0], np.array([-0.3, 0, 0.02])),
    ]
    rays = np.array([((R @ X + t)[:2] / (R @ X + t)[2]) for R, t in poses])
    assert np.allclose(triangulate_dlt(rays, poses), X, atol=1e-9)


def test_bundle_adjustment_reaches_noise_floor():
    rng = np.random.default_rng(0)
    cam = Camera(3000, 3000, 2016, 1512, np.array([0.1, -0.2, 0, 0, 0.0]))
    P = rng.normal(size=(300, 3)) * 0.2 + [0, 0, 3]
    C = 6
    rv, tv = rng.normal(size=(C, 3)) * 0.1, rng.normal(size=(C, 3)) * 0.3
    rv[0] = tv[0] = 0
    ci = np.repeat(np.arange(C), len(P))
    pi = np.tile(np.arange(len(P)), C)
    obs = np.vstack(
        [cam.project(rotate(P, np.repeat(rv[c : c + 1], len(P), 0)) + tv[c]) for c in range(C)]
    )
    obs += rng.normal(size=obs.shape) * 0.5
    noisy_rv = rv + rng.normal(size=rv.shape) * 0.01 * (np.arange(C)[:, None] > 0)
    prob = ba.BAProblem(
        noisy_rv, tv.copy(), P + rng.normal(size=P.shape) * 0.01, ci, pi, obs, cam.copy()
    )
    stats = ba.solve(prob, max_nfev=100)
    assert stats["rms_before_px"] > 10
    assert stats["rms_after_px"] < 0.8  # noise sigma 0.5 px per axis -> ~0.7 px RMS


def test_p3_to_srgb_keeps_neutrals():
    grey = np.array([[0, 0, 0], [128, 128, 128], [255, 255, 255]], np.uint8)
    assert np.abs(io_utils.p3_to_srgb(grey).astype(int) - grey).max() <= 1


if __name__ == "__main__":
    # Minimal runner so the tests also work without pytest installed.
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
