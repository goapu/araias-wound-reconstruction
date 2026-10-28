"""Regression tests for numerical and reproducibility contracts.

Each test isolates a boundary condition or invalid input so that the expected
numerical and reproducibility contracts remain enforced.

Run with `python -m pytest tests` or, without pytest, `python tests/test_regressions.py`.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woundrecon import bundle_adjustment as ba  # noqa: E402
from woundrecon import features as feat  # noqa: E402
from woundrecon import measure as ms  # noqa: E402
from woundrecon import provenance  # noqa: E402
from woundrecon import scale as sc  # noqa: E402
from woundrecon.calibration import board_points, calibrate, heldout_rms  # noqa: E402
from woundrecon.camera import Camera  # noqa: E402
from woundrecon.dense import StereoParams, _sgbm, left_right_check  # noqa: E402
from woundrecon.mesh import TSDFVolume, point_to_mesh_distance  # noqa: E402
from woundrecon.sfm import IncrementalSfM, SfMParams  # noqa: E402
from woundrecon.stages.mesh import current_depth_maps  # noqa: E402


def _raises(exc, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


# --------------------------------------------------------------------------- #
# Mesh extraction only where the TSDF was observed
# --------------------------------------------------------------------------- #
def test_tsdf_isolated_sample_gives_no_surface():
    vol = TSDFVolume(np.zeros(3), np.full(3, 4.0), 1.0, 4.0)
    vol.tsdf[2, 2, 2], vol.weight[2, 2, 2] = -0.5, 3  # one observed sample, unknown around it
    verts, faces, _, stats = vol.extract_mesh(min_weight=2)
    assert len(faces) == 0 and stats["faces_unsupported"] >= 1


def test_tsdf_fully_observed_plane_is_recovered_and_stops_at_unobserved_region():
    vol = TSDFVolume(np.zeros(3), np.full(3, 6.0), 1.0, 1.0)
    z = vol.origin[2] + (np.arange(vol.dims[2]) + 0.5) * vol.voxel
    vol.tsdf[:] = np.clip(2.3 - np.broadcast_to(z, vol.dims), -1, 1)
    vol.weight[:] = 3
    verts, faces, _, _ = vol.extract_mesh(min_weight=2)
    assert len(faces) > 0 and np.allclose(verts[:, 2], 2.3, atol=1e-6)
    vol.tsdf[4:], vol.weight[4:] = 1.0, 0  # x >= 4.5 never observed
    verts, faces, _, _ = vol.extract_mesh(min_weight=2)
    assert verts[:, 0].max() <= 3.5 + 1e-9


def test_tsdf_integration_of_one_view_has_no_back_sheet():
    """A depth map of a plane must give one sheet, not a second one at the back of the truncation band.

    Behind the band the volume stays unobserved (+1). Whether scikit-image's
    per-cell mask lets such cells through depends on the viewing direction
    relative to the grid axes, so both directions are tested. Support filtering
    must retain the observed plane without creating a second sheet in either view.
    """
    K = np.array([[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1]])
    depth = np.full((128, 128), 1.0, np.float32)
    for R, lo, hi, z_plane in (
        (np.eye(3), [-0.3, -0.3, 0.8], [0.3, 0.3, 1.3], 1.0),
        (np.diag([1.0, -1.0, -1.0]), [-0.3, -0.3, -1.3], [0.3, 0.3, -0.8], -1.0),
    ):
        vol = TSDFVolume(np.array(lo), np.array(hi), 0.01, 0.04)
        for _ in range(2):  # two observations -> weight 2
            vol.integrate(depth, K, R, np.zeros(3))
        verts, faces, _, _ = vol.extract_mesh(min_weight=2)
        assert len(faces) > 0 and np.abs(verts[:, 2] - z_plane).max() < 0.011


# --------------------------------------------------------------------------- #
# Stereo left-right check
# --------------------------------------------------------------------------- #
def test_lr_check_rejects_matches_outside_the_right_image():
    disp = np.full((80, 300), -10.0)  # agreeing negative disparities
    valid = left_right_check(disp, disp.copy(), invalid_value=-20, tolerance=1.0)
    assert not valid[:, -10:].any() and valid[:, :-10].all()


def test_lr_check_rejects_invalid_right_disparity():
    dl, dr = np.full((10, 200), 5.0), np.full((10, 200), 5.0)
    dr[:, 100] = 4.0  # SGBM invalid value = min_disp - 1 = 4
    valid = left_right_check(dl, dr, invalid_value=4.0, tolerance=1.0)
    assert not valid[:, 105].any() and valid[:, 104].all()


def test_sgbm_recovers_shifted_texture():
    """Recover a known disparity from a synthetic textured plane."""
    rng = np.random.default_rng(7)
    left = cv2.GaussianBlur(rng.integers(0, 256, (160, 400)).astype(np.uint8), (3, 3), 0)
    right = np.zeros_like(left)
    right[:, :-12] = left[:, 12:]
    disp = _sgbm(left, right, 0, 32, StereoParams())
    interior = disp[20:-20, 60:-40]
    assert abs(np.median(interior[interior > 0]) - 12) < 0.2


def test_matching_is_reproducible():
    """Randomised k-d tree matching must give identical matches on identical inputs."""
    rng = np.random.default_rng(5)
    d1 = rng.random((800, 128)).astype(np.float32)
    d2 = (d1[rng.permutation(800)] + rng.normal(0, 0.02, (800, 128))).astype(np.float32)
    first = feat.match_descriptors(d1, d2, 0.8)
    for _ in range(3):
        again = feat.match_descriptors(d1, d2, 0.8)
        assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))


# --------------------------------------------------------------------------- #
# Cache lineage
# --------------------------------------------------------------------------- #
def test_fingerprint_detects_changed_inputs():
    a = provenance.fingerprint("features", {"img.png": "abc"}, 0.5, 12000)
    assert a == provenance.fingerprint("features", {"img.png": "abc"}, 0.5, 12000)
    assert a != provenance.fingerprint("features", {"img.png": "abd"}, 0.5, 12000)
    assert a != provenance.fingerprint("features", {"img.png": "abc"}, 0.5, 1000)
    with tempfile.TemporaryDirectory() as d:
        np.savez(Path(d) / "c.npz", x=np.zeros(1))  # cache without fingerprint
        assert not provenance.cache_is_valid(np.load(Path(d) / "c.npz"), a, "Test")
        np.savez(Path(d) / "c.npz", x=np.zeros(1), fingerprint=np.array(a))
        assert provenance.cache_is_valid(np.load(Path(d) / "c.npz"), a, "Test")


def test_pair_cache_with_out_of_range_keypoints_is_rejected():
    cam = Camera(1000, 1000, 500, 400, np.zeros(5), 1000, 800)
    feats = [
        feat.ImageFeatures(
            f"i{k}.png",
            (800, 1000),
            np.random.default_rng(k).uniform(0, 800, (1000, 2)),
            np.zeros((1000, 128), np.float32),
            np.zeros((1000, 3), np.uint8),
        )
        for k in range(2)
    ]
    sfm = IncrementalSfM(feats, cam, ["none", "none"], SfMParams())
    with tempfile.TemporaryDirectory() as d:
        sfm.pairs = {(0, 1): feat.PairMatches(0, 1, np.array([5, 11999]), np.array([3, 4]), 2, 0)}
        sfm.save_pairs(Path(d) / "pairs.npz", "fp")
        sfm.pairs = {}
        assert sfm.load_pairs(Path(d) / "pairs.npz") is False and sfm.pairs == {}


def test_mesh_stage_refuses_depth_maps_from_other_poses():
    with tempfile.TemporaryDirectory() as d:
        dense = Path(d)
        (dense / "depth").mkdir()
        (dense / "dense_stats.json").write_text(
            json.dumps({"per_view": {"a.png": {"fused_px": 10}, "b.png": {"fused_px": 0}}})
        )
        sfm = {
            "names": np.array(["a.png", "b.png"]),
            "rvecs": np.zeros((2, 3)),
            "tvecs": np.zeros((2, 3)),
        }
        np.savez(
            dense / "depth" / "a.npz",
            depth=np.zeros((2, 2)),
            K=np.eye(3),
            R=np.eye(3),
            t=np.zeros(3),
        )
        np.savez(
            dense / "depth" / "b.npz",
            depth=np.zeros((2, 2)),
            K=np.eye(3),
            R=np.eye(3),
            t=np.zeros(3),
        )
        assert [f.name for f in current_depth_maps(dense, sfm)] == ["a.npz"]  # stale b.npz ignored
        np.savez(
            dense / "depth" / "a.npz",
            depth=np.zeros((2, 2)),
            K=np.eye(3),
            R=np.eye(3),
            t=np.ones(3),
        )
        assert _raises(ValueError, current_depth_maps, dense, sfm)


# --------------------------------------------------------------------------- #
# Scale references
# --------------------------------------------------------------------------- #
def _two_view_sfm():
    cam = Camera(1000.0, 1000.0, 500.0, 400.0, np.zeros(5), 1000, 800)
    R2 = cv2.Rodrigues(np.array([0.0, -0.3, 0.0]))[0]
    t2 = -R2 @ np.array([0.5, 0.0, 0.0])
    sfm = {
        "K": cam.K,
        "dist": cam.dist,
        "names": np.array(["a.png", "b.png"]),
        "registered": np.array([True, True]),
        "rvecs": np.array([np.zeros(3), cv2.Rodrigues(R2)[0].ravel()]),
        "tvecs": np.array([np.zeros(3), t2]),
        "rules": np.array(["none", "none"]),
        "image_size": np.array([1000, 800]),
    }
    return cam, sfm, [(np.eye(3), np.zeros(3)), (R2, t2)]


def _annotation(cam, poses, A, B, length_mm, noise=None):
    obs = []
    for name, (R, t) in zip(("a.png", "b.png"), poses, strict=True):
        pa, pb = (cam.project((R @ X + t)[None])[0] for X in (A, B))
        if noise is not None and name == "b.png":
            pb = pb + noise
        obs.append({"image": name, "endpoint_a": pa.tolist(), "endpoint_b": pb.tolist()})
    return {"length_mm": length_mm, "observations": obs}


def test_scale_rejects_invalid_references():
    for bad in (-50.0, 0.0, float("nan"), float("inf"), "ten", None):
        assert _raises(sc.ScaleError, sc.scale_from_reference_points, [[0, 0, 0], [1, 0, 0]], bad)
    assert _raises(sc.ScaleError, sc.scale_from_reference_points, [[1, 2, 3], [1, 2, 3]], 50.0)
    assert _raises(sc.ScaleError, sc.scale_from_reference_points, [[0, 0, np.nan], [1, 0, 0]], 50.0)
    assert _raises(sc.ScaleError, sc.check_scale, float("inf"))


def test_scale_from_marked_length_recovers_known_scale_and_rejects_bad_clicks():
    cam, sfm, poses = _two_view_sfm()
    A, B = np.array([0.1, 0.05, 3.0]), np.array([0.4, -0.05, 3.2])
    res = sc.scale_from_annotation(_annotation(cam, poses, A, B, 20.0), sfm)
    assert abs(res["mm_per_unit"] - 20.0 / np.linalg.norm(A - B)) < 1e-6
    assert _raises(
        sc.ScaleError,
        sc.scale_from_annotation,
        _annotation(cam, poses, A, B, 20.0, noise=[0, 40.0]),
        sfm,
    )
    assert _raises(
        sc.ScaleError, sc.scale_from_annotation, _annotation(cam, poses, A, B, -20.0), sfm
    )


# --------------------------------------------------------------------------- #
# Measurement integration and topology-aware label filling
# --------------------------------------------------------------------------- #
def test_fill_holes_only_fills_regions_enclosed_by_the_wound():
    # Grid mesh 6x6 vertices: wound = ring of vertices around an interior vertex, plus a separate triangle.
    n = 6
    idx = np.arange(n * n).reshape(n, n)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            faces += [[a, b, d], [a, d, c]]
    faces = np.array(faces + [[36, 37, 38]])  # disconnected triangle
    N = 39
    adj = ms.vertex_adjacency(N, faces)
    boundary = ms.boundary_vertices(N, faces)
    wound = np.zeros(N, bool)
    wound[
        [idx[1, 1], idx[1, 2], idx[1, 3], idx[2, 1], idx[2, 3], idx[3, 1], idx[3, 2], idx[3, 3]]
    ] = True
    out = ms.fill_holes(wound, adj, boundary)
    assert out[idx[2, 2]]  # enclosed hole filled
    assert not out[36:39].any()  # disconnected object untouched
    assert not out[idx[0, 0]]  # surrounding skin untouched


def test_reference_order_chosen_by_cross_validation_not_residual():
    rng = np.random.default_rng(3)
    ang = rng.uniform(-np.pi, np.pi, 4000)
    r = rng.uniform(1.0, 1.4, 4000)
    xy = np.c_[r * np.cos(ang), r * np.sin(ang)]
    z = 0.1 * xy[:, 0] ** 2 + 0.05 * xy[:, 1] ** 2 + rng.normal(0, 0.002, 4000)  # quadratic skin
    order, scores, used = ms.select_reference_order(
        xy, z, ms.ring_sectors(xy, np.zeros(2), 8), (1, 2, 3, 4), 50
    )
    assert order == 2 and used == 8 and scores[1] > 5 * scores[2]


def test_measurement_is_rejected_when_the_wound_is_not_below_the_reference():
    r = {
        "depth_p99": -0.01,
        "area_fraction_above_reference": 1.0,
        "reference_sectors_used": 8,
        "reference_cv_scores": {1: 0.001},
        "reference_order": 1,
    }
    status, reasons = ms._quality(r, ms.MeasureParams())
    assert status == "rejected" and len(reasons) >= 2


def test_weighted_percentile():
    v, w = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0, 97.0])
    assert (
        ms.weighted_percentile(v, w, 50) == 4.0 and ms.weighted_percentile(v, np.ones(4), 50) == 2.0
    )


def test_point_to_triangle_distance():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2]])
    pts = np.array([[0.2, 0.2, 0.5], [0.5, -0.3, 0.0], [-1.0, -1.0, 0.0], [2.0, 0.0, 0.0]])
    d = point_to_mesh_distance(pts, verts, faces)
    assert np.allclose(d, [0.5, 0.3, np.sqrt(2), 1.0])


# --------------------------------------------------------------------------- #
# Bundle adjustment acceptance and status
# --------------------------------------------------------------------------- #
def test_ba_rejects_nonfinite_and_records_status():
    cam = Camera(800.0, 800.0, 400.0, 300.0, np.zeros(5), 800, 600)
    rng = np.random.default_rng(4)
    P = rng.uniform([-0.5, -0.5, 3.0], [0.5, 0.5, 4.0], (60, 3))
    prob = ba.BAProblem(
        np.zeros((2, 3)),
        np.array([[0, 0, 0], [-0.3, 0, 0.0]]),
        P,
        np.repeat([0, 1], 60),
        np.tile(np.arange(60), 2),
        np.vstack([cam.project(P), cam.project(P + [-0.3, 0, 0])]),
        cam.copy(),
    )
    assert ba.acceptance_problems(prob, np.array([np.nan]), 1.0, 1.0, 1.0) == [
        "non-finite parameters"
    ]
    assert ba.acceptance_problems(prob, np.zeros(3), 1.0, 2.0, 1.0)  # RMS got worse
    assert ba.acceptance_problems(prob, np.zeros(3), 1.0, 1.0, 1.5)  # implausible focal change
    stats = ba.solve(prob, max_nfev=50)
    assert {"status", "stopped", "accepted", "max_nfev"} <= set(stats) and stats["accepted"]


# --------------------------------------------------------------------------- #
# Calibration hold-out on disjoint corners; synthetic camera recovery
# --------------------------------------------------------------------------- #
def _synthetic_views(K, dist, n_views=12, seed=11):
    obj = board_points((9, 6), 1.0)
    rng = np.random.default_rng(seed)
    views = []
    for _ in range(n_views):
        r = rng.uniform(-0.5, 0.5, 3)
        t = np.array([rng.uniform(-5, -2), rng.uniform(-3, -1), rng.uniform(14, 22)])
        views.append(cv2.projectPoints(obj, r, t, K, dist)[0].reshape(-1, 2))
    return obj, views


def test_synthetic_calibration_is_recovered_and_heldout_uses_unseen_corners():
    K = np.array([[3080.0, 0, 2016], [0, 3075, 1512], [0, 0, 1]])
    dist = np.array([0.2, -1.0, 0, 0, 0.0])
    obj, views = _synthetic_views(K, dist)
    res = calibrate(obj, views, (4032, 3024), "k1k2")
    assert np.allclose(res.K, K, rtol=1e-4, atol=0.5)
    assert heldout_rms(obj, views[0], res.K, res.dist) < 1e-3
    wrong = res.K.copy()
    wrong[0, 0] *= 1.02
    assert heldout_rms(obj, views[0], wrong, res.dist) > heldout_rms(obj, views[0], res.K, res.dist)


# --------------------------------------------------------------------------- #
# Submission export
# --------------------------------------------------------------------------- #
def test_export_stops_before_writing_when_inputs_are_missing():
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / "submission"
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "export_submission.py"),
                "--output-dir",
                str(Path(d) / "empty"),
                "--dest",
                str(dest),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 1 and "missing inputs" in r.stderr
        assert not dest.exists() and not dest.with_name("submission.partial").exists()


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
