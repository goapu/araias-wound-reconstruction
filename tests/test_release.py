"""Cache, integration and release boundaries tested without confidential data."""

import cv2
import numpy as np
import pytest

from woundrecon import calibration, dense, geometry, io_utils, render
from woundrecon.delivery import source_files
from woundrecon.stages import calibrate as calibration_stage
from woundrecon.validation import mesh_topology


def test_ply_roundtrip_preserves_close_world_vertices(tmp_path):
    """World-coordinate storage must not collapse a valid narrow triangle."""
    from woundrecon.validation import verify_mesh_normals

    vertices = np.array([[1.6, 0.5, 7.0], [1.6 + 1e-8, 0.5, 7.0], [1.6, 0.500001, 7.0]])
    faces = np.array([[0, 1, 2]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
    normals = np.tile([0.0, 0, 1.0], (3, 1))
    assert np.array_equal(vertices[0].astype("f4"), vertices[1].astype("f4"))
    path = tmp_path / "surface.ply"
    io_utils.write_ply(path, vertices, colors, normals, faces)
    saved = io_utils.read_ply_vertices(path)
    np.testing.assert_array_equal(saved["points"], vertices)
    np.testing.assert_array_equal(saved["colors"], colors)
    np.testing.assert_array_equal(io_utils.read_ply_faces(path), faces)
    verify_mesh_normals(saved, faces, "narrow triangle")


def test_ply_readers_support_legacy_float_coordinates(tmp_path):
    """Existing float-coordinate files retain correct vertex and face offsets."""
    import struct

    path = tmp_path / "legacy.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    path.write_bytes(header + struct.pack("<9fB3i", 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0, 1, 2))
    np.testing.assert_array_equal(
        io_utils.read_ply_vertices(path)["points"], [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    )
    np.testing.assert_array_equal(io_utils.read_ply_faces(path), [[0, 1, 2]])


def test_camera_rejects_unsupported_distortion_instead_of_truncating():
    """A rational calibration cannot silently become a different five-coefficient camera."""
    from woundrecon.camera import Camera

    calibration_data = {
        "K": [[1000.0, 0, 600], [0, 1000.0, 450], [0, 0, 1.0]],
        "image_size": [1200, 900],
        "dist": [0.1, -0.2, 0, 0, 0.01, 0.4, -0.1, 0.03],
    }
    with pytest.raises(ValueError, match="at most five OpenCV distortion coefficients"):
        Camera.from_calibration(calibration_data)


def test_tsdf_normals_face_observing_camera_and_match_winding():
    """Both camera directions expose a sign error hidden by two-sided shading."""
    from woundrecon.mesh import TSDFVolume
    from woundrecon.validation import verify_mesh_normals

    K = np.array([[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1]])
    depth = np.full((128, 128), 1.0, np.float32)
    for R, lo, hi, outward in (
        (np.eye(3), [-0.3, -0.3, 0.8], [0.3, 0.3, 1.3], [0, 0, -1]),
        (np.diag([1.0, -1.0, -1.0]), [-0.3, -0.3, -1.3], [0.3, 0.3, -0.8], [0, 0, 1]),
    ):
        volume = TSDFVolume(np.array(lo), np.array(hi), 0.02, 0.08)
        for _ in range(2):
            volume.integrate(depth, K, R, np.zeros(3))
        vertices, faces, normals, _ = volume.extract_mesh()
        assert len(faces) > 0
        assert np.min(normals @ np.array(outward)) > 0.99
        checked = verify_mesh_normals({"points": vertices, "normals": normals}, faces, "plane")
        assert checked["orientation"]["face_agreement_fraction"] == 1.0
        with pytest.raises(ValueError, match="winding"):
            verify_mesh_normals({"points": vertices, "normals": -normals}, faces, "reversed")


def test_missing_gradient_normals_are_repaired_without_moving_points():
    x, y = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    original = points.copy()
    normals = np.full_like(points, np.nan)
    normals[::7] = [0, 0, -2]
    fused, _, repaired, _ = dense.voxel_downsample(
        points, np.full(points.shape, 128, np.uint8), normals, 0.01
    )
    np.testing.assert_array_equal(points, original)
    assert set(map(tuple, fused)) == set(map(tuple, points))
    np.testing.assert_allclose(repaired, np.tile([0, 0, -1], (len(points), 1)), atol=1e-10)


def test_missing_normals_reject_unsupported_orientation_and_rank():
    points = np.c_[np.arange(5), np.zeros((5, 2))]
    with pytest.raises(ValueError, match="oriented normal"):
        dense.repair_normals(points, np.zeros((5, 3)))
    normals = np.zeros((5, 3))
    normals[0] = [0, 0, 1]
    with pytest.raises(ValueError, match="collinear"):
        dense.repair_normals(points, normals)


def test_normal_release_gate_rejects_nonfinite_and_zero_normals():
    from woundrecon.validation import verify_normals

    for bad in (np.array([[np.nan, 0, 1]]), np.zeros((1, 3))):
        with pytest.raises(ValueError):
            verify_normals({"points": np.zeros((1, 3)), "normals": bad}, "fixture")


def test_reuse_proves_original_sources_and_rejects_changed_prerequisites(tmp_path):
    import sys

    from woundrecon import provenance, reuse
    from woundrecon.stages import STAGE_ORDER

    baseline, current, output = (tmp_path / name for name in ("baseline", "current", "output"))
    for directory in (baseline, current, output):
        directory.mkdir()
    for directory in (baseline, current):
        (directory / "camera.py").write_text("# original camera\n")
        (directory / "dense.py").write_text("# original dense\n")
    (current / "dense.py").write_text("# repaired downstream dense\n")
    artifact = output / "camera.npz"
    artifact.write_bytes(b"camera solution")
    cfg, images = {"value": 1}, {"images/one.png": "digest"}
    previous = dict(
        code_version=provenance.code_version(root=baseline),
        stages=[dict(stage=name, status="ok") for name in STAGE_ORDER],
        finished_utc="recorded",
        python=sys.version.split()[0],
        libraries=provenance.library_versions(),
        opencv_threads=4,
        config=cfg,
        inputs_sha256=images,
        outputs_sha256={"camera.npz": provenance.sha256_file(artifact)},
    )

    def check():
        return reuse.validate_reuse(
            previous, cfg, images, output, STAGE_ORDER[3:], 4, baseline, current
        )

    assert check()["reused_stages"] == ["audit", "calibrate", "sfm"]
    (current / "camera.py").write_text("# altered upstream camera\n")
    with pytest.raises(ValueError, match="Upstream source"):
        check()
    (current / "camera.py").write_text("# original camera\n")
    artifact.write_bytes(b"corrupted solution")
    with pytest.raises(ValueError, match="changed prerequisite"):
        check()
    artifact.write_bytes(b"camera solution")
    (baseline / "camera.py").write_text("# unverified replacement\n")
    with pytest.raises(ValueError, match="Source snapshot"):
        check()


def corner_config(folder):
    return {
        "paths": {"checkerboard": folder},
        "calibration": {"pattern_size": [9, 6], "detect_scale": 0.5, "subpix_window": 11},
    }


def test_corner_cache_tracks_bytes_inventory_settings_and_legacy_files(tmp_path, monkeypatch):
    images, cache = tmp_path / "images", tmp_path / "cache"
    images.mkdir()
    cache.mkdir()
    image = images / "a.png"
    image.write_bytes(b"first source content")
    calls = []

    def detect(paths, *args):
        calls.append([p.name for p in paths])
        return calibration.Detections(
            [p.name for p in paths],
            [np.zeros((54, 2)) for _ in paths],
            [(100, 120)] * len(paths),
            [],
        )

    monkeypatch.setattr(calibration_stage.cal, "detect_all", detect)
    config = corner_config(images)
    calibration_stage._load_or_detect(config, cache)
    calibration_stage._load_or_detect(config, cache)
    assert len(calls) == 1
    image.write_bytes(b"changed source content")
    calibration_stage._load_or_detect(config, cache)
    assert len(calls) == 2
    (images / "b.png").write_bytes(b"another image")
    calibration_stage._load_or_detect(config, cache)
    assert len(calls) == 3 and calls[-1] == ["a.png", "b.png"]
    config["calibration"]["detect_scale"] = 0.25
    calibration_stage._load_or_detect(config, cache)
    assert len(calls) == 4
    config["calibration"]["subpix_window"] = 7
    calibration_stage._load_or_detect(config, cache)
    assert len(calls) == 5
    # An unsigned file cannot become trusted just because it exists.
    np.savez(
        cache / "corners.npz",
        names=["a.png"],
        corners=np.zeros((1, 54, 2)),
        shapes=[[100, 120]],
        rejected=[],
    )
    calibration_stage._load_or_detect(config, cache)
    assert len(calls) == 6


def test_corrupt_corner_cache_is_recomputed(tmp_path, monkeypatch):
    images, cache = tmp_path / "images", tmp_path / "cache"
    images.mkdir()
    cache.mkdir()
    (images / "a.png").write_bytes(b"input")
    (cache / "corners.npz").write_bytes(b"not a numpy archive")
    monkeypatch.setattr(
        calibration_stage.cal,
        "detect_all",
        lambda *args: calibration.Detections(["a.png"], [np.zeros((54, 2))], [(100, 120)], []),
    )
    assert calibration_stage._load_or_detect(corner_config(images), cache).names == ["a.png"]
    with np.load(cache / "corners.npz", allow_pickle=True) as data:
        assert "fingerprint" in data


def test_boundary_clipping_preserves_partial_triangle_area():
    vertices = np.array([[0.0, 0, 0], [2.0, 0, 0], [0, 2.0, 0]])
    pieces = geometry.boundary_triangles(
        vertices, np.array([[0, 1, 2]]), np.array([True, False, False]), np.ones(3)
    )
    positive, _, _ = geometry.clipped_depth_integrals(pieces)
    assert positive == pytest.approx(0.5)  # one quarter of a triangle of area 2


def test_mixed_sign_volume_is_integrated_on_both_sides_of_zero():
    triangles = np.array([[[0.0, 0, 0, -1], [2.0, 0, 0, 1], [0, 2.0, 0, 1]]])
    positive, negative_area, negative_volume = geometry.clipped_depth_integrals(triangles)
    assert positive == pytest.approx(5 / 6)
    assert negative_area == pytest.approx(0.5)
    assert negative_volume == pytest.approx(1 / 6)
    assert positive - negative_volume == pytest.approx(2 / 3)


def test_complete_triangle_search_finds_large_distant_centroid():
    # The large triangle contains the query projection, but its centroid is far.
    vertices = [[-20.0, -20, 0], [20.0, -20, 0], [0, 20.0, 0]]
    faces = [[0, 1, 2]]
    for i in range(30):
        z = 1 + i * 0.01
        offset = len(vertices)
        vertices += [[-0.01, -0.01, z], [0.01, -0.01, z], [0, 0.01, z]]
        faces.append([offset, offset + 1, offset + 2])
    distance = geometry.point_to_mesh_distance(
        np.array([[0.0, 0, 0.1]]), np.array(vertices), np.array(faces)
    )
    assert distance[0] == pytest.approx(0.1)


def test_footprint_membership_uses_triangles_and_keeps_true_holes():
    vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    query = np.array([[0.1, 0.1, 100], [0.9, 0.9, 0]])
    np.testing.assert_array_equal(
        geometry.footprint_membership(query, vertices, np.array([[0, 1, 2]])), [True, False]
    )


def test_voxel_colour_average_uses_linear_light():
    points = np.array([[0.0, 0, 0], [0.01, 0, 0]])
    colors = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)
    _, rgb, _, counts = dense.voxel_downsample(points, colors, np.tile([0.0, 0, 1.0], (2, 1)), 1)
    np.testing.assert_allclose(rgb[0], [188, 188, 188], atol=1)
    assert counts[0] == 2


@pytest.mark.parametrize("vertical", [False, True])
def test_rectification_and_stereo_recover_a_known_plane(vertical):
    rng = np.random.default_rng(8)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (160, 400), dtype=np.uint8), (3, 3), 0)
    other = np.zeros_like(gray)
    if vertical:
        other[:-12, :] = gray[12:, :]
    else:
        other[:, :-12] = gray[:, 12:]
    K = np.array([[180.0, 0, 199.5], [0, 180.0, 79.5], [0, 0, 1.0]])
    t = np.array([0.0, -0.1, 0]) if vertical else np.array([-0.1, 0, 0])

    def view(name, image, shift):
        return dense.View(
            name,
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
            np.full(image.shape, 255, np.uint8),
            np.eye(3),
            shift,
            K,
            np.zeros(5),
        )

    world = np.array(
        [[x, y, 1.5] for x in np.linspace(-0.4, 0.4, 5) for y in np.linspace(-0.2, 0.2, 5)]
    )
    depths, stats = dense.pair_depth(
        view("left", gray, np.zeros(3)),
        view("right", other, t),
        world,
        dense.StereoParams(clahe_clip=0),
    )
    assert "skipped" not in stats and np.isfinite(depths).sum() > 1000
    assert np.nanmedian(depths) == pytest.approx(1.5, rel=0.01)


def test_topology_distinguishes_open_wound_edges_from_outer_edges():
    vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
    faces = np.array([[0, 1, 2]])
    assert (
        mesh_topology(vertices, faces, np.array([False, False, False]))[
            "boundary_edges_touching_wound"
        ]
        == 0
    )
    assert (
        mesh_topology(vertices, faces, np.array([True, False, False]))[
            "boundary_edges_touching_wound"
        ]
        == 2
    )


def test_source_archive_allowlist_excludes_data_and_credentials(tmp_path):
    for name in [
        "src/package/main.py",
        "src/woundrecon/resources/default.yaml",
        "tests/test_math.py",
        "images/scene.png",
        "outputs/models.ply",
        "report/report.pdf",
        "report/latex/main.tex",
        "report/latex/sections/plan.tex",
        "report/latex/results.tex",
        "report/latex/images/scene.png",
        "report/build_report.py",
        "report/latex/README.md",
        "docs/internal.md",
        "INTERNAL_NOTES.md",
        "scripts/local_helper.py",
        "development/private/helper.py",
        "development/private/notes.md",
        ".env",
        "private.key",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    selected = {p.relative_to(tmp_path).as_posix() for p in source_files(tmp_path)}
    assert selected == {
        "src/package/main.py",
        "src/woundrecon/resources/default.yaml",
        "tests/test_math.py",
    }


@pytest.mark.parametrize(
    "case", ["empty", "missing", "duplicate", "failed", "unfinished", "empty_inventory"]
)
def test_release_rejects_incomplete_run_before_reading_geometry(tmp_path, case):
    """An interrupted or incomplete stage record cannot release leftover geometry."""
    from woundrecon.stages import STAGE_ORDER
    from woundrecon.validation import verify_outputs

    stages = [{"stage": name, "status": "ok"} for name in STAGE_ORDER]
    manifest = {
        "stages": stages,
        "finished_utc": "2026-10-28T00:00:00Z",
        "outputs_sha256": {"mesh.ply": "placeholder"},
    }
    if case == "empty":
        stages.clear()
    elif case == "missing":
        stages.pop()
    elif case == "duplicate":
        stages[-1] = stages[0].copy()
    elif case == "failed":
        stages[3]["status"] = "failed"
    elif case == "unfinished":
        del manifest["finished_utc"]
    else:
        manifest["outputs_sha256"] = {}
    io_utils.save_json(tmp_path / "run_manifest.json", manifest)
    with pytest.raises(ValueError, match="Release requires|cannot be released"):
        verify_outputs(tmp_path)


def test_export_rejects_unreadable_pdf_before_creating_delivery(tmp_path, monkeypatch):
    """Report parsing errors produce a clear preflight failure with no partial delivery."""
    from woundrecon import delivery

    project, output = tmp_path / "project", tmp_path / "output"
    report = tmp_path / "invalid.pdf"
    for path in delivery.required_inputs(output, project, report):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"invalid fixture")
    monkeypatch.setattr(delivery, "verify_outputs", lambda path: {"status": "passed"})
    destination = tmp_path / "delivery"
    with pytest.raises(ValueError, match="Report PDF cannot be read"):
        delivery.export(project, output, destination, report=report)
    assert not destination.exists()


def test_completed_cli_run_hashes_calibration_and_render_evidence(tmp_path, monkeypatch):
    """A successful run must bind calibration evidence and inspection renders to their bytes."""
    import json
    from types import SimpleNamespace

    from woundrecon import cli, validation
    from woundrecon.provenance import sha256_file

    data, output = tmp_path / "data", tmp_path / "outputs"
    for folder in ("checkerboard", "images"):
        (data / folder).mkdir(parents=True)
        (data / folder / "image.png").write_bytes(b"input hash fixture")
    args = SimpleNamespace(
        config=None,
        data_root=data,
        output_dir=output,
        threads=1,
        stages=cli.STAGE_ORDER.copy(),
        source_snapshot=None,
        verbose=False,
    )
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli, "setup_logging", lambda folder, verbose: folder / "test.log")

    # Folder names written by the real calibrate and render stages.
    folders = {"calibrate": "calibration", "render": "renders"}

    def stage(name):
        def run_stage(cfg):
            folder = output / folders.get(name, name)
            folder.mkdir(parents=True)
            (folder / "stage.json").write_text("{}\n")
            if name == "calibrate":
                (folder / "calibration.json").write_text('{"model": "k1"}\n')
                (folder / "diagnostics.json").write_text('{"views": 2}\n')
                np.savez_compressed(folder / "bootstrap_samples.npz", params=np.arange(6))
            if name == "render":
                (folder / "01_textured_top.png").write_bytes(b"render fixture")

        return SimpleNamespace(run=run_stage)

    monkeypatch.setattr(cli, "get_stage", stage)
    monkeypatch.setattr(validation, "verify_outputs", lambda *args: {"status": "passed"})
    assert cli.main() == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    expected = [
        "calibration/calibration.json",
        "calibration/diagnostics.json",
        "calibration/bootstrap_samples.npz",
        "renders/01_textured_top.png",
    ]
    for key in expected:
        assert key in manifest["outputs_sha256"]
        assert manifest["outputs_sha256"][key] == sha256_file(output / key)


def test_calibration_model_subset_keeps_labels_and_bootstrap_evidence(tmp_path, monkeypatch):
    """A supported model subset must fit, label diagnostics and retain its actual bootstrap draws."""
    from woundrecon.config import load_config

    cfg = load_config()
    cfg["paths"]["output"] = tmp_path
    cfg["camera"]["native_size"] = [1200, 900]
    cfg["calibration"]["models"] = ["k1"]
    cfg["calibration"]["bootstrap_samples"] = 20
    obj = calibration.board_points((9, 6), 1.0)
    K = np.array([[1000.0, 0, 600], [0, 1000.0, 450], [0, 0, 1.0]])
    rng, corners = np.random.default_rng(9), []
    for i in range(10):
        rv = np.array([-0.22 + 0.045 * i, 0.15 * np.sin(i), 0.04 * np.cos(i)])
        tv = np.array([-4.0 + 0.2 * np.sin(i), -2.5 + 0.15 * np.cos(i), 22.0 + i])
        uv = cv2.projectPoints(obj, rv, tv, K, np.array([0.01, 0, 0, 0, 0]))[0].reshape(-1, 2)
        uv += rng.normal(0, 0.02, uv.shape)
        # Invert the declared clockwise displayed-to-native rotation.
        corners.append(np.c_[uv[:, 1], 1199 - uv[:, 0]].astype(np.float32))
    detections = calibration.Detections(
        names=[f"board{i}.png" for i in range(10)], corners=corners, shapes=[(1200, 900)] * 10
    )
    monkeypatch.setattr(calibration_stage, "_load_or_detect", lambda *args: detections)
    labels = []
    for name in [
        "apply_style",
        "plot_coverage",
        "plot_per_view",
        "plot_distortion_maps",
        "plot_residual_field",
    ]:
        monkeypatch.setattr(calibration_stage.plots, name, lambda *args: None)
    monkeypatch.setattr(
        calibration_stage.plots, "plot_model_selection", lambda names, *args: labels.append(names)
    )
    result = calibration_stage.run(cfg)
    diagnostics = io_utils.load_json(tmp_path / "calibration/diagnostics.json")
    assert result["model"] == "k1" and labels == [["k1"]]
    for key in ["train_planar", "train_ro", "heldout_planar", "heldout_ro"]:
        assert set(diagnostics[key]) == {"k1"}
    with np.load(tmp_path / "calibration/bootstrap_samples.npz", allow_pickle=False) as data:
        names = ["fx", "fy", "cx", "cy", *result["dist_names"]]
        np.testing.assert_allclose(
            data["params"].std(axis=0), [result["std_bootstrap"][name] for name in names]
        )
        assert data["probes"].shape == (41 * 31, 2)
        assert data["probe_displacement_px"].shape[1] == 41 * 31


def test_export_uses_external_pdf_without_report_authoring_files(tmp_path, monkeypatch):
    """A code-only checkout can package a separately prepared report and verified outputs."""
    import json
    import zipfile

    from PIL import Image
    from pypdf import PdfWriter

    from woundrecon import delivery

    project, output = tmp_path / "project", tmp_path / "output"
    project.mkdir()
    output.mkdir()
    readme = "# Reconstruction\n\nRun and inspect the model as described here.\n"
    (project / "README.md").write_text(readme)
    (project / "pyproject.toml").write_text('[project]\nname = "fixture"\n')
    (output / "run_manifest.json").write_text("{}\n")
    for name in delivery.MODELS:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"verified model fixture")
    for name in delivery.SCREENSHOTS:
        path = output / "renders" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (800, 600), "white").save(path)
    report = tmp_path / "separate.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with report.open("wb") as stream:
        writer.write(stream)
    # Numerical release checks have their own tests; exercise packaging without
    # report sources and without coupling the fixture to a reconstruction run.
    monkeypatch.setattr(delivery, "verify_outputs", lambda path: {"status": "passed"})
    destination = tmp_path / "delivery"
    result = delivery.export(project, output, destination, report=report)
    assert result["status"] == "passed"
    assert not (project / "report").exists()
    assert (destination / "report.pdf").read_bytes() == report.read_bytes()
    assert (destination / "README.md").read_text() == readme
    assert not (destination / "report_latex.zip").exists()
    assert not (destination / "MODEL_GUIDE.md").exists()
    with zipfile.ZipFile(destination / "source_code.zip") as archive:
        assert set(archive.namelist()) == {"README.md", "pyproject.toml"}
    with zipfile.ZipFile(destination / "submission_bundle.zip") as archive:
        assert archive.testzip() is None and "README.md" in archive.namelist()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["report.pdf"]["sha256"] == delivery.sha256_file(report)


def test_triangle_rendering_fills_faces_without_sampling_gaps():
    camera = render.OrbitCamera(
        np.zeros(3),
        np.array([0.0, 0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        fov_deg=90,
        width=32,
        height=32,
    )
    vertices = np.array([[-1.0, -1, 2], [1.0, -1, 2], [1.0, 1, 2], [-1.0, 1, 2]])
    image, depth = render.rasterize_mesh(
        vertices, np.array([[0, 1, 2], [0, 2, 3]]), np.tile([50, 100, 150], (4, 1)), camera
    )
    assert np.isfinite(depth).sum() == 256
    np.testing.assert_allclose(depth[8:24, 8:24], 2)
