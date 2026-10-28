"""Actual stage output paths, artifact hashes and partial-run failure records."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from woundrecon import cli, io_utils, validation
from woundrecon.config import REPO_ROOT, load_config, stage_dir
from woundrecon.provenance import sha256_file
from woundrecon.stages import STAGE_ORDER
from woundrecon.stages import render as render_stage


def _args_and_config(tmp_path, monkeypatch):
    data, output = tmp_path / "data", tmp_path / "outputs"
    for name in ("checkerboard", "images"):
        (data / name).mkdir(parents=True)
        (data / name / "input.png").write_bytes(b"input fingerprint fixture")
    args = SimpleNamespace(
        config=None,
        data_root=data,
        output_dir=output,
        threads=1,
        stages=STAGE_ORDER.copy(),
        source_snapshot=None,
        verbose=False,
    )
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli, "setup_logging", lambda folder, verbose: folder / "test.log")
    return output, load_config(data_root=data, output_dir=output)


def _render_inputs(output):
    scale = output / "scale"
    scale.mkdir(parents=True)
    io_utils.save_json(scale / "scale.json", {"metric": False})
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0.1], [-1, 1, 0.1]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    normals = np.tile([0.0, -0.05, 1.0], (4, 1))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    colors = np.tile([170, 70, 50], (4, 1)).astype(np.uint8)
    io_utils.write_ply(scale / "mesh_aligned.ply", vertices, colors, normals, faces)
    io_utils.write_ply(scale / "sparse_aligned.ply", vertices, colors)
    camera = np.array(
        [[0, 0, 4], [-0.3, -0.2, 3.7], [0.3, -0.2, 3.7], [0.3, 0.2, 3.7], [-0.3, 0.2, 3.7]], float
    )
    io_utils.write_ply(scale / "cameras_aligned.ply", camera, np.full((5, 3), 150, np.uint8))
    measure = output / "measure"
    measure.mkdir()
    io_utils.write_ply(measure / "wound_depth.ply", vertices, colors, normals, faces)
    np.savez(
        measure / "wound_masks.npz",
        wound=[True, True, False, False],
        ring=[False, False, True, True],
    )
    io_utils.save_json(
        measure / "measurements.json",
        {
            "scale_free": {"depth_p99": 0.1},
            "absolute": {"depth_p99": 0.1},
        },
    )


def test_real_render_files_are_hashed_and_tampering_is_rejected(tmp_path, monkeypatch):
    output, _ = _args_and_config(tmp_path, monkeypatch)
    _render_inputs(output)
    real_verify = validation.verify_outputs
    # Run the actual renderer with small canvases, retaining its real folder
    # selection, triangle rasterizer, captions and PNG writer.
    frame_points = render_stage.rd.frame_points

    def small_frame(*args, **kwargs):
        return frame_points(*args, **kwargs, width=320, height=240)

    monkeypatch.setattr(render_stage.rd, "frame_points", small_frame)
    monkeypatch.setattr(
        cli,
        "get_stage",
        lambda name: render_stage if name == "render" else SimpleNamespace(run=lambda cfg: {}),
    )
    monkeypatch.setattr(validation, "verify_outputs", lambda *args: {"status": "passed"})
    assert cli.main() == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    images = sorted((output / "renders").glob("*.png"))
    assert len(images) == 5
    for path in images:
        key = path.relative_to(output).as_posix()
        assert manifest["outputs_sha256"][key] == sha256_file(path)
        with Image.open(path) as image:
            image.verify()
    images[0].write_bytes(images[0].read_bytes() + b"changed screenshot")
    with pytest.raises(ValueError, match="Missing or changed artifact: renders/"):
        real_verify(output)


def test_failed_stage_still_hashes_partial_renders_and_cannot_be_released(tmp_path, monkeypatch):
    output, _ = _args_and_config(tmp_path, monkeypatch)

    def fail_after_writing(cfg):
        folder = stage_dir(cfg, "render")
        (folder / "partial.png").write_bytes(b"partial render evidence")
        raise RuntimeError("renderer interrupted")

    monkeypatch.setattr(
        cli,
        "get_stage",
        lambda name: SimpleNamespace(
            run=fail_after_writing if name == "render" else lambda cfg: {}
        ),
    )
    assert cli.main() == 1
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["stages"][-1]["status"] == "failed"
    key = "renders/partial.png"
    assert manifest["outputs_sha256"][key] == sha256_file(output / key)
    with pytest.raises(ValueError, match="failed pipeline stage"):
        validation.verify_outputs(output)


def test_stage_directory_names_follow_the_public_output_layout(tmp_path):
    cfg = {"paths": {"output": tmp_path}}
    assert stage_dir(cfg, "calibrate") == tmp_path / "calibration"
    assert stage_dir(cfg, "render") == tmp_path / "renders"


def test_packaged_defaults_match_the_documented_checkout_defaults():
    assert (REPO_ROOT / "configs/default.yaml").read_bytes() == (
        REPO_ROOT / "src/woundrecon/resources/default.yaml"
    ).read_bytes()
