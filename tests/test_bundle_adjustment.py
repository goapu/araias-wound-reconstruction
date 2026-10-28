"""Transactional bundle-adjustment acceptance with controlled solver outcomes."""

from types import SimpleNamespace

import numpy as np
import pytest

from woundrecon import bundle_adjustment as ba
from woundrecon.camera import Camera


def _problem():
    rng = np.random.default_rng(13)
    camera = Camera(1000.0, 950.0, 400.0, 300.0, np.zeros(5), 800, 600)
    points = rng.uniform([-0.5, -0.5, 3.0], [0.5, 0.5, 5.0], (20, 3))
    rotations = np.zeros((2, 3))
    translations = np.array([[0.0, 0.0, 0.0], [-0.3, 0.01, 0.0]])
    observations = np.vstack([camera.project(points + t) for t in translations])
    observations += rng.normal(0.0, 0.2, observations.shape)
    return ba.BAProblem(
        rotations,
        translations,
        points,
        np.repeat([0, 1], len(points)),
        np.tile(np.arange(len(points)), 2),
        observations,
        camera,
    )


def _snapshot(problem):
    return (
        problem.camera.copy(),
        problem.rvecs.copy(),
        problem.tvecs.copy(),
        problem.points.copy(),
        ba.residuals(problem, ()).copy(),
        id(problem.camera),
    )


def _assert_unchanged(problem, before):
    camera, rotations, translations, points, residuals, camera_id = before
    assert id(problem.camera) == camera_id
    assert (problem.camera.fx, problem.camera.fy, problem.camera.cx, problem.camera.cy) == (
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
    )
    np.testing.assert_array_equal(problem.camera.dist, camera.dist)
    np.testing.assert_array_equal(problem.rvecs, rotations)
    np.testing.assert_array_equal(problem.tvecs, translations)
    np.testing.assert_array_equal(problem.points, points)
    np.testing.assert_array_equal(ba.residuals(problem, ()), residuals)


def _solver(monkeypatch, candidate, status=1):
    monkeypatch.setattr(
        ba,
        "least_squares",
        lambda *args, **kwargs: SimpleNamespace(
            x=candidate.copy(), status=status, nfev=1, message="controlled solver outcome"
        ),
    )


def test_rejected_refinement_restores_state_and_reports_retained_rms(monkeypatch):
    problem = _problem()
    before = _snapshot(problem)
    free = ("f", "cx", "cy", "k1", "k2", "k3")
    candidate = ba._pack(problem, free)
    candidate[:6] += 0.01
    candidate[6:-6] += 0.02
    candidate[-6:] = [2.0, 420.0, 290.0, 0.1, -0.05, 0.01]
    _solver(monkeypatch, candidate)

    stats = ba.solve(problem, free=free)

    assert not stats["accepted"]
    assert any("focal" in reason for reason in stats["rejection_reasons"])
    _assert_unchanged(problem, before)
    actual_rms = np.sqrt(np.mean(np.sum(ba.residuals(problem, ()) ** 2, axis=1)))
    assert stats["rms_after_px"] == pytest.approx(actual_rms)
    assert stats["candidate_rms_px"] > stats["rms_after_px"]


@pytest.mark.parametrize("status", [-2, -1])
def test_unsuccessful_solver_status_cannot_be_accepted(monkeypatch, status):
    problem = _problem()
    before = _snapshot(problem)
    _solver(monkeypatch, ba._pack(problem, ("f",)), status=status)

    stats = ba.solve(problem, free=("f",))

    assert not stats["accepted"]
    assert stats["status"] == status
    assert "converged" not in stats["stopped"]
    assert any("solver" in reason for reason in stats["rejection_reasons"])
    _assert_unchanged(problem, before)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_candidate_does_not_replace_valid_state(monkeypatch, value):
    problem = _problem()
    before = _snapshot(problem)
    candidate = ba._pack(problem, ("f",))
    candidate[6] = value
    _solver(monkeypatch, candidate)

    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        stats = ba.solve(problem, free=("f",))

    assert not stats["accepted"]
    assert np.isfinite(stats["rms_after_px"])
    assert stats["candidate_rms_px"] is None
    _assert_unchanged(problem, before)


def test_exception_during_candidate_validation_leaves_original_state_intact(monkeypatch):
    problem = _problem()
    before = _snapshot(problem)
    candidate = ba._pack(problem, ("f",))
    candidate[-1] = 1.1
    candidate[6] += 0.1
    _solver(monkeypatch, candidate)

    def fail_validation(*args, **kwargs):
        raise RuntimeError("validation interrupted")

    monkeypatch.setattr(ba, "acceptance_problems", fail_validation)
    with pytest.raises(RuntimeError, match="validation interrupted"):
        ba.solve(problem, free=("f",))
    _assert_unchanged(problem, before)


@pytest.mark.parametrize("status", [0, 1])
def test_valid_focal_update_is_committed_once_including_at_evaluation_cap(monkeypatch, status):
    problem = _problem()
    initial = problem.camera.copy()
    camera_id = id(problem.camera)
    before_residuals = ba.residuals(problem, ()).copy()
    candidate = ba._pack(problem, ("f",))
    scale = 1.1
    # Compensating x/y coordinates retain the projections while testing one
    # relative focal update and preserving the initial camera's aspect ratio.
    candidate[3:5] /= scale
    candidate[6:-1].reshape(-1, 3)[:, :2] /= scale
    candidate[-1] = scale
    _solver(monkeypatch, candidate, status=status)

    stats = ba.solve(problem, free=("f",))

    assert stats["accepted"]
    assert id(problem.camera) == camera_id
    assert problem.camera.fx == pytest.approx(initial.fx * scale)
    assert problem.camera.fy == pytest.approx(initial.fy * scale)
    np.testing.assert_allclose(ba.residuals(problem, ()), before_residuals, atol=1e-12)
    actual_rms = np.sqrt(np.mean(np.sum(ba.residuals(problem, ()) ** 2, axis=1)))
    assert stats["rms_after_px"] == pytest.approx(actual_rms)
    if status == 0:
        assert stats["stopped"] == "evaluation cap reached"
