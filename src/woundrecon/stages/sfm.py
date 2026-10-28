"""Stage 2: sparse reconstruction (structure-from-motion + bundle adjustment).

Two variants are computed from the same incremental reconstruction:

    A  intrinsics fixed to the checkerboard calibration
    B  focal length and k1, k2 refined in the final bundle adjustment
       (self-calibration on the wound images)

Which variant is used downstream is decided on held-out images (see
``heldout.py``): B is fitted to the training observations, so its lower
training error is not evidence on its own.

Outputs (``outputs/sfm/``):
    features.npz           cached keypoints / descriptors
    sfm.npz                final cameras, points and observations
    sparse.ply             coloured sparse point cloud (arbitrary units)
    cameras.ply            camera frusta as a triangle mesh
    sfm_stats.json         registration, error and self-calibration statistics
    fig_*.png              figures for the report
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .. import features as feat
from .. import heldout, io_utils, plots, provenance
from ..camera import Camera, camera_centre
from ..config import stage_dir
from ..sfm import IncrementalSfM, SfMParams

log = logging.getLogger(__name__)

# Shared-intrinsic refinement couples all cameras, so it receives more evaluations
# than incremental pose updates; a shorter fixed-intrinsic polish follows it.
SELF_CALIB_BA_ROUNDS = 2
SELF_CALIB_MAX_NFEV = 200
FINAL_POLISH_MAX_NFEV = 100
FRUSTUM_SCALE_FRACTION = 0.08  # Display size relative to scene extent; no geometric constraint.


def _features_fingerprint(cfg: dict[str, Any], paths: list[Path]) -> str:
    """Bind features to image bytes, detector settings and the extraction implementation."""
    s = cfg["sfm"]
    return provenance.fingerprint(
        "features",
        provenance.input_hashes(paths),
        s["work_scale"],
        s["sift_features"],
        provenance.code_version(("features", "io_utils")),
    )


def _load_or_extract(cfg: dict[str, Any], out: Path) -> tuple[list[feat.ImageFeatures], str]:
    """Extract features, or reuse the cache if it was made from the same images, settings and code."""
    cache = out / "features.npz"
    paths = io_utils.list_images(cfg["paths"]["images"])
    fp = _features_fingerprint(cfg, paths)
    d = np.load(cache, allow_pickle=True) if cache.exists() else None
    if provenance.cache_is_valid(d, fp, "Feature"):
        log.info("Reusing cached features (%s)", cache.name)
        return [
            feat.ImageFeatures(n, tuple(s), k, ds, c)
            for n, s, k, ds, c in zip(
                d["names"], d["shapes"], d["keypoints"], d["descriptors"], d["colors"], strict=True
            )
        ], fp
    s = cfg["sfm"]
    feats = []
    for path in paths:
        f = feat.extract(path, s["work_scale"], s["sift_features"])
        log.info("Features %s: %d", path.name, len(f.keypoints))
        feats.append(f)
    np.savez(
        cache,
        names=np.array([f.name for f in feats]),
        shapes=np.array([f.displayed_shape for f in feats]),
        keypoints=np.array([f.keypoints for f in feats], dtype=object),
        descriptors=np.array([f.descriptors for f in feats], dtype=object),
        colors=np.array([f.colors for f in feats], dtype=object),
        fingerprint=np.array(fp),
    )
    return feats, fp


def _pairs_fingerprint(cfg: dict[str, Any], features_fp: str, calib_path: Path) -> str:
    """Bind verified pairs to features, calibration, matching settings and geometry code."""
    s = cfg["sfm"]
    keys = ("ratio_test", "min_inliers", "ransac_threshold_px", "valid_radius_scale")
    return provenance.fingerprint(
        "pairs",
        features_fp,
        {k: s[k] for k in keys},
        provenance.sha256_file(calib_path),
        provenance.code_version(("features", "sfm")),
    )


def frustum_mesh(
    R: np.ndarray, t: np.ndarray, cam: Camera, size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Small pyramid (apex = camera centre) visualising one camera."""
    c = camera_centre(R, t)
    w, h = cam.width, cam.height
    corners_px = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    rays = np.c_[
        (corners_px[:, 0] - cam.cx) / cam.fx, (corners_px[:, 1] - cam.cy) / cam.fy, np.ones(4)
    ]
    corners_w = (rays * size) @ R + c  # R^T applied row-wise
    verts = np.vstack([c, corners_w])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
    return verts, faces


def _write_cameras(path: Path, rec, cam: Camera, size: float) -> None:
    """Write world-frame camera frusta at the chosen SfM-unit visualization size."""
    verts, faces, colors = [], [], []
    for i in np.flatnonzero(rec.registered):
        R, _ = cv2.Rodrigues(rec.rvecs[i])
        v, f = frustum_mesh(R, rec.tvecs[i], cam, size)
        faces.append(f + sum(len(x) for x in verts))
        verts.append(v)
        colors.append(np.tile([235, 104, 52], (len(v), 1)))
    io_utils.write_ply(
        path,
        np.vstack(verts),
        np.vstack(colors).astype(np.uint8),
        faces=np.vstack(faces),
        comments=["camera frusta, same arbitrary units as sparse.ply"],
    )


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Register wound views and compare fixed versus refined shared intrinsics.

    The stage fails when no supported initial reconstruction exists or required
    inputs/geometry are invalid.

    Args:
        cfg: Validated configuration with resolved paths; reads
            calibration/calibration.json, photographs and fingerprinted feature/pair caches.

    Returns:
        Writes sfm/sfm.npz, sparse/camera PLYs, caches and sfm_stats.json; returns
        reconstruction statistics. Points/translations share arbitrary SfM length units;
        intrinsics and errors use full-resolution native sensor pixels.

    Raises:
        OSError: Required inputs or output paths cannot be accessed.
        RuntimeError: No verified initial image pair supports reconstruction.
    """
    plots.apply_style()
    out = stage_dir(cfg, "sfm")
    calib = io_utils.load_json(cfg["paths"]["output"] / "calibration" / "calibration.json")
    params = SfMParams.from_config(cfg["sfm"])
    camera = Camera.from_calibration(calib, params.valid_radius_scale)
    feats, features_fp = _load_or_extract(cfg, out)

    rules = [
        calib["orientation"]["portrait"]
        if io_utils.is_portrait(f.displayed_shape)
        else calib["orientation"]["landscape"]
        for f in feats
    ]
    sfm = IncrementalSfM(feats, camera, rules, params)
    pair_cache = out / "pairs.npz"
    pairs_fp = _pairs_fingerprint(
        cfg, features_fp, cfg["paths"]["output"] / "calibration" / "calibration.json"
    )
    cached = np.load(pair_cache, allow_pickle=True) if pair_cache.exists() else None
    if provenance.cache_is_valid(cached, pairs_fp, "Pair") and sfm.load_pairs(pair_cache):
        log.info("Reusing cached verified pairs (%s)", pair_cache.name)
    else:
        sfm.pairs = {}
        sfm.match_all()
        sfm.save_pairs(pair_cache, pairs_fp)
    sfm.run()

    # Variant A: checkerboard intrinsics fixed -----------------------------------
    obs_a = sfm.observations()
    stats_a = {
        "rms_px": float(np.sqrt(np.mean(obs_a["obs_error_px"] ** 2))),
        "median_px": float(np.median(obs_a["obs_error_px"])),
        "points": len(obs_a["points"]),
        "observations": len(obs_a["obs_xy"]),
    }
    camera_a = sfm.cam.copy()
    rec_a = copy.deepcopy(sfm.rec)

    # Variant B: self-calibration of focal length and radial distortion -----------
    free = tuple(cfg["sfm"].get("refine_parameters", ["f", "k1", "k2"]))
    for _ in range(SELF_CALIB_BA_ROUNDS):
        sfm.bundle_adjust(free=free, max_nfev=SELF_CALIB_MAX_NFEV)
    sfm.bundle_adjust(max_nfev=FINAL_POLISH_MAX_NFEV)
    obs_b = sfm.observations()
    stats_b = {
        "rms_px": float(np.sqrt(np.mean(obs_b["obs_error_px"] ** 2))),
        "median_px": float(np.median(obs_b["obs_error_px"])),
        "points": len(obs_b["points"]),
        "observations": len(obs_b["obs_xy"]),
        "fx": sfm.cam.fx,
        "fy": sfm.cam.fy,
        "dist": sfm.cam.dist,
        "focal_change_pct": (sfm.cam.fx / camera_a.fx - 1) * 100,
    }
    log.info(
        "Variant A (fixed): RMS %.3f px | Variant B (self-calibrated): RMS %.3f px, focal %+.2f %%",
        stats_a["rms_px"],
        stats_b["rms_px"],
        stats_b["focal_change_pct"],
    )
    # The training RMS of A and B is computed on different surviving observations
    # and B is fitted to them, so it cannot choose the variant. With
    # refine_intrinsics: auto the choice is made on held-out images instead.
    choice = cfg["sfm"].get("refine_intrinsics", "auto")
    heldout_result = None
    if choice == "auto":
        hc = heldout.HeldOutConfig(**cfg["sfm"].get("heldout", {}))
        held = heldout.heldout_indices(len(feats), hc)
        train = [i for i in range(len(feats)) if i not in held]
        log.info(
            "Held-out evaluation: training model from %d images, %d held out", len(train), len(held)
        )
        model = heldout.reconstruct_training(feats, camera_a, rules, params, sfm.pairs, train, free)
        heldout_result = heldout.evaluate(feats, sfm.rec.rules, params, sfm.pairs, model, held, hc)
        heldout_result["training_registered"] = model["registered"]
        use_b = heldout_result["decision"] == "B"
    else:
        use_b = bool(choice)
    if not use_b:
        sfm.rec, sfm.cam = rec_a, camera_a
    final_obs = obs_b if use_b else obs_a
    rec, cam = sfm.rec, sfm.cam

    orient_check = sfm.orientation_check_loo()

    # Statistics ---------------------------------------------------------------
    track_len = np.bincount(final_obs["obs_point"])
    per_image = {}
    for i in range(len(feats)):
        m = final_obs["obs_image"] == i
        per_image[feats[i].name] = {
            "registered": bool(rec.registered[i]),
            "rule": rec.rules[i],
            "observations": int(m.sum()),
            "rms_px": float(np.sqrt(np.mean(final_obs["obs_error_px"][m] ** 2)))
            if m.any()
            else None,
        }
    centres = np.array(
        [
            camera_centre(cv2.Rodrigues(rec.rvecs[i])[0], rec.tvecs[i])
            for i in np.flatnonzero(rec.registered)
        ]
    )
    stats = {
        "images": len(feats),
        "registered": int(rec.registered.sum()),
        "unregistered": [f.name for f, r in zip(feats, rec.registered, strict=True) if not r],
        "points": int(len(final_obs["points"])),
        "mean_track_length": float(track_len.mean()),
        "variant_A_fixed_intrinsics": stats_a,
        "variant_B_self_calibrated": stats_b,
        "final_variant": "B" if use_b else "A",
        "variant_choice": ("held-out evaluation" if heldout_result else "configuration"),
        "heldout_evaluation": heldout_result,
        "camera_final": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy, "dist": cam.dist},
        "per_image": per_image,
        "orientation_diagnostic": orient_check,
        "log": sfm.log,
        "verified_pairs": len(sfm.pairs),
    }
    io_utils.save_json(out / "sfm_stats.json", stats)

    # Save geometry --------------------------------------------------------------
    np.savez(
        out / "sfm.npz",
        names=np.array(rec.names),
        rules=np.array(rec.rules),
        registered=rec.registered,
        rvecs=rec.rvecs,
        tvecs=rec.tvecs,
        K=cam.K,
        dist=cam.dist,
        image_size=np.array([cam.width, cam.height]),
        valid_radius_px=np.array(cam.valid_radius_px or 0.0),
        **{k: v for k, v in final_obs.items()},
    )
    keep = track_len >= 3
    io_utils.write_ply(
        out / "sparse.ply",
        final_obs["points"][keep],
        final_obs["colors"][keep],
        comments=["sparse SfM points (track length >= 3), arbitrary units, sRGB colours"],
    )
    extent = np.percentile(np.linalg.norm(centres - centres.mean(0), axis=1), 90)
    _write_cameras(out / "cameras.ply", rec, cam, FRUSTUM_SCALE_FRACTION * extent)

    plots.plot_histogram(
        final_obs["obs_error_px"],
        "reprojection error [px]",
        f"Sparse reprojection errors (RMS {stats_b['rms_px'] if use_b else stats_a['rms_px']:.2f} px)",
        out / "fig_reprojection_hist.png",
        bins=80,
    )
    plots.plot_histogram(
        track_len,
        "track length [images]",
        "Track lengths",
        out / "fig_track_length.png",
        bins=np.arange(1.5, track_len.max() + 1.5),
    )
    return stats
