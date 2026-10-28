"""Stage 1: camera calibration from the checkerboard images.

Outputs (``outputs/calibration/``):
    calibration.json      final intrinsics, distortion, uncertainty, orientation rules
    diagnostics.json      every number behind the model choice and the sufficiency verdict
    corners.npz           cached corner detections (displayed frame)
    bootstrap_samples.npz recorded parameter draws and probe displacements
    overlays/             detection overlays for visual inspection
    fig_*.png             figures used in the report
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .. import calibration as cal
from .. import io_utils, plots, provenance
from ..config import stage_dir

log = logging.getLogger(__name__)

# Models ordered by number of distortion parameters (used for the parsimony rule).
MODEL_ORDER = ["k1", "k1k2", "k1k2k3", "opencv5", "rational"]
PARSIMONY_TOLERANCE = 0.02  # accept a simpler model within 2 % of the best held-out error
# Fixed reporting screens, not fitted constraints or validated accuracy tolerances.
MIN_RADIAL_REACH = 0.85  # Near-edge corners limit distortion extrapolation across the sensor.
MAX_CORNER_DISP_SD_PX = 1.0  # Flag corner-distortion instability exceeding one native pixel.
MAX_REL_FOCAL_SD = 0.002  # Flag bootstrap focal spread above 0.2%; this is precision only.
MIN_MAX_TILT_DEG = 45  # Oblique target poses help separate focal length and target distance.
MIN_DISTANCE_SPAN_SQUARES = 10  # Distance diversity complements tilt; unit is board squares.
MAX_BOARD_BEND_SQUARES = 0.005  # Flag released-target relief above 0.5% of a square side.
MAX_LENS_PRIOR_REL_DIFF = 0.05  # Broad 5% lens-prior screen; focus/crop effects remain untested.


def _corners_fingerprint(cfg: dict[str, Any], paths: list) -> str:
    """Bind displayed-frame detections to image bytes and detector settings.

    Orientation hypotheses are applied after detection and therefore do not
    alter this cache's coordinate contract. Unsigned caches are rejected.
    """
    c = cfg["calibration"]
    return provenance.fingerprint(
        "corners-v1",
        provenance.input_hashes(paths),
        list(c["pattern_size"]),
        c["detect_scale"],
        c["subpix_window"],
        provenance.code_version(("calibration", "io_utils")),
        provenance.library_versions(),
    )


def _load_or_detect(cfg: dict[str, Any], out_dir) -> cal.Detections:
    """Reuse a complete corner cache only when its dependencies still match."""
    cache = out_dir / "corners.npz"
    c = cfg["calibration"]
    paths = io_utils.list_images(cfg["paths"]["checkerboard"])
    expected = _corners_fingerprint(cfg, paths)
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as data:
                if provenance.cache_is_valid(data, expected, "Corner"):
                    det = cal.Detections(
                        names=list(data["names"]),
                        corners=list(data["corners"]),
                        shapes=[tuple(s) for s in data["shapes"]],
                        rejected=list(data["rejected"]),
                    )
                    log.info("Loaded %d verified cached detections", len(det.names))
                    return det
        except (OSError, ValueError, KeyError, EOFError, TypeError) as error:
            log.warning("Corner cache is unreadable (%s); detecting again", error)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    det = cal.detect_all(
        paths, tuple(c["pattern_size"]), c["detect_scale"], c["subpix_window"], overlay_dir
    )
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("wb") as file:
        np.savez(
            file,
            names=np.array(det.names),
            corners=np.array(det.corners, dtype=np.float32),
            shapes=np.array(det.shapes),
            rejected=np.array(det.rejected),
            fingerprint=np.array(expected),
        )
    temporary.replace(cache)
    return det


def _select_model(heldout: dict[str, float]) -> str:
    """Simplest model whose mean held-out error is within tolerance of the best."""
    best = min(heldout.values())
    for model in MODEL_ORDER:
        if model in heldout and heldout[model] <= best * (1 + PARSIMONY_TOLERANCE):
            return model
    return min(heldout, key=heldout.get)


def _sufficiency(diag: dict[str, Any], prior_fx: float) -> list[dict[str, str]]:
    """Turn the diagnostics into explicit pass/fail statements for the report."""
    findings = []

    def add(name: str, ok: bool, detail: str) -> None:
        """Append a named screening verdict and its numerical evidence."""
        findings.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})

    cov = diag["coverage"]
    add(
        "Sensor coverage",
        cov["max_radial_reach"] > MIN_RADIAL_REACH,
        f"corners reach {cov['max_radial_reach']:.0%} of the half-diagonal; "
        f"{cov['fraction_cells_covered']:.0%} of a 16x12 sensor grid is covered",
    )
    corner_sd = diag["bootstrap"]["corner_displacement_sd_px"]
    add(
        "Distortion at image corners",
        corner_sd < MAX_CORNER_DISP_SD_PX,
        f"bootstrap 1-sigma of the distortion shift at the image corner is {corner_sd:.1f} px",
    )
    rel_f = diag["bootstrap"]["std"]["fx"] / diag["final"]["K"][0][0]
    add(
        "Focal length precision",
        rel_f < MAX_REL_FOCAL_SD,
        f"bootstrap 1-sigma of fx is {rel_f:.2%} (covariance estimate: "
        f"{diag['final']['covariance_std']['fx'] / diag['final']['K'][0][0]:.2%})",
    )
    tilt = np.array(diag["poses"]["tilt_deg"])
    add(
        "Pose diversity",
        tilt.max() > MIN_MAX_TILT_DEG
        and np.ptp(diag["poses"]["distance_board_units"]) > MIN_DISTANCE_SPAN_SQUARES,
        f"tilt {tilt.min():.0f}-{tilt.max():.0f} deg; distance "
        f"{min(diag['poses']['distance_board_units']):.0f}-{max(diag['poses']['distance_board_units']):.0f} squares",
    )
    defo = diag["board_deformation"]
    add(
        "Target flatness",
        defo["out_of_plane_ptp"] < MAX_BOARD_BEND_SQUARES,
        f"refined board deviates {defo['out_of_plane_ptp']:.3f} squares out of plane (peak-to-peak); "
        f"held-out error drops {diag['heldout_planar'][diag['final']['model']]:.2f} -> "
        f"{diag['heldout_ro'][diag['final']['model']]:.2f} px when the board is released",
    )
    fx = diag["final"]["K"][0][0]
    add(
        "Plausibility vs lens prior",
        abs(fx / prior_fx - 1) < MAX_LENS_PRIOR_REL_DIFF,
        f"fx = {fx:.0f} px vs {prior_fx:.0f} px from 5.7 mm / 1.9 um (infinity focus); "
        f"relative difference {fx / prior_fx - 1:+.1%}; its cause requires controlled testing",
    )
    add(
        "Single orientation",
        diag["n_landscape"] == 0,
        f"{diag['n_portrait']} portrait + {diag['n_landscape']} landscape views; rotation relation "
        "had to be inferred from the data",
    )
    return findings


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit shared native-sensor intrinsics and assess calibration sufficiency.

    The stage fails when calibration/validation support is insufficient or files
    are unreadable. Sufficiency screens are reported rather than aborting a fit.

    Args:
        cfg: Validated configuration with resolved paths; reads checkerboard images and
            calibration/camera settings.

    Returns:
        Writes calibration.json, diagnostics, bootstrap draws, cached corners and figures;
        returns calibration. Intrinsics/errors use native sensor pixels; distortion is
        dimensionless and board coordinates use the configured square unit.

    Raises:
        OSError: Required inputs or output paths cannot be accessed.
    """
    plots.apply_style()
    out = stage_dir(cfg, "calibration")
    c = cfg["calibration"]
    models = [name for name in MODEL_ORDER if name in c["models"]]
    size = tuple(cfg["camera"]["native_size"])  # (width, height)
    pattern = tuple(c["pattern_size"])
    obj = cal.board_points(pattern, c["square_size"])

    # 1. Corners --------------------------------------------------------------
    det = _load_or_detect(cfg, out)
    log.info("Usable views: %d, rejected: %s", len(det.names), det.rejected)

    # 2. Orientation: portrait -> native is "cw" by convention; test landscape views
    orient = cal.test_landscape_rotation(det, obj, size, "k1k2k3")
    rules = list(orient["best_rule_per_view"].values())
    landscape_rule = max(set(rules), key=rules.count) if rules else "none"
    log.info("Landscape rotation test: %s -> %s", orient["views"], landscape_rule)
    pts = det.native("cw", landscape_rule)

    # 3. Model selection: planar vs release-object, train vs held-out ---------
    train = {"planar": [], "ro": []}
    heldout = {"planar": [], "ro": []}
    heldout_planar, heldout_ro = {}, {}
    for model in models:
        for key, ro in (("planar", False), ("ro", True)):
            res = cal.calibrate(obj, pts, size, model, release_object=ro)
            loo = cal.leave_one_out(obj, pts, size, model, release_object=ro).mean()
            train[key].append(res.rms)
            heldout[key].append(loo)
            (heldout_ro if ro else heldout_planar)[model] = float(loo)
            log.info("%-9s %-7s train %.3f px  held-out %.3f px", model, key, res.rms, loo)

    use_ro = min(heldout_ro.values()) < min(heldout_planar.values())
    model = _select_model(heldout_ro if use_ro else heldout_planar)
    final = cal.calibrate(obj, pts, size, model, release_object=use_ro)
    planar_same_model = cal.calibrate(obj, pts, size, model, release_object=False)
    log.info(
        "Selected model %s (release_object=%s): K=\n%s\ndist=%s", model, use_ro, final.K, final.dist
    )

    # 4. Uncertainty: bootstrap over views ------------------------------------
    w, h = size
    gx, gy = np.meshgrid(np.linspace(0, w - 1, 41), np.linspace(0, h - 1, 31))
    probes = np.c_[gx.ravel(), gy.ravel()]
    boot = cal.bootstrap(
        obj, pts, size, model, use_ro, c["bootstrap_samples"], c["random_seed"], probes
    )
    # Retain actual draws so uncertainty figures use this completed run's samples.
    # The successful-run artifact inventory includes this calibration evidence.
    np.savez_compressed(
        out / "bootstrap_samples.npz",
        params=boot["params"],
        probe_displacement_px=boot["probe_displacement_px"],
        probes=probes,
    )
    names = ["fx", "fy", "cx", "cy"] + final.dist_names
    boot_std = dict(zip(names, boot["params"].std(axis=0), strict=True))
    disp_sd = boot["probe_displacement_px"].std(axis=0).reshape(gx.shape)
    disp = cal.distortion_displacement(probes, final.K, final.dist).reshape(gx.shape)

    # 5. Diagnostics -----------------------------------------------------------
    cov = cal.coverage(pts, size)
    poses = cal.pose_statistics(final)
    focal_per_view = cal.per_view_focal(pts, size, final)
    deformation = cal.board_deformation(obj, final.board)
    residuals = cal.residual_vectors(pts, final)
    prior_fx = cal.physical_focal_prior(
        cfg["camera"]["focal_length_mm"], cfg["camera"]["pixel_pitch_um"]
    )
    reach_px = cov["max_radial_reach"] * float(np.hypot(w / 2, h / 2))
    portrait = det.portrait_mask()

    diag = {
        "views_used": det.names,
        "views_rejected": det.rejected,
        "n_portrait": int(portrait.sum()),
        "n_landscape": int((~portrait).sum()),
        "orientation_test": orient,
        "landscape_rule": landscape_rule,
        "train_planar": dict(zip(models, train["planar"], strict=True)),
        "train_ro": dict(zip(models, train["ro"], strict=True)),
        "heldout_planar": heldout_planar,
        "heldout_ro": heldout_ro,
        "final": final.summary(),
        "planar_same_model": planar_same_model.summary(),
        "per_view_rms_px": dict(zip(det.names, final.per_view_rms, strict=True)),
        "bootstrap": {
            "n_samples": len(boot["params"]),
            "std": boot_std,
            "corner_displacement_sd_px": float(disp_sd[0, 0]),
            "centre_region_displacement_sd_px": float(np.median(disp_sd[10:21, 15:26])),
        },
        "coverage": {k: v for k, v in cov.items() if k != "cell_counts"},
        "poses": {
            "tilt_deg": poses["tilt_deg"],
            "distance_board_units": poses["distance_board_units"],
        },
        "per_view_focal_px": focal_per_view,
        "per_view_focal_corr_with_distance": float(
            np.corrcoef(focal_per_view, poses["distance_board_units"])[0, 1]
        ),
        "per_view_focal_corr_with_tilt": float(
            np.corrcoef(focal_per_view, poses["tilt_deg"])[0, 1]
        ),
        "board_deformation": deformation,
        "prior_fx_px": prior_fx,
        "covered_radius_px": reach_px,
    }
    diag["sufficiency"] = _sufficiency(diag, prior_fx)

    # 6. Final calibration file used downstream --------------------------------
    calib = {
        "image_size": size,
        "K": final.K,
        "dist": final.dist,
        "dist_names": final.dist_names,
        "model": model,
        "release_object": use_ro,
        "rms_px": final.rms,
        "heldout_rms_px": (heldout_ro if use_ro else heldout_planar)[model],
        "std_bootstrap": boot_std,
        "std_covariance": final.summary()["covariance_std"],
        "orientation": {"portrait": "cw", "landscape": landscape_rule},
        "covered_radius_px": reach_px,
    }
    io_utils.save_json(out / "calibration.json", calib)
    io_utils.save_json(out / "diagnostics.json", diag)

    # 7. Figures ---------------------------------------------------------------
    plots.plot_coverage(pts, size, cov["max_radial_reach"], out / "fig_coverage.png")
    plots.plot_model_selection(models, train, heldout, out / "fig_model_selection.png")
    plots.plot_per_view(
        det.names,
        final.per_view_rms,
        poses["tilt_deg"],
        poses["distance_board_units"],
        focal_per_view,
        out / "fig_per_view.png",
    )
    plots.plot_distortion_maps(gx, gy, disp, disp_sd, reach_px, size, out / "fig_distortion.png")
    plots.plot_residual_field(
        pts,
        residuals,
        size,
        out / "fig_residuals.png",
        f"Residuals, {model}, release-object={use_ro}",
    )
    return calib
