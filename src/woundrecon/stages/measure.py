"""Stage 6: wound measurements, their sensitivity to the measurement definition, and validation.

Measurements are made in whatever unit the scale stage provides: millimetres
when a reference of known size exists, otherwise the arbitrary reconstruction
unit. Every quantity is also reported relative to the wound length L
(scale-free). All distances used by the algorithm are fractions of the length
of the red wound region, so they do not depend on unknown scale. Sampling-resolution effects remain.

Each nominal value comes with a *sensitivity range*: the spread of the result
when one definition choice is changed (reference-surface order, colour
threshold, ring and crater distances). It shows how much a number depends on
these choices; it is not a confidence interval or an accuracy bound.

Outputs (``outputs/measure/``):
    measurements.json     wound size, depth, volume, sensitivity, status, validation
    wound_depth.ply       mesh coloured by depth below the periwound reference surface
    wound_masks.npz       wound / periwound-ring vertex masks and per-vertex depth
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np

from .. import io_utils, render
from .. import measure as ms
from ..config import stage_dir
from ..geometry import footprint_membership, point_to_mesh_distance
from ..validation import mesh_topology

log = logging.getLogger(__name__)

# Quantities summarised in the report, as (key, how it is made scale-free).
QUANTITIES = (
    ("length", None),
    ("width", 1),
    ("planimetric_area", 2),
    ("surface_area", 2),
    ("depth_p99", 1),
    ("mean_depth", 1),
    ("volume_below_reference", 3),
)


def params_from_config(m: dict[str, Any]) -> ms.MeasureParams:
    """Map measurement YAML values to colour, reference-order and region-size-relative definitions."""
    return ms.MeasureParams(
        a_min=m["a_min"],
        ring_inner=m["ring_inner"],
        ring_outer=m["ring_outer"],
        crater_depth=m["crater_depth"],
        crater_reach=m["crater_reach"],
        order=m["reference_surface_order"],
        orders=tuple(m["reference_orders"]),
        n_sectors=m["cv_sectors"],
    )


def summarise(r: dict) -> dict[str, float]:
    """Scale-free summary of one measurement (lengths / L, areas / L^2, volumes / L^3)."""
    L = r["length"]
    out = {"length": L}
    for key, power in QUANTITIES[1:]:
        out[key] = r[key] / L**power
    out["length_to_width"] = L / r["width"]
    return out


def sensitivity(verts, faces, colors, base: ms.MeasureParams, cfg_s: dict[str, Any]) -> list[dict]:
    """Re-measure with one definition choice changed at a time."""
    variants = [(f"reference order {o}", {"order": o}) for o in base.orders]
    d = cfg_s["a_min_delta"]
    variants += [
        (f"a* > {base.a_min - d:g}", {"a_min": base.a_min - d}),
        (f"a* > {base.a_min + d:g}", {"a_min": base.a_min + d}),
    ]
    for s in cfg_s["distance_scales"]:
        variants.append(
            (
                f"ring and crater distances x{s:g}",
                {
                    k: getattr(base, k) * s
                    for k in ("ring_inner", "ring_outer", "crater_depth", "crater_reach")
                },
            )
        )
    runs = []
    for name, change in variants:
        try:
            r = ms.measure_wound(verts, faces, colors, dataclasses.replace(base, **change))
        except (ValueError, np.linalg.LinAlgError) as error:
            r = {"status": "rejected", "reasons": [str(error)]}
        entry = {"variant": name, "status": r["status"], "reasons": r.get("reasons", [])}
        if "length" in r:
            entry["scale_free"] = summarise(r)
            entry["reference_order"] = r["reference_order"]
        runs.append(entry)
        log.info(
            "Sensitivity %-34s %-9s %s",
            name,
            r["status"],
            ""
            if "length" not in r
            else f"depth P99 {entry['scale_free']['depth_p99']:.2%} L, "
            f"volume {entry['scale_free']['volume_below_reference']:.4f} L^3",
        )
    return runs


def sensitivity_range(runs: list[dict], nominal: dict[str, float]) -> dict[str, list[float]]:
    """[min, max] of each quantity over the nominal result and all non-rejected variants."""
    ok = [r["scale_free"] for r in runs if r["status"] != "rejected"] + [nominal]
    return {k: [min(r[k] for r in ok), max(r[k] for r in ok)] for k in nominal}


def validate_against_sparse(
    sparse: np.ndarray, verts: np.ndarray, faces: np.ndarray, L: float, cell_rel: float
) -> dict[str, Any]:
    """Distance from SfM points to the mesh surface, for every point over the mesh footprint.

    Membership is decided only by the (x, y) footprint of the mesh, never by the
    distance itself, and the full distribution is reported. The SfM points share
    the camera poses with the dense reconstruction and set its disparity search
    ranges, so this is an internal-consistency check, not an independent accuracy test.
    """
    over = footprint_membership(sparse, verts, faces)
    d = point_to_mesh_distance(sparse[over], verts, faces)
    rel = d / L
    return {
        "sparse_points": int(len(sparse)),
        "over_mesh_footprint": int(over.sum()),
        "outside_footprint": int((~over).sum()),
        "distance": "complete closest-triangle search; internal consistency only",
        "membership": "exact projected triangle footprint; no distance-based selection",
        "relative_to_wound_length": {
            "median": float(np.median(rel)),
            "rms": float(np.sqrt(np.mean(rel**2))),
            "p95": float(np.percentile(rel, 95)),
            "max": float(rel.max()),
        },
        "absolute": {
            "median": float(np.median(d)),
            "rms": float(np.sqrt(np.mean(d**2))),
            "p95": float(np.percentile(d, 95)),
            "max": float(d.max()),
        },
    }


def _depth_colours(
    colors: np.ndarray, depth: np.ndarray, wound: np.ndarray, vmax: float
) -> np.ndarray:
    """Grey-scale texture outside the wound, depth colour map (0..vmax) inside."""
    grey = np.repeat(colors.mean(axis=1, keepdims=True), 3, axis=1).astype(np.uint8)
    out = grey.copy()
    out[wound] = render.colormap(np.clip(depth[wound], 0, None), 0.0, vmax)
    return out


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute reference-relative wound descriptors and their definition sensitivity.

    The stage fails when nominal segmentation/reference support is insufficient.
    Failed sensitivity variants are retained with their rejection reason.

    Args:
        cfg: Validated configuration with resolved paths; reads scale/mesh_aligned.ply,
            aligned sparse points, scale.json and measurement settings.

    Returns:
        Writes measure/measurements.json, wound_masks.npz and wound_depth.ply; returns
        descriptors. The anatomical-frame length unit is inherited from scale.json; areas
        and volumes use its square and cube.

    Raises:
        OSError: Required inputs or output paths cannot be accessed.
        RuntimeError: The nominal wound segmentation is rejected.
        ValueError: Reference support or projected wound geometry is degenerate.
    """
    out = stage_dir(cfg, "measure")
    o = cfg["paths"]["output"]
    scale = io_utils.load_json(o / "scale" / "scale.json")
    mesh_path = o / "scale" / "mesh_aligned.ply"
    mesh = io_utils.read_ply_vertices(mesh_path)
    faces = io_utils.read_ply_faces(mesh_path)
    v, col = mesh["points"], mesh["colors"]
    m = cfg["measure"]
    params = params_from_config(m)

    res = ms.measure_wound(v, faces, col, params)
    if "length" not in res:
        raise RuntimeError(f"Wound measurement failed: {res['reasons']}")
    L = res["length"]
    nominal = summarise(res)
    runs = sensitivity(v, faces, col, params, m["sensitivity"])
    validation = validate_against_sparse(
        io_utils.read_ply_vertices(o / "scale" / "sparse_aligned.ply")["points"],
        v,
        faces,
        L,
        m["validation_footprint_cell"],
    )
    validation["periwound_reference_rms_relative"] = res["reference_fit_rms"] / L

    coverage = mesh_topology(v, faces, res["wound_mask"])
    reasons = [
        "heuristic wound boundary lacks independent annotation validation",
        "counterfactual reference surface lacks external geometry validation",
    ] + res["reasons"]
    if coverage["boundary_edges_touching_wound"]:
        reasons.append("missing observed surface intersects the saved wound region")
    status = (
        "rejected"
        if res["status"] == "rejected" or coverage["boundary_edges_touching_wound"]
        else "uncertain"
    )
    record = {
        "units": scale["units"],
        "metric": scale["metric"],
        "status": status,
        "algorithm_status": res["status"],
        "reasons": reasons,
        "coverage": coverage,
        "conventions": {
            "depth": "along +z of the anatomical frame, below a polynomial height field fitted to the periwound ring",
            "depth_p99": "projected-area-weighted 99th percentile of clipped triangle mean depths",
            "mean_depth": "signed mean face depth weighted by projected area (parts above the reference count negative)",
            "volume_below_reference": "piecewise-linear positive-depth integral over clipped boundary triangles",
            "length_width": "longest extent along the principal axis x largest perpendicular extent, in projection",
            "distances": "ring and crater distances are fractions of the red-region length",
        },
        "absolute": {k: res[k] for k, _ in QUANTITIES}
        | {"volume_above_reference": res["volume_above_reference"]},
        "scale_free": nominal,
        "area_fraction_above_reference": res["area_fraction_above_reference"],
        "reference_surface": {
            "order": res["reference_order"],
            "selection": "leave-one-sector-out cross-validation (mean of per-sector median |residual|)",
            "cv_score_relative": {
                str(k): (s / L if np.isfinite(s) else None)
                for k, s in res["reference_cv_scores"].items()
            },
            "sectors_used": res["reference_sectors_used"],
            "fit_rms_relative": res["reference_fit_rms"] / L,
            "inlier_fraction": res["reference_fit_inlier_fraction"],
        },
        "sensitivity": {
            "note": "spread over one-at-a-time definition changes; not a confidence interval",
            "range": sensitivity_range(runs, nominal),
            "runs": runs,
        },
        "segmentation": {
            "wound_vertices": res["n_wound_vertices"],
            "red_vertices": res["red_vertices"],
            "crater_vertices_added": res["crater_vertices_added"],
            "ring_vertices": res["ring_vertices"],
            "red_region_length": res["length_red_region"],
        },
        "validation": validation,
    }
    io_utils.save_json(out / "measurements.json", record)
    np.savez_compressed(
        out / "wound_masks.npz", wound=res["wound_mask"], ring=res["ring_mask"], depth=res["depth"]
    )
    vmax = max(res["depth_p99"], 1e-9)
    io_utils.write_ply(
        out / "wound_depth.ply",
        v,
        _depth_colours(col, res["depth"], res["wound_mask"], vmax),
        normals=mesh.get("normals"),
        faces=faces,
        comments=[
            f"wound depth below periwound surface, viridis colour map from 0 to "
            f"{vmax:.4g} {scale['units']} (area-weighted P99 of clipped triangle means)"
        ],
    )
    rng = record["sensitivity"]["range"]
    log.info(
        "Wound (%s, public status %s): L %.4g, depth P99 %.2f%% L (range %.2f-%.2f), volume %.4f L^3 (range %.4f-%.4f)",
        scale["units"],
        record["status"],
        L,
        nominal["depth_p99"] * 100,
        rng["depth_p99"][0] * 100,
        rng["depth_p99"][1] * 100,
        nominal["volume_below_reference"],
        *rng["volume_below_reference"],
    )
    return record
