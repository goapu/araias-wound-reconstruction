"""Experimental colour/geometry labels and reference-relative descriptors.

A polynomial surface fitted to the surrounding ring is a counterfactual model,
not a measured intact wound surface. Returned numerical quality gates do not
establish clinical validity, physical accuracy or a reviewed wound boundary.
All distances are relative to region size; sampling-resolution effects remain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .geometry import boundary_triangles, clipped_depth_integrals

log = logging.getLogger(__name__)

# Robust polynomial-fit safeguards; model/region choices remain in MeasureParams/YAML.
INITIAL_SUPPORT_PER_TERM = 4  # Overdetermine the fit before robust rejection.
RETAINED_SUPPORT_PER_TERM = 2  # Retain redundancy after rejection, beyond full rank alone.
MIN_REFERENCE_EXTENT = 1e-10  # Length-unit floor to prevent division by collapsed extent.
GAUSSIAN_MAD_SCALE = 1.4826  # Converts Gaussian MAD to the standard-deviation scale.
REFERENCE_REJECTION_SIGMA = 2.5  # Robust residual cutoff, in estimated z standard deviations.
MIN_REFERENCE_SIGMA = 1e-9  # Length-unit floor for a numerically exact planar fit.
MIN_RED_VERTICES = 10  # Reject tiny colour islands before defining a wound coordinate scale.
MIN_PROJECTED_WOUND_AREA = 1e-12  # Squared-length floor against zero-area integration.
MAX_ABOVE_REFERENCE_FRACTION = 0.25  # Reject a reference above which >25% of wound area lies.
MAX_REFERENCE_ERROR_DEPTH_RATIO = 0.25  # Flag ring prediction error above 25% of P99 depth.
MAX_MISSING_RING_SECTORS = 2  # More missing sectors leave angular extrapolation uncertain.
MIN_CV_SECTORS = 3  # Require at least three spatially separated test sectors.


def lab(colors_rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (N, 3) -> CIELAB float (L in 0-100, a/b approx. -128..127)."""
    lab8 = (
        cv2.cvtColor(colors_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    )
    return np.c_[lab8[:, 0] * 100 / 255, lab8[:, 1] - 128, lab8[:, 2] - 128]


def wound_colour_score(colors_rgb: np.ndarray, a_min: float = 22.0) -> np.ndarray:
    """Boolean per vertex: red granulation tissue (high CIELAB a*).

    Sampled Lab values on the moulage: wound bed a* ~ 25-35, pad a* ~ 12-17,
    skin a* ~ 15, table a* ~ 0. Slough (L ~ 40, a* ~ 9, b* ~ 21) cannot be
    separated from skin by colour alone, so it is not detected here; slough
    islands enclosed by granulation are added by hole filling on the mesh.
    Small reddish "bruise" spots on the pad pass the threshold but are removed
    by keeping the largest connected region.
    """
    return lab(colors_rgb)[:, 1] > a_min


def vertex_adjacency(n: int, faces: np.ndarray) -> coo_matrix:
    """Build an undirected mesh-vertex adjacency matrix.

    Args:
        n: Number of mesh vertices.
        faces: (F, 3) integer vertex indices; geometry/frame/units are not used.

    Returns:
        (n, n) CSR adjacency with edge multiplicities used by label smoothing.
    """
    e = np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    e = np.r_[e, e[:, ::-1]]
    return coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()


def smooth_labels(labels: np.ndarray, adj, iterations: int = 6) -> np.ndarray:
    """Majority-vote smoothing of a binary vertex label over the mesh graph."""
    x = labels.astype(float)
    # A self-vote preserves an isolated label and keeps the denominator nonzero.
    deg = np.asarray(adj.sum(axis=1)).ravel() + 1
    for _ in range(iterations):
        x = (adj @ x + x) / deg
    return x > 0.5


def largest_region(mask: np.ndarray, adj) -> np.ndarray:
    """Keep the largest connected component of a vertex mask."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return mask
    sub = adj[idx][:, idx]
    _, comp = connected_components(sub, directed=False)
    keep = idx[comp == np.argmax(np.bincount(comp))]
    out = np.zeros_like(mask)
    out[keep] = True
    return out


def boundary_vertices(n: int, faces: np.ndarray) -> np.ndarray:
    """Boolean per vertex: lies on an open mesh boundary (an edge used by one face only)."""
    e = np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1)
    edges, counts = np.unique(e, axis=0, return_counts=True)
    out = np.zeros(n, bool)
    out[edges[counts == 1].ravel()] = True
    return out


def fill_holes(mask: np.ndarray, adj, boundary: np.ndarray) -> np.ndarray:
    """Add non-wound regions that are fully enclosed by the wound (e.g. glossy spots in the bed).

    A region of non-wound vertices is filled only if it (a) touches the wound,
    (b) contains no vertex on an open mesh boundary (otherwise it may continue
    into unobserved space) and (c) is not the largest non-wound region (the
    surrounding skin). Disconnected objects never touch the wound and are left alone.
    """
    idx = np.flatnonzero(~mask)
    if len(idx) == 0:
        return mask.copy()
    _, comp = connected_components(adj[idx][:, idx], directed=False)
    n_comp = comp.max() + 1
    outside = np.argmax(np.bincount(comp))
    touches = np.zeros(n_comp, bool)
    touches[comp[np.asarray(adj[idx][:, np.flatnonzero(mask)].sum(axis=1)).ravel() > 0]] = True
    open_edge = np.zeros(n_comp, bool)
    open_edge[comp[boundary[idx]]] = True
    fill = touches & ~open_edge
    fill[outside] = False
    out = mask.copy()
    out[idx[fill[comp]]] = True
    return out


def poly_terms(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Build the total-degree polynomial design matrix in normalized x/y coordinates.

    Args:
        x: (N,) dimensionless normalized anatomical x coordinates.
        y: (N,) corresponding normalized anatomical y coordinates.
        order: Maximum total degree, a nonnegative integer.

    Returns:
        (N, (order + 1) * (order + 2) / 2) dimensionless monomial columns,
        ordered by increasing x power and then increasing y power.
    """
    return np.stack([x**i * y**j for i in range(order + 1) for j in range(order + 1 - i)], axis=1)


def fit_reference_surface(
    xy: np.ndarray, z: np.ndarray, order: int, iterations: int = 5
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a robust normalized polynomial height field to the periwound ring.

    Args:
        xy: (N, 2) anatomical-frame ring coordinates in reconstruction units or mm.
        z: (N,) ring heights in the same length units.
        order: Total polynomial degree.
        iterations: Robust least-squares/rejection passes.

    Returns:
        Packed polynomial coefficients followed by x/y centre and coordinate
        scale; inlier RMS in z units; and an (N,) inlier mask. Coefficients have
        height units because their monomial inputs are dimensionless.

    Raises:
        ValueError: Support is insufficient, non-finite, degenerate or rank deficient.
    """
    xy = np.asarray(xy, float)
    z = np.asarray(z, float)
    n_terms = (order + 1) * (order + 2) // 2
    if (
        len(xy) < INITIAL_SUPPORT_PER_TERM * n_terms
        or not np.isfinite(xy).all()
        or not np.isfinite(z).all()
    ):
        raise ValueError("Insufficient finite periwound support")
    centre = xy.mean(axis=0)
    scale = np.abs(xy - centre).max()
    if scale <= MIN_REFERENCE_EXTENT:
        raise ValueError("Degenerate periwound extent")
    # Centring/scaling to [-1, 1] reduces powers' dynamic range and improves
    # Vandermonde conditioning; full rank is still checked for each retained set.
    u = (xy - centre) / scale
    A = poly_terms(u[:, 0], u[:, 1], order)
    keep = np.ones(len(z), bool)
    coef = np.zeros(A.shape[1])
    for _ in range(iterations):
        if (
            keep.sum() < RETAINED_SUPPORT_PER_TERM * n_terms
            or np.linalg.matrix_rank(A[keep]) < n_terms
        ):
            raise ValueError("Rank-deficient periwound reference")
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef
        # Gaussian-consistent MAD estimates residual spread without large-error leverage.
        sigma = GAUSSIAN_MAD_SCALE * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r) < REFERENCE_REJECTION_SIGMA * max(sigma, MIN_REFERENCE_SIGMA)
    if keep.sum() < RETAINED_SUPPORT_PER_TERM * n_terms:
        raise ValueError("Robust fit retained insufficient reference support")
    rms = float(np.sqrt(np.mean((z[keep] - A[keep] @ coef) ** 2)))
    return np.r_[coef, centre, scale], rms, keep


def eval_reference_surface(params: np.ndarray, xy: np.ndarray, order: int) -> np.ndarray:
    """Evaluate the fitted periwound reference surface z = f(x, y).

    Args:
        params: (P + 3,) packed result from fit_reference_surface: P polynomial
            coefficients, anatomical x/y centre, and coordinate-normalization scale.
        xy: (N, 2) anatomical-frame query coordinates in the fit's length units.
        order: Polynomial degree used for the fit; P = (order + 1) * (order + 2) / 2.

    Returns:
        (N,) reference heights in the fit's length units. Wound depth is this
        reference height minus the reconstructed surface height.
    """
    n_terms = (order + 1) * (order + 2) // 2
    coef, centre, scale = params[:n_terms], params[n_terms : n_terms + 2], params[n_terms + 2]
    u = (xy - centre) / scale
    return poly_terms(u[:, 0], u[:, 1], order) @ coef


def length_width(xy: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Longest extent along the principal axis and the largest perpendicular extent."""
    c = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    p = (xy - c) @ vt.T
    return float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])), vt[0]


@dataclass
class MeasureParams:
    """Region-size-relative definitions, independent of physical scale.

    Mesh sampling and heuristic labels still affect the resulting descriptors.
    """

    a_min: float = 22.0  # CIELAB a* threshold for red granulation tissue
    ring_inner: float = 0.04  # periwound ring used for the reference surface ...
    ring_outer: float = 0.13  # ... between these distances from the wound
    crater_depth: float = 0.014  # vertices this far below the reference ...
    crater_reach: float = 0.095  # ... and this close to the red region join the wound
    order: int | str = "auto"  # polynomial order of the reference surface, or "auto"
    orders: tuple[int, ...] = (1, 2, 3, 4)
    n_sectors: int = 8  # angular sectors for cross-validating the order
    min_sector_vertices: int = 50  # Avoid unstable medians from sparsely sampled ring sectors.


def ring_sectors(xy: np.ndarray, centre: np.ndarray, n: int) -> np.ndarray:
    """Angular sector (0..n-1) of each point around ``centre``."""
    ang = np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])
    return ((ang + np.pi) / (2 * np.pi) * n).astype(int) % n


def select_reference_order(
    xy: np.ndarray,
    z: np.ndarray,
    sectors: np.ndarray,
    orders: tuple[int, ...],
    min_vertices: int,
    tie: float = 0.05,
) -> tuple[int, dict[int, float], int]:
    """Choose the reference-surface order by leave-one-sector-out cross-validation.

    For every candidate order the surface is fitted to all ring sectors but one
    and evaluated on the left-out sector (median absolute residual); the score is
    the mean over sectors. Greater flexibility can reduce training residuals
    without improving prediction; predicting a part of the ring that was not
    used for fitting is closer to the actual task (predicting the skin surface
    inside the wound from its surroundings). The lowest order within ``tie`` of
    the best score is chosen.

    Returns the order, the score per order, and the number of sectors used.
    """
    # Spatial sectors test extrapolation across an omitted region rather than
    # predicting near-neighbour vertices from the same local patch.
    used = [s for s in np.unique(sectors) if (sectors == s).sum() >= min_vertices]
    scores = {}
    for order in orders:
        errs = []
        for s in used:
            test = sectors == s
            try:
                params, _, _ = fit_reference_surface(xy[~test], z[~test], order)
                errs.append(
                    np.median(np.abs(z[test] - eval_reference_surface(params, xy[test], order)))
                )
            except ValueError:
                errs.append(float("inf"))
        scores[order] = float(np.mean(errs)) if errs else float("inf")
    best = min(scores.values())
    if not np.isfinite(best) or len(used) < MIN_CV_SECTORS:
        raise ValueError("Insufficient angular support for reference cross-validation")
    order = min(o for o, v in scores.items() if v <= best * (1 + tie))
    return order, scores, len(used)


def _reference(
    verts: np.ndarray,
    region: np.ndarray,
    inner: float,
    outer: float,
    p: MeasureParams,
    order: int | str,
):
    """Fit the periwound reference surface around ``region``; return depth and fit details."""
    d, _ = cKDTree(verts[region, :2]).query(verts[:, :2], k=1)
    ring = (~region) & (d > inner) & (d < outer)
    xy, z = verts[ring, :2], verts[ring, 2]
    sectors = ring_sectors(xy, verts[region, :2].mean(axis=0), p.n_sectors)
    chosen, scores, n_used = select_reference_order(xy, z, sectors, p.orders, p.min_sector_vertices)
    if order != "auto":
        chosen = int(order)
    params, rms, inl = fit_reference_surface(xy, z, chosen)
    depth = eval_reference_surface(params, verts[:, :2], chosen) - verts[:, 2]  # > 0: below
    info = {
        "order": chosen,
        "cv_scores": scores,
        "sectors_used": n_used,
        "rms": rms,
        "inlier_fraction": float(inl.mean()),
        "ring": ring,
    }
    return depth, d, info


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Percentile ``q`` (0-100) of ``values`` weighted by ``weights``."""
    if (
        not len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or np.sum(weights) <= 0
    ):
        raise ValueError("Invalid area weights")
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, q / 100 * cum[-1])])


def measure_wound(
    verts: np.ndarray, faces: np.ndarray, colors: np.ndarray, p: MeasureParams | None = None
) -> dict:
    """Segment the wound and compute area, size, depth and volume (in mesh units).

    Segmentation combines colour and geometry:
      1. red granulation tissue (colour) -> smoothed, largest region, enclosed
         holes filled; its length L_red sets all measurement distances;
      2. a first reference surface is fitted around it;
      3. vertices deeper than ``crater_depth`` below that surface and within
         ``crater_reach`` of the red region are added (pale slough on the crater
         wall is part of the wound bed but is not red);
      4. the reference surface is refitted around the final wound.

    Conventions: the mesh is expected in the anatomical frame (z away from the
    table); the reference is a height field z = f(x, y) and depth is measured
    along z (depth = f - z, positive below the reference). Areas, means and
    percentiles are weighted by the projected (planimetric) area of the wound
    faces. Volume below the reference counts only the parts below it; the
    volume above it is reported separately.

    The result carries a ``status``: "ok", "uncertain" (usable, with the listed
    reasons) or "rejected" (no defensible depth/volume).
    """
    p = p or MeasureParams()
    adj = vertex_adjacency(len(verts), faces)
    boundary = boundary_vertices(len(verts), faces)
    red = largest_region(smooth_labels(wound_colour_score(colors, p.a_min), adj), adj)
    red = fill_holes(red, adj, boundary)
    if red.sum() < MIN_RED_VERTICES:
        return {"status": "rejected", "reasons": ["no red wound region found"]}
    L_red = length_width(verts[red, :2])[0]
    inner, outer = p.ring_inner * L_red, p.ring_outer * L_red

    depth0, dist0, _ = _reference(verts, red, inner, outer, p, p.order)
    crater = (depth0 > p.crater_depth * L_red) & (dist0 < p.crater_reach * L_red)
    wound = largest_region(smooth_labels(red | crater, adj), adj)
    wound = fill_holes(wound, adj, boundary)

    depth, _, ref = _reference(verts, wound, inner, outer, p, p.order)

    # Clip boundary triangles at the linearly interpolated label threshold.
    triangles = boundary_triangles(verts, faces, wound, depth)
    cross = np.cross(
        triangles[:, 1, :3] - triangles[:, 0, :3], triangles[:, 2, :3] - triangles[:, 0, :3]
    )
    a3d = 0.5 * np.linalg.norm(cross, axis=1)
    a2d = 0.5 * np.abs(cross[:, 2])
    face_depth = triangles[:, :, 3].mean(axis=1)
    area = float(a2d.sum())
    if area <= MIN_PROJECTED_WOUND_AREA:
        raise ValueError("No projected wound area")
    volume_positive, area_negative, volume_negative = clipped_depth_integrals(triangles)
    length, width, axis = length_width(verts[wound, :2])
    res = {
        "wound_mask": wound,
        "ring_mask": ref["ring"],
        "depth": depth,
        "n_wound_vertices": int(wound.sum()),
        "red_vertices": int(red.sum()),
        "crater_vertices_added": int((wound & ~red).sum()),
        "ring_vertices": int(ref["ring"].sum()),
        "length_red_region": float(L_red),
        "planimetric_area": area,
        "surface_area": float(a3d.sum()),
        "length": length,
        "width": width,
        "major_axis_xy": axis,
        "depth_p99": weighted_percentile(face_depth, a2d, 99),
        "mean_depth": float(np.sum(a2d * face_depth) / area),
        "volume_below_reference": float(volume_positive),
        "volume_above_reference": float(volume_negative),
        "area_fraction_above_reference": float(area_negative / area),
        "reference_order": ref["order"],
        "reference_cv_scores": ref["cv_scores"],
        "reference_sectors_used": ref["sectors_used"],
        "reference_fit_rms": ref["rms"],
        "reference_fit_inlier_fraction": ref["inlier_fraction"],
    }
    res["status"], res["reasons"] = _quality(res, p)
    return res


def _quality(r: dict, p: MeasureParams) -> tuple[str, list[str]]:
    """Rejection / uncertainty rules for a measurement."""
    rejected, uncertain = [], []
    if r["depth_p99"] <= 0:
        rejected.append("the wound does not lie below the reference surface")
    if r["area_fraction_above_reference"] > MAX_ABOVE_REFERENCE_FRACTION:
        rejected.append(
            f"{r['area_fraction_above_reference']:.0%} of the wound area lies above the reference surface"
        )
    if r["reference_sectors_used"] < p.n_sectors - MAX_MISSING_RING_SECTORS:
        uncertain.append(
            f"periwound ring covers only {r['reference_sectors_used']} of {p.n_sectors} sectors"
        )
    cv = r["reference_cv_scores"].get(r["reference_order"], float("inf"))
    if r["depth_p99"] > 0 and cv > MAX_REFERENCE_ERROR_DEPTH_RATIO * r["depth_p99"]:
        uncertain.append(
            f"reference prediction error is {cv / r['depth_p99']:.0%} of the 99th-percentile depth"
        )
    if rejected:
        return "rejected", rejected + uncertain
    return ("uncertain" if uncertain else "ok"), uncertain
