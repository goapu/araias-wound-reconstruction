"""Triangle geometry used by surface validation and experimental integration.

Clipping is exact for linearly interpolated fields on the supplied triangles.
It does not validate a physical boundary or an inferred reference surface.
The distance search verifies the centroid shortlist with a complete candidate
ball so large or irregular triangles cannot be silently missed.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# Numerical tolerances in the caller's input units, not physical error bounds.
CANDIDATE_RADIUS_PADDING = 1e-10  # Length-unit slack at the candidate-ball boundary.
MIN_PROJECTED_DOUBLE_AREA = 1e-14  # Squared-length floor for projected degeneracy.
BARYCENTRIC_SLACK = 1e-7  # Dimensionless allowance at shared triangle edges.
LABEL_CONTOUR = 0.5  # Halfway contour of linearly interpolated binary labels.


def closest_points_on_triangles(
    P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """Find the closest point on each corresponding triangle (Ericson, section 5.1.5).

    Args:
        P: (N, 3) query points in the same Cartesian frame and length units as A/B/C.
        A: (N, 3) first vertices of non-degenerate triangles.
        B: (N, 3) second vertices.
        C: (N, 3) third vertices.

    Returns:
        (N, 3) closest points in the input frame and length units.
    """

    def dot(u, v):
        """Compute one Euclidean dot product per corresponding row."""
        return np.einsum("ij,ij->i", u, v)

    ab, ac, ap, bp, cp = B - A, C - A, P - A, P - B, P - C
    d1, d2, d3, d4, d5, d6 = (
        dot(ab, ap),
        dot(ac, ap),
        dot(ab, bp),
        dot(ac, bp),
        dot(ab, cp),
        dot(ac, cp),
    )
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        inside = A + ab * (vb / denom)[:, None] + ac * (vc / denom)[:, None]
        on_ab = A + ab * (d1 / (d1 - d3))[:, None]
        on_ac = A + ac * (d2 / (d2 - d6))[:, None]
        on_bc = B + (C - B) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]
    # Voronoi regions in Ericson's order; the first match wins.
    regions = [
        ((d1 <= 0) & (d2 <= 0), A),
        ((d3 >= 0) & (d4 <= d3), B),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), on_ab),
        ((d6 >= 0) & (d5 <= d6), C),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), on_ac),
        ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), on_bc),
    ]
    out = inside.copy()
    done = np.zeros(len(P), bool)
    for cond, q in regions:
        m = cond & ~done
        out[m] = q[m]
        done |= m
    return out


def point_to_mesh_distance(points, verts, faces, k=24):
    """Compute complete nearest-triangle distances for a Cartesian mesh.

    Args:
        points: (N, 3) queries in the same frame and length units as verts.
        verts: (V, 3) finite mesh coordinates, either SfM or anatomical coordinates.
        faces: (F, 3) integer indices of non-degenerate triangles; F must be positive.
        k: Initial centroid shortlist size, used only to obtain an upper bound.

    Returns:
        (N,) Euclidean distances in the input length units. The candidate-ball
        expansion checks all triangles able to improve the provisional bound.
    """
    triangles = verts[faces]
    centres = triangles.mean(axis=1)
    radius = float(np.linalg.norm(triangles - centres[:, None], axis=2).max())
    tree = cKDTree(centres)
    distances = []
    for p in points:
        _, ids = tree.query(p, k=min(k, len(faces)))
        ids = np.atleast_1d(ids)
        tri = triangles[ids]
        q = closest_points_on_triangles(
            np.repeat(p[None], len(ids), axis=0), tri[:, 0], tri[:, 1], tri[:, 2]
        )
        best = float(np.linalg.norm(q - p, axis=1).min())
        # Triangle inequality: a closer point can only lie in a triangle whose
        # centroid is within best + radius. Check that complete candidate ball.
        ids = tree.query_ball_point(p, best + radius + CANDIDATE_RADIUS_PADDING)
        tri = triangles[ids]
        q = closest_points_on_triangles(
            np.repeat(p[None], len(ids), axis=0), tri[:, 0], tri[:, 1], tri[:, 2]
        )
        distances.append(float(np.linalg.norm(q - p, axis=1).min()))
    return np.array(distances)


def clip_polygon(data, values, threshold):
    """Clip a convex polygon by an affine scalar threshold, preserving attributes.

    Args:
        data: (N, D) ordered vertices and interpolated attributes; spatial columns
            retain their caller-defined frame and units.
        values: (N,) scalar field at those vertices.
        threshold: Inclusive lower bound, in the same units as values.

    Returns:
        (K, D) clipped vertices, or an empty array when nothing survives.
        All columns are interpolated at edge crossings.
    """
    output = []
    for i in range(len(data)):
        a, b = data[i - 1], data[i]
        va, vb = values[i - 1], values[i]
        ina, inb = va >= threshold, vb >= threshold
        if ina != inb:
            alpha = (threshold - va) / (vb - va)
            output.append(a + alpha * (b - a))
        if inb:
            output.append(b)
    return np.asarray(output)


def fan_triangles(poly):
    """Triangulate a convex ordered polygon without changing vertex attributes.

    Args:
        poly: (N, D) polygon vertices/attributes in a common frame and units.

    Returns:
        List of (3, D) triangles sharing the first vertex; empty for N below three.
    """
    return (
        [np.stack([poly[0], poly[i], poly[i + 1]]) for i in range(1, len(poly) - 1)]
        if len(poly) >= 3
        else []
    )


def boundary_triangles(verts, faces, wound, depth):
    """Clip mesh triangles to the halfway contour of binary vertex labels.

    Args:
        verts: (V, 3) anatomical-frame coordinates in the scale stage's length units.
        faces: (F, 3) triangle vertex indices.
        wound: (V,) boolean wound labels, interpolated linearly across each face.
        depth: (V,) signed reference-minus-surface height in the same length units.

    Returns:
        (K, 3, 4) triangles with columns x, y, z, depth. Partial boundary faces
        retain their geometrically clipped contribution.

    Raises:
        ValueError: No triangle intersects the labelled region.
    """
    labels = wound[faces]
    complete = faces[labels.all(axis=1)]
    output = (
        [np.concatenate([verts[complete], depth[complete, None]], axis=2)] if len(complete) else []
    )
    partial = faces[labels.any(axis=1) & ~labels.all(axis=1)]
    clipped = []
    for face in partial:
        data = np.c_[verts[face], depth[face]]
        clipped.extend(fan_triangles(clip_polygon(data, wound[face].astype(float), LABEL_CONTOUR)))
    if clipped:
        output.append(np.asarray(clipped))
    if not output:
        raise ValueError("No wound triangles")
    return np.concatenate(output)


def clipped_depth_integrals(triangles):
    """Integrate a signed affine depth field over its positive and negative parts.

    Args:
        triangles: (K, 3, 4) anatomical-frame x/y/z/depth vertices. The x/y plane
            defines projected area; all spatial/depth columns share length units.

    Returns:
        Positive-depth volume, negative-depth projected area, and the magnitude
        of negative-depth volume, in cubed/squared/cubed input units.
    """
    xyz = triangles[:, :, :3]
    depth = triangles[:, :, 3]
    # Height-field volume uses x/y projected area, rather than curved surface area.
    areas = 0.5 * np.abs(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])[:, 2])
    positive = depth.min(axis=1) >= 0
    negative = depth.max(axis=1) <= 0
    vp = float(np.sum(areas[positive] * depth[positive].mean(axis=1)))
    vn = float(np.sum(areas[negative] * (-depth[negative]).mean(axis=1)))
    an = float(areas[negative].sum())
    # Split mixed-sign faces at depth zero so positive and negative relief do not cancel.
    for tri in triangles[~positive & ~negative]:
        for sign in (1, -1):
            pieces = fan_triangles(clip_polygon(tri, sign * tri[:, 3], 0.0))
            for t in pieces:
                area = 0.5 * abs(np.cross(t[1, :3] - t[0, :3], t[2, :3] - t[0, :3])[2])
                integral = area * sign * t[:, 3].mean()
                if sign == 1:
                    vp += integral
                else:
                    vn += integral
                    an += area
    return vp, an, vn


def footprint_membership(points, vertices, faces):
    """Test projected triangle coverage without selecting by 3D residual size.

    Args:
        points: (N, 3) query points in the mesh's anatomical frame and length units.
        vertices: (V, 3) mesh vertices in the same frame.
        faces: (F, 3) triangle indices; F must be positive.

    Returns:
        (N,) boolean membership in the union of x/y-projected triangles. Query
        heights are ignored and holes outside that union remain outside.
    """
    from scipy.spatial import cKDTree

    tri = vertices[faces, :2]
    centres = tri.mean(axis=1)
    radius = float(np.linalg.norm(tri - centres[:, None], axis=2).max())
    tree = cKDTree(centres)
    result = np.zeros(len(points), bool)
    for i, p in enumerate(points[:, :2]):
        ids = tree.query_ball_point(p, radius + CANDIDATE_RADIUS_PADDING)
        if not ids:
            continue
        t = tri[ids]
        a = t[:, 1] - t[:, 0]
        b = t[:, 2] - t[:, 0]
        q = p - t[:, 0]
        # Cramer's rule solves the projected 2D triangle barycentric coordinates.
        den = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        valid = np.abs(den) > MIN_PROJECTED_DOUBLE_AREA
        den = np.where(valid, den, 1)
        u = (q[:, 0] * b[:, 1] - q[:, 1] * b[:, 0]) / den
        v = (a[:, 0] * q[:, 1] - a[:, 1] * q[:, 0]) / den
        result[i] = bool(
            np.any(
                valid
                & (u >= -BARYCENTRIC_SLACK)
                & (v >= -BARYCENTRIC_SLACK)
                & (u + v <= 1 + BARYCENTRIC_SLACK)
            )
        )
    return result
