"""Surface reconstruction by TSDF fusion (Curless & Levoy 1996) and marching cubes.

Each fused depth map is integrated into a truncated signed distance volume.
A surface is extracted only in grid cells whose eight corner samples were all
observed by at least ``min_weight`` depth maps, so no surface is invented where
there is no data. Vertex colours are sampled from the photographs.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure

from . import io_utils
from .camera import project_points
from .geometry import closest_points_on_triangles, point_to_mesh_distance  # noqa: F401

log = logging.getLogger(__name__)


class TSDFVolume:
    """Axis-aligned TSDF volume with running weighted average."""

    def __init__(
        self, bounds_min: np.ndarray, bounds_max: np.ndarray, voxel: float, truncation: float
    ):
        """Allocate a positive-free-space TSDF grid in the SfM world frame.

        Args:
            bounds_min: (3,) lower world-coordinate bound in SfM length units.
            bounds_max: (3,) upper bound in the same frame and units.
            voxel: Positive isotropic voxel pitch in those length units.
            truncation: Signed-distance clipping band in those length units.
        """
        self.origin = np.asarray(bounds_min, float)
        self.voxel = float(voxel)
        self.trunc = float(truncation)
        self.dims = np.ceil((np.asarray(bounds_max) - self.origin) / voxel).astype(int) + 1
        self.tsdf = np.ones(self.dims, np.float32)
        self.weight = np.zeros(self.dims, np.float32)
        log.info(
            "TSDF volume %s voxels (%.1f M), voxel %.4g",
            self.dims.tolist(),
            self.dims.prod() / 1e6,
            voxel,
        )

    def integrate(
        self, depth: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, slab: int = 16
    ) -> None:
        """Integrate one depth map (undistorted grid, pinhole K, world->camera R, t)."""
        h, w = depth.shape
        nx, ny, nz = self.dims
        gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        for z0 in range(0, nz, slab):  # process in slabs to bound memory
            zs = np.arange(z0, min(z0 + slab, nz))
            P = np.stack(
                [
                    np.repeat(gx[..., None], len(zs), 2),
                    np.repeat(gy[..., None], len(zs), 2),
                    np.broadcast_to(zs, (nx, ny, len(zs))),
                ],
                axis=-1,
            ).reshape(-1, 3)
            Xw = self.origin + (P + 0.5) * self.voxel
            Xc = Xw @ R.T + t
            z = Xc[:, 2]
            front = z > 1e-9
            u = np.full(len(z), -1, int)
            v = np.full(len(z), -1, int)
            u[front] = np.round(K[0, 0] * Xc[front, 0] / z[front] + K[0, 2]).astype(int)
            v[front] = np.round(K[1, 1] * Xc[front, 1] / z[front] + K[1, 2]).astype(int)
            inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            d = np.full(len(z), np.nan, np.float32)
            d[inside] = depth[v[inside], u[inside]]
            sdf = d - z
            upd = np.isfinite(sdf) & (sdf > -self.trunc)  # skip voxels far behind the surface
            if not upd.any():
                continue
            val = np.clip(sdf[upd] / self.trunc, -1, 1).astype(np.float32)
            idx = P[upd]
            ii, jj, kk = idx[:, 0], idx[:, 1], idx[:, 2]
            w_old = self.weight[ii, jj, kk]
            self.tsdf[ii, jj, kk] = (self.tsdf[ii, jj, kk] * w_old + val) / (w_old + 1)
            self.weight[ii, jj, kk] = w_old + 1

    def extract_mesh(
        self, min_weight: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Marching cubes restricted to fully observed grid cells.

        A cell may produce a surface only if all eight of its corner samples were
        observed by at least ``min_weight`` depth maps. scikit-image's ``mask``
        gates a cell by a single corner sample, so a cell touching never-observed
        samples (which keep the +1 initialisation) could still create faces: an
        invented boundary, and a hidden second sheet at the back of the
        truncation band. Those faces are removed here.

        Returns vertices (world), faces, vertex normals and extraction statistics.
        """
        supported = self.weight >= min_weight
        try:
            verts, faces, normals, _ = measure.marching_cubes(
                self.tsdf,
                level=0.0,
                spacing=(self.voxel,) * 3,
                mask=supported,
                allow_degenerate=False,
            )
        except RuntimeError:  # no zero crossing at all
            return (
                np.zeros((0, 3)),
                np.zeros((0, 3), int),
                np.zeros((0, 3)),
                {"faces_extracted": 0, "faces_unsupported": 0},
            )
        keep = cells_fully_supported(verts / self.voxel, faces, supported)
        stats = {"faces_extracted": int(len(faces)), "faces_unsupported": int((~keep).sum())}
        verts, faces, used = compact_mesh(verts, faces[keep])
        verts = verts + self.origin + 0.5 * self.voxel
        # Our TSDF is positive in observed free space (depth - camera z).
        # scikit-image's returned gradient normals have the opposite sign to
        # the retained face winding for this convention. Flip the normals so
        # they face the observing cameras, preserving all positions and faces.
        return verts, faces, -normals[used], stats


def cells_fully_supported(
    grid_verts: np.ndarray, faces: np.ndarray, supported: np.ndarray
) -> np.ndarray:
    """True for faces whose grid cell has all eight corner samples ``supported``.

    ``grid_verts`` are vertex positions in grid-index units (sample i at i).
    Every marching-cubes triangle lies inside one cell, identified from the
    triangle centroid.
    """
    if len(faces) == 0:
        return np.zeros(0, bool)
    dims = np.array(supported.shape)
    cell = np.floor(grid_verts[faces].mean(axis=1) + 1e-9).astype(int)
    cell = np.clip(cell, 0, dims - 2)
    ok = np.ones(len(faces), bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ok &= supported[cell[:, 0] + dx, cell[:, 1] + dy, cell[:, 2] + dz]
    return ok


def compact_mesh(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop vertices no face uses; returns new vertices, re-indexed faces, used vertex ids."""
    used = np.unique(faces)
    remap = -np.ones(len(verts), int)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces], used


def largest_components(verts: np.ndarray, faces: np.ndarray, keep_fraction: float = 0.02):
    """Remove small disconnected mesh fragments (smaller than ``keep_fraction`` of the largest)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(verts)
    e = np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    face_label = labels[faces[:, 0]]
    sizes = np.bincount(face_label)
    keep_labels = np.flatnonzero(sizes >= keep_fraction * sizes.max())
    return compact_mesh(verts, faces[np.isin(face_label, keep_labels)])


def transfer_colors(
    verts: np.ndarray, points: np.ndarray, colors: np.ndarray, k: int = 4
) -> np.ndarray:
    """Vertex colours as the distance-weighted mean of the k nearest dense points."""
    tree = cKDTree(points)
    d, idx = tree.query(verts, k=k)
    wts = 1.0 / (d + 1e-9)
    wts /= wts.sum(axis=1, keepdims=True)
    linear = io_utils._srgb_eotf(colors[idx].astype(float) / 255.0)
    mixed = np.sum(linear * wts[..., None], axis=1)
    return np.rint(255 * io_utils._srgb_oetf(np.clip(mixed, 0, 1))).astype(np.uint8)


def bilinear_sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a (H, W, C) uint8 image at sub-pixel positions (N, 2)."""
    h, w = image.shape[:2]
    x = np.clip(uv[:, 0], 0, w - 1.001)
    y = np.clip(uv[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    img = image.astype(np.float32)
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bottom = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return np.clip(top * (1 - fy) + bottom * fy + 0.5, 0, 255).astype(np.uint8)


def color_from_views(
    verts: np.ndarray,
    normals: np.ndarray,
    views: list,
    top_k: int = 3,
    zbuffer_scale: float = 0.25,
    depth_tol_rel: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertex colours sampled directly from the photographs (view selection + blending).

    For every view, vertices are projected with the full camera model; a
    z-buffer built from the mesh itself decides visibility. Each vertex takes a
    weighted mean of its ``top_k`` best views, weighted by cos^4 of the angle
    between the surface normal and the viewing ray, and down-weighted on
    specular highlights. This is sharper than interpolating from the dense
    cloud and avoids baking highlights into the texture.

    Args:
        views: objects with ``image`` (BGR, native frame), ``R``, ``t``, ``K``
            (for that image size) and ``dist``.

    Returns:
        (N, 3) uint8 sRGB colours and (N,) number of views that saw each vertex.
    """
    n = len(verts)
    scores = np.zeros((n, len(views)), np.float32)
    cols = np.zeros((n, len(views), 3), np.float32)
    for k, v in enumerate(views):
        Xc = verts @ v.R.T + v.t
        z = Xc[:, 2]
        front = z > 1e-9
        f4 = np.array([v.K[0, 0], v.K[1, 1], v.K[0, 2], v.K[1, 2]])
        uv = np.full((n, 2), -1.0)
        uv[front] = project_points(Xc[front], f4, v.dist)
        h, w = v.image.shape[:2]
        inside = (
            front & (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
        )
        # z-buffer at reduced resolution: nearest mesh depth per cell.
        zw, zh = int(w * zbuffer_scale) + 1, int(h * zbuffer_scale) + 1
        cell = np.floor(uv[inside, 1] * zbuffer_scale).astype(int) * zw + np.floor(
            uv[inside, 0] * zbuffer_scale
        ).astype(int)
        zbuf = np.full(zw * zh, np.inf)
        np.minimum.at(zbuf, cell, z[inside])
        visible = np.zeros(n, bool)
        visible[inside] = z[inside] <= zbuf[cell] * (1 + depth_tol_rel)
        # Facing term: cosine between the normal and the direction to the camera.
        centre = -v.R.T @ v.t
        to_cam = centre - verts
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        cosang = np.abs(np.sum(normals * to_cam, axis=1))
        idx = np.flatnonzero(visible)
        if len(idx) == 0:
            continue
        sample = bilinear_sample(v.image, uv[idx])
        rgb = io_utils.p3_to_srgb(sample[:, ::-1]).astype(np.float32)
        hsv = cv2.cvtColor(sample.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        specular = (hsv[:, 2] > 235) & (hsv[:, 1] < 40)
        scores[idx, k] = cosang[idx] ** 4 * np.where(specular, 0.05, 1.0)
        cols[idx, k] = rgb
    order = np.argsort(-scores, axis=1)[:, :top_k]
    s = np.take_along_axis(scores, order, axis=1)
    c = np.take_along_axis(cols, order[..., None], axis=1)
    wsum = s.sum(axis=1, keepdims=True)
    out_linear = np.where(
        wsum > 0,
        (io_utils._srgb_eotf(c / 255.0) * s[..., None]).sum(axis=1) / np.maximum(wsum, 1e-12),
        io_utils._srgb_eotf(np.array(128 / 255.0)),
    )
    out = 255 * io_utils._srgb_oetf(np.clip(out_linear, 0, 1))
    return np.clip(out, 0, 255).astype(np.uint8), (scores > 0).sum(axis=1)


def mesh_area(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area of every triangle."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
