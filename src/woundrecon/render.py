"""Software rendering of point clouds and meshes for report screenshots.

Point clouds use z-buffer splatting; meshes use perspective-correct triangle
rasterization. No OpenGL context is required. Lambertian and ambient shading
use stored normals to make surface relief visible without colour.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class OrbitCamera:
    """Look-at camera for rendering."""

    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_deg: float = 35.0
    width: int = 1600
    height: int = 1200

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World-to-camera R, t (OpenCV convention: x right, y down, z forward)."""
        z = self.target - self.eye
        z /= np.linalg.norm(z)
        x = np.cross(z, self.up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        return R, -R @ self.eye

    def K(self) -> np.ndarray:
        """Return a (3, 3) pinhole matrix in rendered-image pixels, independent of capture calibration."""
        f = 0.5 * self.width / np.tan(np.radians(self.fov_deg) / 2)
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1]])


def frame_points(
    points: np.ndarray,
    view_dir: np.ndarray,
    up: np.ndarray,
    fov_deg: float = 35.0,
    width: int = 1600,
    height: int = 1200,
    margin: float = 1.15,
    centre: np.ndarray | None = None,
) -> OrbitCamera:
    """Camera looking along ``view_dir`` at ``centre`` (default: cloud median), far enough to frame it."""
    if centre is None:
        centre = np.median(points, axis=0)
    radius = np.max(np.linalg.norm(points - centre, axis=1))
    # Frame the full bounding sphere in the narrower image dimension.
    tangent = np.tan(np.radians(fov_deg) / 2) * min(1, height / width)
    dist = margin * radius / np.sin(np.arctan(tangent))
    d = view_dir / np.linalg.norm(view_dir)
    return OrbitCamera(centre - d * dist, centre, up, fov_deg, width, height)


def splat(
    points: np.ndarray,
    colors: np.ndarray,
    cam: OrbitCamera,
    point_px: int = 2,
    background=(255, 255, 255),
) -> tuple[np.ndarray, np.ndarray]:
    """Render coloured points with a z-buffer. Returns (RGB image, depth buffer)."""
    R, t = cam.pose()
    K = cam.K()
    Xc = points @ R.T + t
    front = Xc[:, 2] > 1e-6
    Xc, colors = Xc[front], colors[front]
    u = K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]
    w, h = cam.width, cam.height
    depth = np.full(h * w, np.inf)
    index = np.full(h * w, -1, np.int64)
    order = np.argsort(-Xc[:, 2])  # far to near: near points overwrite
    half = point_px // 2
    for dx in range(-half, point_px - half):
        for dy in range(-half, point_px - half):
            ui = np.round(u[order]).astype(int) + dx
            vi = np.round(v[order]).astype(int) + dy
            ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            flat = vi[ok] * w + ui[ok]
            z = Xc[order[ok], 2]
            closer = z < depth[flat]
            # Assignment with repeated indices keeps the last write = nearest point (sorted far->near).
            depth[flat[closer]] = z[closer]
            index[flat[closer]] = order[ok][closer]
    img = np.empty((h * w, 3), np.uint8)
    img[:] = background
    hit = index >= 0
    img[hit] = colors[index[hit]]
    return img.reshape(h, w, 3), depth.reshape(h, w)


def draw_polylines(
    img: np.ndarray,
    polylines_world: list[np.ndarray],
    cam: OrbitCamera,
    color=(235, 104, 52),
    thickness: int = 2,
) -> np.ndarray:
    """Project 3D polylines (e.g. camera frusta) and draw them on an RGB image."""
    R, t = cam.pose()
    K = cam.K()
    out = img.copy()
    for line in polylines_world:
        Xc = line @ R.T + t
        if np.any(Xc[:, 2] <= 0):
            continue
        uv = (Xc[:, :2] / Xc[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        cv2.polylines(out, [np.round(uv).astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    return out


def shade(
    normals: np.ndarray, base_rgb: np.ndarray, light_dir: np.ndarray, ambient: float = 0.35
) -> np.ndarray:
    """Lambertian shading of per-point colours (two-sided)."""
    ld = light_dir / np.linalg.norm(light_dir)
    lam = np.abs(normals @ ld)
    k = ambient + (1 - ambient) * lam
    return np.clip(base_rgb.astype(float) * k[:, None], 0, 255).astype(np.uint8)


def colormap(
    values: np.ndarray, vmin: float, vmax: float, cmap: int = cv2.COLORMAP_VIRIDIS
) -> np.ndarray:
    """Map scalars to RGB colours with an OpenCV colour map."""
    x = np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    rgb = cv2.applyColorMap((x * 255).astype(np.uint8).reshape(-1, 1), cmap).reshape(-1, 3)[:, ::-1]
    return rgb


def add_caption(
    img: np.ndarray, text: str, scale_bar: tuple[float, str] | None = None
) -> np.ndarray:
    """Draw a caption (and optionally a scale bar of ``length_px`` labelled ``label``)."""
    out = img.copy()
    cv2.putText(out, text, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
    if scale_bar is not None:
        length_px, label = scale_bar
        h = out.shape[0]
        x0, y0 = 40, h - 50
        cv2.line(out, (x0, y0), (int(x0 + length_px), y0), (20, 20, 20), 4)
        cv2.putText(
            out, label, (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA
        )
    return out


def rasterize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    cam: OrbitCamera,
    background=(255, 255, 255),
) -> tuple[np.ndarray, np.ndarray]:
    """Fill only observed triangles, preserving real holes and depth occlusion.

    Attributes and depth are interpolated in inverse camera depth. Filling
    triangles avoids gaps caused by point-sampling density while preserving
    unobserved mesh boundaries. Inspection cameras frame the complete mesh;
    triangles crossing the near plane are skipped.
    """
    rotation, translation = cam.pose()
    xyz = vertices @ rotation.T + translation
    k = cam.K()
    safe_z = np.maximum(xyz[:, 2], 1e-9)
    uv = xyz[:, :2] / safe_z[:, None] * [k[0, 0], k[1, 1]] + k[:2, 2]
    depth = np.full((cam.height, cam.width), np.inf)
    out = np.empty((cam.height, cam.width, 3), np.uint8)
    out[:] = background
    for face in faces:
        z = xyz[face, 2]
        if np.any(z <= 1e-9):
            continue
        triangle = uv[face]
        x0, y0 = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(
            np.ceil(triangle.max(axis=0)).astype(int), [cam.width - 1, cam.height - 1]
        )
        if x1 < x0 or y1 < y0:
            continue
        a, b = triangle[1] - triangle[0], triangle[2] - triangle[0]
        determinant = a[0] * b[1] - a[1] * b[0]
        if abs(determinant) < 1e-12:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        qx, qy = xx + 0.5 - triangle[0, 0], yy + 0.5 - triangle[0, 1]
        u = (qx * b[1] - qy * b[0]) / determinant
        v = (a[0] * qy - a[1] * qx) / determinant
        bary = np.stack([1 - u - v, u, v], axis=-1)
        inside = np.all(bary >= -1e-9, axis=-1)
        reciprocal = bary / z
        inverse_depth = reciprocal.sum(axis=-1)
        pixel_depth = 1 / np.maximum(inverse_depth, 1e-12)
        target_depth = depth[y0 : y1 + 1, x0 : x1 + 1]
        visible = inside & (pixel_depth < target_depth)
        if visible.any():
            rgb = reciprocal[visible] @ colors[face].astype(float)
            rgb /= inverse_depth[visible, None]
            out[y0 : y1 + 1, x0 : x1 + 1][visible] = np.clip(rgb, 0, 255).astype(np.uint8)
            target_depth[visible] = pixel_depth[visible]
    return out, depth
