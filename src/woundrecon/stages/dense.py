"""Stage 3: dense reconstruction.

Outputs (``outputs/dense/``):
    depth/<image>.npz        fused depth map + agreement count per reference view
    depth/<image>.png        depth preview
    dense.ply                fused dense point cloud with colours and normals (arbitrary units)
    dense_stats.json         per-view and global statistics
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from .. import dense as mvs
from .. import io_utils
from ..config import stage_dir

log = logging.getLogger(__name__)


def load_sfm(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Read registered poses, tracks and native intrinsics from the current SfM NPZ artifact."""
    return dict(np.load(cfg["paths"]["output"] / "sfm" / "sfm.npz", allow_pickle=True))


def load_views(
    cfg: dict[str, Any], sfm: dict[str, np.ndarray], scale: float
) -> tuple[list[mvs.View], list[int]]:
    """Registered images as dense views (native frame, working scale)."""
    views, index = [], []
    paths = {p.name: p for p in io_utils.list_images(cfg["paths"]["images"])}
    for i, name in enumerate(sfm["names"]):
        if not sfm["registered"][i]:
            continue
        R, _ = cv2.Rodrigues(sfm["rvecs"][i])
        views.append(
            mvs.load_view(
                paths[str(name)],
                str(sfm["rules"][i]),
                scale,
                R,
                sfm["tvecs"][i],
                sfm["K"],
                sfm["dist"],
                float(sfm["valid_radius_px"]) or None,
            )
        )
        index.append(i)
    return views, index


def shared_point_matrix(sfm: dict[str, np.ndarray], index: list[int]) -> np.ndarray:
    """Number of sparse points observed by each pair of registered images."""
    pos = {img: k for k, img in enumerate(index)}
    n = len(index)
    vis = np.zeros((n, len(sfm["points"])), bool)
    for img, pt in zip(sfm["obs_image"], sfm["obs_point"], strict=True):
        if img in pos:
            vis[pos[img], pt] = True
    v = vis.astype(np.int32)
    return v @ v.T


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct observed dense depths from registered image pairs.

    The stage fails when no usable dense samples remain or required inputs are
    unreadable. A view with too few consistent pairs is recorded and skipped.

    Args:
        cfg: Validated configuration with resolved paths; reads sfm/sfm.npz, wound
            photographs and dense stereo settings.

    Returns:
        Writes dense/depth maps, dense.ply and dense_stats.json; returns fusion statistics.
        Depths and world points use arbitrary SfM length units; depth-map intrinsics use
        working native-frame pixels.

    Raises:
        OSError: Required inputs or output paths cannot be accessed.
        ValueError: No samples can be fused, or missing normals lack local support.
    """
    out = stage_dir(cfg, "dense")
    depth_dir = out / "depth"
    depth_dir.mkdir(exist_ok=True)
    # Remove depth maps of any earlier run: a view that fails this time must not
    # leave an old depth map (computed with old poses) behind for the mesh stage.
    for stale in list(depth_dir.glob("*.npz")) + list(depth_dir.glob("*.png")):
        stale.unlink()
    d = cfg["dense"]
    params = mvs.StereoParams(
        block_size=d["block_size"],
        max_disparities=d["num_disparities"],
        uniqueness=d.get("uniqueness", 5),
        clahe_clip=d.get("clahe_clip", 3.0),
        depth_rel_tolerance=d["depth_rel_tolerance"],
        min_agreeing_pairs=d["min_consistent_views"],
    )
    sfm = load_sfm(cfg)
    views, index = load_views(cfg, sfm, d["work_scale"])
    shared = shared_point_matrix(sfm, index)
    centre = np.median(sfm["points"], axis=0)
    neighbours = mvs.select_neighbours(
        views,
        shared,
        d["neighbours_per_view"],
        d["min_pair_angle_deg"],
        d["max_pair_angle_deg"],
        centre,
    )

    # Sparse points visible in each registered image (for disparity ranges).
    pos = {img: k for k, img in enumerate(index)}
    seen = [set() for _ in views]
    for img, pt in zip(sfm["obs_image"], sfm["obs_point"], strict=True):
        if img in pos:
            seen[pos[img]].add(int(pt))

    all_pts, all_rgb, all_nrm = [], [], []
    per_view = {}
    for r, ref in enumerate(views):
        depths, pair_stats = [], {}
        for n in neighbours[r]:
            common = np.array(sorted(seen[r] & seen[n]), int)
            depth, st = mvs.pair_depth(ref, views[n], sfm["points"][common], params)
            pair_stats[views[n].name] = st
            if "skipped" not in st:
                depths.append(depth)
        if len(depths) < params.min_agreeing_pairs:
            log.warning("%s: only %d usable stereo pairs, skipped", ref.name, len(depths))
            per_view[ref.name] = {"pairs": pair_stats, "fused_px": 0}
            continue
        fused, count = mvs.fuse_reference(depths, params)
        np.savez_compressed(
            depth_dir / f"{ref.name[:-4]}.npz", depth=fused, count=count, K=ref.K, R=ref.R, t=ref.t
        )
        mvs.save_depth_preview(fused, depth_dir / f"{ref.name[:-4]}.png")

        colour = io_utils.p3_to_srgb(mvs.undistorted_image(ref)[..., ::-1])
        normals = mvs.depth_normals(fused, ref)
        pts, flat = mvs.backproject(fused, ref)
        all_pts.append(pts)
        all_rgb.append(colour.reshape(-1, 3)[flat])
        all_nrm.append(normals.reshape(-1, 3)[flat])
        per_view[ref.name] = {
            "pairs": pair_stats,
            "neighbours": [views[n].name for n in neighbours[r]],
            "fused_px": int(len(pts)),
        }
        log.info("%s: %d neighbours, %d fused depth pixels", ref.name, len(depths), len(pts))

    points = np.vstack(all_pts)
    colors = np.vstack(all_rgb)
    normals = np.vstack(all_nrm)

    # Remove far outliers (beyond the 0.5-99.5 percentile box, enlarged by 10 %).
    lo, hi = np.percentile(points, [0.5, 99.5], axis=0)
    pad = 0.1 * (hi - lo)
    keep = np.all((points > lo - pad) & (points < hi + pad), axis=1)
    points, colors, normals = points[keep], colors[keep], normals[keep]

    extent = float(np.linalg.norm(hi - lo))
    voxel = d.get("fusion_voxel_rel", 0.0008) * extent
    p, c, nrm, counts = mvs.voxel_downsample(points, colors, normals, voxel)
    io_utils.write_ply(
        out / "dense.ply",
        p,
        c,
        normals=nrm,
        comments=["fused dense point cloud, arbitrary SfM units, sRGB colours"],
    )
    stats = {
        "views_fused": sum(1 for v in per_view.values() if v["fused_px"] > 0),
        "raw_points": int(len(points)),
        "points": int(len(p)),
        "voxel": voxel,
        "extent": extent,
        "normal_method": "finite camera-facing gradient votes; missing voxel normals use oriented local PCA (k=24)",
        "normals_finite": bool(np.isfinite(nrm).all()),
        "per_view": per_view,
    }
    io_utils.save_json(out / "dense_stats.json", stats)
    log.info("Dense cloud: %d raw points -> %d after voxel fusion", len(points), len(p))
    return stats
