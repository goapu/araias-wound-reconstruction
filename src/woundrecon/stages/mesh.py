"""Stage 4: surface mesh from the fused depth maps (TSDF + marching cubes).

Outputs (``outputs/mesh/``):
    mesh.ply           triangle mesh with vertex colours and normals (arbitrary SfM units)
    mesh_stats.json    volume size, vertex/face counts, faces removed for missing support
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .. import io_utils
from .. import mesh as surf
from ..config import stage_dir
from .dense import load_sfm, load_views

log = logging.getLogger(__name__)


def current_depth_maps(dense_dir: Path, sfm: dict[str, np.ndarray]) -> list[Path]:
    """Depth maps written by the latest dense run, verified against the current SfM poses.

    The dense statistics list the views that produced a fused depth map; any
    other file in the folder is ignored. A listed file that is missing, or whose
    stored pose differs from the current reconstruction, stops the stage, so
    depth maps of an earlier reconstruction can never be mixed in.
    """
    stats = io_utils.load_json(dense_dir / "dense_stats.json")
    names = [str(n) for n in sfm["names"]]
    files = []
    for view, info in stats["per_view"].items():
        if info.get("fused_px", 0) == 0:
            continue
        f = dense_dir / "depth" / f"{Path(view).stem}.npz"
        if not f.exists():
            raise FileNotFoundError(f"Depth map listed by the dense stage is missing: {f}")
        d = np.load(f)
        i = names.index(view)
        R = cv2.Rodrigues(sfm["rvecs"][i])[0]
        if not (
            np.allclose(d["R"], R, atol=1e-9)
            and np.allclose(d["t"].ravel(), sfm["tvecs"][i].ravel(), atol=1e-9)
        ):
            raise ValueError(
                f"{f.name} was computed with different camera poses; rerun the dense stage"
            )
        if "K" in sfm and "image_size" in sfm:
            from ..dense import scale_intrinsics

            scale = d["depth"].shape[1] / sfm["image_size"][0]
            expected = scale_intrinsics(sfm["K"], float(scale))
            if not np.allclose(d["K"], expected, rtol=0, atol=1e-8):
                raise ValueError(
                    f"{f.name} has stale camera intrinsics; rerun dense reconstruction"
                )
        files.append(f)
    return sorted(files)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fuse current depth observations into a supported coloured TSDF mesh.

    The stage fails when depth poses/intrinsics do not match SfM, or no
    sufficiently supported surface remains.

    Args:
        cfg: Validated configuration with resolved paths; reads dense/dense.ply, current
            depth maps, SfM poses and texture photographs.

    Returns:
        Writes mesh/mesh.ply and mesh_stats.json; returns extraction/texture statistics.
        Vertices and normals use the SfM world frame; vertex lengths retain arbitrary SfM
        units.

    Raises:
        OSError: Required inputs or output paths cannot be accessed.
        ValueError: Depth metadata is stale or supported surface extraction is empty.
    """
    out = stage_dir(cfg, "mesh")
    dense_dir = cfg["paths"]["output"] / "dense"
    cloud = io_utils.read_ply_vertices(dense_dir / "dense.ply")
    pts, cols = cloud["points"], cloud["colors"]

    lo, hi = np.percentile(pts, [0.2, 99.8], axis=0)
    extent = float(np.linalg.norm(hi - lo))
    voxel = cfg["mesh"]["voxel_size_rel"] * extent
    pad = 3 * voxel
    vol = surf.TSDFVolume(lo - pad, hi + pad, voxel, cfg["mesh"]["truncation_voxels"] * voxel)

    # Integrate exactly the depth maps the current dense run produced (listed in
    # its statistics), and check that each was computed with the current poses.
    sfm = load_sfm(cfg)
    depth_files = current_depth_maps(dense_dir, sfm)
    for f in depth_files:
        d = np.load(f)
        vol.integrate(d["depth"], d["K"], d["R"], d["t"])
        log.info("Integrated %s", f.stem)

    verts, faces, normals, extract_stats = vol.extract_mesh(
        min_weight=cfg["mesh"].get("min_weight", 2)
    )
    log.info(
        "Marching cubes: %d faces, %d removed for incomplete observation support",
        extract_stats["faces_extracted"],
        extract_stats["faces_unsupported"],
    )
    if not len(faces):
        raise ValueError("No sufficiently observed surface; recapture or inspect depth support")
    verts, faces, used = surf.largest_components(
        verts, faces, cfg["mesh"].get("keep_fraction", 0.02)
    )
    normals = normals[used]
    # Texture: colours sampled from the photographs (best views, highlight-aware);
    # vertices no image sees keep the colour interpolated from the dense cloud.
    views, _ = load_views(cfg, sfm, cfg["mesh"].get("texture_scale", 0.5))
    colors, n_views = surf.color_from_views(verts, normals, views)
    unseen = n_views == 0
    if unseen.any():
        colors[unseen] = surf.transfer_colors(verts[unseen], pts, cols)
    io_utils.write_ply(
        out / "mesh.ply",
        verts,
        colors,
        normals=normals,
        faces=faces,
        comments=["TSDF mesh, arbitrary SfM units, sRGB vertex colours"],
    )
    stats = {
        "voxel": voxel,
        "volume_dims": vol.dims.tolist(),
        "depth_maps": len(depth_files),
        "min_weight": cfg["mesh"].get("min_weight", 2),
        **extract_stats,
        "unsupported_faces_retained": 0,
        "colour_fusion": "view-selected linear-light sRGB averaging",
        "vertices": int(len(verts)),
        "faces": int(len(faces)),
        "vertices_textured_from_images": int((~unseen).sum()),
        "median_views_per_vertex": float(np.median(n_views)),
        "surface_area_units2": float(surf.mesh_area(verts, faces).sum()),
    }
    io_utils.save_json(out / "mesh_stats.json", stats)
    log.info("Mesh: %d vertices, %d faces", len(verts), len(faces))
    return stats
