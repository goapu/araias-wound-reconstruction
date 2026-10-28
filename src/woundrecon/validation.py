"""Release checks for geometry, wound coverage and artifact integrity.

A topologically valid mesh can still be unusable for wound measurement. These
checks therefore distinguish outer open boundaries from boundaries touching
an explicitly saved wound mask. They establish internal consistency, not
physical accuracy or validity of the heuristic mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import io_utils
from .provenance import code_version, sha256_file
from .stages import STAGE_ORDER

MAX_NORMAL_LENGTH_ERROR = 1e-5  # Float32 PLY normalisation tolerance; dimensionless.
MIN_NORMAL_WINDING_AGREEMENT = 0.95  # Detect global reversal while allowing local noise.


def verify_normals(cloud: dict, label: str) -> dict:
    """A PLY is usable for shading only if every saved normal is finite and unit length."""
    normals = cloud.get("normals")
    if normals is None or normals.shape != cloud["points"].shape:
        raise ValueError(f"{label}: missing or mismatched normals")
    if not np.isfinite(normals).all():
        raise ValueError(f"{label}: non-finite normals")
    error = float(np.max(np.abs(np.linalg.norm(normals, axis=1) - 1)))
    if error > MAX_NORMAL_LENGTH_ERROR:
        raise ValueError(f"{label}: normals must have unit length")
    return {"count": len(normals), "maximum_unit_length_error": error}


def verify_mesh_normals(
    cloud: dict, faces: np.ndarray, label: str, minimum=MIN_NORMAL_WINDING_AGREEMENT
) -> dict:
    """Require normal/winding agreement for at least 95% of faces and surface area.

    Local gradient/smooth-normal disagreements are possible at sharp/noisy features,
    so the gate detects global reversal rather than requiring exact face alignment.
    Winding agreement alone is not proof of outward orientation on an arbitrary
    open surface; the TSDF camera-facing sign is checked by a known-plane regression.
    """
    result = verify_normals(cloud, label)
    vertices = cloud["points"]
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError(f"{label}: invalid triangle indices")
    triangle = vertices[faces]
    cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    if not np.isfinite(area2).all() or np.any(area2 <= 0):
        raise ValueError(f"{label}: invalid or degenerate triangles")
    # Compare saved normals to geometric winding, independently of the renderer.
    average = cloud["normals"][faces].mean(axis=1)
    dot = np.einsum("ij,ij->i", cross, average)
    agreement = dot > 0
    face_fraction = float(agreement.mean())
    area_fraction = float(area2[agreement].sum() / area2.sum())
    result["orientation"] = {
        "definition": "positive dot product of face cross product and mean vertex normal",
        "face_agreement_fraction": face_fraction,
        "area_weighted_agreement_fraction": area_fraction,
        "minimum_required_fraction": minimum,
    }
    if min(face_fraction, area_fraction) < minimum:
        raise ValueError(
            f"{label}: normals disagree with triangle winding "
            f"(faces {face_fraction:.3%}, area {area_fraction:.3%}; require {minimum:.0%})"
        )
    return result


def mesh_topology(vertices: np.ndarray, faces: np.ndarray, wound=None) -> dict:
    """Count components and edge incidence without filling missing surface."""
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite 3D points")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("A nonempty triangle surface is required")
    if (
        not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("Triangle indices are invalid")
    triangles = vertices[faces]
    area2 = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    if np.any(area2 <= 0):
        raise ValueError("Degenerate triangles cannot support measurement")
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices))
    )
    components, labels = connected_components(graph, directed=False)
    result = dict(
        vertices=len(vertices),
        faces=len(faces),
        components=components,
        component_vertices=sorted(np.bincount(labels).tolist(), reverse=True),
        nonmanifold_edges=int((counts > 2).sum()),
        boundary_edges=int((counts == 1).sum()),
    )
    if wound is not None:
        wound = np.asarray(wound, bool)
        if wound.shape != (len(vertices),):
            raise ValueError("Wound mask does not match the mesh vertex ordering")
        result["boundary_edges_touching_wound"] = int(wound[edges[counts == 1]].any(axis=1).sum())
        result["mask_scope"] = "heuristic labels; not an independently reviewed clinical boundary"
    return result


def verify_outputs(output: Path, data_root: Path | None = None, check_code=True) -> dict:
    """Verify a completed run without mutating the evidence being checked."""
    output = Path(output)
    manifest = io_utils.load_json(output / "run_manifest.json")
    stages = manifest.get("stages", [])
    if [stage.get("stage") for stage in stages] != STAGE_ORDER:
        raise ValueError("Release requires every pipeline stage once, in execution order")
    if any(stage.get("status") != "ok" for stage in stages):
        raise ValueError("A failed pipeline stage cannot be released")
    if not manifest.get("finished_utc") or not manifest.get("outputs_sha256"):
        raise ValueError("An unfinished run or missing artifact inventory cannot be released")
    if check_code and manifest["code_version"] != code_version():
        raise ValueError("Source code differs from the completed run")
    for name, expected in manifest["outputs_sha256"].items():
        path = output / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Missing or changed artifact: {name}")
    if data_root is not None:
        for name, expected in manifest["inputs_sha256"].items():
            path = Path(data_root) / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Missing or changed source image: {name}")
    sparse = io_utils.load_json(output / "sfm/sfm_stats.json")
    if sparse["registered"] != sparse["images"]:
        raise ValueError("Incomplete sparse registration")
    dense = io_utils.load_json(output / "dense/dense_stats.json")
    cloud = io_utils.read_ply_vertices(output / "dense/dense.ply")
    if len(cloud["points"]) != dense["points"] or not np.isfinite(cloud["points"]).all():
        raise ValueError("Dense cloud count or coordinates differ from the run evidence")
    normal_checks = {"dense": verify_normals(cloud, "Dense cloud")}
    thinned = io_utils.read_ply_vertices(output / "scale/dense_aligned_thinned.ply")
    if not np.isfinite(thinned["points"]).all():
        raise ValueError("Thinned inspection cloud contains non-finite coordinates")
    normal_checks["inspection_cloud"] = verify_normals(thinned, "Inspection cloud")
    path = output / "scale/mesh_aligned.ply"
    mesh = io_utils.read_ply_vertices(path)
    faces = io_utils.read_ply_faces(path)
    normal_checks["mesh"] = verify_mesh_normals(mesh, faces, "Aligned surface mesh")
    raw_path = output / "mesh/mesh.ply"
    raw = io_utils.read_ply_vertices(raw_path)
    normal_checks["raw_mesh"] = verify_mesh_normals(
        raw, io_utils.read_ply_faces(raw_path), "Raw surface mesh"
    )
    depth_path = output / "measure/wound_depth.ply"
    depth_mesh = io_utils.read_ply_vertices(depth_path)
    depth_faces = io_utils.read_ply_faces(depth_path)
    if not np.array_equal(depth_mesh["points"], mesh["points"]) or not np.array_equal(
        depth_faces, faces
    ):
        raise ValueError("Experimental depth mesh does not match the aligned surface")
    normal_checks["wound_depth"] = verify_mesh_normals(
        depth_mesh, depth_faces, "Depth-coloured mesh"
    )
    masks = np.load(output / "measure/wound_masks.npz")
    topology = mesh_topology(mesh["points"], faces, masks["wound"])
    if topology["nonmanifold_edges"] or topology["boundary_edges_touching_wound"]:
        raise ValueError("Surface coverage gate failed in the saved wound region")
    surface = io_utils.load_json(output / "mesh/mesh_stats.json")
    if topology["vertices"] != surface["vertices"] or topology["faces"] != surface["faces"]:
        raise ValueError("Surface counts differ from the run evidence")
    if surface.get("unsupported_faces_retained", 0) != 0:
        raise ValueError("Unsupported surface faces cannot be released")
    measure = io_utils.load_json(output / "measure/measurements.json")
    if measure["status"] == "ok":
        raise ValueError("Unvalidated wound descriptors must retain an uncertain/rejected status")
    for name in ["01_textured_top.png", "02_geometry_oblique.png", "05_textured_oblique.png"]:
        if not (output / "renders" / name).is_file():
            raise ValueError(f"Missing inspection image: {name}")
    return dict(
        status="passed",
        source_images=len(manifest["inputs_sha256"]),
        registered_images=sparse["registered"],
        dense_points=dense["points"],
        normals=normal_checks,
        topology=topology,
        measurement_status=measure["status"],
        physical_accuracy="not independently established",
        units=measure["units"],
    )


def main(argv=None):
    """Verify saved artifacts from CLI paths and print the resulting evidence as JSON."""
    parser = argparse.ArgumentParser(description="Verify a completed reconstruction")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "outputs")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(verify_outputs(args.output_dir, args.data_root), indent=2))
