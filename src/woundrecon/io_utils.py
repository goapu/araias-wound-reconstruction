"""Image and geometry I/O.

Orientation handling
--------------------
The images were saved "rotated the way the phone displayed them" and their
EXIF orientation tags were removed. Portrait images (3024 x 4032) are rotated
copies of the native landscape sensor frame (4032 x 3024). A single camera
model is only valid in the native frame, so every image (and every detected
2D point) is mapped into that frame before any geometry is computed.

The mapping is described by a transform name:

    "none"  image already in the native frame
    "ccw"   native = image rotated 90 degrees counter-clockwise
    "cw"    native = image rotated 90 degrees clockwise
    "180"   native = image rotated by 180 degrees

Colour management
-----------------
The PNGs carry a Display P3 ICC profile. OpenCV ignores ICC profiles, so
colours used for textured output are converted explicitly from Display P3 to
sRGB (both share the sRGB transfer curve and the D65 white point).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

_ROTATE_CODE = {
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "cw": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
}

# Linear-light conversion matrix from Display P3 (D65) to sRGB (D65).
_P3_TO_SRGB = np.array(
    [
        [1.2249401, -0.2249404, 0.0000000],
        [-0.0420569, 1.0420571, 0.0000000],
        [-0.0196376, -0.0786361, 1.0982735],
    ]
)


# --------------------------------------------------------------------------- #
# Listing and loading
# --------------------------------------------------------------------------- #
def list_images(folder: Path) -> list[Path]:
    """Return the image files of a folder, sorted by name."""
    return sorted(
        p
        for p in Path(folder).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an 8-bit image as displayed (BGR or grayscale). Raises if unreadable."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise OSError(f"Cannot read image: {path}")
    return img


def is_portrait(shape: tuple[int, ...]) -> bool:
    """True when an image (H, W, ...) is taller than wide."""
    return shape[0] > shape[1]


# --------------------------------------------------------------------------- #
# Orientation: displayed frame <-> native sensor frame
# --------------------------------------------------------------------------- #
def native_transform(shape: tuple[int, ...], portrait_rule: str, landscape_rule: str) -> str:
    """Select the transform that maps a displayed image into the native frame."""
    return portrait_rule if is_portrait(shape) else landscape_rule


def to_native_image(img: np.ndarray, transform: str) -> np.ndarray:
    """Rotate a displayed image into the native sensor frame."""
    return img if transform == "none" else cv2.rotate(img, _ROTATE_CODE[transform])


def to_native_points(
    pts: np.ndarray, displayed_shape: tuple[int, ...], transform: str
) -> np.ndarray:
    """Map pixel coordinates (N, 2) from the displayed image into the native frame.

    Pixel centres are at integer coordinates; the mapping is exact for the
    rotations performed by ``cv2.rotate``.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    h, w = displayed_shape[:2]
    x, y = pts[:, 0], pts[:, 1]
    if transform == "none":
        out = np.stack([x, y], axis=1)
    elif transform == "ccw":
        out = np.stack([y, (w - 1) - x], axis=1)
    elif transform == "cw":
        out = np.stack([(h - 1) - y, x], axis=1)
    elif transform == "180":
        out = np.stack([(w - 1) - x, (h - 1) - y], axis=1)
    else:
        raise ValueError(f"Unknown transform: {transform}")
    return out


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #
def _srgb_eotf(v: np.ndarray) -> np.ndarray:
    """sRGB / Display P3 encoded value -> linear light (inputs in [0, 1])."""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb_oetf(v: np.ndarray) -> np.ndarray:
    """Linear light -> sRGB encoded value (inputs in [0, 1])."""
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def p3_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert Display P3 colours (uint8 RGB, any leading shape) to sRGB uint8 RGB.

    Out-of-gamut colours (the saturated reds of the wound bed) are clipped.
    """
    lin = _srgb_eotf(rgb.astype(np.float64) / 255.0)
    lin = np.clip(lin @ _P3_TO_SRGB.T, 0.0, 1.0)
    return np.round(_srgb_oetf(lin) * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars, arrays and paths."""

    def default(self, o: Any) -> Any:
        """Convert NumPy arrays/scalars to standard JSON containers and numbers."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(path: Path, data: Any) -> None:
    """Write ``data`` as indented JSON (numpy types are converted)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def load_json(path: Path) -> Any:
    """Read a UTF-8 JSON artifact and return its decoded Python value."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    comments: Iterable[str] = (),
) -> None:
    """Write a binary little-endian PLY point cloud or triangle mesh.

    Args:
        path: Destination PLY file.
        points: (N, 3) float vertex positions in the caller's frame and length
            units; stored as 64-bit doubles to preserve world-coordinate geometry.
        colors: optional (N, 3) uint8 RGB vertex colours.
        normals: optional (N, 3) float vertex normals.
        faces: optional (M, 3) int triangle indices.
        comments: header comments (units, provenance).
    """
    # Translating a valid tiny triangle away from the origin can make distinct
    # vertices round to the same float32 world coordinate. Preserve float64 XYZ
    # through storage/alignment; normal vectors need no world-position offset.
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    fields = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertex = np.empty(n, dtype=fields)
    vertex["x"], vertex["y"], vertex["z"] = points.T
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        vertex["nx"], vertex["ny"], vertex["nz"] = normals.T
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        vertex["red"], vertex["green"], vertex["blue"] = colors.T

    ply_type = {"<f8": "double", "<f4": "float", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0"]
    header += [f"comment {c}" for c in comments]
    header.append(f"element vertex {n}")
    header += [f"property {ply_type[t]} {name}" for name, t in fields]
    if faces is not None:
        header.append(f"element face {len(faces)}")
        header.append("property list uchar int vertex_indices")
    header.append("end_header")

    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(vertex.tobytes())
        if faces is not None:
            face = np.empty(len(faces), dtype=[("n", "u1"), ("idx", "<i4", (3,))])
            face["n"] = 3
            face["idx"] = np.asarray(faces, dtype=np.int32)
            f.write(face.tobytes())


def read_ply_vertices(path: Path) -> dict[str, np.ndarray]:
    """Read the vertex element of a binary PLY written by :func:`write_ply`."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        props, n = [], 0
        in_vertex = False
        for line in header_lines:
            tok = line.split()
            if tok[:2] == ["element", "vertex"]:
                n, in_vertex = int(tok[2]), True
            elif tok and tok[0] == "element":
                in_vertex = False
            elif in_vertex and tok[0] == "property":
                props.append((tok[2], {"float": "<f4", "double": "<f8", "uchar": "u1"}[tok[1]]))
        data = np.frombuffer(f.read(n * np.dtype(props).itemsize), dtype=props, count=n)
    out = {"points": np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)}
    if "red" in data.dtype.names:
        out["colors"] = np.stack([data["red"], data["green"], data["blue"]], axis=1)
    if "nx" in data.dtype.names:
        out["normals"] = np.stack([data["nx"], data["ny"], data["nz"]], axis=1)
    return out


def read_ply_faces(path: Path) -> np.ndarray:
    """Read the triangle list of a binary PLY written by :func:`write_ply` (empty if none)."""
    with open(path, "rb") as f:
        n_v, n_f, vertex_bytes, in_vertex = 0, 0, 0, False
        while True:
            line = f.readline().decode("ascii").strip()
            tok = line.split()
            if tok[:2] == ["element", "vertex"]:
                n_v, in_vertex = int(tok[2]), True
            elif tok[:2] == ["element", "face"]:
                n_f, in_vertex = int(tok[2]), False
            elif in_vertex and tok and tok[0] == "property":
                vertex_bytes += {"float": 4, "double": 8, "uchar": 1}[tok[1]]
            if line == "end_header":
                break
        f.seek(n_v * vertex_bytes, 1)
        face = np.frombuffer(f.read(n_f * 13), dtype=[("n", "u1"), ("idx", "<i4", (3,))], count=n_f)
    return face["idx"].astype(np.int64)
