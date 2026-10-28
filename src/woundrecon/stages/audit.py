"""Stage 0: data audit.

Checks every input image before any geometry is computed: resolution and
orientation, colour profile, exposure, clipping, sharpness and specular
highlights, and produces contact sheets for visual inspection.

Outputs (``outputs/audit/``):
    audit.json                per-image statistics and set-level findings
    contact_<set>.jpg         thumbnails of each image set
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms

from .. import io_utils
from ..config import stage_dir

log = logging.getLogger(__name__)

# Acquisition diagnostics on 8-bit images; these flags do not reject geometry.
HIGHLIGHT_MIN_VALUE = 235  # Near-white HSV value with low saturation suggests glare.
HIGHLIGHT_MAX_SATURATION = 40  # Excludes bright saturated colours from the glare flag.
CLIPPED_INTENSITY_MIN = 254  # Top two codes flag near-saturation in any RGB channel.
DARK_LUMA_MAX = 5  # Near-black luma has little usable photometric detail.
LOW_SHARPNESS_MEDIAN_RATIO = 0.5  # Flag blur below half the set's median Laplacian variance.


def _icc_description(path: Path) -> str:
    """Name of the embedded ICC profile, or 'none'."""
    with Image.open(path) as im:
        icc = im.info.get("icc_profile")
    if not icc:
        return "none"
    return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()


def image_statistics(path: Path) -> dict[str, Any]:
    """Exposure, clipping, sharpness and highlight statistics of one image.

    Sharpness is the variance of the Laplacian on a 1/4-scale grey image
    (a relative measure: it also depends on scene texture, so it is compared
    within a set, not across sets).
    """
    img = io_utils.read_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Specular highlights: very bright and nearly colourless pixels.
    highlight = (hsv[..., 2] > HIGHLIGHT_MIN_VALUE) & (hsv[..., 1] < HIGHLIGHT_MAX_SATURATION)
    return {
        "name": path.name,
        "height": int(img.shape[0]),
        "width": int(img.shape[1]),
        "orientation": "portrait" if io_utils.is_portrait(img.shape) else "landscape",
        "icc_profile": _icc_description(path),
        "mean_luma": float(gray.mean()),
        "clipped_pct": float((img.max(axis=2) >= CLIPPED_INTENSITY_MIN).mean() * 100),
        "dark_pct": float((gray <= DARK_LUMA_MAX).mean() * 100),
        "sharpness_laplacian_var": float(cv2.Laplacian(small, cv2.CV_64F).var()),
        "highlight_pct": float(highlight.mean() * 100),
    }


def contact_sheet(paths: list[Path], out_path: Path, cols: int = 6, cell: int = 400) -> None:
    """Grid of thumbnails labelled with the image number."""
    tiles = []
    for p in paths:
        img = io_utils.read_image(p)
        scale = cell / max(img.shape[:2])
        thumb = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tile = np.full((cell + 30, cell, 3), 255, np.uint8)
        tile[: thumb.shape[0], : thumb.shape[1]] = thumb
        cv2.putText(
            tile,
            p.stem,
            (6, cell + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Audit readable input images and write acquisition diagnostics.

    The stage fails when input folders/images are missing or unreadable.

    Args:
        cfg: Validated configuration with resolved paths; reads checkerboard/ and images/
            PNG/JPEG inputs.

    Returns:
        Writes audit/audit.json and per-set contact sheets; returns the audit dictionary.
        Image dimensions are displayed-frame pixels; percentages and counts carry no length
        unit.

    Raises:
        OSError: Required inputs or output paths cannot be accessed.
    """
    out = stage_dir(cfg, "audit")
    report: dict[str, Any] = {}
    for key in ("checkerboard", "images"):
        paths = io_utils.list_images(cfg["paths"][key])
        stats = [image_statistics(p) for p in paths]
        sharp = np.array([s["sharpness_laplacian_var"] for s in stats])
        report[key] = {
            "count": len(stats),
            "portrait": sum(s["orientation"] == "portrait" for s in stats),
            "landscape": sum(s["orientation"] == "landscape" for s in stats),
            "icc_profiles": sorted({s["icc_profile"] for s in stats}),
            "sharpness_median": float(np.median(sharp)),
            # Relative outliers within the set (possible blur): < 50 % of the median.
            "low_sharpness": [
                s["name"]
                for s in stats
                if s["sharpness_laplacian_var"] < LOW_SHARPNESS_MEDIAN_RATIO * np.median(sharp)
            ],
            "max_clipped_pct": max(s["clipped_pct"] for s in stats),
            "per_image": stats,
        }
        contact_sheet(paths, out / f"contact_{key}.jpg")
        log.info(
            "%s: %d images (%d portrait, %d landscape), profiles %s",
            key,
            len(stats),
            report[key]["portrait"],
            report[key]["landscape"],
            report[key]["icc_profiles"],
        )
    io_utils.save_json(out / "audit.json", report)
    return report
