"""Configuration loading.

The configuration is a plain nested dictionary read from YAML. Paths in the
config are resolved against the repository root so that the pipeline can be
started from any working directory.
"""

from __future__ import annotations

import copy
import math
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from .stages import STAGE_OUTPUT_FOLDERS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
if not DEFAULT_CONFIG.is_file():
    DEFAULT_CONFIG = files("woundrecon").joinpath("resources/default.yaml")


def _resolve(path_like: str | Path) -> Path:
    """Return an absolute path; relative paths are taken relative to the repo root."""
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(
    config_path: str | Path | None = None,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load the YAML config and resolve all paths.

    Args:
        config_path: YAML file; defaults to ``configs/default.yaml``.
        data_root: optional override for ``data.root`` (folder with checkerboard/ and images/).
        output_dir: optional override for ``output_dir``.

    Returns:
        The configuration dictionary with additional resolved ``Path`` entries:
        ``paths.checkerboard``, ``paths.images`` and ``paths.output``.
    """
    with open(config_path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    validate_config(cfg)

    root = _resolve(data_root if data_root is not None else cfg["data"]["root"])
    out = _resolve(output_dir if output_dir is not None else cfg["output_dir"])
    cfg["paths"] = {
        "data_root": root,
        "checkerboard": root / cfg["data"]["checkerboard_dir"],
        "images": root / cfg["data"]["images_dir"],
        "output": out,
    }
    return cfg


def stage_dir(cfg: dict[str, Any], stage: str) -> Path:
    """Create (if needed) and return the output folder of a pipeline stage."""
    path = cfg["paths"]["output"] / STAGE_OUTPUT_FOLDERS.get(stage, stage)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_config(cfg: dict[str, Any]) -> None:
    """Reject unsupported or non-finite settings before native computation."""
    required = {
        "data",
        "output_dir",
        "camera",
        "calibration",
        "sfm",
        "dense",
        "mesh",
        "scale",
        "measure",
    }
    if not isinstance(cfg, dict) or not required <= cfg.keys():
        raise ValueError("Configuration is missing required pipeline sections")

    def finite(value):
        """Reject nonnumeric or non-finite configuration values before execution."""
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Configuration values must be finite")

    finite(cfg)
    for section, key in [
        ("calibration", "detect_scale"),
        ("sfm", "work_scale"),
        ("dense", "work_scale"),
        ("dense", "fusion_voxel_rel"),
        ("mesh", "voxel_size_rel"),
        ("mesh", "truncation_voxels"),
    ]:
        if not isinstance(cfg[section][key], (int, float)) or cfg[section][key] <= 0:
            raise ValueError(f"{section}.{key} must be positive")
    models = cfg["calibration"]["models"]
    if not models or set(models) - {"k1", "k1k2", "k1k2k3", "opencv5", "rational"}:
        raise ValueError("Unsupported calibration model")
    for value in [
        cfg["calibration"]["bootstrap_samples"],
        cfg["dense"]["neighbours_per_view"],
        cfg["dense"]["min_consistent_views"],
        cfg["dense"]["block_size"],
    ]:
        if type(value) is not int or value < 2:
            raise ValueError(
                "Bootstrap, neighbour, support and window counts must be integers >= 2"
            )
    if cfg["dense"]["block_size"] % 2 != 1:
        raise ValueError("Stereo block size must be odd")
    if cfg["dense"]["min_consistent_views"] > cfg["dense"]["neighbours_per_view"]:
        raise ValueError("Stereo support exceeds the available neighbour count")
    if cfg["mesh"]["min_weight"] < 2:
        raise ValueError("Surface extraction requires at least two observed depth maps")
