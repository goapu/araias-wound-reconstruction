"""Static figures for the report (matplotlib, non-interactive backend).

All figures share one visual style: thin marks, recessive axes, a fixed
categorical colour order and single-hue sequential maps for magnitudes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed categorical order (never cycled), text inks and surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#d9d8d4"
SEQ_CMAP = "Blues"
DIV_CMAP = "RdBu_r"


def apply_style() -> None:
    """Global matplotlib style used by every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    """Save a publication figure at the configured resolution and close its resources."""
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Calibration figures
# --------------------------------------------------------------------------- #
def plot_coverage(
    img_pts: Sequence[np.ndarray], image_size: tuple[int, int], reach: float, path: Path
) -> None:
    """All detected corners on the sensor, with the radius they reach."""
    w, h = image_size
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for pts in img_pts:
        ax.plot(pts[:, 0], pts[:, 1], ".", ms=1.6, color=SERIES[0], alpha=0.6)
    centre = np.array([w / 2, h / 2])
    r = reach * np.linalg.norm(centre)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        centre[0] + r * np.cos(t),
        centre[1] + r * np.sin(t),
        "--",
        color=SERIES[1],
        lw=1.2,
        label=f"max corner radius ({reach:.0%} of half-diagonal)",
    )
    ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False, ec=TEXT_2, lw=1))
    ax.set_xlim(-50, w + 50)
    ax.set_ylim(h + 50, -50)
    ax.set_aspect("equal")
    ax.set_title("Checkerboard corners on the sensor (native frame)")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32))
    _save(fig, path)


def plot_model_selection(
    models: Sequence[str],
    train: dict[str, Sequence[float]],
    heldout: dict[str, Sequence[float]],
    path: Path,
) -> None:
    """Training vs held-out RMS per distortion model, for planar and release-object boards."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(models))
    width = 0.2
    series = [
        ("planar board, train", train["planar"], SERIES[0], 0.45),
        ("planar board, held-out", heldout["planar"], SERIES[0], 1.0),
        ("release-object, train", train["ro"], SERIES[1], 0.45),
        ("release-object, held-out", heldout["ro"], SERIES[1], 1.0),
    ]
    for i, (label, vals, col, alpha) in enumerate(series):
        ax.bar(x + (i - 1.5) * width, vals, width * 0.9, color=col, alpha=alpha, label=label)
    ax.set_xticks(x, models)
    ax.set_ylabel("RMS reprojection error [px]")
    ax.set_title("Distortion model selection (leave-one-image-out)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    _save(fig, path)


def plot_per_view(
    names: Sequence[str],
    rms: np.ndarray,
    tilt: np.ndarray,
    dist: np.ndarray,
    focal: np.ndarray,
    path: Path,
) -> None:
    """Per-view error, board pose and single-view focal length."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
    short = [n.replace("IMG_", "").replace(".png", "") for n in names]
    axes[0].bar(short, rms, color=SERIES[0])
    axes[0].set_title("Per-view RMS [px]")
    axes[0].tick_params(axis="x", rotation=90, labelsize=6)
    axes[1].scatter(dist, tilt, s=18, color=SERIES[0])
    axes[1].set_xlabel("board distance [squares]")
    axes[1].set_ylabel("board tilt [deg]")
    axes[1].set_title("Pose diversity")
    sc = axes[2].scatter(
        dist, focal, c=tilt, cmap=SEQ_CMAP, s=22, edgecolors=TEXT_2, linewidths=0.3
    )
    axes[2].set_xlabel("board distance [squares]")
    axes[2].set_ylabel("single-view focal [px]")
    axes[2].set_title("Focal vs distance (colour = tilt)")
    fig.colorbar(sc, ax=axes[2], label="tilt [deg]")
    fig.tight_layout()
    _save(fig, path)


def plot_distortion_maps(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    disp: np.ndarray,
    disp_std: np.ndarray,
    reach_px: float,
    image_size: tuple[int, int],
    path: Path,
) -> None:
    """Distortion displacement of the final model and its bootstrap uncertainty."""
    w, h = image_size
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    t = np.linspace(0, 2 * np.pi, 200)
    for ax, data, title, label in [
        (axes[0], np.minimum(disp, 100), "Distortion displacement (clipped at 100)", "px"),
        (
            axes[1],
            np.minimum(disp_std, 50),
            "Bootstrap 1-sigma of displacement (clipped at 50)",
            "px",
        ),
    ]:
        im = ax.pcolormesh(grid_x, grid_y, data, cmap=SEQ_CMAP, shading="auto")
        ax.plot(
            w / 2 + reach_px * np.cos(t),
            h / 2 + reach_px * np.sin(t),
            "--",
            color=SERIES[1],
            lw=1.2,
        )
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    fig.suptitle("Dashed: region covered by checkerboard corners", color=TEXT_2, fontsize=8, y=0.02)
    _save(fig, path)


def plot_residual_field(
    img_pts: Sequence[np.ndarray],
    residuals: np.ndarray,
    image_size: tuple[int, int],
    path: Path,
    title: str,
) -> None:
    """Reprojection residual vectors over the sensor (exaggerated)."""
    w, h = image_size
    pts = np.concatenate(img_pts)
    res = residuals.reshape(-1, 2)
    mag = np.linalg.norm(res, axis=1)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    q = ax.quiver(
        pts[:, 0],
        pts[:, 1],
        res[:, 0],
        -res[:, 1],
        mag,
        cmap=SEQ_CMAP,
        angles="xy",
        scale_units="xy",
        scale=1 / 60,
        width=0.0025,
    )
    fig.colorbar(q, ax=ax, label="residual [px]")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title(title + " (vectors x60)")
    _save(fig, path)


# --------------------------------------------------------------------------- #
# Generic helpers used by later stages
# --------------------------------------------------------------------------- #
def plot_histogram(
    values: np.ndarray,
    xlabel: str,
    title: str,
    path: Path,
    bins: int = 60,
    vline: float | None = None,
) -> None:
    """Save a distribution plot without modifying the numerical input.

    Args:
        values: One-dimensional sample values in the units stated by xlabel.
        xlabel: Axis label including the sample units.
        title: Figure title.
        path: Output image path.
        bins: Histogram bin count.
        vline: Optional reference value in the same units as values.
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.hist(values, bins=bins, color=SERIES[0], edgecolor="white", linewidth=0.4)
    if vline is not None:
        ax.axvline(vline, color=SERIES[1], ls="--", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    _save(fig, path)
