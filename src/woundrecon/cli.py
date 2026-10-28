"""Command-line entry point for the reconstruction pipeline.

Examples:
    # full pipeline, data folder next to the repository (default config)
    python scripts/run_pipeline.py

    # verified downstream rerun, explicit data location
    python scripts/run_pipeline.py --data-root /path/to/data --stages dense mesh scale measure render

Every run writes ``<output>/logs/run_<UTC time>.log`` (full debug log) and
``<output>/run_manifest.json`` (code version, resolved configuration, library
versions, SHA-256 of every input image, status and duration of every stage,
SHA-256 of the main outputs). A failing stage is recorded with its error and
the run stops with a non-zero exit code; later stages are not run on
incomplete inputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from woundrecon.config import load_config
from woundrecon.provenance import RunManifest, portable_path, sha256_file
from woundrecon.stages import STAGE_ORDER, get_stage

# Main outputs whose checksums are recorded in the run manifest.
KEY_OUTPUTS = (
    "calibration/calibration.json",
    "sfm/sfm.npz",
    "sfm/sfm_stats.json",
    "dense/dense_stats.json",
    "mesh/mesh.ply",
    "mesh/mesh_stats.json",
    "scale/scale.json",
    "scale/mesh_aligned.ply",
    "scale/dense_aligned_thinned.ply",
    "scale/sparse_aligned.ply",
    "scale/cameras_aligned.ply",
    "measure/measurements.json",
    "measure/wound_depth.ply",
)


def parse_args() -> argparse.Namespace:
    """Parse input/output paths, stage selection, worker count and logging options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML config (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="folder containing checkerboard/ and images/ (overrides the config)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="output folder (overrides the config)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=STAGE_ORDER,
        choices=STAGE_ORDER,
        help="stages to run, in pipeline order",
    )
    parser.add_argument("--threads", type=int, default=4, help="OpenCV worker threads (1-32)")
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        help="original package source directory for verifying reuse of an older run manifest",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="debug messages on the console as well"
    )
    return parser.parse_args()


def setup_logging(log_dir: Path, verbose: bool) -> Path:
    """Console (INFO, or DEBUG with -v) plus a full DEBUG log file for this run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    to_file = logging.FileHandler(log_file, encoding="utf-8")
    to_file.setLevel(logging.DEBUG)
    to_file.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers[:] = [console, to_file]
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def main() -> int:
    """Execute requested stages, record lineage and verify a completed pipeline.

    Returns:
        Zero on success or one after a recorded stage failure.

    Raises:
        ValueError: Configuration, worker count or partial-run prerequisites fail
            validation before stage execution.
        OSError: Required input or output files cannot be accessed.
    """
    args = parse_args()
    cfg = load_config(args.config, args.data_root, args.output_dir)
    out = cfg["paths"]["output"]
    if not 1 <= args.threads <= 32:
        raise ValueError("Use 1-32 OpenCV threads")
    import cv2

    cv2.setNumThreads(args.threads)
    from woundrecon import io_utils
    from woundrecon.provenance import input_hashes
    from woundrecon.reuse import archive_evidence, validate_reuse

    requested = [s for s in STAGE_ORDER if s in args.stages]
    previous_path = out / "run_manifest.json"
    reused, archived = None, None
    if requested != STAGE_ORDER:
        if not previous_path.is_file():
            raise ValueError("A partial run requires a completed run manifest")
        previous = io_utils.load_json(previous_path)
        current_images = {
            f"{folder}/{name}": digest
            for folder in ("checkerboard", "images")
            for name, digest in input_hashes(io_utils.list_images(cfg["paths"][folder])).items()
        }

        reused = validate_reuse(
            previous, cfg, current_images, out, requested, args.threads, args.source_snapshot
        )
        archived = archive_evidence(out, previous_path, args.source_snapshot)
    log_file = setup_logging(out / "logs", args.verbose)
    logging.info("Data root: %s", portable_path(cfg["paths"]["data_root"]))
    logging.info("Output:    %s", portable_path(out))
    stages = [s for s in STAGE_ORDER if s in args.stages]
    manifest = RunManifest(cfg, stages, log_file)
    manifest.data["opencv_threads"] = args.threads
    if reused is not None:
        relative = archived.relative_to(out).as_posix()
        manifest.data["reuse"] = {
            **reused,
            "previous_manifest": relative,
            "previous_manifest_sha256": sha256_file(archived),
            "previous_code_version": previous["code_version"],
        }
        manifest.data["stages"] = [
            {**entry, "execution": "reused", "reused_from": relative}
            for entry in previous["stages"]
            if entry["stage"] in reused["reused_stages"]
        ]

    for name in stages:
        logging.info("=== Stage: %s ===", name)
        start = time.time()
        try:
            get_stage(name).run(cfg)
        except Exception as exc:  # record, then stop the run
            logging.error("Stage %s failed: %s\n%s", name, exc, traceback.format_exc())
            manifest.stage(name, "failed", time.time() - start, f"{type(exc).__name__}: {exc}")
            manifest.finish([out / k for k in KEY_OUTPUTS])
            logging.error("Run stopped. Details: %s", portable_path(log_file))
            return 1
        seconds = time.time() - start
        manifest.stage(name, "ok", seconds)
        logging.info("=== Stage %s finished in %.0f s ===", name, seconds)
    # Stage names and output folders differ for calibration. Hash its parameters,
    # diagnostics and bootstrap draws alongside geometry, renders and archived evidence.
    artifact_folders = ["calibration" if name == "calibrate" else name for name in STAGE_ORDER]
    artifact_folders += ["provenance"]
    manifest.finish(
        [
            path
            for folder in artifact_folders
            for path in (out / folder).rglob("*")
            if path.is_file()
        ]
    )
    if stages == STAGE_ORDER or reused is not None:
        from woundrecon.validation import verify_outputs

        io_utils.save_json(
            out / "verification.json", verify_outputs(out, cfg["paths"]["data_root"])
        )
    logging.info("Run manifest: %s", portable_path(manifest.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
