"""Conservative, explicit reuse of a completed prefix of the pipeline.

An unchanged camera solution need not be recomputed to repair a downstream
export. Reuse still requires the exact original source snapshot, unchanged
upstream numerical modules, input bytes, configuration, environment and saved
prerequisites. The old run and source are retained as immutable evidence.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .provenance import (
    PACKAGE_DIR,
    code_version,
    library_versions,
    package_sources,
    sha256_file,
)
from .stages import STAGE_ORDER

# These modules are used only by the indicated stage and its successors.
# Shared numerical modules (camera, io, calibration, features, BA, etc.) are
# deliberately absent: any edit there requires a fresh full run.
DOWNSTREAM = {
    "dense": {"dense.py", "stages/dense.py"},
    "mesh": {"mesh.py", "stages/mesh.py"},
    "scale": {"scale.py", "stages/scale.py"},
    "measure": {"measure.py", "geometry.py", "stages/measure.py"},
    "render": {"render.py", "stages/render.py"},
}
ORCHESTRATION = {"cli.py", "provenance.py", "reuse.py", "validation.py", "delivery.py"}


def validate_reuse(
    previous, cfg, images, output, stages, threads, snapshot=None, current=PACKAGE_DIR
):
    """Reject changed or incomplete evidence before any output is overwritten.

    The source snapshot argument is necessary only for runs created before
    per-file source inventories were recorded. Its aggregate source hash must
    equal the original completed manifest; an arbitrary lookalike is refused.
    """
    first = STAGE_ORDER.index(stages[0])
    if stages != STAGE_ORDER[first:] or first < STAGE_ORDER.index("dense"):
        raise ValueError("Partial runs must be a complete suffix beginning at dense or later")
    completed = {entry["stage"] for entry in previous["stages"] if entry["status"] == "ok"}
    if set(STAGE_ORDER) - completed or "finished_utc" not in previous:
        raise ValueError("Reuse requires a previously completed pipeline")

    def semantic(value):
        """Compare numerical settings independently of where input/output folders are mounted."""
        return {key: val for key, val in value.items() if key != "paths"}

    if images != previous["inputs_sha256"] or semantic(cfg) != semantic(previous["config"]):
        raise ValueError("Partial rerun refused: input bytes/inventory or settings changed")
    if (
        previous["python"] != sys.version.split()[0]
        or previous["libraries"] != library_versions()
        or previous.get("opencv_threads") != threads
    ):
        raise ValueError("Partial rerun refused: numerical environment or thread setting changed")
    for name, digest in previous["outputs_sha256"].items():
        path = Path(output) / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Partial rerun refused: changed prerequisite {name}")
    baseline = previous.get("source_files_sha256")
    if snapshot is not None:
        if code_version(root=Path(snapshot)) != previous["code_version"]:
            raise ValueError("Source snapshot does not reproduce the completed run's code identity")
        baseline = package_sources(Path(snapshot))
    if baseline is None:
        raise ValueError("Original source snapshot required for this older manifest")
    now = package_sources(Path(current))
    changed = {name for name in set(baseline) | set(now) if baseline.get(name) != now.get(name)}
    allowed = ORCHESTRATION | set().union(*(DOWNSTREAM.get(stage, set()) for stage in stages))
    if changed - allowed:
        raise ValueError("Upstream source changed: " + ", ".join(sorted(changed - allowed)))
    return {"reused_stages": STAGE_ORDER[:first], "changed_sources": sorted(changed)}


def archive_evidence(output: Path, previous_path: Path, snapshot: Path | None) -> Path:
    """Keep original manifest bytes and, when supplied, matching source bytes."""
    folder = output / "provenance" / sha256_file(previous_path)[:16]
    folder.mkdir(parents=True, exist_ok=True)
    archived = folder / "run_manifest.json"
    if archived.exists() and sha256_file(archived) != sha256_file(previous_path):
        raise ValueError("Provenance archive collision")
    shutil.copy2(previous_path, archived)
    if snapshot is not None:
        target = folder / "source_snapshot"
        for name in package_sources(snapshot):
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot / name, destination)
        (folder / "source_inventory.json").write_text(
            json.dumps(package_sources(snapshot), indent=2) + "\n"
        )
    return archived
