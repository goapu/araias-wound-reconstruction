"""Preflight-checked, atomic assembly of source and confidential deliverables.

The source archive uses a positive allowlist. Images, reconstruction outputs,
reports, local work caches and credentials cannot enter it through a broad
recursive archive operation. The separate delivery contains the visual evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .config import REPO_ROOT
from .provenance import sha256_file
from .validation import verify_outputs

SCREENSHOTS = ("01_textured_top.png", "02_geometry_oblique.png", "05_textured_oblique.png")
MODELS = (
    "scale/mesh_aligned.ply",
    "scale/dense_aligned_thinned.ply",
    "scale/sparse_aligned.ply",
    "scale/cameras_aligned.ply",
    "measure/wound_depth.ply",
)


def required_inputs(
    output: Path, project: Path = REPO_ROOT, report: Path | None = None
) -> list[Path]:
    """List geometry, source documentation and the separately prepared PDF needed for export."""
    return (
        [
            report if report is not None else project / "report/report.pdf",
            project / "README.md",
            output / "run_manifest.json",
        ]
        + [output / "renders" / name for name in SCREENSHOTS]
        + [output / name for name in MODELS]
    )


def source_files(project: Path) -> list[Path]:
    """Select reproducible source files without confidential generated evidence.

    Args:
        project: Repository root containing the package, tests, scripts and README.

    Returns:
        Sorted unique paths from the source allowlist. Generated artifacts,
        local development files, task data and credentials are excluded.
    """
    files = []
    for folder in ("src", "tests"):
        files += [p for p in (project / folder).rglob("*.py") if "__pycache__" not in p.parts]
    for name in ("run_pipeline.py", "verify_outputs.py", "export_submission.py"):
        path = project / "scripts" / name
        if path.is_file():
            files.append(path)
    files += list((project / "src/woundrecon/resources").glob("*.yaml"))
    for folder, suffix in [("configs", ".yaml"), (".github/workflows", ".yml")]:
        files += list((project / folder).rglob("*" + suffix))
    for name in (
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements.lock.txt",
        ".gitignore",
    ):
        if (project / name).is_file():
            files.append(project / name)
    return sorted(set(files))


def write_zip(path: Path, items: list[tuple[Path, str]]) -> None:
    """Write a CRC-checked ZIP; members are explicit relative names."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, name in items:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe archive member")
            archive.write(source, name)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")


def export(project: Path, output: Path, destination: Path, *, report: Path | None = None) -> dict:
    """Assemble an atomic confidential delivery after artifact preflight.

    Args:
        project: Repository root containing installable source and its README.
        output: Completed pipeline output root, including its run manifest.
        destination: Delivery directory; an existing directory is archived first.
        report: Separately prepared PDF; defaults to the local report/report.pdf.

    Returns:
        Export status, destination and number of hash-inventoried files.

    Raises:
        ValueError: Geometry, provenance, report, image or ZIP checks fail.
        OSError: Required files cannot be read or the delivery cannot be written.
    """
    report = Path(report) if report is not None else project / "report/report.pdf"
    missing = [str(p) for p in required_inputs(output, project, report) if not p.is_file()]
    if missing:
        raise ValueError("Export missing inputs: " + ", ".join(missing))
    verified = verify_outputs(output)
    try:
        page_count = len(PdfReader(report).pages)
    except PyPdfError as error:
        raise ValueError("Report PDF cannot be read") from error
    if not page_count:
        raise ValueError("Report PDF has no pages")
    for name in SCREENSHOTS:
        with Image.open(output / "renders" / name) as image:
            if min(image.size) < 600:
                raise ValueError("Inspection image is too small")
            image.verify()
    files = source_files(project)
    if not files or not (project / "pyproject.toml").is_file():
        raise ValueError("Installable source package is missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent)
    )
    try:
        (temporary / "screenshots").mkdir()
        shutil.copy2(report, temporary / "report.pdf")
        for name in SCREENSHOTS:
            with Image.open(output / "renders" / name) as image:
                image.convert("RGB").save(
                    temporary / "screenshots" / name.replace(".png", ".jpg"), quality=92
                )
        write_zip(
            temporary / "source_code.zip", [(p, p.relative_to(project).as_posix()) for p in files]
        )
        write_zip(
            temporary / "3d_models.zip", [(output / name, Path(name).name) for name in MODELS]
        )
        (temporary / "metrics").mkdir()
        for stage in ("calibration", "sfm", "dense", "mesh", "scale", "measure"):
            for path in (output / stage).glob("*.json"):
                shutil.copy2(path, temporary / "metrics" / (stage + "_" + path.name))
        shutil.copy2(output / "run_manifest.json", temporary / "metrics/run_manifest.json")
        if (output / "provenance").is_dir():
            shutil.copytree(output / "provenance", temporary / "metrics/provenance")
        (temporary / "verification.json").write_text(json.dumps(verified, indent=2) + "\n")
        shutil.copy2(project / "README.md", temporary / "README.md")
        manifest = {
            p.relative_to(temporary).as_posix(): {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in sorted(temporary.rglob("*"))
            if p.is_file()
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        write_zip(
            temporary / "submission_bundle.zip",
            [
                (p, p.relative_to(temporary).as_posix())
                for p in sorted(temporary.rglob("*"))
                if p.is_file() and p.name != "submission_bundle.zip"
            ],
        )
        if destination.exists():
            previous = destination.with_name(
                destination.name + ".archive-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            )
            if previous.exists():
                raise FileExistsError(previous)
            destination.rename(previous)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return dict(status="passed", destination=str(destination), files=len(manifest))


def main(argv=None):
    """Parse delivery paths, run export preflight and return zero on success."""
    parser = argparse.ArgumentParser(description="Assemble the verified submission")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "submission")
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "report/report.pdf",
        help="already prepared PDF; report authoring sources are not required",
    )
    args = parser.parse_args(argv)
    try:
        result = export(
            REPO_ROOT, args.output_dir.resolve(), args.dest.resolve(), report=args.report.resolve()
        )
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, indent=2))
    return 0
