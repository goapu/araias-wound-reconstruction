"""Provenance: fingerprints for caches and an audit record for every pipeline run.

Measurement software has to be able to answer "which inputs, settings and code
produced this number?". Two mechanisms support that here:

* **Cache fingerprints.** Every cached intermediate (features, verified pairs)
  stores a SHA-256 fingerprint of everything it depends on: the input images
  (file content), the relevant settings and the code that produced it. A cache
  is reused only when the fingerprint matches; otherwise it is recomputed and
  the reason is logged. A stale cache cannot be mixed silently into a
  new run.
* **Run manifest.** ``outputs/run_manifest.json`` records, per run, the code
  version, the resolved configuration, library versions, the SHA-256 of every
  input image, each stage's status and duration (and error, if any) and the
  SHA-256 of the main outputs. A log file is written next to it. Paths are
  recorded relative to the repository, never as absolute machine paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]


def portable_path(path: Path | str) -> str:
    """Path relative to the repository, so records carry no machine-specific absolute paths."""
    return Path(os.path.relpath(Path(path).resolve(), REPO_ROOT)).as_posix()


LIBRARIES = (
    "numpy",
    "scipy",
    "opencv-python",
    "opencv-python-headless",
    "scikit-image",
    "matplotlib",
    "pyyaml",
    "pillow",
    "pypdf",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hash file contents without loading the whole artifact into memory.

    Args:
        path: Existing file to read.
        chunk: Positive read-block size in bytes.

    Returns:
        Lowercase hexadecimal SHA-256 digest of the exact stored bytes.

    Raises:
        OSError: The file cannot be read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(*parts: Any) -> str:
    """SHA-256 of a JSON-serialisable description (paths are reduced to their names)."""

    def norm(x):
        """Canonicalize paths, mappings and NumPy scalars before hashing settings."""
        if isinstance(x, Path):
            return x.name
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x

    return hashlib.sha256(json.dumps(norm(list(parts)), sort_keys=True).encode()).hexdigest()


def code_version(modules: tuple[str, ...] | None = None, root: Path = PACKAGE_DIR) -> str:
    """SHA-256 over the source of the package (or of the named modules)."""
    files = (
        sorted(root.rglob("*.py"))
        if modules is None
        else [root / f"{m.replace('.', '/')}.py" for m in modules]
    )
    # Relative names bind renames; exact bytes include comments/docstrings so
    # the identity describes the exact source that ran.
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def package_sources(root: Path = PACKAGE_DIR) -> dict[str, str]:
    """Per-file source identities, supporting conservative stage dependency checks."""
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*.py"))}


def input_hashes(paths: list[Path]) -> dict[str, str]:
    """Return a basename-to-SHA-256 map for the supplied, uniquely named input files."""
    return {p.name: sha256_file(p) for p in paths}


def cache_is_valid(stored: np.lib.npyio.NpzFile | None, expected: str, what: str) -> bool:
    """True if a cache file carries the expected fingerprint; logs why it is rejected otherwise."""
    if stored is None:
        return False
    found = str(stored["fingerprint"]) if "fingerprint" in stored.files else None
    if found != expected:
        log.warning(
            "%s cache ignored: %s; recomputing",
            what,
            "no fingerprint (written by an older version)"
            if found is None
            else "inputs, settings or code changed since it was written",
        )
        return False
    return True


def library_versions() -> dict[str, str]:
    """Return installed versions of the libraries used by reconstruction and export."""
    out = {}
    for name in LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


class RunManifest:
    """Audit record of one pipeline run, written after every stage (so a crash still leaves a record)."""

    def __init__(self, cfg: dict[str, Any], stages: list[str], log_file: Path):
        """Create the initial source/input/settings record at the configured output root."""
        self.path = cfg["paths"]["output"] / "run_manifest.json"
        images = sorted(
            p
            for d in (cfg["paths"]["checkerboard"], cfg["paths"]["images"])
            for p in Path(d).glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        )
        self.data = {
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "code_version": code_version(),
            "source_files_sha256": package_sources(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "libraries": library_versions(),
            "config": _jsonable(cfg),
            "inputs_sha256": {f"{p.parent.name}/{p.name}": sha256_file(p) for p in images},
            "stages_requested": stages,
            "stages": [],
            "log_file": log_file.name,
        }
        self._save()

    def stage(self, name: str, status: str, seconds: float, error: str | None = None) -> None:
        """Append a stage outcome and elapsed seconds, then persist the manifest atomically."""
        entry = {"stage": name, "status": status, "seconds": round(seconds, 1)}
        if error:
            entry["error"] = error
        # Persist each outcome so an interrupted run retains its completed-stage history.
        self.data["stages"].append(entry)
        self._save()

    def finish(self, outputs: list[Path]) -> None:
        """Record completion time and hashes of existing artifacts relative to the output root."""
        self.data["finished_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        root = self.path.parent
        self.data["outputs_sha256"] = {
            p.relative_to(root).as_posix(): sha256_file(p) for p in outputs if p.exists()
        }
        self._save()

    def _save(self) -> None:
        """Atomically replace the JSON manifest so interrupted writes cannot truncate it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Replace on the same filesystem to avoid a partially written JSON manifest.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)


def _jsonable(x: Any) -> Any:
    """Convert configuration values to JSON-compatible values with portable relative paths."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return portable_path(x)
    return x
