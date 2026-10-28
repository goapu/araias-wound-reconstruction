#!/usr/bin/env python3
"""Source-checkout entry point; the installed equivalent is wound-reconstruct."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from woundrecon.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
