#!/usr/bin/env python3
"""Source-checkout wrapper for the checked submission exporter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from woundrecon.delivery import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
