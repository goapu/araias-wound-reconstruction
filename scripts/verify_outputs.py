#!/usr/bin/env python3
"""Source-checkout wrapper for artifact and wound-coverage validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from woundrecon.validation import main  # noqa: E402

if __name__ == "__main__":
    main()
