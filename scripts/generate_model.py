#!/usr/bin/env python3
"""Generate a FixedFontOCR template model from a TTF/OTF font.

Usage:
    python scripts/generate_model.py font.ttf model_dir [--charset charset.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fixedfontocr.cli import main


if __name__ == "__main__":
    sys.exit(main())
