#!/usr/bin/env python3
"""Register a font file under ``fonts/`` into ``fonts/registry.json``.

Usage:
    python tools/register_font.py fonts/MyFont/MyFont.otf

The registry stores the font's path relative to ``fonts/`` together with
its SHA256. Every consumer then verifies that only registered fonts from
``fonts/`` are used for rendering, training, and model metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fixedfontocr import defaults  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("font", type=Path, help="font file under fonts/")
    args = parser.parse_args()

    resolved = defaults.ensure_font_path(args.font, require_registered=False)
    rel = resolved.relative_to(defaults.FONTS_DIR).as_posix()
    sha256 = defaults.compute_font_sha256(resolved)

    registry = defaults._load_font_registry()
    registry[rel] = sha256
    defaults.FONT_REGISTRY_PATH.write_text(
        json.dumps({"fonts": dict(sorted(registry.items()))}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"registered {rel} (sha256 {sha256})")


if __name__ == "__main__":
    main()
