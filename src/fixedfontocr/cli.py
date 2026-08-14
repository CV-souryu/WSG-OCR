"""Command-line tools: generate template models from a TTF font."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import defaults
from .fontgen import build_templates, write_model


def _read_charset(path: str | None, default: str) -> list[str]:
    if path:
        raw = Path(path).read_text(encoding="utf-8")
        chars: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for ch in line:
                if not ch.isspace():
                    chars.append(ch)
        if not chars:
            raise SystemExit(f"charset file {path} contains no characters")
        return chars
    # No charset given: use the project default. Missing glyphs are reported
    # by build_templates instead of being silently filtered out.
    return list(default)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fixedfontocr-generate",
        description="Render a TTF font into the FixedFontOCR template model format.",
    )
    parser.add_argument(
        "font",
        nargs="?",
        default=str(defaults.FONT_PATH),
        help="path to a registered .ttf/.otf font under fonts/ "
        "(default: bundled game font)",
    )
    parser.add_argument("output", help="output model directory (config.json + weights.bin)")
    parser.add_argument(
        "--charset",
        default=str(defaults.CHARSET_PATH),
        help="text file listing characters (default: charsets/sets/combined.txt)",
    )
    parser.add_argument(
        "--size", type=int, default=24, help="normalized glyph size (default 24)"
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=32,
        help="pixel size used when rasterizing the font (default 32)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=140,
        help="anti-aliased alpha threshold for binarization (default 140)",
    )
    args = parser.parse_args()

    default_charset = defaults.read_charset()
    font = defaults.resolve_font(args.font)
    chars = _read_charset(args.charset, default_charset)
    used, templates = build_templates(
        font,
        chars,
        target_size=args.size,
        render_size=args.render_size,
        threshold=args.threshold,
    )
    write_model(args.output, used, templates, target_size=args.size, font_path=font)
    print(f"wrote {len(used)} templates to {args.output}")


if __name__ == "__main__":
    main()
