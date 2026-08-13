"""Command-line tools: generate template models from a TTF font."""

from __future__ import annotations

import argparse
from pathlib import Path

from .fontgen import build_templates, write_model


def _read_charset(path: str | None, font: str, default: str) -> list[str]:
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
    # No charset given: use the font's ASCII coverage as a convenient default.
    from PIL import ImageFont

    font_obj = ImageFont.truetype(font, 24)
    chars = [ch for ch in default if font_obj.getlength(ch) > 0]
    return chars


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fixedfontocr-generate",
        description="Render a TTF font into the FixedFontOCR template model format.",
    )
    parser.add_argument("font", help="path to a .ttf/.otf font file")
    parser.add_argument("output", help="output model directory (config.json + weights.bin)")
    parser.add_argument("--charset", help="text file listing characters to include")
    parser.add_argument(
        "--size", type=int, default=24, help="normalized glyph size (default 24)"
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=64,
        help="pixel size used when rasterizing the font (default 64)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=140,
        help="anti-aliased alpha threshold for binarization (default 140)",
    )
    args = parser.parse_args()

    default_charset = (
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "获得金币消耗数量提示确定取消返回攻击防御生命法力"
        "+-×÷%.,:;!?()[]{}「」"
    )
    chars = _read_charset(args.charset, args.font, default_charset)
    used, templates = build_templates(
        args.font,
        chars,
        target_size=args.size,
        render_size=args.render_size,
        threshold=args.threshold,
    )
    write_model(args.output, used, templates, target_size=args.size)
    print(f"wrote {len(used)} templates to {args.output}")


if __name__ == "__main__":
    main()
