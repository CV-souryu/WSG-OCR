"""Command-line tools: generate template models from a TTF font."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import defaults
from .fontgen import (
    TEMPLATE_V2_DOWNSAMPLE_MODES,
    TEMPLATE_V2_SIZES,
    build_template_v2,
    build_templates,
    write_model,
    write_template_v2_model,
)


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
    parser.add_argument(
        "--template-v1",
        action="store_true",
        help="write the legacy one-template-per-character format "
        "(default is Goal 9 Template V2 multi-prototype)",
    )
    parser.add_argument(
        "--render-sizes",
        default=",".join(str(s) for s in TEMPLATE_V2_SIZES),
        help="comma-separated prototype render sizes in px "
        f"(default {','.join(str(s) for s in TEMPLATE_V2_SIZES)})",
    )
    parser.add_argument(
        "--downsample",
        default=",".join(TEMPLATE_V2_DOWNSAMPLE_MODES),
        help="comma-separated downsample modes: bilinear,area",
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=3,
        help="supersampling factor used before downsampling prototypes",
    )
    args = parser.parse_args()

    default_charset = defaults.read_charset()
    font = defaults.resolve_font(args.font)
    chars = _read_charset(args.charset, default_charset)
    if args.template_v1:
        used, templates = build_templates(
            font,
            chars,
            target_size=args.size,
            render_size=args.render_size,
            threshold=args.threshold,
        )
        write_model(args.output, used, templates, target_size=args.size, font_path=font)
        print(f"wrote {len(used)} templates to {args.output}")
        return
    sizes = tuple(int(v) for v in args.render_sizes.split(",") if v.strip())
    modes = tuple(m for m in args.downsample.split(",") if m.strip())
    for mode in modes:
        if mode not in TEMPLATE_V2_DOWNSAMPLE_MODES:
            parser.error(
                f"unknown downsample mode {mode!r}; use bilinear/area"
            )
    used, data = build_template_v2(
        font,
        chars,
        target_size=args.size,
        render_sizes=sizes,
        downsample_modes=modes,
        supersample=args.supersample,
        threshold=args.threshold,
    )
    write_template_v2_model(
        args.output,
        used,
        data,
        target_size=args.size,
        font_path=font,
        render_size=args.render_size,
        threshold=args.threshold,
    )
    print(
        f"wrote {len(used)} characters × {data.prototypes_per_char} "
        f"prototypes (Template V2) to {args.output}"
    )


if __name__ == "__main__":
    main()
