#!/usr/bin/env python3
"""Generate a synthetic fixed-font training dataset from a TTF/OTF font.

Every sample renders one character with a randomized combination of:
font size +/-2 px, x/y offset, outline, shadow, gaussian blur, a
pre-normalization scale, alpha/text luminance, background level + noise and
a binarization threshold. Output is a compressed ``.npz``:

    x:      uint8  [N, 24, 24] normalized glyphs (0/255)
    y:      int64  [N] label index into ``chars``
    chars:  unicode [C] charset
    font_sha256: unicode [1] SHA256 of the registered source font
    input_mode: unicode [1] "binary" (default) or "soft"

``--soft`` stores 0..255 foreground-strength glyphs (anti-aliasing, edge
gray and alpha preserved) instead of 0/255 masks. The Visual Frontend
(Goal 2) feeds soft glyphs to the TinyCNN; the model config records which
mode it was trained with.

Usage:
    python tools/dataset/generate_font_dataset.py font.ttf dataset.npz \\
        --charset charset.txt --samples-per-char 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fixedfontocr import defaults  # noqa: E402
from fixedfontocr.defaults import (  # noqa: E402
    compute_font_sha256,
    resolve_font,
)
from fixedfontocr.preprocess import (  # noqa: E402
    Component,
    compute_normalize_spec,
    glyph_normalize_geometry,
    normalize,
    normalize_grayscale,
)


def _scale(mask: NDArray[np.bool_], factor: float) -> NDArray[np.bool_]:
    """Nearest-neighbor resize of a boolean ink mask."""
    h, w = mask.shape
    nh = max(1, int(round(h * factor)))
    nw = max(1, int(round(w * factor)))
    ys = (np.arange(nh)[:, None] * h / nh).astype(np.int32)
    xs = (np.arange(nw)[None, :] * w / nw).astype(np.int32)
    return mask[ys, xs]


def _shift(glyph: NDArray[np.uint8], dx: int, dy: int) -> NDArray[np.uint8]:
    out = np.zeros_like(glyph)
    y0, y1 = max(0, dy), min(glyph.shape[0], glyph.shape[0] + dy)
    x0, x1 = max(0, dx), min(glyph.shape[1], glyph.shape[1] + dx)
    sy0, sy1 = max(0, -dy), min(glyph.shape[0], glyph.shape[0] - dy)
    sx0, sx1 = max(0, -dx), min(glyph.shape[1], glyph.shape[1] - dx)
    out[y0:y1, x0:x1] = glyph[sy0:sy1, sx0:sx1]
    return out


def generate_dataset(
    font: str | Path,
    charset: list[str],
    samples_per_char: int,
    output: str | Path,
    size: int = 24,
    seed: int = 0,
    threshold_min: int = 100,
    threshold_max: int = 180,
    soft: bool = False,
    render_size: int | None = None,
) -> None:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    font = resolve_font(font)
    font_sha256 = compute_font_sha256(font)
    spec = compute_normalize_spec(font, charset, size, size)
    rng = np.random.default_rng(seed)
    if render_size is None:
        render_size = size
    x_list: list[NDArray[np.uint8]] = []
    y_list: list[int] = []

    for label, char in enumerate(charset):
        for _ in range(samples_per_char):
            render_px = int(
                rng.integers(max(2, render_size - 2), render_size + 3)
            )
            font_obj = ImageFont.truetype(str(font), render_px)
            pad = 10
            bbox = font_obj.getbbox(char)
            w = bbox[2] - bbox[0] + pad * 2 + 4
            h = bbox[3] - bbox[1] + pad * 2 + 4
            bg = int(rng.integers(0, 71))
            img = Image.new("L", (max(1, w), max(1, h)), bg)
            draw = ImageDraw.Draw(img)
            fill = int(rng.integers(190, 256))
            dx = int(rng.integers(-2, 3))
            dy = int(rng.integers(-2, 3))

            # Shadow: a darker copy offset by 1-2 px underneath the glyph.
            if rng.random() < 0.4:
                sh = int(rng.integers(1, 3))
                draw.text(
                    (pad - bbox[0] + dx + sh, pad - bbox[1] + dy + sh),
                    char,
                    font=font_obj,
                    fill=int(rng.integers(30, 100)),
                )

            # Outline: Pillow stroke around the glyph.
            stroke_width = int(rng.integers(0, 3))
            stroke_fill = int(rng.integers(80, 180)) if stroke_width else None
            draw.text(
                (pad - bbox[0] + dx, pad - bbox[1] + dy),
                char,
                font=font_obj,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )

            if rng.random() < 0.5:
                radius = float(rng.uniform(0.0, 1.6))
                if radius >= 0.05:
                    img = img.filter(ImageFilter.GaussianBlur(radius))

            # Sparse background speckles (stays below most thresholds).
            if rng.random() < 0.3:
                arr = np.array(img, dtype=np.uint8).copy()
                noise = rng.random(arr.shape) < 0.004
                arr[noise] = rng.integers(0, 90, size=int(noise.sum())).astype(
                    np.uint8
                )
                img = Image.fromarray(arr)

            arr = np.asarray(img, dtype=np.uint8)
            threshold = int(rng.integers(threshold_min, threshold_max + 1))
            mask = arr.astype(np.float32) >= threshold
            ys, xs = np.where(mask)
            # Blur + a high threshold can erase a thin glyph's anti-aliased
            # core (e.g. ``#`` at 22 px); retry at lower thresholds before
            # declaring the font unable to render the character.
            for _ in range(8):
                if ys.size:
                    break
                threshold = max(1, threshold - 20)
                mask = arr.astype(np.float32) >= threshold
                ys, xs = np.where(mask)
            if ys.size == 0:
                raise ValueError(
                    f"font {font} cannot render {char!r}: no ink rendered"
                )
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            tight = mask[y0:y1, x0:x1]
            tight_gray = arr[y0:y1, x0:x1]

            # Pre-normalization scale (Goal 3 baseline frame is applied next).
            if rng.random() < 0.5:
                factor = float(rng.uniform(0.85, 1.15))
                tight = _scale(tight, factor)
                tight_gray = _scale_gray(tight_gray, factor)

            th, tw = tight.shape
            candidate = Component(mask=tight, x=0, y=0, w=tw, h=th)
            baseline_offset, scale = glyph_normalize_geometry(
                candidate, spec, size
            )
            glyph = (
                normalize_grayscale(
                    tight_gray,
                    size,
                    baseline_offset=baseline_offset,
                    scale=scale,
                    baseline_row=spec.baseline_row,
                )
                if soft
                else normalize(
                    tight,
                    size,
                    baseline_offset=baseline_offset,
                    scale=scale,
                    baseline_row=spec.baseline_row,
                )
            )
            if rng.random() < 0.6:
                glyph = _shift(glyph, int(rng.integers(-1, 2)), int(rng.integers(-1, 2)))
            x_list.append(glyph)
            y_list.append(label)

    x = np.stack(x_list)
    y = np.asarray(y_list, dtype=np.int64)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x=x,
        y=y,
        chars=np.asarray(charset, dtype="<U8"),
        size=np.asarray(size, dtype=np.int64),
        font_sha256=np.asarray([font_sha256], dtype="<U64"),
        input_mode=np.asarray(["soft" if soft else "binary"], dtype="<U16"),
    )
    print(
        f"wrote {x.shape[0]} synthetic samples ({len(charset)} classes) "
        f"to {out}"
    )


def _scale_gray(glyph: NDArray[np.uint8], factor: float) -> NDArray[np.uint8]:
    """Nearest-neighbor resize of a grayscale glyph."""

    h, w = glyph.shape
    nh = max(1, int(round(h * factor)))
    nw = max(1, int(round(w * factor)))
    ys = (np.arange(nh)[:, None] * h / nh).astype(np.int32)
    xs = (np.arange(nw)[None, :] * w / nw).astype(np.int32)
    return glyph[ys, xs]


def _read_charset(path: str | None, default: str) -> list[str]:
    if path:
        chars: list[str] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for ch in line:
                if not ch.isspace():
                    chars.append(ch)
        if not chars:
            raise SystemExit(f"charset file {path} contains no characters")
        return chars
    # Missing glyphs are reported by generate_dataset instead of being
    # silently filtered out here.
    return list(default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "font",
        nargs="?",
        default=str(defaults.FONT_PATH),
        help="path to a registered .ttf/.otf font under fonts/ "
        "(default: bundled game font)",
    )
    parser.add_argument("output", help="output .npz dataset")
    parser.add_argument(
        "--charset",
        default=str(defaults.CHARSET_PATH),
        help="text file listing characters (default: charsets/sets/combined.txt)",
    )
    parser.add_argument("--samples-per-char", type=int, default=300)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument(
        "--render-size",
        type=int,
        default=None,
        help="raster size (px) used for the font; defaults to --size",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold-min", type=int, default=100)
    parser.add_argument("--threshold-max", type=int, default=180)
    parser.add_argument(
        "--soft",
        action="store_true",
        help="store 0..255 foreground-strength glyphs instead of 0/255 masks",
    )
    args = parser.parse_args()

    default_charset = defaults.read_charset()
    font = defaults.resolve_font(args.font)
    charset = _read_charset(args.charset, default_charset)
    generate_dataset(
        font,
        charset,
        args.samples_per_char,
        args.output,
        size=args.size,
        seed=args.seed,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        soft=args.soft,
        render_size=args.render_size,
    )


if __name__ == "__main__":
    main()
