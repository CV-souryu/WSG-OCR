#!/usr/bin/env python3
"""Generate a synthetic fixed-font training dataset from a TTF/OTF font.

Every sample renders one character through the Goal 7 low-resolution
training domain: the font is rendered large (2-3x supersampled), then
downsampled to the final 10-18 px screenshot size with bilinear/area-like
degradation. Per-sample augmentation randomizes:

* font size (10, 11, 12, 13, 14, 15, 16, 17, 18 px),
* sub-pixel x/y offset (float, not whole-pixel),
* render scale / UI-scale variation,
* downsampling method (bilinear or area-like),
* gaussian blur before and after downsampling,
* alpha / text opacity,
* text brightness,
* background level, background mixing and sparse noise,
* outline and shadow changes.

This simulates ``font -> game render -> UI scale -> GPU sampling ->
screenshot`` instead of only rendering the font itself. Output is a
compressed ``.npz``:

    x:                 uint8  [N, 24, 24] normalized glyphs (0/255)
    y:                 int64  [N] label index into ``chars``
    chars:             unicode [C] charset
    font_sha256:       unicode [1] SHA256 of the registered source font
    input_mode:        unicode [1] "binary" (default) or "soft"
    render_sizes:      int64  [N] final 10..18 px font size per sample
    source_sizes:      int64  [N] supersampled render size before downsample
    downsample_modes:  unicode [N] bilinear/area degradation per sample
    augmentations:     unicode [N] comma-separated applied augmentation tags
    render_size_min:   int64  [1] configured minimum final size
    render_size_max:   int64  [1] configured maximum final size

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

LOW_RES_SIZES = tuple(range(10, 19))
DEFAULT_RENDER_SIZE_MIN = LOW_RES_SIZES[0]
DEFAULT_RENDER_SIZE_MAX = LOW_RES_SIZES[-1]
DEFAULT_SUPERSAMPLE_MIN = 2
DEFAULT_SUPERSAMPLE_MAX = 3
DOWNSAMPLE_MODES = ("bilinear", "area", "random")


def _scale(mask: NDArray[np.bool_], factor: float) -> NDArray[np.bool_]:
    """Nearest-neighbor resize of a boolean ink mask."""
    h, w = mask.shape
    nh = max(1, int(round(h * factor)))
    nw = max(1, int(round(w * factor)))
    ys = (np.arange(nh)[:, None] * h / nh).astype(np.int32)
    xs = (np.arange(nw)[None, :] * w / nw).astype(np.int32)
    return mask[ys, xs]


def _scale_gray(glyph: NDArray[np.uint8], factor: float) -> NDArray[np.uint8]:
    """Bilinear resize of a grayscale glyph (keeps soft edge intensities)."""

    from PIL import Image

    h, w = glyph.shape
    nh = max(1, int(round(h * factor)))
    nw = max(1, int(round(w * factor)))
    img = Image.fromarray(glyph, mode="L")
    return np.asarray(
        img.resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.uint8
    )


def _shift(glyph: NDArray[np.uint8], dx: int, dy: int) -> NDArray[np.uint8]:
    out = np.zeros_like(glyph)
    y0, y1 = max(0, dy), min(glyph.shape[0], glyph.shape[0] + dy)
    x0, x1 = max(0, dx), min(glyph.shape[1], glyph.shape[1] + dx)
    sy0, sy1 = max(0, -dy), min(glyph.shape[0], glyph.shape[0] - dy)
    sx0, sx1 = max(0, -dx), min(glyph.shape[1], glyph.shape[1] - dx)
    out[y0:y1, x0:x1] = glyph[sy0:sy1, sx0:sx1]
    return out


def _pick_downsample(mode: str, rng: np.random.Generator) -> str:
    """Resolve the downsample mode, including the random choice."""

    mode = mode.lower()
    if mode not in DOWNSAMPLE_MODES:
        raise ValueError(
            f"downsample must be one of {DOWNSAMPLE_MODES}, got {mode!r}"
        )
    if mode == "random":
        return str(rng.choice(("bilinear", "area")))
    return mode


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
    render_size_min: int | None = None,
    render_size_max: int | None = None,
    supersample_min: int = DEFAULT_SUPERSAMPLE_MIN,
    supersample_max: int = DEFAULT_SUPERSAMPLE_MAX,
    downsample: str = "random",
    scale_min: float = 0.85,
    scale_max: float = 1.15,
) -> None:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    font = resolve_font(font)
    font_sha256 = compute_font_sha256(font)
    spec = compute_normalize_spec(font, charset, size, size)
    rng = np.random.default_rng(seed)
    if render_size is not None:
        render_size_min = render_size_max = render_size
    if render_size_min is None:
        render_size_min = DEFAULT_RENDER_SIZE_MIN
    if render_size_max is None:
        render_size_max = DEFAULT_RENDER_SIZE_MAX
    if render_size_min < 2 or render_size_max < render_size_min:
        raise ValueError(
            f"invalid render size range: {render_size_min}..{render_size_max}"
        )
    if supersample_min < 1 or supersample_max < supersample_min:
        raise ValueError(
            f"invalid supersample range: {supersample_min}..{supersample_max}"
        )
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError(f"invalid scale range: {scale_min}..{scale_max}")
    _pick_downsample(downsample, rng)  # validate mode early
    x_list: list[NDArray[np.uint8]] = []
    y_list: list[int] = []
    render_sizes: list[int] = []
    source_sizes: list[int] = []
    downsample_modes: list[str] = []
    augmentation_tags: list[str] = []

    for label, char in enumerate(charset):
        for _ in range(samples_per_char):
            render_px = int(
                rng.integers(render_size_min, render_size_max + 1)
            )
            ss = int(rng.integers(supersample_min, supersample_max + 1))
            font_scale = float(rng.uniform(scale_min, scale_max))
            hi_size = max(2, int(round(render_px * ss * font_scale)))
            font_obj = ImageFont.truetype(str(font), hi_size)
            pad = max(8, int(round(hi_size * 0.35)))
            bbox = font_obj.getbbox(char)
            stroke_width = int(rng.integers(0, 3)) * ss
            margin = stroke_width * 2 + 4
            w = bbox[2] - bbox[0] + pad * 2 + margin
            h = bbox[3] - bbox[1] + pad * 2 + margin
            bg = int(rng.integers(0, 71))
            fill = int(rng.integers(190, 256))
            alpha = float(rng.uniform(0.7, 1.0))
            brightness = float(rng.uniform(0.7, 1.3))
            # Sub-pixel offset: fractional in high-res units, so after
            # downsampling it lands on a sub-pixel phase of the final grid.
            dx = float(rng.uniform(-0.4, 0.4)) * ss
            dy = float(rng.uniform(-0.4, 0.4)) * ss
            tags = ["subpixel", "alpha", "brightness", "background"]

            canvas = Image.new("RGBA", (max(1, w), max(1, h)), (bg, bg, bg, 255))
            text_layer = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_layer)

            # Shadow: a darker copy offset by 1-2 px underneath the glyph.
            if rng.random() < 0.4:
                sh = int(rng.integers(1, 3)) * ss
                draw.text(
                    (pad - bbox[0] + dx + sh, pad - bbox[1] + dy + sh),
                    char,
                    font=font_obj,
                    fill=(0, 0, 0, int(alpha * 255)),
                )
                tags.append("shadow")

            # Outline: Pillow stroke around the glyph.
            stroke_fill = (
                tuple(int(rng.integers(80, 180)) for _ in range(3))
                if stroke_width
                else None
            )
            draw.text(
                (pad - bbox[0] + dx, pad - bbox[1] + dy),
                char,
                font=font_obj,
                fill=(fill, fill, fill, int(alpha * 255)),
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            if stroke_width:
                tags.append("outline")

            img = Image.alpha_composite(canvas, text_layer).convert("L")
            arr = np.asarray(img, dtype=np.float32) * brightness
            arr = np.clip(arr, 0.0, 255.0)
            img = Image.fromarray(arr.astype(np.uint8), mode="L")

            # Pre-downsample blur (high-res anti-aliasing / GPU sampling).
            if rng.random() < 0.5:
                radius = float(rng.uniform(0.0, 1.6)) * ss
                if radius >= 0.05:
                    img = img.filter(ImageFilter.GaussianBlur(radius))
                    tags.append("blur")

            # Sparse background speckles (stays below most thresholds).
            if rng.random() < 0.3:
                arr = np.array(img, dtype=np.uint8).copy()
                noise = rng.random(arr.shape) < 0.004
                arr[noise] = rng.integers(0, 90, size=int(noise.sum())).astype(
                    np.uint8
                )
                img = Image.fromarray(arr)

            # Downsample to the final screenshot size with bilinear/area-like
            # degradation (GPU sampling / UI-scale simulation).
            target_w = max(1, round(w / ss))
            target_h = max(1, round(h / ss))
            mode = _pick_downsample(downsample, rng)
            resample = (
                Image.Resampling.BILINEAR
                if mode == "bilinear"
                else Image.Resampling.BOX
            )
            img = img.resize((target_w, target_h), resample)
            downsample_modes.append(mode)
            tags.append("downsample")

            # A slight post-downsample blur is common in final screenshots.
            if rng.random() < 0.3:
                radius = float(rng.uniform(0.0, 0.8))
                if radius >= 0.05:
                    img = img.filter(ImageFilter.GaussianBlur(radius))
                    tags.append("blur")

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

            # Extra per-glyph scale variation (Goal 3 baseline frame is
            # applied next, but the raster phase should not be identical).
            if rng.random() < 0.5:
                factor = float(rng.uniform(scale_min, scale_max))
                tight = _scale(tight, factor)
                tight_gray = _scale_gray(tight_gray, factor)
                tags.append("scale")

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
            render_sizes.append(render_px)
            source_sizes.append(hi_size)
            augmentation_tags.append(",".join(tags))

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
        render_sizes=np.asarray(render_sizes, dtype=np.int64),
        source_sizes=np.asarray(source_sizes, dtype=np.int64),
        downsample_modes=np.asarray(downsample_modes, dtype="<U16"),
        augmentations=np.asarray(augmentation_tags, dtype="<U128"),
        render_size_min=np.asarray(render_size_min, dtype=np.int64),
        render_size_max=np.asarray(render_size_max, dtype=np.int64),
    )
    print(
        f"wrote {x.shape[0]} synthetic samples ({len(charset)} classes) "
        f"to {out}"
    )
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
        help="fixed final raster size (px); overrides --render-size-min/max",
    )
    parser.add_argument(
        "--render-size-min",
        type=int,
        default=DEFAULT_RENDER_SIZE_MIN,
        help=f"minimum final font size (default: {DEFAULT_RENDER_SIZE_MIN})",
    )
    parser.add_argument(
        "--render-size-max",
        type=int,
        default=DEFAULT_RENDER_SIZE_MAX,
        help=f"maximum final font size (default: {DEFAULT_RENDER_SIZE_MAX})",
    )
    parser.add_argument(
        "--supersample-min",
        type=int,
        default=DEFAULT_SUPERSAMPLE_MIN,
        help="minimum supersample factor before downsampling",
    )
    parser.add_argument(
        "--supersample-max",
        type=int,
        default=DEFAULT_SUPERSAMPLE_MAX,
        help="maximum supersample factor before downsampling",
    )
    parser.add_argument(
        "--downsample",
        choices=DOWNSAMPLE_MODES,
        default="random",
        help="degradation used when resizing to the final screenshot size",
    )
    parser.add_argument(
        "--scale-min",
        type=float,
        default=0.85,
        help="minimum font/UI scale variation multiplier",
    )
    parser.add_argument(
        "--scale-max",
        type=float,
        default=1.15,
        help="maximum font/UI scale variation multiplier",
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
        render_size_min=args.render_size_min,
        render_size_max=args.render_size_max,
        supersample_min=args.supersample_min,
        supersample_max=args.supersample_max,
        downsample=args.downsample,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )


if __name__ == "__main__":
    main()
