"""Render font glyphs into normalized bitset templates."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .defaults import (
    compute_font_sha256,
    ensure_font_path,
    resolve_font,
)
from .classifier import (
    DOWNSAMPLE_AREA,
    DOWNSAMPLE_BILINEAR,
    DOWNSAMPLE_CLEAN,
    TemplateV2Data,
)
from .geometry import write_geometry_json
from .model import save_template_v2
from .preprocess import (
    Component,
    NormalizeSpec,
    compute_normalize_spec,
    glyph_normalize_geometry,
    normalize,
)


# Goal 9 Template V2 prototype grid. 11..16 px are the real UI sizes from
# Goal 7; every size is rasterized on a supersampled canvas, shifted by a
# sub-pixel phase, downsampled with bilinear/area-like degradation, then
# normalized into the Goal 3 24x24 baseline frame. A clean high-resolution
# prototype (the old V1 render) is appended so V2 is never worse than V1 on
# clean screenshots.
TEMPLATE_V2_SIZES = tuple(range(11, 17))
TEMPLATE_V2_SUBPIXEL_PHASES = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
TEMPLATE_V2_DOWNSAMPLE_MODES = ("bilinear", "area")
TEMPLATE_V2_SUPERSAMPLE = 3
TEMPLATE_V2_HIGH_RES_SIZE = 32


@functools.lru_cache(maxsize=64)
def _truetype(font_path: str, size: int):
    return ImageFont.truetype(font_path, size)


@functools.lru_cache(maxsize=16)
def _cmap_codepoints(font_path: str) -> frozenset[int]:
    """Return the set of Unicode code points covered by a font's cmap."""
    font = TTFont(font_path)
    try:
        return frozenset(font.getBestCmap() or {})
    finally:
        font.close()


def render_glyph(
    font_path: str | Path,
    char: str,
    render_size: int = 64,
    threshold: int = 140,
) -> NDArray[np.bool_]:
    """Render one character to a tight boolean ink mask."""

    font_path = ensure_font_path(font_path)
    if ord(char) not in _cmap_codepoints(str(font_path)):
        raise ValueError(f"font {font_path} cannot render {char!r}: missing glyph")
    font = ImageFont.truetype(str(font_path), render_size)
    # Use a generous canvas; the ink bbox below gives the tight mask.
    canvas = Image.new("L", (render_size * 2, render_size * 2), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 0), char, font=font, fill=255)
    arr = np.asarray(canvas, dtype=np.uint8)
    mask = arr >= threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows):
        return np.zeros((0, 0), dtype=bool)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    return mask[int(ys[0]) : int(ys[-1]) + 1, int(xs[0]) : int(xs[-1]) + 1]


def render_prototype(
    font_path: str | Path,
    char: str,
    render_size: int,
    subpixel_dx: float = 0.0,
    subpixel_dy: float = 0.0,
    downsample: str = "bilinear",
    supersample: int = TEMPLATE_V2_SUPERSAMPLE,
    threshold: int = 140,
    threshold_min: int = 60,
) -> NDArray[np.bool_]:
    """Render one Goal 9 prototype: supersampled glyph -> downsample -> mask.

    The font is rasterized at ``render_size * supersample`` px (so a
    fractional sub-pixel offset and the downsample phase land on the final
    pixel grid), then resized to the target screenshot size with bilinear or
    area-like (BOX) resampling and binarized at ``threshold``. The result is
    a tight boolean ink mask in the same form as :func:`render_glyph`. A
    thin glyph whose anti-aliased core disappears at the configured
    threshold is retried at lower thresholds (like the Goal 7 dataset
    generator) before an explicit "no ink" error is raised.
    """

    font_path = ensure_font_path(font_path)
    if ord(char) not in _cmap_codepoints(str(font_path)):
        raise ValueError(f"font {font_path} cannot render {char!r}: missing glyph")
    if downsample not in TEMPLATE_V2_DOWNSAMPLE_MODES:
        raise ValueError(
            f"downsample must be one of {TEMPLATE_V2_DOWNSAMPLE_MODES}, "
            f"got {downsample!r}"
        )
    if render_size < 2:
        raise ValueError(f"render_size must be >= 2, got {render_size}")
    if supersample < 1:
        raise ValueError(f"supersample must be >= 1, got {supersample}")
    hi_size = render_size * supersample
    font = _truetype(str(font_path), hi_size)
    bbox = font.getbbox(char)
    pad = max(4, int(round(hi_size * 0.25)))
    w = bbox[2] - bbox[0] + pad * 2 + 4
    h = bbox[3] - bbox[1] + pad * 2 + 4
    canvas = Image.new("L", (max(1, w), max(1, h)), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (
            pad - bbox[0] + subpixel_dx * supersample,
            pad - bbox[1] + subpixel_dy * supersample,
        ),
        char,
        font=font,
        fill=255,
    )
    target_w = max(1, int(round(w / supersample)))
    target_h = max(1, int(round(h / supersample)))
    resample = (
        Image.Resampling.BILINEAR
        if downsample == "bilinear"
        else Image.Resampling.BOX
    )
    img = canvas.resize((target_w, target_h), resample)
    arr = np.asarray(img, dtype=np.uint8)
    t = threshold
    while t >= threshold_min:
        mask = arr >= t
        if np.any(mask):
            break
        t -= 20
    else:
        return np.zeros((0, 0), dtype=bool)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    return mask[int(ys[0]) : int(ys[-1]) + 1, int(xs[0]) : int(xs[-1]) + 1]


def build_template_v2(
    font_path: str | Path,
    charset: list[str],
    target_size: int = 24,
    render_sizes: tuple[int, ...] = TEMPLATE_V2_SIZES,
    subpixel_phases: tuple[tuple[float, float], ...] = TEMPLATE_V2_SUBPIXEL_PHASES,
    downsample_modes: tuple[str, ...] = TEMPLATE_V2_DOWNSAMPLE_MODES,
    supersample: int = TEMPLATE_V2_SUPERSAMPLE,
    threshold: int = 140,
    include_high_res: bool = True,
    high_res_size: int = TEMPLATE_V2_HIGH_RES_SIZE,
) -> tuple[list[str], TemplateV2Data]:
    """Render ``charset`` into a Goal 9 multi-prototype template set.

    Every character gets ``P = len(render_sizes) * len(subpixel_phases) *
    len(downsample_modes)`` low-resolution prototypes plus (by default) one
    clean high-resolution prototype, all normalized into the same Goal 3
    24x24 baseline frame. Returns ``(chars, TemplateV2Data)`` where the data
    carries the packed bitsets and per-prototype metadata (render size,
    sub-pixel phase in 1/8 px, downsample mode).
    """

    font_path = resolve_font(font_path)
    spec = compute_normalize_spec(font_path, charset, target_size, high_res_size)
    chars: list[str] = []
    rows: list[NDArray[np.uint8]] = []
    render_sizes_arr: list[int] = []
    dx_arr: list[int] = []
    dy_arr: list[int] = []
    mode_arr: list[int] = []
    for char in charset:
        per_char: list[NDArray[np.uint8]] = []
        for size in render_sizes:
            for dx, dy in subpixel_phases:
                for mode in downsample_modes:
                    ink = render_prototype(
                        font_path,
                        char,
                        size,
                        subpixel_dx=dx,
                        subpixel_dy=dy,
                        downsample=mode,
                        supersample=supersample,
                        threshold=threshold,
                    )
                    if ink.size == 0:
                        raise ValueError(
                            f"font {font_path} cannot render {char!r} at "
                            f"{size}px: no ink"
                        )
                    h, w = ink.shape
                    candidate = Component(mask=ink, x=0, y=0, w=w, h=h)
                    baseline_offset, scale = glyph_normalize_geometry(
                        candidate, spec, target_size
                    )
                    normalized = normalize(
                        ink,
                        target_size,
                        baseline_offset=baseline_offset,
                        scale=scale,
                        baseline_row=spec.baseline_row,
                    )
                    per_char.append(
                        np.packbits(normalized.reshape(-1), bitorder="little")
                    )
                    render_sizes_arr.append(size)
                    dx_arr.append(int(round(dx * 8)))
                    dy_arr.append(int(round(dy * 8)))
                    mode_arr.append(
                        DOWNSAMPLE_BILINEAR if mode == "bilinear" else DOWNSAMPLE_AREA
                    )
        if include_high_res:
            ink = render_glyph(font_path, char, high_res_size, threshold)
            if ink.size == 0:
                raise ValueError(
                    f"font {font_path} cannot render {char!r}: no ink"
                )
            h, w = ink.shape
            candidate = Component(mask=ink, x=0, y=0, w=w, h=h)
            baseline_offset, scale = glyph_normalize_geometry(
                candidate, spec, target_size
            )
            normalized = normalize(
                ink,
                target_size,
                baseline_offset=baseline_offset,
                scale=scale,
                baseline_row=spec.baseline_row,
            )
            per_char.append(np.packbits(normalized.reshape(-1), bitorder="little"))
            render_sizes_arr.append(high_res_size)
            dx_arr.append(0)
            dy_arr.append(0)
            mode_arr.append(DOWNSAMPLE_CLEAN)
        if not per_char:
            raise ValueError("no prototypes rendered; check the font and charset")
        rows.append(np.stack(per_char))
        chars.append(char)
    if not rows:
        raise ValueError("no characters rendered; check the font and charset")
    p = len(rows[0])
    data = TemplateV2Data(
        bits=np.stack(rows),
        render_sizes=np.asarray(render_sizes_arr, dtype=np.uint8).reshape(-1, p),
        dx=np.asarray(dx_arr, dtype=np.uint8).reshape(-1, p),
        dy=np.asarray(dy_arr, dtype=np.uint8).reshape(-1, p),
        downsample_modes=np.asarray(mode_arr, dtype=np.uint8).reshape(-1, p),
    )
    return chars, data


def build_templates(
    font_path: str | Path,
    charset: list[str],
    target_size: int = 24,
    render_size: int = 64,
    threshold: int = 140,
) -> tuple[list[str], np.ndarray]:
    """Render ``charset`` into normalized bitset templates.

    Returns ``(chars, templates)`` where each template row is a uint8 bit
    field of ``target_size * target_size`` bits, little-endian bit order.
    A character missing from the font is an explicit error.
    """

    font_path = resolve_font(font_path)
    spec = compute_normalize_spec(font_path, charset, target_size, render_size)
    chars: list[str] = []
    rows: list[NDArray[np.uint8]] = []
    for char in charset:
        ink = render_glyph(font_path, char, render_size, threshold)
        if ink.size == 0:
            raise ValueError(f"font {font_path} cannot render {char!r}: no ink")
        h, w = ink.shape
        candidate = Component(mask=ink, x=0, y=0, w=w, h=h)
        baseline_offset, scale = glyph_normalize_geometry(
            candidate, spec, target_size
        )
        normalized = normalize(
            ink,
            target_size,
            baseline_offset=baseline_offset,
            scale=scale,
            baseline_row=spec.baseline_row,
        )
        rows.append(np.packbits(normalized.reshape(-1), bitorder="little"))
        chars.append(char)
    if not rows:
        raise ValueError("no characters rendered; check the font and charset")
    return chars, np.stack(rows)


def write_model(
    model_dir: str | Path,
    chars: list[str],
    templates: NDArray[np.uint8],
    target_size: int = 24,
    version: int = 1,
    font_path: str | Path | None = None,
    font_sha256: str | None = None,
    render_size: int = 32,
    threshold: int = 140,
) -> None:
    """Write ``config.json``, ``charset.txt``, ``weights.bin`` and ``geometry.json``.

    ``font_path`` is resolved against ``fonts/`` and its SHA256 is stored in
    the model metadata unless an explicit ``font_sha256`` is given. When a
    font is provided, the Goal 8 font-geometry database is generated offline
    and stored alongside the model so the runtime never needs fontTools.
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if font_path is not None and font_sha256 is None:
        font_sha256 = compute_font_sha256(font_path)

    config = {
        "input_width": target_size,
        "input_height": target_size,
        "classes": len(chars),
        "version": version,
        "dtype": "f32",
    }
    if font_sha256:
        config["font_sha256"] = font_sha256
    if font_path is not None:
        config["normalize"] = compute_normalize_spec(
            font_path, chars, target_size, render_size
        ).to_dict()
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=4) + "\n",
        encoding="utf-8",
    )
    (model_dir / "charset.txt").write_text(
        "".join(chars) + "\n",
        encoding="utf-8",
    )
    if font_path is not None:
        write_geometry_json(
            model_dir,
            font_path,
            chars,
            render_size=render_size,
            threshold=threshold,
            font_sha256=font_sha256,
        )

    header = np.array([len(chars), templates.shape[1]], dtype="<u4")
    payload = np.concatenate([header.view(np.uint8), templates.reshape(-1)])
    (model_dir / "weights.bin").write_bytes(payload.tobytes())


def write_template_v2_model(
    model_dir: str | Path,
    chars: list[str],
    data: TemplateV2Data,
    target_size: int = 24,
    font_path: str | Path | None = None,
    font_sha256: str | None = None,
    render_size: int = TEMPLATE_V2_HIGH_RES_SIZE,
    threshold: int = 140,
) -> None:
    """Write a Goal 9 template V2 model directory.

    Same layout as :func:`write_model` (``config.json`` + ``charset.txt`` +
    ``geometry.json``), but ``weights.bin`` holds the V2 multi-prototype
    template set (``template_version: 2``) instead of one template per
    character. Old readers keep working because V1 files are detected by
    the absence of the V2 magic.
    """

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if font_path is not None and font_sha256 is None:
        font_sha256 = compute_font_sha256(font_path)
    if data.num_classes != len(chars):
        raise ValueError("templates_v2 and charset must have the same length")

    config = {
        "input_width": target_size,
        "input_height": target_size,
        "classes": len(chars),
        "version": 1,
        "dtype": "f32",
        "template_version": 2,
    }
    if font_sha256:
        config["font_sha256"] = font_sha256
    if font_path is not None:
        config["normalize"] = compute_normalize_spec(
            font_path, chars, target_size, render_size
        ).to_dict()
        config["template_v2"] = {
            "render_sizes": sorted(
                {int(s) for s in data.render_sizes.reshape(-1)}
            ),
            "downsample_modes": ["bilinear", "area", "clean"],
            "prototypes_per_char": data.prototypes_per_char,
            "high_res_size": render_size,
        }
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=4) + "\n",
        encoding="utf-8",
    )
    (model_dir / "charset.txt").write_text(
        "".join(chars) + "\n",
        encoding="utf-8",
    )
    if font_path is not None:
        write_geometry_json(
            model_dir,
            font_path,
            chars,
            render_size=render_size,
            threshold=threshold,
            font_sha256=font_sha256,
        )
    save_template_v2(model_dir / "weights.bin", data)
