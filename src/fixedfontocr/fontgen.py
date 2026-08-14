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
from .preprocess import (
    Component,
    NormalizeSpec,
    compute_normalize_spec,
    glyph_normalize_geometry,
    normalize,
)


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
) -> None:
    """Write ``config.json``, ``charset.txt`` and ``weights.bin``.

    ``font_path`` is resolved against ``fonts/`` and its SHA256 is stored in
    the model metadata unless an explicit ``font_sha256`` is given.
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

    header = np.array([len(chars), templates.shape[1]], dtype="<u4")
    payload = np.concatenate([header.view(np.uint8), templates.reshape(-1)])
    (model_dir / "weights.bin").write_bytes(payload.tobytes())
