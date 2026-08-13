"""Render font glyphs into normalized bitset templates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from .preprocess import normalize


def render_glyph(
    font_path: str | Path,
    char: str,
    render_size: int = 64,
    threshold: int = 140,
) -> NDArray[np.bool_]:
    """Render one character to a tight boolean ink mask."""

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
    Characters the font cannot render are skipped with a warning.
    """

    chars: list[str] = []
    rows: list[NDArray[np.uint8]] = []
    for char in charset:
        ink = render_glyph(font_path, char, render_size, threshold)
        if ink.size == 0:
            print(f"skip {char!r}: no ink rendered")
            continue
        normalized = normalize(ink, target_size)
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
) -> None:
    """Write ``config.json``, ``charset.txt`` and ``weights.bin``."""

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input_width": target_size,
        "input_height": target_size,
        "classes": len(chars),
        "version": version,
        "dtype": "f32",
    }
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
