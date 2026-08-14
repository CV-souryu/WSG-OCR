#!/usr/bin/env python3
"""Regenerate the synthetic regression samples under tests/game_samples/.

The P8 regression set mixes real game-level badges (tests/game_samples/level,
kept as committed fixtures) with deterministic synthetic renders of the
registered font covering: normal Chinese, digits, mixed Chinese+ASCII,
punctuation, fragment-heavy glyphs (鲃/小), confusable pairs (甲/申, 未/末),
light/dark backgrounds, anti-aliasing and multiple font sizes.

Usage:
    python tools/dataset/generate_synthetic_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

from fixedfontocr.defaults import resolve_font  # noqa: E402


def render(
    font_path: Path,
    text: str,
    size: int,
    fill=(255, 255, 255),
    bg=(0, 0, 0),
    blur: float = 0.0,
) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), size)
    bbox = font.getbbox(text)
    pad = 8
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGB", (max(1, w), max(1, h)), bg)
    ImageDraw.Draw(img).text(
        (pad - bbox[0], pad - bbox[1]), text, font=font, fill=fill
    )
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(img, dtype=np.uint8)


def main() -> None:
    font_path = resolve_font(ROOT / "fonts" / "SourceHanSansSC" / "SourceHanSansSC-Bold.otf")
    out_dir = ROOT / "tests" / "game_samples" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "zh_32.png": ("获得金币1000", 32, (255, 255, 255), (0, 0, 0), 0.0),
        "digits_32.png": ("1234567890", 32, (255, 255, 255), (0, 0, 0), 0.0),
        "mixed_32.png": ("舰船Lv.99", 32, (255, 255, 255), (0, 0, 0), 0.0),
        "punct_32.png": ("。，！？", 32, (255, 255, 255), (0, 0, 0), 0.0),
        "frag_ba_32.png": ("鲃", 32, (255, 255, 255), (0, 0, 0), 0.0),
        "frag_xiao_32.png": ("小", 32, (255, 255, 255), (0, 0, 0), 0.0),
        "jiashen_32.png": ("甲申", 32, (255, 255, 255), (0, 0, 0), 0.0),
        "weimo_32.png": ("未末", 32, (255, 255, 255), (0, 0, 0), 0.0),
        "light_bg_32.png": ("获得金币1000", 32, (0, 0, 0), (210, 210, 210), 0.0),
        "dark_bg_32.png": ("生命值1000", 32, (255, 255, 255), (0, 0, 0), 0.0),
        "antialias_32.png": ("获得金币1000", 32, (255, 255, 255), (0, 0, 0), 1.2),
        "small_16.png": ("获得金币1000", 16, (255, 255, 255), (0, 0, 0), 0.0),
        "large_48.png": ("获得金币1000", 48, (255, 255, 255), (0, 0, 0), 0.0),
    }
    for name, (text, size, fill, bg, blur) in cases.items():
        arr = render(font_path, text, size, fill=fill, bg=bg, blur=blur)
        Image.fromarray(arr).save(out_dir / name)
    print(f"wrote {len(cases)} synthetic samples to {out_dir}")


if __name__ == "__main__":
    main()
