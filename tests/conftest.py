from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from fixedfontocr.fontgen import build_templates, write_model


def _find_font() -> Path:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/System/Library/Fonts/Geneva.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Courier.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return Path(path)
    raise pytest.skip("no suitable system font found for tests")


@pytest.fixture(scope="session")
def font_path() -> Path:
    return _find_font()


@pytest.fixture(scope="session")
def model_dir(font_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    charset = (
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "+-×÷%.,:;!?()[]"
    )
    out = tmp_path_factory.mktemp("model")
    chars, templates = build_templates(font_path, list(charset), render_size=28)
    write_model(out, chars, templates)
    return out


def render_text(
    text: str,
    font_path: Path,
    font_size: int = 28,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Render text on black and return an RGB uint8 array."""

    font = ImageFont.truetype(str(font_path), font_size)
    bbox = font.getbbox(text)
    pad = 8
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGB", (max(1, w), max(1, h)), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=color)
    return np.asarray(img, dtype=np.uint8)
