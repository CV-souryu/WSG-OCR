from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from fixedfontocr import defaults
from fixedfontocr.fontgen import build_templates, write_model


@pytest.fixture(scope="session")
def font_path() -> Path:
    if not defaults.FONT_PATH.exists():
        pytest.skip(
            "fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf not found — "
            "drop the registered font into fonts/ to run these tests"
        )
    return defaults.resolve_font(defaults.FONT_PATH)


@pytest.fixture(scope="session")
def model_dir(font_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    charset = (
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "+-×÷%.,:;!?()[]"
    )
    out = tmp_path_factory.mktemp("model")
    chars, templates = build_templates(font_path, list(charset), render_size=32)
    write_model(out, chars, templates, font_path=font_path)
    return out


def render_text(
    text: str,
    font_path: Path,
    font_size: int = 32,
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
