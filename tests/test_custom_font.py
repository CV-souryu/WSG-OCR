"""Recognition tests driven by the project's bundled game font and charset.

The template model is generated from
``fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf`` (Source Han Sans SC Bold)
with ``charsets/sets/combined.txt`` — the game-UI font and CN charset this project
ships with. The font file itself is gitignored (see ``.gitignore``), so these
tests skip cleanly when it is not present on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr import defaults
from fixedfontocr.fontgen import build_templates, write_model

from conftest import render_text

CUSTOM_FONT = defaults.FONT_PATH

# The full CN config charset: ship names, equipment names, Chinese UI copy,
# and the ASCII letters/digits/punctuation set.
CHARSET = defaults.read_charset()

TEST_STRINGS = [
    "俾斯麦",
    "提尔比茨",
    "威尔士亲王",
    "大型单装炮",
    "获得金币1000",
    "返回主菜单",
    "生命值1000",
    "装备名称",
    "港区出击",
    "0123456789",
]

# Rendering at the same size for templates and test images is what keeps the
# fixed-font matching exact (validated: 14/14 strings, confidence 1.0).
RENDER_SIZE = 32


@pytest.fixture(scope="session")
def custom_font() -> Path:
    if not CUSTOM_FONT.exists():
        pytest.skip(
            "fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf not found — the "
            "font is gitignored; drop it into fonts/ to run these tests"
        )
    return defaults.resolve_font(CUSTOM_FONT)


@pytest.fixture(scope="session")
def custom_model_dir(custom_font: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("custom_font_model")
    chars, templates = build_templates(
        custom_font, list(CHARSET), render_size=RENDER_SIZE
    )
    write_model(out, chars, templates, font_path=custom_font)
    return out


@pytest.fixture(scope="session")
def custom_ocr(custom_model_dir: Path) -> FixedFontOCR:
    return FixedFontOCR(model_path=custom_model_dir, backend="cpu")


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_recognize_custom_font(custom_ocr: FixedFontOCR, custom_font: Path, text: str) -> None:
    result = custom_ocr.recognize(
        render_text(text, custom_font, font_size=RENDER_SIZE)
    )
    assert result.text == text
    assert len(result.chars) == len(text)
    assert result.confidence > 0.95


def test_custom_font_model_covers_charset(custom_model_dir: Path) -> None:
    """The generated model directory must carry the full requested charset."""
    charset_txt = (custom_model_dir / "charset.txt").read_text(encoding="utf-8")
    assert set(charset_txt.strip()) == set(CHARSET)
