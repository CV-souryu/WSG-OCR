"""Chinese recognition tests driven by the project's custom font.

The template model is generated from ``fonts/SourceHanSansSC-Bold.otf``
(Source Han Sans SC Bold) — the game-UI font this project ships with. The
font file itself is gitignored (see ``.gitignore``), so these tests skip
cleanly when it is not present on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.fontgen import build_templates, write_model

from conftest import render_text

CUSTOM_FONT = (
    Path(__file__).resolve().parents[1] / "fonts" / "SourceHanSansSC-Bold.otf"
)

# Every character appearing in TEST_STRINGS must be listed here: the
# template baseline can only recognize characters it has a template for.
CHARSET = "获得金币消耗数量提示确定取消返回攻击防御生命法力值力量点要吗主菜单0123456789"

TEST_STRINGS = [
    "获得金币1000",
    "数量提示",
    "确定取消返回",
    "确定要取消吗",
    "攻击防御生命法力",
    "生命值1000",
    "返回主菜单",
    "消耗法力10点",
    "金币500",
    "攻击力15",
    "获得10金币",
    "防御力50",
    "获得金币1000数量提示确定取消返回攻击防御生命法力",  # long single line
    "0123456789",
]

# Rendering at the same size for templates and test images is what keeps the
# fixed-font matching exact (validated: 14/14 strings, confidence 1.0).
RENDER_SIZE = 32


@pytest.fixture(scope="session")
def custom_font() -> Path:
    if not CUSTOM_FONT.exists():
        pytest.skip(
            "fonts/SourceHanSansSC-Bold.otf not found — the font is gitignored; "
            "drop it into fonts/ to run these tests"
        )
    return CUSTOM_FONT


@pytest.fixture(scope="session")
def custom_model_dir(custom_font: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("custom_font_model")
    chars, templates = build_templates(
        custom_font, list(CHARSET), render_size=RENDER_SIZE
    )
    write_model(out, chars, templates)
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
