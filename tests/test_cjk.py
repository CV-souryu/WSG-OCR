from __future__ import annotations

from pathlib import Path

import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.fontgen import build_templates, write_model

from conftest import render_text


def _cjk_font() -> Path:
    for path in (
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ):
        if Path(path).exists():
            return Path(path)
    raise pytest.skip("no CJK-capable system font found")


def test_recognize_chinese(tmp_path):
    font_path = _cjk_font()
    text = "获得金币1000"
    charset = "获得金币1000"
    model_dir = tmp_path / "model"
    chars, templates = build_templates(font_path, list(charset), render_size=28)
    write_model(model_dir, chars, templates)

    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    result = ocr.recognize(render_text(text, font_path))
    assert result.text == text
    assert len(result.chars) == len(text)
