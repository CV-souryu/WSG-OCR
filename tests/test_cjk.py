from __future__ import annotations

from fixedfontocr import FixedFontOCR
from fixedfontocr.fontgen import build_templates, write_model

from conftest import render_text


def test_recognize_chinese(tmp_path, font_path):
    text = "获得金币1000"
    charset = "获得金币1000"
    model_dir = tmp_path / "model"
    chars, templates = build_templates(font_path, list(charset), render_size=32)
    write_model(model_dir, chars, templates, font_path=font_path)

    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    result = ocr.recognize(render_text(text, font_path))
    assert result.text == text
    assert len(result.chars) == len(text)
