from __future__ import annotations

from pathlib import Path

import pytest

from fixedfontocr import FixedFontOCR

from conftest import render_text


def test_template_allowed_chars(font_path, model_dir):
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("13579", font_path)
    assert ocr.recognize(image, allowed_chars="13579").text == "13579"


def test_template_allowed_chars_empty(font_path, model_dir):
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("42", font_path)
    assert ocr.recognize(image, allowed_chars="").text == "??"


def test_cnn_allowed_chars(font_path):
    model_dir = Path(__file__).parent / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        pytest.skip("cnn_digits fixture not generated")
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("13579", font_path)
    assert ocr.recognize(image, allowed_chars="13579").text == "13579"


def test_cnn_allowed_chars_excludes_unknown(font_path):
    model_dir = Path(__file__).parent / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        pytest.skip("cnn_digits fixture not generated")
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("42", font_path)
    result = ocr.recognize(image, allowed_chars="XYZ")
    assert result.text == "??"
