from __future__ import annotations

import numpy as np

from fixedfontocr import FixedFontOCR
from fixedfontocr.types import Profile

from .conftest import render_text


def test_recognize_digits(font_path, model_dir):
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("1234567890", font_path)
    result = ocr.recognize(image)

    assert result.text == "1234567890"
    assert result.confidence > 0.8
    assert len(result.chars) == 10
    for c in result.chars:
        assert c.w > 0 and c.h > 0


def test_recognize_alphanumeric(font_path, model_dir):
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("Ab3Zz9", font_path)
    result = ocr.recognize(image)
    assert result.text == "Ab3Zz9"


def test_template_classifier_distinguishes_chars(font_path, model_dir):
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    # 'l' is excluded: in Arial its bitmap is identical to 'I', so any
    # pixel-exact baseline is allowed to map them together.
    for ch in "O01I1":
        image = render_text(ch, font_path)
        result = ocr.recognize(image)
        assert result.text == ch, f"expected {ch!r}, got {result.text!r}"


def test_input_validation(font_path, model_dir):
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    bad = np.zeros((10, 10), dtype=np.uint8)
    try:
        ocr.recognize(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-RGB image")


def test_empty_image_returns_empty_result(font_path, model_dir):
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    result = ocr.recognize(image)
    assert result.text == ""
    assert result.chars == ()
    assert result.confidence == 0.0


def test_wgpu_backend_not_implemented_yet(font_path, model_dir):
    try:
        FixedFontOCR(model_path=model_dir, backend="wgpu")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unimplemented wgpu backend")


def test_custom_profile_is_used(font_path, model_dir):
    profile = Profile(
        name="white-on-dark",
        use_grayscale=True,
        grayscale_threshold=100,
        char_height_min=8,
        char_height_max=64,
        char_width_min=3,
        char_width_max=72,
        target_size=24,
    )
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu", profile=profile)
    image = render_text("42", font_path)
    result = ocr.recognize(image)
    assert result.text == "42"
