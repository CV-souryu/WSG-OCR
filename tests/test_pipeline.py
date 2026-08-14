from __future__ import annotations

import numpy as np

from fixedfontocr import FixedFontOCR
from fixedfontocr.types import Profile

from conftest import render_text


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


def test_recognize_decimal_point(font_path, model_dir):
    """A '.' must be recognized when the model charset contains it."""
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    for text in ("3.14", "0.5", ".5", "5.", "3..14", "."):
        result = ocr.recognize(render_text(text, font_path))
        assert result.text == text, f"{text!r} -> {result.text!r}"
        assert all(c.confidence >= 0.9 for c in result.chars)


def test_recognize_distinguishes_decimal_and_middle_dot(font_path, tmp_path):
    """'.' (baseline) and '·' (vertical center) must stay distinct in a line."""
    from fixedfontocr.fontgen import build_templates, write_model

    charset = "0123456789·."
    chars, templates = build_templates(font_path, list(charset), render_size=32)
    out = tmp_path / "dot_model"
    write_model(out, chars, templates, font_path=font_path)
    ocr = FixedFontOCR(model_path=out, backend="cpu")
    for text in ("3.14", "3·14", "3.14·5", "12·3.4", "3.14·5.2"):
        result = ocr.recognize(render_text(text, font_path))
        assert result.text == text, f"{text!r} -> {result.text!r}"


def test_template_classifier_distinguishes_chars(font_path, model_dir):
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    # 'l' is excluded: in the bundled font its bitmap is identical to 'I',
    # so any pixel-exact baseline is allowed to map them together.
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


def test_wgpu_backend_requires_tinycnn_model(font_path, model_dir):
    """Template models stay on CPU; 'wgpu' must reject them clearly."""
    try:
        FixedFontOCR(model_path=model_dir, backend="wgpu")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for template model + wgpu backend")


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


def test_hsl_profile_keeps_white_text_and_drops_colored_ui():
    """HSL binarization: white game text survives, colored UI/1px noise does not."""
    image = np.full((30, 128, 3), (16, 125, 214), dtype=np.uint8)  # blue box
    image[10:26, 20:108] = (255, 255, 255)  # white text band
    image[0, :] = (120, 160, 100)  # colored 1px noise row (not white)

    profile = Profile(name="hsl-white", use_hsl=True, use_grayscale=False)
    mask = profile.color_mask(image)

    assert mask[10:26, 20:108].all()  # white text kept
    assert not mask[0, :].any()  # colored noise row dropped
    assert not mask[1:10, :].any()  # blue background dropped
    assert not mask[26:, :].any()
