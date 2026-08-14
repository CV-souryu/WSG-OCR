from __future__ import annotations

import numpy as np

from fixedfontocr.classifier import TemplateClassifier
from fixedfontocr.fontgen import build_templates, render_glyph
from fixedfontocr.types import default_profile


def test_candidate_filter_preserves_results(font_path):
    charset = (
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "+-×÷%.,:;!?()[]"
    )
    chars, templates = build_templates(font_path, list(charset), render_size=28)
    full = TemplateClassifier(templates, chars, candidate_filter=False)
    filtered = TemplateClassifier(templates, chars, candidate_filter=True)
    profile = default_profile()
    total = 0
    reduced = 0
    for ch in chars:
        mask = render_glyph(font_path, ch, render_size=28)
        a = full(mask, profile)
        b = filtered(mask, profile)
        assert a == b, f"filter changed result for {ch!r}: {a} vs {b}"
        total += 1
        if filtered.last_candidates is not None and filtered.last_candidates < len(charset):
            reduced += 1
    assert reduced == total, "candidate filter should narrow at least every glyph"


def test_candidate_filter_keeps_ambiguous_chars_together(font_path):
    charset = "O0Il"
    chars, templates = build_templates(font_path, list(charset), render_size=28)
    filtered = TemplateClassifier(templates, chars, candidate_filter=True)
    profile = default_profile()
    for ch in chars:
        mask = render_glyph(font_path, ch, render_size=28)
        filtered(mask, profile)
        assert filtered.last_candidates >= 1


def test_candidate_filter_signed_comparison_safe(font_path):
    """A thick glyph (low threshold) must not underflow compact dtypes."""
    from PIL import Image, ImageDraw, ImageFont

    chars, templates = build_templates(font_path, list("0123456789"), render_size=28)
    full = TemplateClassifier(templates, chars, candidate_filter=False)
    filtered = TemplateClassifier(templates, chars, candidate_filter=True)
    profile = default_profile()
    font = ImageFont.truetype(str(font_path), 28)
    canvas = Image.new("RGB", (120, 80), (0, 0, 0))
    ImageDraw.Draw(canvas).text((8, 8), "2", font=font, fill=(255, 255, 255))
    arr = np.asarray(canvas, dtype=np.uint8)
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mask = gray >= 100  # thicker than the render_size=28 / threshold=140 template
    ys, xs = np.where(mask)
    tight = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    assert filtered(tight, profile) == full(tight, profile)
