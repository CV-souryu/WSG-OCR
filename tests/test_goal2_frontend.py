"""Goal 2 acceptance: the Visual Frontend (binary + soft in one pass).

These tests pin the contract from ``fonts/goal`` Goal 2:

* one ``np.ndarray[H, W, 3]`` input produces both a binary mask and a soft
  foreground in a single pass;
* the binary mask keeps driving connected components / font geometry /
  Template;
* the soft foreground preserves anti-aliasing, alpha, edge gray and
  low-resolution intensity information for the TinyCNN;
* the production OCR path feeds soft glyphs to the CNN while the template
  path keeps binary glyphs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fixedfontocr import FixedFontOCR, extract_frontend
from fixedfontocr.backends import CPUBackend
from fixedfontocr.model import load_model
from fixedfontocr.preprocess import find_lines, normalize, normalize_grayscale
from fixedfontocr.scorer import SegmentScorer
from fixedfontocr.segmentation import connected_components
from fixedfontocr.types import Profile, default_profile

from conftest import render_text


def test_frontend_extracts_both_representations_in_one_pass(font_path):
    image = render_text("小", font_path, font_size=16)
    profile = default_profile()

    frontend = extract_frontend(image, profile)

    assert frontend.image is image
    assert frontend.binary_mask.shape == image.shape[:2]
    assert frontend.binary_mask.dtype == np.bool_
    assert frontend.soft_foreground.shape == image.shape[:2]
    assert frontend.soft_foreground.dtype == np.uint8

    # Both representations come from the same profile and image.
    assert np.array_equal(frontend.binary_mask, profile.color_mask(image))
    assert np.array_equal(
        frontend.soft_foreground, profile.soft_foreground(image)
    )


def test_frontend_validates_rgb_uint8_input(font_path):
    profile = default_profile()
    for bad in (
        np.zeros((10, 10), dtype=np.uint8),  # not RGB
        np.zeros((10, 10, 4), dtype=np.uint8),  # RGBA
        np.zeros((10, 10, 3), dtype=np.float32),  # wrong dtype
    ):
        try:
            extract_frontend(bad, profile)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad.shape}/{bad.dtype}")


def test_soft_preserves_antialiasing_without_hard_cutoff(font_path):
    """Soft keeps intermediate edge intensities; binary is a hard decision."""

    image = render_text("A", font_path, font_size=16)
    frontend = extract_frontend(image, default_profile())

    soft = frontend.soft_foreground
    binary = frontend.binary_mask

    # Anti-aliased rendering produces edge pixels that are neither pure
    # background nor pure ink in the soft map.
    assert np.any((soft > 0) & (soft < 255))
    # The hard mask has no intermediate state.
    assert binary.dtype == np.bool_
    # Some pixels below the binarization threshold are still visible to the
    # CNN as partial ink (anti-aliasing / alpha / edge gray preserved).
    assert np.any(~binary & (soft > 0) & (soft < 255))


def test_soft_foreground_hsl_profile():
    """HSL soft keeps white text strong and colored UI weak (no hard cut)."""

    image = np.full((30, 128, 3), (16, 125, 214), dtype=np.uint8)  # blue box
    image[10:26, 20:108] = (255, 255, 255)  # white text band
    image[0, :] = (120, 160, 100)  # colored 1px noise row

    profile = Profile(name="hsl-white", use_hsl=True, use_grayscale=False)
    soft = profile.soft_foreground(image)

    assert soft[10:26, 20:108].min() >= 200  # white text is strong
    assert soft[1:10, :].max() <= 20  # blue background is weak
    # A mid-lightness, low-saturation colored pixel is neither pure ink nor
    # pure background in the soft map (no hard cutoff, like an edge pixel).
    assert 1 <= soft[0, :].max() < 200


def test_binary_and_soft_glyph_normalization(font_path):
    image = render_text("鲃", font_path, font_size=16)
    frontend = extract_frontend(image, default_profile())
    line = find_lines(frontend.binary_mask, frontend.profile)[0]
    comps = connected_components(line)
    seg = comps[0]

    target = frontend.profile.target_size
    assert np.array_equal(frontend.binary_glyph(seg), normalize(seg.mask, target))

    roi = frontend.soft_foreground[seg.y : seg.y + seg.h, seg.x : seg.x + seg.w]
    assert np.array_equal(
        frontend.soft_glyph(seg), normalize_grayscale(roi, target)
    )

    # The soft glyph keeps intensity information the binary glyph throws away.
    soft_glyph = frontend.soft_glyph(seg)
    binary_glyph = frontend.binary_glyph(seg)
    assert np.any((soft_glyph > 0) & (soft_glyph < 255))
    assert set(np.unique(binary_glyph)) <= {0, 255}
    assert not np.array_equal(soft_glyph, binary_glyph)


def test_segmentation_consumes_binary_mask(font_path):
    image = render_text("潜甲", font_path)
    frontend = extract_frontend(image, default_profile())

    lines = find_lines(frontend.binary_mask, frontend.profile)
    assert len(lines) == 1
    comps = connected_components(lines[0])
    assert len(comps) >= 2
    # Components are built from the binary mask, not the soft map.
    assert all(np.any(c.mask) for c in comps)


def test_scorer_feeds_cnn_soft_glyphs(font_path, monkeypatch):
    model_dir = Path(__file__).parent / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        import pytest

        pytest.skip("cnn_digits fixture not generated; run scripts/train_tinycnn.py")
    model = load_model(model_dir)
    scorer = SegmentScorer(model)

    image = render_text("2", font_path)
    frontend = extract_frontend(image, default_profile())
    line = find_lines(frontend.binary_mask, frontend.profile)[0]
    comps = connected_components(line)

    original = SegmentScorer._cnn_batch
    captured: dict[str, np.ndarray | None] = {}

    def spy(self, glyphs, allowed_ids, soft_batch=None, image=None,
            segments=None, geometries=None):
        captured["binary"] = glyphs
        captured["soft"] = soft_batch
        return original(
            self, glyphs, allowed_ids, soft_batch, image, segments, geometries
        )

    monkeypatch.setattr(SegmentScorer, "_cnn_batch", spy)
    scorer.score(comps, soft=frontend.soft_foreground)

    assert captured["soft"] is not None
    assert captured["soft"].shape == (len(comps), 24, 24)
    assert np.array_equal(captured["soft"][0], frontend.soft_glyph(comps[0]))
    # The CNN input must not silently fall back to the binary batch.
    assert not np.array_equal(captured["soft"], captured["binary"])


def test_binary_trained_model_keeps_binary_cnn_input(font_path, monkeypatch):
    """Old checkpoints keep their exact 0/255 training-domain input."""

    model_dir = Path("model/game_cn")
    if not (model_dir / "config.json").exists():
        import pytest

        pytest.skip("model/game_cn not built; run tools/train/build_model.py")
    model = load_model(model_dir)
    assert model.input_mode == "binary"
    scorer = SegmentScorer(model)

    image = render_text("鲃", font_path)
    frontend = extract_frontend(image, default_profile())
    line = find_lines(frontend.binary_mask, frontend.profile)[0]
    comps = connected_components(line)

    original = SegmentScorer._cnn_batch
    captured: dict[str, np.ndarray | None] = {}

    def spy(self, glyphs, allowed_ids, soft_batch=None, image=None,
            segments=None, geometries=None):
        captured["binary"] = glyphs
        captured["soft"] = soft_batch
        return original(
            self, glyphs, allowed_ids, soft_batch, image, segments, geometries
        )

    monkeypatch.setattr(SegmentScorer, "_cnn_batch", spy)
    scorer.score(comps, soft=frontend.soft_foreground)
    assert captured["soft"] is None
    assert captured["binary"] is not None
    assert set(np.unique(captured["binary"])) <= {0, 255}


def test_recognize_feeds_soft_glyphs_to_cnn(font_path, monkeypatch):
    model_dir = Path(__file__).parent / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        import pytest

        pytest.skip("cnn_digits fixture not generated; run scripts/train_tinycnn.py")

    captured: dict[str, np.ndarray | None] = {}
    original = CPUBackend.forward_logits

    def spy(self, glyphs):
        captured["glyphs"] = np.asarray(glyphs)
        return original(self, glyphs)

    monkeypatch.setattr(CPUBackend, "forward_logits", spy)
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("42", font_path)
    result = ocr.recognize(image)

    assert result.text == "42"
    glyphs = captured["glyphs"]
    assert glyphs is not None and glyphs.ndim == 3
    # The CNN batch is soft-normalized: it contains intermediate intensities
    # that a binary 0/255 batch would not.
    assert np.any((glyphs > 0) & (glyphs < 255))


def test_goal2_regressions(font_path):
    """The Visual Frontend must not break the core acceptance strings."""

    model_dir = Path("model/game_cn")
    if not (model_dir / "config.json").exists():
        import pytest

        pytest.skip("model/game_cn not built; run tools/train/build_model.py")
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    for text in ("鲃", "鲃鱼。", "小", "潜甲", "潜乙", "Z17", "巴尔的摩"):
        result = ocr.recognize(render_text(text, font_path))
        assert result.text == text, f"{text!r} -> {result.text!r}"
        assert len(result.chars) == len(text)
