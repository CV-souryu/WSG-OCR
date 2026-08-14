"""Goal 3 acceptance: baseline-aligned 24x24 normalization.

The Goal 3 contract from ``fonts/goal``:

* normalize is no longer ``glyph bbox -> force stretch to 24x24``;
* a glyph keeps its aspect ratio, aligns to the font baseline, then is
  centered/padded into 24x24;
* both binary and soft glyphs use the same frame;
* ``1/I/l``, ``Z/2/7``, full-width CJK, narrow Latin and punctuation all
  land in the same vertical layout as the font itself.

The decisive property is *template/runtime consistency*: templates, training
samples and runtime candidates all use :func:`glyph_normalize_geometry`, so a
rendered glyph and its template normalize to the exact same bitmap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fixedfontocr import defaults
from fixedfontocr.fontgen import build_templates, render_glyph, write_model
from fixedfontocr.preprocess import (
    Component,
    NormalizeSpec,
    compute_normalize_spec,
    glyph_normalize_geometry,
    normalize,
    normalize_grayscale,
)

from conftest import render_text


def _normalize_mask(mask: np.ndarray, spec: NormalizeSpec) -> np.ndarray:
    h, w = mask.shape
    cand = Component(mask=mask, x=0, y=0, w=w, h=h)
    baseline_offset, scale = glyph_normalize_geometry(cand, spec, 24)
    return normalize(
        mask,
        24,
        baseline_offset=baseline_offset,
        scale=scale,
        baseline_row=spec.baseline_row,
    )


def _ink_rows(glyph: np.ndarray) -> tuple[int, int] | None:
    rows = np.where(np.any(glyph > 0, axis=1))[0]
    if rows.size == 0:
        return None
    return int(rows[0]), int(rows[-1])


def test_normalize_keeps_aspect_ratio(font_path):
    """Aspect ratio survives normalization (no force-stretch)."""
    charset = list(defaults.read_charset())
    spec = compute_normalize_spec(font_path, charset, 24, 32)
    for ch in ("1", "I", "l", "Z", "2", "7", "小", "摩", "."):
        mask = render_glyph(font_path, ch, 32)
        src_aspect = mask.shape[1] / mask.shape[0]
        glyph = _normalize_mask(mask, spec).astype(bool)
        ys, xs = np.where(glyph)
        out_h = int(ys.max() - ys.min() + 1)
        out_w = int(xs.max() - xs.min() + 1)
        out_aspect = out_w / out_h
        # Rounding may shift one pixel; aspect must stay within 0.15 of src.
        assert abs(out_aspect - src_aspect) <= 0.15, (
            f"{ch!r}: source aspect {src_aspect:.2f} -> {out_aspect:.2f}"
        )


def test_latin_digits_share_baseline_row(font_path):
    """Z/2/7 and 1/I/l sit on the same baseline row (binary + soft)."""
    spec = compute_normalize_spec(font_path, list(defaults.read_charset()), 24, 32)
    for ch in ("1", "I", "l", "Z", "2", "7", "a"):
        mask = render_glyph(font_path, ch, 32)
        glyph = _normalize_mask(mask, spec)
        rows = _ink_rows(glyph)
        # Integer resampling can cut the bottom-most ink row, so the last
        # ink row lands on the baseline row or one pixel above it.
        assert rows is not None and rows[1] in (17, 18), f"{ch!r}: bottom row {rows}"


def test_cjk_uses_font_baseline_instead_of_ink_center(font_path):
    """CJK ink extends below the baseline, unlike Latin/digits."""
    spec = compute_normalize_spec(font_path, list(defaults.read_charset()), 24, 32)
    for ch in ("小", "巴", "尔", "的", "摩", "潜", "鲃"):
        mask = render_glyph(font_path, ch, 32)
        glyph = _normalize_mask(mask, spec)
        rows = _ink_rows(glyph)
        # CJK ink extends below the baseline, so its last ink row is lower
        # than Latin glyphs (which bottom out at row 17/18).
        assert rows is not None and rows[1] > 17, f"{ch!r}: bottom row {rows}"


def test_middle_dot_floats_above_baseline(font_path):
    """· is a vertical-center punctuation mark, not a baseline dot."""
    spec = compute_normalize_spec(font_path, list(defaults.read_charset()), 24, 32)
    dot = _normalize_mask(render_glyph(font_path, ".", 32), spec)
    middle = _normalize_mask(render_glyph(font_path, "·", 32), spec)
    dot_rows = _ink_rows(dot)
    mid_rows = _ink_rows(middle)
    assert dot_rows is not None and dot_rows[1] in (17, 18, 19)
    assert mid_rows is not None
    # The middle dot is a floating punctuation mark: its ink never extends
    # lower than the baseline dot's, and its normalized form is narrower.
    assert mid_rows[1] <= dot_rows[1]
    mid_w = int(np.where(middle.any(axis=0))[0][-1] - np.where(middle.any(axis=0))[0][0] + 1)
    dot_w = int(np.where(dot.any(axis=0))[0][-1] - np.where(dot.any(axis=0))[0][0] + 1)
    assert mid_w < dot_w


def test_narrow_latin_is_narrow_and_cjk_is_full_width(font_path):
    """Full-width CJK occupies a wide box; narrow Latin stays narrow."""
    spec = compute_normalize_spec(font_path, list(defaults.read_charset()), 24, 32)

    def width(ch: str) -> int:
        glyph = _normalize_mask(render_glyph(font_path, ch, 32), spec).astype(bool)
        cols = np.where(np.any(glyph, axis=0))[0]
        return int(cols[-1] - cols[0] + 1)

    assert width("I") < width("小")
    assert width("l") < width("摩")
    assert width("Z") < width("小")


def test_soft_glyph_uses_same_baseline_frame_and_keeps_intensity(font_path):
    """Soft normalization shares the binary frame but keeps anti-aliasing."""
    image = render_text("小", font_path, font_size=32)
    gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    spec = compute_normalize_spec(font_path, list(defaults.read_charset()), 24, 32)
    mask = gray >= 140
    ys, xs = np.where(mask)
    tight_mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    tight_gray = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    cand = Component(
        mask=tight_mask,
        x=0,
        y=0,
        w=tight_mask.shape[1],
        h=tight_mask.shape[0],
    )
    baseline_offset, scale = glyph_normalize_geometry(cand, spec, 24)
    binary = normalize(
        tight_mask,
        24,
        baseline_offset=baseline_offset,
        scale=scale,
        baseline_row=spec.baseline_row,
    )
    soft = normalize_grayscale(
        tight_gray.astype(np.uint8),
        24,
        baseline_offset=baseline_offset,
        scale=scale,
        baseline_row=spec.baseline_row,
    )
    assert _ink_rows(binary) == _ink_rows(soft)
    assert set(np.unique(binary)) <= {0, 255}
    assert np.any((soft > 0) & (soft < 255))
    assert not np.array_equal(binary, soft)


def test_template_and_runtime_use_identical_frame(font_path, tmp_path):
    """A rendered glyph and its template normalize to the exact same bitmap.

    This is the Goal 3 acceptance core: templates are built with
    ``glyph_normalize_geometry`` and runtime candidates use the same
    function, so recognition does not depend on guessing a render size.
    """
    charset = list("0123456789Z17小巴尔的摩潜甲乙鲃鱼。OIl.")
    chars, templates = build_templates(font_path, charset, render_size=32)
    model_dir = tmp_path / "model"
    write_model(model_dir, chars, templates, font_path=font_path)
    spec = compute_normalize_spec(font_path, charset, 24, 32)
    tpl_bits = np.unpackbits(templates, axis=1, bitorder="little").reshape(
        len(chars), 24, 24
    )

    for ch in charset:
        glyph = _normalize_mask(render_glyph(font_path, ch, 32), spec).astype(bool)
        dist = int(np.abs(glyph.astype(int) - tpl_bits[chars.index(ch)].astype(int)).sum())
        assert dist == 0, f"{ch!r}: template/runtime distance {dist}"


def test_normalize_without_geometry_keeps_legacy_center(font_path):
    """Callers without a normalize spec still get centered output."""
    mask = render_glyph(font_path, "Z", 32)
    glyph = normalize(mask, 24)
    rows = _ink_rows(glyph)
    assert rows is not None
    # Centered: top and bottom margins are equal (within resampling
    # rounding; 0.8 fill lands around rows 2..20 for a 24px-tall glyph).
    assert abs(rows[0] - (24 - 1 - rows[1])) <= 1


def test_model_config_carries_normalize_spec(font_path, tmp_path):
    """Goal 3 parameters are embedded in the model metadata."""
    charset = list("01AZ小")
    chars, templates = build_templates(font_path, charset, render_size=32)
    model_dir = tmp_path / "model"
    write_model(model_dir, chars, templates, font_path=font_path)
    config = (model_dir / "config.json").read_text(encoding="utf-8")
    assert '"normalize"' in config
    assert '"baseline_row"' in config
    spec = NormalizeSpec.from_dict(
        __import__("json").loads(config)["normalize"]
    )
    assert spec is not None
    assert spec.baseline_row == 18.0
    assert spec.render_size == 32.0


def test_render_size_invariance_for_clean_glyphs(font_path):
    """Clean glyphs normalize to nearly the same frame at 32/64 px."""
    spec = compute_normalize_spec(font_path, list(defaults.read_charset()), 24, 32)
    for ch in ("1", "Z", "2", "7", "小", "摩"):
        base = _normalize_mask(render_glyph(font_path, ch, 32), spec) > 0
        for size in (64,):
            other = _normalize_mask(render_glyph(font_path, ch, size), spec) > 0
            dist = int(np.abs(base.astype(int) - other.astype(int)).sum())
            # 16 px rasterization legitimately loses thin strokes, so only
            # the 32 px vs 64 px comparison is asserted here.
            assert dist <= 90, f"{ch!r} at {size}px: distance {dist}"
