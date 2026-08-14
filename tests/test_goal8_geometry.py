"""Goal 8 acceptance: offline font geometry database.

The Goal 8 contract from ``fonts/goal``:

* the database is generated offline from ``fonts/`` with per-character
  ``char_id``, ``advance``, ``bbox width``, ``bbox height``, ``aspect
  ratio``, ``ink count``, ``component count`` and ``baseline``;
* it is embedded in model metadata so the NumPy runtime never needs
  fontTools/Pillow;
* it drives candidate pruning, the geometry score and split/merge width
  priors, so narrow ``1 I l i !`` glyphs never share a full-width CJK prior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.fontgen import (
    _cmap_codepoints,
    build_templates,
    render_glyph,
    write_model,
)
from fixedfontocr.geometry import (
    FontGeometryDatabase,
    build_geometry_database,
)
from fixedfontocr.model import load_model
from fixedfontocr.preprocess import (
    _bbox_segment,
    _connected_components,
    _union,
)
from fixedfontocr.segmentation import _geometry_bbox_ok, geometry_score
from fixedfontocr.types import Component, VisualCandidate, default_profile

from conftest import render_text


CHARSET = list("1Il i!Z2.·小巴鲃潜。甲乙")
CHARSET = [ch for ch in CHARSET if not ch.isspace()]


def test_database_contains_every_goal8_field_and_separates_scripts(font_path):
    db = build_geometry_database(font_path, CHARSET)

    assert len(db.entries) == len(CHARSET)
    assert db.charset == "".join(CHARSET)
    for ch in ("1", "I", "l", "i", "!", "小", "潜", "巴"):
        e = db.by_id[CHARSET.index(ch)]
        for value in (
            e.advance,
            e.bbox_width,
            e.bbox_height,
            e.aspect_ratio,
            e.ink_count,
            e.component_count,
            e.baseline,
            e.baseline_ratio,
            e.ink_ratio,
        ):
            assert value is not None

    # Narrow Latin/digits/punctuation have a genuinely narrower prior than
    # full-width CJK, and fragmented CJK records its real component count.
    assert db.by_id[CHARSET.index("I")].bbox_width < db.by_id[
        CHARSET.index("小")
    ].bbox_width
    assert db.by_id[CHARSET.index("1")].aspect_ratio < 0.8
    assert db.by_id[CHARSET.index("小")].component_count >= 3
    assert db.by_id[CHARSET.index("潜")].component_count >= 3
    assert db.by_id[CHARSET.index("I")].component_count == 1
    assert db.narrow_width_ratio < db.full_width_ratio


def test_geometry_json_roundtrip_and_model_embedding(font_path, tmp_path):
    db = build_geometry_database(font_path, list("01A小"))
    path = tmp_path / "geometry.json"
    db.save(path)
    loaded = FontGeometryDatabase.load(path)
    assert loaded == db

    charset = list("01A小")
    chars, templates = build_templates(font_path, charset, render_size=32)
    out = tmp_path / "model"
    write_model(out, chars, templates, font_path=font_path)
    assert (out / "geometry.json").exists()

    model = load_model(out)
    assert model.geometry is not None
    assert len(model.geometry.entries) == len(charset)
    assert model.geometry.charset == "".join(charset)
    assert model.geometry.font_sha256 == model.font_sha256


def test_missing_glyph_is_an_explicit_error(font_path):
    covered = _cmap_codepoints(str(font_path))
    missing_cp = next(cp for cp in range(0xE000, 0xF8FF) if cp not in covered)
    with pytest.raises(ValueError, match="missing glyph"):
        build_geometry_database(font_path, [chr(missing_cp)])


def test_geometry_bbox_prior_rejects_impossible_shapes(font_path):
    db = build_geometry_database(font_path, list("1小"))

    wide_blob = Component(
        mask=np.ones((30, 120), dtype=bool),
        x=0,
        y=0,
        w=120,
        h=30,
    )
    assert _geometry_bbox_ok(wide_blob, db) is False

    cjk = Component(mask=np.ones((30, 30), dtype=bool), x=0, y=0, w=30, h=30)
    assert _geometry_bbox_ok(cjk, db) is True
    one = Component(mask=np.ones((24, 14), dtype=bool), x=0, y=0, w=14, h=24)
    assert _geometry_bbox_ok(one, db) is True


def test_geometry_score_rewards_real_glyph_shape(font_path):
    db = build_geometry_database(font_path, CHARSET)
    profile = default_profile()

    small_mask = render_glyph(font_path, "小", 32)
    parts = [_bbox_segment(c, 0) for c in _connected_components(small_mask)]
    merged = parts[0]
    for part in parts[1:]:
        merged = _union(merged, part)

    small_id = CHARSET.index("小")
    one_id = CHARSET.index("1")

    def scored(seg, comps, n_components, char_id) -> float:
        cand = VisualCandidate(
            start=0,
            end=n_components,
            components=tuple(range(n_components)),
            atoms=(0, n_components),
            segment=seg,
        )
        return geometry_score(
            cand,
            comps,
            profile,
            geometry=db,
            char_id=char_id,
        )

    correct = scored(merged, [merged] * 3, 3, small_id)
    wrong_script = scored(merged, [merged] * 3, 3, one_id)
    fragment = scored(parts[0], [parts[0]], 1, small_id)
    assert wrong_script < correct
    assert fragment < correct

    one = Component(
        mask=np.ones((24, 14), dtype=bool),
        x=0,
        y=0,
        w=14,
        h=24,
    )
    correct_narrow = scored(one, [one], 1, one_id)
    wrong_narrow = scored(one, [one], 1, small_id)
    assert wrong_narrow < correct_narrow


def test_goal8_regressions_still_pass(font_path):
    model_dir = Path("model/game_cn")
    if not (model_dir / "config.json").exists():
        import pytest

        pytest.skip("model/game_cn not built; run tools/train/build_model.py")
    model = load_model(model_dir)
    if model.geometry is None:
        import pytest

        pytest.skip("model/game_cn has no geometry.json; rebuild it")

    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    for text in ("鲃", "鲃鱼。", "小", "潜甲", "潜乙", "Z17", "巴尔的摩"):
        result = ocr.recognize(render_text(text, font_path))
        assert result.text == text, f"{text!r} -> {result.text!r}"
        assert len(result.chars) == len(text)
