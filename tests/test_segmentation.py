"""Tests for the candidate-lattice segmentation + visual DP (P0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.postprocess import top2
from fixedfontocr.preprocess import (
    Segment,
    _connected_components,
    _connected_components_pixel,
)
from fixedfontocr.segmentation import (
    Candidate,
    build_candidates,
    connected_components,
    decode,
    geometry_score,
    segment_line,
)
from fixedfontocr.types import default_profile

from conftest import render_text


MODEL = Path("model/game_cn")


def _require_game_model() -> None:
    if not (MODEL / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")


def _mask_for_text(text: str, font_path) -> Segment:
    gray = render_text(text, font_path) @ np.array(
        [0.299, 0.587, 0.114], dtype=np.float32
    )
    mask = gray >= 140
    return Segment(mask=mask, x=0, y=0, w=mask.shape[1], h=mask.shape[0])


def test_acceptance_cases(font_path):
    """The five strings from the P0 acceptance must recognize exactly."""

    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu")
    for text in ("鲃", "鲃鱼。", "小", "潜甲", "潜乙"):
        result = ocr.recognize(render_text(text, font_path))
        assert result.text == text, f"{text!r} -> {result.text!r}"
        assert len(result.chars) == len(text)
        assert all(c.confidence > 0.9 for c in result.chars)


def test_acceptance_does_not_need_stroke_width_special_case(font_path):
    """鲃 must segment without the old stroke_width=2 hack."""

    from dataclasses import replace

    _require_game_model()
    profile = default_profile()
    profile = replace(profile, stroke_width=1)
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", profile=profile)
    result = ocr.recognize(render_text("鲃", font_path))
    assert result.text == "鲃"


def test_adjacent_chinese_characters_not_merged(font_path):
    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu")
    for text in ("潜甲", "潜乙", "鲃鱼。", "获得金币1000", "俾斯麦提尔比茨大型单装炮"):
        result = ocr.recognize(render_text(text, font_path))
        assert result.text == text, f"{text!r} -> {result.text!r}"
        assert len(result.chars) == len(text)


def test_connected_components_matches_pixel_reference():
    rng = np.random.default_rng(3)
    for _ in range(20):
        mask = rng.random((31, 47)) < 0.22
        got = sorted(
            (m.sum(), tuple(m.shape)) for m in _connected_components(mask)
        )
        ref = sorted(
            (m.sum(), tuple(m.shape)) for m in _connected_components_pixel(mask)
        )
        assert got == ref


def test_connected_components_on_rendered_text(font_path):
    line = _mask_for_text("鲃鱼。", font_path)
    got = sorted(m.sum() for m in _connected_components(line.mask))
    ref = sorted(m.sum() for m in _connected_components_pixel(line.mask))
    assert got == ref


def test_candidates_are_consecutive_and_bounded(font_path):
    line = _mask_for_text("鲃鱼。", font_path)
    comps = connected_components(line)
    cands = build_candidates(comps, default_profile(), max_merge_components=3)
    for cand in cands:
        assert cand.end - cand.start <= 3
        assert cand.components == tuple(range(cand.start, cand.end))
    # Single components always exist.
    singles = [c for c in cands if len(c.components) == 1]
    assert len(singles) == len(comps)


def test_dp_prefers_fewer_segments_on_tie():
    n = 3
    cands = [
        Candidate(
            segment=None,  # type: ignore[arg-type]
            start=0,
            end=1,
            components=(0,),
            geometry_score=0.0,
        ),
        Candidate(
            segment=None,  # type: ignore[arg-type]
            start=1,
            end=2,
            components=(1,),
            geometry_score=0.0,
        ),
        Candidate(
            segment=None,  # type: ignore[arg-type]
            start=2,
            end=3,
            components=(2,),
            geometry_score=0.0,
        ),
        Candidate(
            segment=None,  # type: ignore[arg-type]
            start=0,
            end=3,
            components=(0, 1, 2),
            geometry_score=0.0,
        ),
    ]
    from fixedfontocr.scorer import SegmentScore

    score = SegmentScore(char_id=0, visual_score=1.0, raw_score=0.0, score_type="template")
    for c in cands:
        c.score = score
    path = decode(cands, n)
    assert len(path.candidates) == 1
    assert path.candidates[0].components == (0, 1, 2)


def test_top2_matches_argsort_reference():
    rng = np.random.default_rng(11)
    for n, c in [(1, 1), (5, 1), (7, 2), (13, 3), (20, 100)]:
        logits = rng.standard_normal((n, c), dtype=np.float32)
        ids, top1, top2v, margins = top2(logits)
        order = np.argsort(-logits, axis=1, kind="stable")
        assert np.array_equal(ids, order[:, 0])
        assert np.allclose(top1, logits[np.arange(n), order[:, 0]], atol=1e-6)
        if c >= 2:
            assert np.allclose(top2v, logits[np.arange(n), order[:, 1]], atol=1e-6)
            assert np.allclose(margins, top1 - top2v, atol=1e-6)


def test_top2_ties_keep_first_occurrence():
    logits = np.array([[3.0, 1.0, 3.0, 2.0]], dtype=np.float32)
    ids, _, _, margins = top2(logits)
    assert ids[0] == 0  # np.argmax returns the first max, like stable argsort
    assert margins[0] == 0.0
