"""Tests for the unified scoring API and P6 margin-aware hybrid gate."""

from __future__ import annotations

from pathlib import Path

from fixedfontocr.preprocess import Segment
from fixedfontocr.scorer import SegmentScorer, cnn_visual, to_confidence
from fixedfontocr.segmentation import build_candidates, connected_components
from fixedfontocr.types import default_profile

from conftest import render_text


def _game_model():
    from fixedfontocr.model import load_model

    model = Path("model/game_cn")
    if not (model / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")
    return load_model(model)


def _line(text, font_path) -> Segment:
    import numpy as np

    gray = render_text(text, font_path) @ np.array(
        [0.299, 0.587, 0.114], dtype=np.float32
    )
    mask = gray >= 140
    return Segment(mask=mask, x=0, y=0, w=mask.shape[1], h=mask.shape[0])


def test_hybrid_margin_gate_falls_back_to_cnn(font_path):
    """High template confidence with a tiny margin must not be trusted."""

    scorer = SegmentScorer(_game_model())
    line = _line("鲃", font_path)
    comps = connected_components(line)
    cands = build_candidates(comps, default_profile(), max_merge_components=4)
    scores = scorer.score([c.segment for c in cands])
    by_span = {(c.start, c.end): s for c, s in zip(cands, scores)}

    # The full 鲃 is a confident template match with a real margin.
    full = by_span[(0, len(comps))]
    assert full.score_type == "template"
    assert full.visual_score > 0.99

    # The right fragment looks like 'E' (conf ~0.90) but its top-1/top-2
    # margin is tiny, so the scorer must use the CNN instead.
    frag = by_span[(len(comps) - 1, len(comps))]
    assert frag.score_type == "cnn"


def test_candidate_score_exposes_shared_scale(font_path):
    scorer = SegmentScorer(_game_model())
    line = _line("小", font_path)
    comps = connected_components(line)
    cands = build_candidates(comps, default_profile(), max_merge_components=4)
    scores = scorer.score([c.segment for c in cands])
    for score in scores:
        cs = score.candidate_score
        assert cs.visual_score == score.visual_score
        assert cs.raw_score == score.raw_score
        assert cs.score_type == score.score_type
        assert 0.0 <= cs.visual_score <= 1.0


def test_cnn_visual_and_confidence_are_monotonic():
    assert cnn_visual(-1.0) < cnn_visual(0.0) < cnn_visual(1.0) < cnn_visual(5.0)
    assert cnn_visual(0.0) == 0.5
    assert 0.0 <= to_confidence(2.0, "cnn") <= 1.0
    assert to_confidence(0.9, "template") == 0.9
