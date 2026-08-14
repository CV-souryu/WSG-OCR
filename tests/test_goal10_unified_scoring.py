"""Goal 10 acceptance: unified visual scoring.

The Goal 10 contract from ``fonts/goal``:

* template ``[0,1]`` and CNN logit margins are never mixed directly -- the
  candidate keeps ``template_raw_score``, ``cnn_logit``, ``cnn_margin`` and
  ``geometry_score`` as separate evidence;
* the decoder consumes one ``visual_score`` computed with tunable weights
  (``a * cnn_score + b * template_score + c * geometry_score``);
* weights are stored in the model and can be adjusted on a real test set;
* the public confidence is calibrated into ``[0, 1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.model import load_model
from fixedfontocr.preprocess import Segment
from fixedfontocr.scorer import (
    SegmentScore,
    SegmentScorer,
    VisualCalibration,
    VisualWeights,
    unified_visual_score,
)
from fixedfontocr.segmentation import (
    build_candidates,
    connected_components,
    segment_line,
)
from fixedfontocr.types import VisualScores, default_profile

from conftest import render_text


MODEL = Path("model/game_cn")


def _game_scorer():
    if not (MODEL / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")
    return SegmentScorer(load_model(MODEL))


def _line(text, font_path):
    gray = render_text(text, font_path) @ np.array(
        [0.299, 0.587, 0.114], dtype=np.float32
    )
    return Segment(mask=gray >= 140, x=0, y=0, w=gray.shape[1], h=gray.shape[0])


def test_unified_visual_score_uses_weighted_formula():
    weights = VisualWeights(cnn=0.5, template=0.3, geometry=0.2)
    score = unified_visual_score(
        template_score=0.8,
        cnn_score=0.6,
        geometry_score=-0.1,
        weights=weights,
    )
    assert score == pytest.approx(0.5 * 0.6 + 0.3 * 0.8 + 0.2 * (-0.1))
    assert 0.0 <= score <= 1.0


def test_calibration_is_monotone_piecewise_linear():
    cal = VisualCalibration(
        ((0.0, 0.0), (0.5, 0.4), (0.75, 0.9), (1.0, 1.0))
    )
    xs = np.linspace(0.0, 1.0, 11)
    ys = [cal(x) for x in xs]
    assert ys == sorted(ys)
    assert 0.0 <= min(ys) <= max(ys) <= 1.0
    assert cal(0.5) == pytest.approx(0.4)
    assert cal(0.625) == pytest.approx(0.65)
    assert VisualCalibration()(0.37) == pytest.approx(0.37)


def test_visual_scores_store_template_cnn_and_geometry_evidence(font_path):
    scorer = _game_scorer()
    line = _line("鲃", font_path)
    comps = connected_components(line)
    cands = build_candidates(comps, default_profile(), max_merge_components=4)
    scores = scorer.score([c.segment for c in cands])
    by_span = {(c.atom_span[0], c.atom_span[1]): s for c, s in zip(cands, scores)}

    full = by_span[(0, len(comps))]
    assert full.score_type == "template"
    vs = full.visual_scores
    assert vs is not None
    assert vs.template_raw_score == pytest.approx(1.0)
    assert np.isfinite(vs.cnn_logit)
    assert vs.cnn_margin > 0.0
    assert 0.0 <= vs.cnn_score <= 1.0
    assert 0.0 <= vs.visual_score <= 1.0
    assert 0.0 <= vs.confidence <= 1.0

    frag = by_span[(len(comps) - 1, len(comps))]
    assert frag.score_type == "cnn"
    assert frag.visual_scores is not None
    # The raw template score is retained even when the CNN is the decision
    # source, so the decoder never sees a collapsed single scale.
    assert 0.0 <= frag.visual_scores.template_raw_score <= 1.0
    assert np.isfinite(frag.visual_scores.cnn_logit)
    assert np.isfinite(frag.visual_scores.cnn_margin)


def test_score_fused_applies_weights_without_trust_gate(font_path):
    """The pure Goal 10 fusion honors supplied weights exactly."""
    scorer = _game_scorer()
    line = _line("鲃", font_path)
    comps = connected_components(line)
    cands = build_candidates(comps, default_profile(), max_merge_components=4)
    scores = scorer.score_fused(
        [c.segment for c in cands],
        visual_weights=VisualWeights(cnn=1.0, template=0.0, geometry=0.0),
    )
    for score in scores:
        assert score.visual_score == pytest.approx(score.cnn_score)
        assert score.template_raw_score >= 0.0


def test_segment_line_finalizes_geometry_into_visual_score(font_path):
    scorer = _game_scorer()
    line = _line("小", font_path)
    path = segment_line(line, default_profile(), scorer)
    assert path.candidates
    for cand in path.candidates:
        assert cand.score is not None
        assert cand.score.geometry_included
        assert cand.scores is not None
        assert cand.scores.geometry_score == pytest.approx(cand.geometry)
        assert cand.scores.visual_score == pytest.approx(cand.score.visual_score)
        assert cand.scores.confidence == pytest.approx(cand.score.confidence)
        assert 0.0 <= cand.total_score <= 1.0


def test_finalize_score_adds_geometry_and_calibrates(font_path):
    scorer = _game_scorer()
    vs = VisualScores(
        char_ids=(0, -1),
        logits=(0.8, 0.0),
        template_raw_score=0.9,
        cnn_logit=1.2,
        cnn_margin=0.5,
        cnn_score=0.62,
        visual_score=0.8,
        confidence=0.8,
    )
    score = scorer.finalize_score(
        SegmentScore(
            char_id=0,
            visual_score=0.8,
            raw_score=0.0,
            score_type="template",
            visual_scores=vs,
            template_raw_score=0.9,
            confidence=0.8,
        ),
        geometry_score=-0.1,
    )
    assert score.visual_score == pytest.approx(0.8 + scorer.visual_weights.geometry * -0.1)
    assert score.geometry_score == pytest.approx(-0.1)
    assert score.visual_scores.geometry_score == pytest.approx(-0.1)
    assert 0.0 <= score.confidence <= 1.0


def test_public_confidence_stays_in_unit_interval(font_path):
    if not (MODEL / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu")
    result = ocr.recognize(render_text("123", font_path))
    assert result.text == "123"
    assert result.chars
    assert all(0.0 <= c.confidence <= 1.0 for c in result.chars)
    assert 0.0 <= result.confidence <= 1.0


def test_hybrid_model_config_carries_visual_weights(font_path, tmp_path):
    from fixedfontocr.fontgen import build_templates
    from fixedfontocr.model import write_hybrid_model

    charset = list("0123")
    chars, templates = build_templates(font_path, charset, render_size=32)
    rng = np.random.default_rng(0)
    weights = {
        "conv1.weight": rng.standard_normal((8, 1, 3, 3), dtype=np.float32),
        "conv1.bias": rng.standard_normal(8, dtype=np.float32),
        "dw1.weight": rng.standard_normal((8, 3, 3), dtype=np.float32),
        "dw1.bias": rng.standard_normal(8, dtype=np.float32),
        "pw1.weight": rng.standard_normal((16, 8), dtype=np.float32),
        "pw1.bias": rng.standard_normal(16, dtype=np.float32),
        "dw2.weight": rng.standard_normal((16, 3, 3), dtype=np.float32),
        "dw2.bias": rng.standard_normal(16, dtype=np.float32),
        "pw2.weight": rng.standard_normal((32, 16), dtype=np.float32),
        "pw2.bias": rng.standard_normal(32, dtype=np.float32),
        "fc.weight": rng.standard_normal((len(chars), 32), dtype=np.float32),
        "fc.bias": rng.standard_normal(len(chars), dtype=np.float32),
    }
    out = tmp_path / "goal10_hybrid"
    write_hybrid_model(
        out,
        chars,
        templates,
        weights,
        visual_weights={"cnn": 0.4, "template": 0.5, "geometry": 0.1},
        visual_calibration=[[0.0, 0.0], [0.8, 0.7], [1.0, 1.0]],
        font_path=font_path,
    )
    model = load_model(out)
    assert model.config["visual_weights"] == {
        "cnn": 0.4,
        "template": 0.5,
        "geometry": 0.1,
    }
    assert model.config["visual_calibration"] == [[0.0, 0.0], [0.8, 0.7], [1.0, 1.0]]
