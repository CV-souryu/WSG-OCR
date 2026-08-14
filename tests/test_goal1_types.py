"""Goal 1 acceptance: core lattice/decoder data structures.

These tests pin the contract from ``fonts/goal`` Goal 1:

* ``Component`` is the atomic connected component;
* ``VisualCandidate`` describes one merge/split hypothesis with a bbox,
  glyph slot and geometry score;
* ``VisualLattice`` keeps all components and candidates without making a
  final segmentation decision;
* ``VisualScores`` keeps Top-K (never only Top-1);
* ``DecodePath``/``OCRResult`` carry the chosen path and explicit
  ``matched_term``/``matched_span``/``alternatives`` fields;
* ``CharResult`` remains only as a public compatibility projection.
"""

from __future__ import annotations

import numpy as np

from fixedfontocr import (
    CharResult,
    Component,
    DecodePath,
    FixedFontOCR,
    LexiconMatch,
    OCRResult,
    VisualCandidate,
    VisualLattice,
    VisualScores,
)
from fixedfontocr.model import load_model
from fixedfontocr.postprocess import second_ids, top2
from fixedfontocr.preprocess import Segment
from fixedfontocr.scorer import SegmentScorer
from fixedfontocr.segmentation import (
    build_candidates,
    build_lattice,
    connected_components,
    decode,
)
from fixedfontocr.types import default_profile

from conftest import render_text


def test_core_types_construct_and_round_trip():
    comp = Component(mask=np.zeros((3, 4), dtype=bool), x=1, y=2, w=4, h=3)
    assert comp.bbox == (1, 2, 4, 3)

    scores = VisualScores(char_ids=(3, 7), logits=(2.0, 1.5))
    assert scores.top_k == (3, 7)  # Top-K, not just Top-1
    assert len(scores.char_ids) == 2

    cand = VisualCandidate(
        start=0,
        end=1,
        segment=comp,
        components=(0,),
        geometry_score=0.2,
        scores=scores,
    )
    assert cand.bbox == (1, 2, 4, 3)
    assert cand.geometry == 0.2
    assert cand.total_score == 0.2

    lattice = VisualLattice(
        components=(comp,),
        candidates=(cand,),
        width=10,
        height=5,
    )
    assert lattice.by_end()[1] == [cand]

    path = DecodePath(
        candidates=(cand,),
        mean_score=0.9,
        text="A",
        confidence=0.8,
        alternatives=("B",),
        lattice=lattice,
    )
    result = OCRResult(
        text="A",
        confidence=0.8,
        chars=(CharResult("A", 1, 2, 4, 3, 0.8),),
        matched_term="AB",
        matched_span=(0, 1),
        alternatives=("B",),
        lexicon_match=LexiconMatch(term="AB", span=(0, 1), confidence=0.7),
        path=path,
    )
    assert result.matched_term == "AB"
    assert result.matched_span == (0, 1)
    assert result.alternatives == ("B",)
    assert result.path is path
    assert result.lexicon_match.term == "AB"


def test_second_ids_returns_true_top2():
    logits = np.array(
        [[3.0, 1.0, 2.0], [1.0, 1.0, 1.0], [5.0, 5.0, 1.0]],
        dtype=np.float32,
    )
    ids, _, _, _ = top2(logits)
    got = second_ids(logits, ids)
    assert got.tolist() == [2, 1, 1]


def test_second_ids_is_minus_one_without_real_second():
    logits = np.array(
        [[3.0, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf]],
        dtype=np.float32,
    )
    ids = np.argmax(logits, axis=-1).astype(np.int32)
    assert second_ids(logits, ids).tolist() == [-1, -1]


def test_segmentation_builds_lattice_of_core_types(font_path):
    image = render_text("小", font_path)
    mask = (
        image @ np.array([0.299, 0.587, 0.114], dtype=np.float32) >= 140
    )
    line = Segment(mask=mask, x=0, y=0, w=mask.shape[1], h=mask.shape[0])
    comps = connected_components(line)
    assert comps and all(isinstance(c, Component) for c in comps)

    cands = build_candidates(comps, default_profile())
    assert cands and all(isinstance(c, VisualCandidate) for c in cands)

    lattice = build_lattice(comps, cands, line)
    assert isinstance(lattice, VisualLattice)
    assert lattice.components == tuple(comps)
    assert lattice.candidates == tuple(cands)
    assert decode(lattice).candidates  # lattice is directly decodable


def test_scorer_exposes_topk_visual_scores(font_path, model_dir):
    model = load_model(model_dir)
    scorer = SegmentScorer(model)
    image = render_text("0", font_path)
    mask = (
        image @ np.array([0.299, 0.587, 0.114], dtype=np.float32) >= 140
    )
    seg = Segment(mask=mask, x=0, y=0, w=mask.shape[1], h=mask.shape[0])
    scores = scorer.score([seg])
    assert scores[0].visual_scores is not None
    assert len(scores[0].visual_scores.char_ids) == 2
    assert len(scores[0].visual_scores.logits) == 2


def test_recognize_result_has_goal1_fields(font_path, model_dir):
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    result = ocr.recognize(render_text("123", font_path))

    # visible text is never replaced by a lexicon term
    assert result.text == "123"
    assert result.matched_term is None
    assert result.matched_span is None
    assert isinstance(result.alternatives, tuple)
    assert result.path is not None
    assert result.path.candidates
    assert all(isinstance(c, VisualCandidate) for c in result.path.candidates)
    assert result.path.lattice is not None

    # ``chars`` is the public compatibility projection, not the internal core
    assert len(result.chars) == 3
    assert all(isinstance(c, CharResult) for c in result.chars)
