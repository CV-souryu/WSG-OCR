"""Goal 13 acceptance: the joint decoder (DP + beam search).

The Goal 13 contract from ``fonts/goal``:

* the decoder is the architecture's core and consumes the Visual Lattice +
  Visual Scores + Font Geometry + Lexicon;
* the first version uses DP, then upgrades to beam search (``beam_width``
  8..32 suggested);
* every path is scored as
  ``total = visual + geometry + lexicon + word_prior
  - segmentation_penalty``;
* the decoder outputs the best path and ``alternatives``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import (
    Component,
    DecodePath,
    DecoderConfig,
    FixedFontOCR,
    Lexicon,
    VisualCandidate,
    VisualLattice,
    VisualScores,
    decode_beam,
    decode_dp,
    decode_lattice,
    score_path,
)
from fixedfontocr.geometry import FontGeometryDatabase

from conftest import render_text


MODEL = Path("model/game_cn")


def _require_game_model() -> None:
    if not (MODEL / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")


def _candidate(
    span: tuple[int, int],
    char_ids: tuple[int, ...],
    logits: tuple[float, ...],
    geometry: float = 0.0,
    segment: Component | None = None,
) -> VisualCandidate:
    components = tuple(range(span[0], span[1]))
    return VisualCandidate(
        start=span[0],
        end=span[1],
        components=components,
        atoms=span,
        segment=segment,
        geometry_score=geometry,
        scores=VisualScores(char_ids=char_ids, logits=logits),
    )


def _lattice(*candidates: VisualCandidate) -> VisualLattice:
    return VisualLattice(candidates=tuple(candidates))


def test_decoder_config_beam_width_matches_goal_suggestion():
    cfg = DecoderConfig()
    assert 8 <= cfg.beam_width <= 32
    assert cfg.beam_width == 16
    with pytest.raises(ValueError, match="beam_width"):
        DecoderConfig(beam_width=0)
    with pytest.raises(ValueError, match="num_alternatives"):
        DecoderConfig(num_alternatives=0)
    with pytest.raises(ValueError, match="weights/penalties"):
        DecoderConfig(visual_weight=-1.0)


def test_score_path_implements_goal13_formula():
    charset = ["甲", "乙"]
    lexicon = Lexicon.from_terms("t", ["甲甲", "甲乙"])
    c0 = _candidate((0, 1), (0, 1), (0.8, 0.7), geometry=-0.02)
    c1 = _candidate((1, 2), (0, 1), (0.6, 0.9), geometry=-0.04)
    path = DecodePath(
        candidates=(c0, c1),
        text="甲甲",
        char_ids=(0, 0),
    )
    cfg = DecoderConfig(
        visual_weight=1.0,
        geometry_weight=2.0,
        lexicon_weight=0.5,
        word_prior_weight=0.3,
        segmentation_penalty=0.1,
    )
    ps = score_path(path, charset, lexicon, cfg)

    assert ps.visual == pytest.approx(0.7)
    assert ps.geometry == pytest.approx(-0.03)
    assert ps.lexicon == pytest.approx(1.0)  # exact term "甲甲"
    # 甲 appears in 3 of 4 lexicon characters -> raw prior 1.0, gated by
    # each position's visual uncertainty: (1.0*0.2 + 1.0*0.4)/2 = 0.3
    assert ps.word_prior == pytest.approx(0.3)
    assert ps.segmentation_penalty == pytest.approx(1.0)  # n_candidates - 1
    uncertainty = 1.0 - 0.7
    expected = (
        1.0 * 0.7
        + 2.0 * (-0.03)
        + 0.5 * 1.0 * uncertainty
        + 0.3 * 0.3
        - 0.1 * 1.0
    )
    assert ps.total == pytest.approx(expected)


def test_lexicon_and_word_prior_break_visual_ties():
    """With equal visual evidence the dictionary decides; clear evidence wins."""

    charset = ["甲", "乙"]
    ambiguous = (
        _candidate((0, 1), (0, 1), (0.55, 0.54)),
        _candidate((1, 2), (0, 1), (0.55, 0.54)),
    )
    lat = _lattice(*ambiguous)

    no_lex = decode_dp(lat, charset)
    assert no_lex.text in ("甲甲", "乙乙")

    lex_a = Lexicon.from_terms("t", ["甲甲"])
    path_a = decode_dp(lat, charset, lex_a)
    beam_a = decode_beam(lat, charset, lex_a)
    assert path_a.text == "甲甲"
    assert beam_a.text == "甲甲"

    lex_b = Lexicon.from_terms("t", ["乙乙"])
    assert decode_dp(lat, charset, lex_b).text == "乙乙"

    # Goal 15: a visually confident 乙乙 must not be rewritten to 甲甲.
    confident = (
        _candidate((0, 1), (0, 1), (0.40, 0.95)),
        _candidate((1, 2), (0, 1), (0.40, 0.95)),
    )
    strong = decode_dp(_lattice(*confident), charset, lex_a)
    assert strong.text == "乙乙"


def test_segmentation_penalty_prevents_fragment_explosion():
    """小-style glyphs: slightly-better fragments must not win the path."""

    charset = ["A"]
    whole = _candidate((0, 3), (0,), (0.90,))
    fragments = (
        _candidate((0, 1), (0,), (0.905,)),
        _candidate((1, 2), (0,), (0.905,)),
        _candidate((2, 3), (0,), (0.905,)),
    )
    lat = _lattice(whole, *fragments)

    no_penalty = DecoderConfig(segmentation_penalty=0.0)
    assert decode_dp(lat, charset, config=no_penalty).text == "AAA"
    assert decode_beam(lat, charset, config=no_penalty).text == "AAA"

    with_penalty = DecoderConfig(segmentation_penalty=0.005)
    assert decode_dp(lat, charset, config=with_penalty).text == "A"
    assert decode_beam(lat, charset, config=with_penalty).text == "A"


def test_beam_search_returns_best_path_and_alternatives():
    charset = ["甲", "乙"]
    cands = (
        _candidate((0, 1), (0, 1), (0.9, 0.6)),
        _candidate((1, 2), (0, 1), (0.85, 0.7)),
    )
    lat = _lattice(*cands)
    path = decode_beam(lat, charset)

    assert path.text == "甲甲"
    assert path.char_ids == (0, 0)
    assert path.alternatives
    assert path.alternatives[0] == "甲乙"
    assert path.alternatives[0] != path.text
    assert path.mean_score == pytest.approx(
        score_path(path, charset).total
    )

    dp = decode_dp(lat, charset)
    assert dp.text == path.text
    assert decode_lattice(lat, charset, beam=False).text == dp.text
    assert decode_lattice(lat, charset).text == path.text


def test_decoder_uses_font_geometry_input(font_path):
    _require_game_model()
    geometry_path = MODEL / "geometry.json"
    if not geometry_path.exists():
        pytest.skip("model/game_cn/geometry.json missing")
    db = FontGeometryDatabase.load(geometry_path)

    seg = Component(
        mask=np.ones((12, 12), dtype=bool),
        x=0,
        y=0,
        w=12,
        h=12,
    )
    cand = _candidate((0, 1), (0,), (1.0,), segment=seg)
    path = decode_beam(_lattice(cand), db.charset)
    assert path.text == db.charset[0]

    without_db = score_path(path, db.charset)
    with_db = score_path(path, db.charset, geometry=db)
    # The database-derived geometry term is a (<= 0) agreement penalty.
    assert with_db.geometry <= without_db.geometry
    assert with_db.total <= without_db.total + 1e-9


def test_goal13_end_to_end_regressions(font_path):
    """Goal 4/7/11/12 acceptance strings survive the joint decoder."""

    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)

    for text in ("鲃", "鲃鱼。", "小", "潜甲", "潜乙", "巴尔的摩", "Z17"):
        result = ocr.recognize(
            render_text(text, font_path),
            lexicon="ships",
            lexicon_mode="prefer",
        )
        assert result.text == text, f"{text!r} -> {result.text!r}"
        assert len(result.chars) == len(text)

    partial = ocr.recognize(
        render_text("尔的摩", font_path),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert partial.text == "尔的摩"
    assert partial.matched_term == "巴尔的摩"
    assert partial.matched_span == (1, 4)
    assert partial.lexicon_match is not None
    assert partial.lexicon_match.kind == "suffix_crop"

    # Goal 15: unknown text is still emitted when visual evidence is clear.
    unknown = ocr.recognize(
        render_text("ABCXYZ999", font_path),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert unknown.text == "ABCXYZ999"
    assert unknown.confidence > 0.0

    # The decoder's alternatives flow into the public result.
    with_alt = ocr.recognize(
        render_text("巴尔的摩", font_path),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert isinstance(with_alt.alternatives, tuple)
