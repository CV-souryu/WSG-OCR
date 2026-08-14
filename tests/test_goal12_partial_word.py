"""Goal 12 acceptance: partial-word support.

The Goal 12 contract from ``fonts/goal``:

* a full dictionary term may appear on screen as only a crop, e.g. the
  six-character term ``C1C2C3C4C5C6`` with visible ``C2C3C4C5``;
* prefix crops and suffix crops are allowed, while internally missing
  characters carry a higher penalty;
* the final result keeps the visible text and only annotates the inferred
  entity::

      text         = "C2C3C4C5"
      matched_term = "C1C2C3C4C5C6"
      matched_span = (1, 5)

* the visible text is never fabricated into the full term.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import (
    CharResult,
    DecodePath,
    FixedFontOCR,
    Lexicon,
    OCRResult,
    VisualCandidate,
    VisualScores,
    apply_lexicon,
)
from fixedfontocr.lexicon import match_term

from conftest import render_text


MODEL = Path("model/game_cn")


def _require_game_model() -> None:
    if not (MODEL / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")


def test_goal12_six_char_term_inner_crop_span_convention():
    """The exact Goal 12 example: term C1..C6, screen C2..C5."""

    lex = Lexicon.from_terms("t", ["ABCDEF"])
    match = lex.best_match("BCDE")
    assert match is not None
    assert match.term == "ABCDEF"
    assert match.span == (1, 5)
    assert match.kind == "inner_crop"
    assert match.text == "BCDE"
    assert match.text_span == (0, 4)


def test_prefix_suffix_inner_and_exact_alignments_use_half_open_spans():
    lex = Lexicon.from_terms("t", ["ABCDEF"])
    cases = (
        ("ABCDEF", "exact", (0, 6)),
        ("ABCDE", "prefix_crop", (0, 5)),
        ("BCDEF", "suffix_crop", (1, 6)),
        ("BCDE", "inner_crop", (1, 5)),
    )
    for visible, kind, span in cases:
        match = lex.best_match(visible)
        assert match is not None
        assert match.term == "ABCDEF"
        assert match.kind == kind
        assert match.span == span


def test_internal_missing_chars_are_penalized_more_than_edge_crops():
    lex = Lexicon.from_terms("t", ["ABCDEF"])
    exact = lex.best_match("ABCDEF").score
    edge = lex.best_match("BCDEF").score
    inner = lex.best_match("BCDE").score
    gap = lex.best_match("BCE").score  # D missing inside the crop

    assert exact == pytest.approx(1.0)
    assert edge == pytest.approx(0.92)
    assert inner == pytest.approx(0.80)
    assert gap < inner < edge < exact


def test_internal_gap_penalty_grows_with_missing_character_count():
    lex = Lexicon.from_terms("t", ["ABCDEF"])
    one_missing = lex.best_match("BCE").score   # D missing
    two_missing = lex.best_match("BDF").score   # C, E missing
    many_missing = lex.best_match("BF").score   # C, D, E missing

    assert one_missing > two_missing > many_missing
    assert many_missing >= 0.2


def test_unrelated_or_reordered_text_is_not_a_partial_match():
    lex = Lexicon.from_terms("t", ["ABCDEF"])
    assert lex.best_match("XYZ") is None
    assert lex.best_match("BA") is None          # reversed order
    assert lex.best_match("ACEB") is None        # B cannot follow E
    assert lex.best_match("ABXDEF") is None      # X is not in the term
    assert match_term("", "ABCDEF") is None
    assert match_term("ABCDEF", "") is None


def test_whitespace_is_normalized_before_partial_matching():
    lex = Lexicon.from_terms("t", ["使 用"])
    assert "使用" in lex
    prefix = lex.best_match("使")
    suffix = lex.best_match("用")
    assert prefix is not None and prefix.kind == "prefix_crop"
    assert prefix.span == (0, 1)
    assert suffix is not None and suffix.kind == "suffix_crop"
    assert suffix.span == (1, 2)


def test_shorter_term_wins_for_the_same_crop():
    """For an identical crop the shorter term is the safer inference."""

    lex = Lexicon.from_terms("t", ["ABCDEF", "ABCD"])
    crop = lex.best_match("ABC")
    assert crop is not None
    assert crop.term == "ABCD"
    assert crop.kind == "prefix_crop"
    assert lex.best_match("ABCD").kind == "exact"


def _synthetic_result(
    text: str,
    confidences: tuple[float, ...],
    top_chars: tuple[tuple[str, ...], ...],
    charset: list[str],
) -> OCRResult:
    chars = tuple(
        CharResult(ch, i, 0, 10, 10, conf)
        for i, (ch, conf) in enumerate(zip(text, confidences))
    )
    candidates = []
    for i, top in enumerate(top_chars):
        ids = tuple(charset.index(ch) for ch in top)
        candidates.append(
            VisualCandidate(
                start=i,
                end=i + 1,
                components=(i,),
                atoms=(i, i + 1),
                scores=VisualScores(char_ids=ids, logits=(1.0,) * len(ids)),
            )
        )
    path = DecodePath(candidates=tuple(candidates), text=text)
    return OCRResult(
        text=text,
        confidence=float(np.mean(confidences)),
        chars=chars,
        path=path,
    )


def test_apply_lexicon_never_fabricates_the_full_term():
    """Weak visual evidence must not turn a crop into the full term."""

    charset = list("ABCDEF")
    lex = Lexicon.from_terms("t", ["ABCDEF"])
    result = _synthetic_result(
        "BCDE",
        (1.0, 0.4, 0.5, 1.0),
        (("B", "A"), ("C", "D"), ("D", "E"), ("E", "F")),
        charset,
    )
    out = apply_lexicon(result, lex, "prefer", charset=charset)

    assert out.text == "BCDE"                       # never fabricated
    assert out.matched_term == "ABCDEF"
    assert out.matched_span == (1, 5)
    assert out.lexicon_match is not None
    assert out.lexicon_match.kind == "inner_crop"


def test_prefer_mode_annotates_partial_word_crops_end_to_end(font_path):
    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)

    for visible, term, span, kind in (
        ("巴尔的摩", "巴尔的摩", (0, 4), "exact"),
        ("巴尔的", "巴尔的摩", (0, 3), "prefix_crop"),
        ("尔的摩", "巴尔的摩", (1, 4), "suffix_crop"),
        ("尔的", "巴尔的摩", (1, 3), "inner_crop"),
    ):
        result = ocr.recognize(
            render_text(visible, font_path),
            lexicon="ships",
            lexicon_mode="prefer",
        )
        assert result.text == visible
        assert result.matched_term == term
        assert result.matched_span == span
        assert result.lexicon_match is not None
        assert result.lexicon_match.kind == kind


def test_goal12_example_six_char_crop_end_to_end(font_path):
    """Visible C2..C5 of 塞瓦斯托波尔 -> span (1, 5), text stays visible."""

    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)
    visible = "瓦斯托波"
    result = ocr.recognize(
        render_text(visible, font_path),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == visible
    assert result.matched_term == "塞瓦斯托波尔"
    assert result.matched_span == (1, 5)
    assert result.lexicon_match is not None
    assert result.lexicon_match.kind == "inner_crop"


def test_internal_gap_crop_is_annotated_but_not_rewritten(font_path):
    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)
    result = ocr.recognize(
        render_text("巴摩", font_path),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == "巴摩"
    assert result.matched_term == "巴尔的摩"
    assert result.matched_span == (0, 4)
    assert result.lexicon_match is not None
    assert result.lexicon_match.kind == "gap_crop"
    # Internal missing characters score below every contiguous crop.
    assert result.lexicon_match.score < 0.92


def test_strict_mode_rejects_partial_crops_without_fabricating(font_path):
    """Goal 11 strict semantics stay exact-only; a crop is not the full term."""

    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)
    result = ocr.recognize(
        render_text("尔的摩", font_path),
        lexicon="ships",
        lexicon_mode="strict",
    )
    assert result.text == ""
    assert result.confidence == 0.0
    assert result.matched_term is None
    assert result.lexicon_match is None
