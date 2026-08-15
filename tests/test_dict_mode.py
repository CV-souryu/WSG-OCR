"""``lexicon_mode="dict"``: associative dictionary inference.

Contract (from the game-screenshot design discussion):

* the visible text is scanned position by position; every dictionary term
  that can explain it -- exact, cropped at either edge, confusable
  (Top-K substitution) or partially damaged (forced completion of a unique
  prefix/suffix) -- competes as one word hypothesis;
* the visible ``text`` is never rewritten; the winning word is annotated
  as ``matched_term`` with its Goal 12 ``matched_span``;
* "return nothing unless a word matches": a result without any surviving
  hypothesis is rejected as empty output (``text=""``, ``confidence=0``);
* the Goal 15 priority holds: a clearly visible term (潜乙) is never
  rewritten into a look-alike dictionary term (潜甲);
* single-char terms require stronger evidence than multi-char terms.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixedfontocr import FixedFontOCR

from conftest import render_text


MODEL = Path("model/game_cn")


def _require_game_model() -> None:
    if not (MODEL / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")


def _ocr():
    _require_game_model()
    return FixedFontOCR(model_path=MODEL, backend="cpu")


def test_dict_mode_annotates_suffix_crop(font_path):
    """Visible 尔的摩 (suffix of 巴尔的摩) keeps text, annotates the word."""

    ocr = _ocr()
    result = ocr.recognize(
        render_text("尔的摩", font_path, font_size=14),
        lexicon="ships",
        lexicon_mode="dict",
    )
    assert result.text == "尔的摩"
    assert result.matched_term == "巴尔的摩"
    assert result.matched_span == (1, 4)
    assert result.lexicon_match is not None
    assert result.lexicon_match.kind == "suffix_crop"


def test_dict_mode_associates_confusable_topk(font_path):
    """14px 波/彼 confusion: the visible text stays, the word is inferred."""

    ocr = _ocr()
    result = ocr.recognize(
        render_text("塞瓦斯托波尔", font_path, font_size=14),
        lexicon="ships",
        lexicon_mode="dict",
    )
    assert result.matched_term == "塞瓦斯托波尔"
    assert result.matched_span == (0, 6)
    # the visible text keeps the scan result (never rewritten)
    assert result.text != "" and "瓦" in result.text


def test_dict_mode_forced_completion_unique_prefix(font_path):
    """华盛 + unique prefix -> the whole word 华盛顿 is put on the table."""

    ocr = _ocr()
    result = ocr.recognize(
        render_text("华盛蜂", font_path, font_size=14),
        lexicon="ships",
        lexicon_mode="dict",
    )
    assert result.matched_term == "华盛顿"
    assert result.matched_span == (0, 3)
    assert result.text == "华盛蜂"


def test_dict_mode_empty_when_no_word(font_path):
    """Dictionary mode returns nothing unless a word matches."""

    ocr = _ocr()
    result = ocr.recognize(
        render_text("!@#", font_path, font_size=14),
        lexicon="ships",
        lexicon_mode="dict",
    )
    assert result.text == ""
    assert result.confidence == 0.0
    assert result.matched_term is None
    assert result.matched_span is None
    assert result.lexicon_match is None


def test_dict_mode_goal15_never_rewrites_clear_term(font_path):
    """A clear 潜乙 stays 潜乙; the look-alike 潜甲 never wins."""

    ocr = _ocr()
    result = ocr.recognize(
        render_text("潜乙", font_path, font_size=12),
        lexicon="ships",
        lexicon_mode="dict",
    )
    assert result.text == "潜乙"
    assert result.matched_term == "潜乙"
    assert result.matched_term != "潜甲"


def test_dict_mode_single_char_term_needs_strong_evidence(font_path):
    """Single-char terms are allowed but only on strong visual support."""

    ocr = _ocr()
    result = ocr.recognize(
        render_text("虎", font_path, font_size=14),
        lexicon="ships",
        lexicon_mode="dict",
    )
    assert result.text == "虎"
    assert result.matched_term == "虎"


def test_dict_mode_rejects_result_without_any_word(font_path):
    """End-to-end: a visible string outside the lexicon yields empty output."""

    ocr = _ocr()
    result = ocr.recognize(
        render_text("XYZ", font_path, font_size=32),
        lexicon="ships",
        lexicon_mode="dict",
    )
    assert result.text == ""
    assert result.confidence == 0.0
    assert result.matched_term is None


def test_dict_mode_rejects_empty_text():
    """An empty decoded text can never produce a dictionary hypothesis."""

    from fixedfontocr import Lexicon, OCRResult, apply_lexicon

    result = apply_lexicon(
        OCRResult(text="", confidence=0.0),
        Lexicon.from_terms("t", ["ABCDEF"]),
        "dict",
    )
    assert result.text == ""
    assert result.matched_term is None
