"""Goal 15 acceptance: the lexicon must not overpower strong visual evidence.

The Goal 15 contract from ``fonts/goal``:

* ``prefer`` mode must not rewrite visually confident text;
* visually ambiguous text may be helped by the lexicon (only when the
  dictionary target is already a plausible Top-K alternative);
* a non-unique dictionary match keeps the OCR text and its alternatives;
* unknown text must still be emitted normally.

The canonical example is ``潜乙``: even though ``潜甲`` exists in the ships
lexicon, a clear ``潜乙`` must never be rewritten into ``潜甲``.
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
    VisualLattice,
    VisualScores,
    apply_lexicon,
    decode_beam,
    decode_dp,
)

from conftest import render_text


MODEL = Path("model/game_cn")
SMALL_SIZES = (12, 14, 16)


def _require_game_model() -> None:
    if not (MODEL / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")


def _ocr() -> FixedFontOCR:
    _require_game_model()
    return FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)


def _candidate(
    span: tuple[int, int],
    char_ids: tuple[int, ...],
    logits: tuple[float, ...],
) -> VisualCandidate:
    return VisualCandidate(
        start=span[0],
        end=span[1],
        components=tuple(range(span[0], span[1])),
        atoms=span,
        scores=VisualScores(char_ids=char_ids, logits=logits),
    )


def _lattice(*candidates: VisualCandidate) -> VisualLattice:
    return VisualLattice(candidates=tuple(candidates))


def _synthetic_result(
    text: str,
    confidences: tuple[float, ...],
    top_chars: tuple[tuple[str, ...], ...],
    charset: list[str],
    alternatives: tuple[str, ...] = (),
) -> OCRResult:
    chars = tuple(
        CharResult(ch, i, 0, 10, 10, conf)
        for i, (ch, conf) in enumerate(zip(text, confidences))
    )
    candidates = tuple(
        VisualCandidate(
            start=i,
            end=i + 1,
            components=(i,),
            atoms=(i, i + 1),
            scores=VisualScores(
                char_ids=tuple(charset.index(ch) for ch in top),
                logits=(1.0,) * len(top),
            ),
        )
        for i, top in enumerate(top_chars)
    )
    path = DecodePath(candidates=candidates, text=text)
    return OCRResult(
        text=text,
        confidence=float(np.mean(confidences)),
        chars=chars,
        path=path,
        alternatives=alternatives,
    )


def test_prefer_never_rewrites_confident_visual_evidence():
    charset = list("ABC")
    lex = Lexicon.from_terms("t", ["AC"])

    # Both characters are visually confident, so even though C is in the
    # Top-K of the second character, prefer must not change B -> C.
    strong = _synthetic_result(
        "AB", (1.0, 1.0), (("A",), ("C", "B")), charset
    )
    kept = apply_lexicon(strong, lex, "prefer", charset=charset)
    assert kept.text == "AB"
    assert kept.alternatives == strong.alternatives


def test_prefer_uses_lexicon_when_visual_is_ambiguous():
    charset = list("ABC")
    lex = Lexicon.from_terms("t", ["AC"])

    # B is visually uncertain and C is already a plausible Top-K
    # alternative -> the lexicon resolves the ambiguity.
    weak = _synthetic_result(
        "AB", (1.0, 0.6), (("A",), ("C", "B")), charset
    )
    corrected = apply_lexicon(weak, lex, "prefer", charset=charset)
    assert corrected.text == "AC"
    assert corrected.matched_term == "AC"
    assert corrected.chars[1].char == "C"


def test_prefer_never_invents_characters_outside_visual_topk():
    charset = list("ABC")
    lex = Lexicon.from_terms("t", ["AC"])

    # C is not in the visual Top-K, so the lexicon cannot fabricate it
    # even though the character is uncertain.
    missing = _synthetic_result(
        "AB", (1.0, 0.6), (("A",), ("B",)), charset
    )
    untouched = apply_lexicon(missing, lex, "prefer", charset=charset)
    assert untouched.text == "AB"


def test_prefer_keeps_ocr_and_alternatives_when_match_is_not_unique():
    charset = list("ABC")
    # Both "AC" and "CB" are equally plausible one-character corrections of
    # "AB"; the dictionary cannot pick a unique winner.
    lex = Lexicon.from_terms("t", ["AC", "CB"])
    result = _synthetic_result(
        "AB",
        (0.5, 0.5),
        (("A", "C"), ("B", "C")),
        charset,
        alternatives=("CB", "AC"),
    )

    out = apply_lexicon(result, lex, "prefer", charset=charset)
    assert out.text == "AB"
    assert out.alternatives == ("CB", "AC")


def test_prefer_keeps_ocr_when_corrections_are_nearly_tied():
    charset = list("ABC")
    # "AC" (B->C) and "CB" (A->C) are both plausible, with scores within
    # the uniqueness margin; prefer keeps the visible OCR text.
    lex = Lexicon.from_terms("t", ["AC", "CB"])
    result = _synthetic_result(
        "AB",
        (0.5, 0.52),
        (("A", "C"), ("B", "C")),
        charset,
        alternatives=("CB", "AC"),
    )

    out = apply_lexicon(result, lex, "prefer", charset=charset)
    assert out.text == "AB"
    assert out.alternatives == ("CB", "AC")

    # A zero margin still allows the best correction to win, proving the
    # margin (not a missing candidate) is what keeps the OCR text.
    permissive = apply_lexicon(
        result,
        lex,
        "prefer",
        charset=charset,
        correction_unique_margin=0.0,
    )
    assert permissive.text == "AC"


def test_decoder_confident_visual_beats_lexicon():
    """A visually confident 乙乙 path survives a lexicon that only knows 甲甲."""

    charset = ["甲", "乙"]
    lex = Lexicon.from_terms("t", ["甲甲"])
    confident = (
        _candidate((0, 1), (0, 1), (0.40, 0.95)),
        _candidate((1, 2), (0, 1), (0.40, 0.95)),
    )

    dp = decode_dp(_lattice(*confident), charset, lex)
    beam = decode_beam(_lattice(*confident), charset, lex)
    assert dp.text == "乙乙"
    assert beam.text == "乙乙"


def test_decoder_lexicon_resolves_visual_ties():
    """With equal visual evidence the lexicon decides between 甲甲/乙乙."""

    charset = ["甲", "乙"]
    ambiguous = (
        _candidate((0, 1), (0, 1), (0.55, 0.54)),
        _candidate((1, 2), (0, 1), (0.55, 0.54)),
    )

    lex_a = Lexicon.from_terms("t", ["甲甲"])
    assert decode_dp(_lattice(*ambiguous), charset, lex_a).text == "甲甲"

    lex_b = Lexicon.from_terms("t", ["乙乙"])
    assert decode_dp(_lattice(*ambiguous), charset, lex_b).text == "乙乙"


@pytest.mark.parametrize("text", ("潜乙", "潜甲"))
@pytest.mark.parametrize("size", (*SMALL_SIZES, 32))
def test_clear_qian_glyph_not_rewritten_to_dictionary_twin(font_path, text, size):
    """Clear 潜乙 must stay 潜乙 even though 潜甲 is in the ships lexicon."""

    ocr = _ocr()
    result = ocr.recognize(
        render_text(text, font_path, font_size=size),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == text, (
        f"{text!r} at {size}px -> {result.text!r}"
    )
    assert result.matched_term == text
    assert len(result.chars) == 2
    # The decoder/beam alternatives survive prefer mode.
    assert isinstance(result.alternatives, tuple)


@pytest.mark.parametrize("size", (*SMALL_SIZES, 32))
def test_unknown_text_is_emitted_normally_in_prefer_mode(font_path, size):
    ocr = _ocr()
    result = ocr.recognize(
        render_text("ABCXYZ999", font_path, font_size=size),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    # Unknown text is never suppressed by the lexicon, even at small sizes
    # where the visual evidence is genuinely ambiguous (C/С, X/Χ).
    assert result.text, f"{size}px: unknown text was suppressed"
    assert len(result.chars) == len(result.text)
    assert result.confidence > 0.0
    assert result.matched_term is None


def test_unknown_text_exact_at_regular_resolution(font_path):
    ocr = _ocr()
    result = ocr.recognize(
        render_text("ABCXYZ999", font_path),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == "ABCXYZ999"
    assert result.confidence > 0.0
    assert result.matched_term is None
