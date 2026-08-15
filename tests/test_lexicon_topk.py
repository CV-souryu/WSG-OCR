"""Top-K lexicon guess (``lexicon_mode="topk"``, "Top-3 取词表")."""

from __future__ import annotations

import fixedfontocr.lexicon as lexicon_mod
from fixedfontocr import (
    CharResult,
    DecodePath,
    Lexicon,
    OCRResult,
    VisualCandidate,
    VisualScores,
    apply_lexicon,
)


def _result(
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
        confidence=float(sum(confidences) / len(confidences)),
        chars=chars,
        path=path,
        alternatives=alternatives,
    )


def test_topk_promotes_alternative_term_with_different_segmentation():
    charset = list("灶7Z17z217")
    lex = Lexicon.from_terms("ships", ["Z17", "Z31"])
    result = _result(
        "灶7",
        (0.918, 0.918),
        (("灶", "Z"), ("7", "1")),
        charset,
        alternatives=("Z17", "z17", "217"),
    )
    out = apply_lexicon(result, lex, "topk", charset=charset)
    assert out.text == "Z17"
    assert out.matched_term == "Z17"
    assert out.matched_span == (0, 3)
    assert len(out.chars) == 3
    assert out.lexicon_match is not None
    assert out.lexicon_match.mode == "topk"


def test_topk_keeps_visible_exact_term_over_alternative():
    charset = list("狮蜩")
    lex = Lexicon.from_terms("ships", ["狮", "蜩"])
    result = _result(
        "狮",
        (0.95,),
        (("狮", "蜩"),),
        charset,
        alternatives=("蜩",),
    )
    out = apply_lexicon(result, lex, "topk", charset=charset)
    assert out.text == "狮"
    assert out.matched_term == "狮"


def test_topk_rewrites_same_length_term_from_top3():
    charset = list("纽伦量堡")
    lex = Lexicon.from_terms("ships", ["纽伦堡"])
    result = _result(
        "纽伦量",
        (0.93, 0.93, 0.93),
        (("纽",), ("伦",), ("量", "堡")),
        charset,
    )
    out = apply_lexicon(result, lex, "topk", charset=charset)
    assert out.text == "纽伦堡"
    assert out.matched_term == "纽伦堡"
    assert len(out.chars) == 3
    assert out.chars[2].char == "堡"


def test_topk_keeps_original_when_guess_is_not_unique(monkeypatch):
    charset = list("ABCD")
    lex = Lexicon.from_terms("ships", ["AC", "AD"])
    result = _result(
        "AB",
        (0.9, 0.9),
        (("A", "B"), ("B", "C", "D")),
        charset,
    )
    # Widen the uniqueness margin so both terms count as near-tied.
    monkeypatch.setattr(lexicon_mod, "TOP_K_GUESS_UNIQUE_MARGIN", 0.2)
    out = apply_lexicon(result, lex, "topk", charset=charset)
    assert out.text == "AB"
    assert out.matched_term is None


def test_topk_does_not_touch_unknown_or_blank_text():
    charset = list("ABCX")
    lex = Lexicon.from_terms("ships", ["ABC"])
    result = _result(
        "AX",
        (0.9, 0.5),
        (("A",), ("X",)),
        charset,
        alternatives=("AB",),
    )
    # "AB" is not a term, and neither X nor B is in the Top-3 set for the
    # second position, so the original text must be preserved.
    out = apply_lexicon(result, lex, "topk", charset=charset)
    assert out.text == "AX"
