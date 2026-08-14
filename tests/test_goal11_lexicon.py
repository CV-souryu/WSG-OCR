"""Goal 11 acceptance: the Lexicon Layer.

The Goal 11 contract from ``fonts/goal``:

* the runtime uses the existing ``charsets/words/`` lists, split by domain
  (ships / equipment / ui);
* ``ocr.recognize(image, lexicon="ships", lexicon_mode="prefer")`` is the
  public API;
* ``none`` / ``prefer`` / ``strict`` are supported without any LLM;
* the visible ``text`` is never rewritten just because a dictionary entry
  exists -- ``matched_term`` / ``matched_span`` carry the inferred entity;
* prefer helps only when the visual evidence is uncertain and the target
  character is already a plausible Top-K alternative (Goal 15);
* partial-word alignment (Goal 12) is designed into the matcher.
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
    LEXICON_DOMAINS,
    OCRResult,
    VisualCandidate,
    VisualScores,
    apply_lexicon,
    load_lexicon,
)
from fixedfontocr.lexicon import match_term

from conftest import render_text


MODEL = Path("model/game_cn")


def _require_game_model() -> None:
    if not (MODEL / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")


def test_builtin_lexicons_load_and_cover_the_goal_domains():
    assert "ships" in LEXICON_DOMAINS
    assert "equipment" in LEXICON_DOMAINS
    assert "ui" in LEXICON_DOMAINS

    ships = load_lexicon("ships")
    equipment = load_lexicon("equipment")
    ui = load_lexicon("ui")

    assert len(ships) > 100
    assert len(equipment) > 100
    assert len(ui) > 500
    assert "巴尔的摩" in ships
    # The "ships" domain covers both the original and harmonized ship-name
    # word lists, not just the un-harmonized subset.
    assert "魟" in ships
    assert "小型单装炮" in equipment
    # The source file writes "使 用"; OCR emits no whitespace, so the
    # normalized dictionary must contain "使用".
    assert "使用" in ui
    for lex in (ships, equipment, ui):
        assert all(not any(ch.isspace() for ch in t) for t in lex.terms)

    for domain, set_files in (
        (
            "ships",
            (
                "ship_names_charset.txt",
                "ship_names_harmonized_charset.txt",
            ),
        ),
        ("equipment", ("equipment_charset.txt",)),
        ("ui", ("ui_texts_charset.txt",)),
    ):
        checked_chars: set[str] = set()
        for set_file in set_files:
            checked_chars.update(
                ch
                for ch in (Path("charsets/sets") / set_file).read_text(
                    encoding="utf-8"
                )
                if not ch.isspace()
            )
        checked = "".join(sorted(checked_chars, key=ord))
        assert load_lexicon(domain).charset == checked

    all_charset = "".join(
        ch
        for ch in Path("charsets/sets/all_charset.txt").read_text(
            encoding="utf-8"
        )
        if not ch.isspace()
    )
    assert load_lexicon("all").charset == all_charset

    # Aliases point at the same word lists.
    ship_names = load_lexicon("ship_names")
    harmonized = load_lexicon("harmonized_ships")
    assert ship_names.terms == ships.terms[: len(ship_names)]
    assert harmonized.terms == load_lexicon("ship_names_harmonized").terms
    assert set(ships.terms) == set(ship_names.terms) | set(harmonized.terms)
    assert load_lexicon("equipment_names").terms == equipment.terms
    assert load_lexicon("ui_texts").terms == ui.terms


def test_lexicon_accepts_explicit_paths_and_deduplicates(tmp_path):
    path = tmp_path / "custom.txt"
    path.write_text("Alpha\nBeta\n\nAl pha\nBeta\n", encoding="utf-8")
    lex = load_lexicon(path)
    assert lex.terms == ("Alpha", "Beta")
    assert lex.charset == "ABaehlpt"
    assert lex.best_match("Alpha").kind == "exact"

    from_terms = Lexicon.from_terms("x", ["A 1", "B2", "B 2", "A1"])
    assert from_terms.terms == ("A1", "B2")
    assert from_terms.charset == "12AB"


def test_matching_supports_exact_full_and_goal12_partial_alignments():
    lex = Lexicon.from_terms("ships", ["巴尔的摩", "巴尔的摩号", "使用"])

    exact = lex.best_match("巴尔的摩")
    assert exact is not None
    assert exact.term == "巴尔的摩"
    assert exact.span == (0, 4)
    assert exact.kind == "exact"
    assert exact.score == pytest.approx(1.0)

    suffix = lex.best_match("尔的摩")
    assert suffix is not None
    assert suffix.term == "巴尔的摩"
    assert suffix.span == (1, 4)  # Goal 12 span convention
    assert suffix.kind == "suffix_crop"
    assert suffix.score == pytest.approx(0.92)

    prefix = lex.best_match("巴尔的")
    assert prefix is not None
    assert prefix.term == "巴尔的摩"
    assert prefix.span == (0, 3)
    assert prefix.kind == "prefix_crop"

    inner = lex.best_match("尔的")
    assert inner is not None
    assert inner.span == (1, 3)
    assert inner.kind == "inner_crop"

    gap = lex.best_match("巴摩")
    assert gap is not None
    assert gap.term == "巴尔的摩"
    assert gap.span == (0, 4)  # internal missing 尔 is inside the span
    assert gap.kind == "gap_crop"
    assert gap.score < suffix.score

    inside = lex.best_match("我的巴尔的摩")
    assert inside is not None
    assert inside.term == "巴尔的摩"
    assert inside.span == (0, 4)
    assert inside.text_span == (2, 6)
    assert inside.kind == "term_in_text"


def test_match_term_returns_none_for_unrelated_text():
    assert match_term("abc", "巴尔的摩") is None
    assert match_term("", "巴尔的摩") is None
    assert match_term("巴尔的摩", "") is None


def test_prefer_mode_annotates_clear_text_without_rewriting(font_path):
    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)

    none_result = ocr.recognize(
        render_text("巴尔的摩", font_path),
        lexicon="ships",
        lexicon_mode="none",
    )
    assert none_result.text == "巴尔的摩"
    assert none_result.matched_term is None
    assert none_result.lexicon_match is None

    result = ocr.recognize(
        render_text("巴尔的摩", font_path),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == "巴尔的摩"
    assert result.matched_term == "巴尔的摩"
    assert result.matched_span == (0, 4)
    assert result.lexicon_match is not None
    assert result.lexicon_match.kind == "exact"
    assert result.lexicon_match.mode == "prefer"
    assert len(result.chars) == 4


def test_prefer_mode_supports_partial_word_crop(font_path):
    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)
    result = ocr.recognize(
        render_text("尔的摩", font_path),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == "尔的摩"
    assert result.matched_term == "巴尔的摩"
    assert result.matched_span == (1, 4)
    assert result.lexicon_match is not None
    assert result.lexicon_match.kind == "suffix_crop"


def test_prefer_mode_allows_unknown_text(font_path):
    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)
    result = ocr.recognize(
        render_text("ABCXYZ999", font_path),
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == "ABCXYZ999"
    assert result.confidence > 0.0


def test_strict_mode_accepts_exact_terms_and_rejects_others(font_path):
    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)

    ok = ocr.recognize(
        render_text("巴尔的摩", font_path),
        lexicon="ships",
        lexicon_mode="strict",
    )
    assert ok.text == "巴尔的摩"
    assert ok.matched_term == "巴尔的摩"
    assert ok.lexicon_match is not None
    assert ok.lexicon_match.mode == "strict"

    rejected = ocr.recognize(
        render_text("ABCXYZ999", font_path),
        lexicon="ships",
        lexicon_mode="strict",
    )
    assert rejected.text == ""
    assert rejected.confidence == 0.0
    assert rejected.chars == ()
    assert rejected.matched_term is None
    assert rejected.lexicon_match is None


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
        cand = VisualCandidate(
            start=i,
            end=i + 1,
            components=(i,),
            atoms=(i, i + 1),
            scores=VisualScores(char_ids=ids, logits=(1.0,) * len(ids)),
        )
        candidates.append(cand)
    path = DecodePath(candidates=tuple(candidates), text=text)
    return OCRResult(
        text=text,
        confidence=float(np.mean(confidences)),
        chars=chars,
        path=path,
    )


def test_prefer_correction_only_uses_uncertain_topk_alternatives():
    charset = list("ABC")
    lex = Lexicon.from_terms("t", ["AC"])

    # B is uncertain and C is in the visual Top-K -> prefer corrects.
    weak = _synthetic_result(
        "AB", (1.0, 0.6), (("A",), ("C", "B")), charset
    )
    corrected = apply_lexicon(weak, lex, "prefer", charset=charset)
    assert corrected.text == "AC"
    assert corrected.matched_term == "AC"
    assert corrected.matched_span == (0, 2)
    assert corrected.chars[1].char == "C"

    # B is visually confident -> prefer must not rewrite it.
    strong = _synthetic_result(
        "AB", (1.0, 1.0), (("A",), ("C", "B")), charset
    )
    kept = apply_lexicon(strong, lex, "prefer", charset=charset)
    assert kept.text == "AB"
    assert kept.lexicon_match is None or kept.lexicon_match.kind != "exact"

    # C is not in the visual Top-K -> lexicon must not invent it.
    missing = _synthetic_result(
        "AB", (1.0, 0.6), (("A",), ("B",)), charset
    )
    untouched = apply_lexicon(missing, lex, "prefer", charset=charset)
    assert untouched.text == "AB"


def test_strict_can_accept_a_visually_uncertain_dictionary_correction():
    charset = list("ABC")
    lex = Lexicon.from_terms("t", ["AC"])
    result = _synthetic_result(
        "AB", (1.0, 0.6), (("A",), ("C", "B")), charset
    )
    strict = apply_lexicon(result, lex, "strict", charset=charset)
    assert strict.text == "AC"
    assert strict.lexicon_match is not None
    assert strict.lexicon_match.mode == "strict"

    rejected = _synthetic_result(
        "AB", (1.0, 1.0), (("A",), ("C", "B")), charset
    )
    strict_rejected = apply_lexicon(rejected, lex, "strict", charset=charset)
    assert strict_rejected.text == ""
    assert strict_rejected.confidence == 0.0


def test_positional_shorthand_and_invalid_modes(font_path):
    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)
    result = ocr.recognize(render_text("巴尔的摩", font_path), "ships", "prefer")
    assert result.matched_term == "巴尔的摩"

    with pytest.raises(ValueError, match="lexicon_mode"):
        ocr.recognize(render_text("123", font_path), lexicon_mode="bogus")
    with pytest.raises(ValueError, match="unknown lexicon"):
        load_lexicon("not_a_domain")
