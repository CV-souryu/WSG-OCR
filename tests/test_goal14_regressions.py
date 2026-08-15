"""Goal 14 acceptance: the typical-problem regression battery.

The Goal 14 contract from ``fonts/goal``:

* ``鲃鱼`` must never decode as ``$E鱼`` (fragment-heavy 鲃 + 鱼);
* ``小`` must never fragment into several characters;
* ``潜甲`` / ``潜乙`` must never be falsely merged;
* ``Z17`` and ``巴尔的摩`` must be correct at small font sizes;
* the battery also covers mixed Chinese+ASCII, digits, punctuation, short
  words, long words and partial-word crops.

Small size here follows the project's own low-res domain (Goal 7: 10..18 px)
and the Goal 9 line-acceptance sizes (12/14/16 px). Every case is also
checked at 32 px so a high-resolution regression cannot slip through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import FixedFontOCR, OCRResult

from conftest import render_text


MODEL = Path("model/game_cn")
SMALL_SIZES = (12, 14, 16)


def _require_game_model() -> None:
    if not (MODEL / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")


def _ocr() -> FixedFontOCR:
    _require_game_model()
    return FixedFontOCR(model_path=MODEL, backend="cpu", auto_benchmark=False)


def _recognize(
    ocr: FixedFontOCR,
    text: str,
    size: int,
    font_path: Path,
    **kwargs,
) -> OCRResult:
    return ocr.recognize(render_text(text, font_path, font_size=size), **kwargs)


@pytest.mark.parametrize("size", (*SMALL_SIZES, 32))
def test_bayu_is_never_dollar_e_fish(font_path, size):
    """鲃鱼: the fragmented 鲃 must stay one glyph, not $+E."""

    ocr = _ocr()
    result = _recognize(ocr, "鲃鱼", size, font_path)
    assert result.text == "鲃鱼", (
        f"{size}px: 鲃鱼 -> {result.text!r}"
    )
    assert result.text != "$E鱼"
    assert len(result.chars) == 2


@pytest.mark.parametrize("size", (*SMALL_SIZES, 32))
def test_xiao_never_fragments(font_path, size):
    """小: three components must decode as one character."""

    ocr = _ocr()
    result = _recognize(ocr, "小", size, font_path)
    assert result.text == "小", (
        f"{size}px: 小 -> {result.text!r}"
    )
    assert len(result.chars) == 1
    assert len(result.path.candidates) == 1


@pytest.mark.parametrize("text", ("潜甲", "潜乙"))
@pytest.mark.parametrize("size", (*SMALL_SIZES, 32))
def test_qian_xy_never_falsely_merged(font_path, text, size):
    """潜甲/潜乙: two characters stay two characters (ships lexicon)."""

    ocr = _ocr()
    result = _recognize(
        ocr,
        text,
        size,
        font_path,
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == text, (
        f"{text!r} at {size}px -> {result.text!r}"
    )
    assert len(result.chars) == 2
    assert result.matched_term == text


@pytest.mark.parametrize("size", (*SMALL_SIZES, 32))
def test_z17_small_size(font_path, size):
    ocr = _ocr()
    result = _recognize(ocr, "Z17", size, font_path)
    assert result.text == "Z17", f"{size}px: Z17 -> {result.text!r}"
    assert len(result.chars) == 3


@pytest.mark.parametrize("size", (*SMALL_SIZES, 32))
def test_baltimore_small_size_chinese(font_path, size):
    ocr = _ocr()
    result = _recognize(
        ocr,
        "巴尔的摩",
        size,
        font_path,
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == "巴尔的摩", (
        f"{size}px: 巴尔的摩 -> {result.text!r}"
    )
    assert len(result.chars) == 4
    assert result.matched_term == "巴尔的摩"
    assert result.matched_span == (0, 4)


def test_mixed_chinese_ascii_small_and_regular(font_path):
    ocr = _ocr()
    for size in (16, 32):
        result = _recognize(ocr, "舰船Lv.99", size, font_path)
        assert result.text == "舰船Lv.99", (
            f"{size}px: 舰船Lv.99 -> {result.text!r}"
        )
        assert len(result.chars) == len("舰船Lv.99")


def test_digits_small_and_regular(font_path):
    ocr = _ocr()
    for size in (16, 32):
        result = _recognize(ocr, "1234567890", size, font_path)
        assert result.text == "1234567890", (
            f"{size}px: 1234567890 -> {result.text!r}"
        )
        assert len(result.chars) == 10


def test_punctuation(font_path):
    ocr = _ocr()
    result = _recognize(ocr, "。，！？", 32, font_path)
    assert result.text == "。，！？", (
        f"punctuation -> {result.text!r}"
    )
    assert len(result.chars) == 4


@pytest.mark.parametrize("size", (*SMALL_SIZES, 32))
def test_short_word_with_ui_lexicon(font_path, size):
    ocr = _ocr()
    result = _recognize(
        ocr,
        "使用",
        size,
        font_path,
        lexicon="ui",
        lexicon_mode="prefer",
    )
    assert result.text == "使用", f"{size}px: 使用 -> {result.text!r}"
    assert result.matched_term == "使用"
    assert len(result.chars) == 2


@pytest.mark.parametrize("size", (16, 32))
def test_long_word_with_ships_lexicon(font_path, size):
    ocr = _ocr()
    result = _recognize(
        ocr,
        "塞瓦斯托波尔",
        size,
        font_path,
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == "塞瓦斯托波尔", (
        f"{size}px: 塞瓦斯托波尔 -> {result.text!r}"
    )
    assert result.matched_term == "塞瓦斯托波尔"
    assert result.matched_span == (0, 6)
    assert len(result.chars) == 6


@pytest.mark.parametrize("size", (*SMALL_SIZES, 32))
def test_partial_word_crop(font_path, size):
    """Goal 12 partial-word stays visible text + annotated entity."""

    ocr = _ocr()
    result = _recognize(
        ocr,
        "尔的摩",
        size,
        font_path,
        lexicon="ships",
        lexicon_mode="prefer",
    )
    assert result.text == "尔的摩", (
        f"{size}px: 尔的摩 -> {result.text!r}"
    )
    assert result.matched_term == "巴尔的摩"
    assert result.matched_span == (1, 4)
    assert result.lexicon_match is not None
    assert result.lexicon_match.kind == "suffix_crop"
