"""Real-glyph NCC arbitration (dict mode fallback).

The bank is built offline by ``tools/lexicon/build_real_glyph_bank.py``
from a labeled crops corpus; runtime matching is pure numpy NCC. These
tests pin the NCC/resize primitives and, when the local crops corpus and
its bank exist, the end-to-end arbitration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import realglyphs
from fixedfontocr.realglyphs import RealGlyphBank, load_real_glyphs, ncc, resize24

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "fonts" / "SourceHanSansSC" / "crops" / "crops_items_recognition.csv"
BANK = CSV.with_suffix(CSV.suffix + ".realglyphs.npz")
CROPS = CSV.parent / "crops_items"


def test_ncc_identity_and_scale_invariance():
    a = np.random.default_rng(0).random((24, 24)).astype(np.float32)
    assert ncc(a, a) == pytest.approx(1.0)
    assert ncc(a * 3 + 7, a * 0.2 - 1) == pytest.approx(1.0)
    b = np.zeros((24, 24), dtype=np.float32)
    assert ncc(a, b) == pytest.approx(0.0)


def test_resize24_shape_and_roundtrip():
    a = np.random.default_rng(1).random((8, 13)).astype(np.float32)
    out = resize24(a)
    assert out.shape == (24, 24)
    out2 = resize24(out)
    assert out2.shape == (24, 24)
    assert np.isfinite(out).all()


def test_load_real_glyphs_missing_is_none(tmp_path):
    assert load_real_glyphs(tmp_path / "nope.npz") is None


def test_ncc_arbitration_rescues_snow_tie():
    """初雪's misread glyph is the real 雪: NCC beats the font gap."""

    if not (BANK.is_file() and (CROPS / "0_y290_y318_item0.png").is_file()):
        pytest.skip("crops_items corpus/bank not present (local dataset)")
    bank = load_real_glyphs(BANK)
    assert bank is not None
    assert isinstance(bank, RealGlyphBank)

    from PIL import Image

    from fixedfontocr import FixedFontOCR

    from fixedfontocr.frontend import extract_frontend

    ocr = FixedFontOCR(model_path=ROOT / "model" / "game_cn", backend="cpu")
    img = np.asarray(
        Image.open(CROPS / "0_y290_y318_item0.png").convert("RGB"),
        dtype=np.uint8,
    )
    result = ocr.recognize(
        img,
        lexicon="ship_names",
        lexicon_mode="dict",
        real_glyph_bank=BANK,
    )
    assert result.matched_term == "初雪"
    assert result.lexicon_match is not None


def test_ncc_arbitration_never_runs_without_bank():
    """Without the bank argument the dict mode stays assoc-only."""

    if not (CROPS / "0_y290_y318_item0.png").is_file():
        pytest.skip("crops_items corpus not present (local dataset)")
    from PIL import Image

    from fixedfontocr import FixedFontOCR

    ocr = FixedFontOCR(model_path=ROOT / "model" / "game_cn", backend="cpu")
    img = np.asarray(
        Image.open(CROPS / "0_y290_y318_item0.png").convert("RGB"),
        dtype=np.uint8,
    )
    result = ocr.recognize(img, lexicon="ship_names", lexicon_mode="dict")
    # without the bank this crop stays an honest empty (tie unresolved)
    assert result.matched_term != "初雪"


# Touching Z+digit game crops (Z17 / Z28 / Z1, plus 47工程): the merged
# blob looks like one character (灶 / “) to the CNN, so the hybrid merge
# veto must keep the single-atom split path on the lattice and let the
# decoder read the real glyphs. The ground truth comes straight from the
# labeled CSV ("the answer is already in the csv").
Z_FAMILY_ROWS = (
    "0_y112_y140_item1.png",  # Z17
    "1_y646_y674_item2.png",  # Z28
    "1_y646_y674_item6.png",  # Z1
    "0_y151_y179_item5.png",  # 47工程
)


def _csv_expected() -> dict[str, str]:
    import csv

    with open(CSV, encoding="utf-8-sig", newline="") as f:
        return {
            row[1]: row[3]
            for row in list(csv.reader(f))[1:]
            if row[3] != ""
        }


def test_touching_zdigit_crops_decode_to_csv_labels():
    """Base mode reads the Z-family crops exactly as the CSV labels them."""

    if not all((CROPS / name).is_file() for name in Z_FAMILY_ROWS):
        pytest.skip("crops_items corpus not present (local dataset)")
    from PIL import Image

    from fixedfontocr import FixedFontOCR

    expected = _csv_expected()
    ocr = FixedFontOCR(model_path=ROOT / "model" / "game_cn", backend="cpu")
    for name in Z_FAMILY_ROWS:
        img = np.asarray(
            Image.open(CROPS / name).convert("RGB"), dtype=np.uint8
        )
        result = ocr.recognize(img)
        assert result.text == expected[name], (
            f"{name}: expected {expected[name]!r} from the CSV, "
            f"got {result.text!r}"
        )


def test_touching_zdigit_crops_resolve_in_dict_mode():
    """dict + real-glyph bank resolves the Z-family crops to CSV labels."""

    if not (
        BANK.is_file()
        and all((CROPS / name).is_file() for name in Z_FAMILY_ROWS)
    ):
        pytest.skip("crops_items corpus/bank not present (local dataset)")
    from PIL import Image

    from fixedfontocr import FixedFontOCR

    expected = _csv_expected()
    ocr = FixedFontOCR(model_path=ROOT / "model" / "game_cn", backend="cpu")
    for name in Z_FAMILY_ROWS:
        img = np.asarray(
            Image.open(CROPS / name).convert("RGB"), dtype=np.uint8
        )
        result = ocr.recognize(
            img,
            lexicon="ship_names",
            lexicon_mode="dict",
            real_glyph_bank=BANK,
        )
        got = result.matched_term or result.text or ""
        assert got == expected[name], (
            f"{name}: expected {expected[name]!r} from the CSV, "
            f"got matched_term={result.matched_term!r} text={result.text!r}"
        )


def test_ncc_evidence_cached_per_glyph_and_char():
    """NCC evidence is memoized: each (glyph, char) pair is scanned once.

    The 乌戈里尼 crop's visible text (，2*维瓦尔迪) puts ~94 terms on the
    candidate table; the uncached arbitration re-scanned the same prototype
    stacks thousands of times. The counting proxy pins the cache contract:
    no (glyph, char) pair may be evaluated twice, and the result must equal
    the real bank's.
    """

    if not (BANK.is_file() and (CROPS / "1_y400_y428_item1.png").is_file()):
        pytest.skip("crops_items corpus/bank not present (local dataset)")
    from PIL import Image

    from fixedfontocr import FixedFontOCR
    from fixedfontocr.frontend import extract_frontend
    from fixedfontocr.lexicon import _ncc_assoc_match, load_lexicon
    from fixedfontocr.realglyphs import resize24

    ocr = FixedFontOCR(model_path=ROOT / "model" / "game_cn", backend="cpu")
    bank = load_real_glyphs(BANK)
    assert bank is not None
    img = np.asarray(
        Image.open(CROPS / "1_y400_y428_item1.png").convert("RGB"),
        dtype=np.uint8,
    )
    frontend = extract_frontend(img, ocr.profile)
    result = ocr.recognize(img)
    soft_glyphs = [
        resize24(
            frontend.soft_foreground[
                cand.segment.y : cand.segment.y + cand.segment.h,
                cand.segment.x : cand.segment.x + cand.segment.w,
            ]
        )
        for cand in result.path.candidates
    ]
    lex = load_lexicon("ship_names")
    expected = _ncc_assoc_match(result, lex, bank, soft_glyphs)

    calls: list[tuple[int, str]] = []

    class CountingBank:
        def best_evidence(self, glyph, char):
            calls.append((id(glyph), char))
            return bank.best_evidence(glyph, char)

    got = _ncc_assoc_match(result, lex, CountingBank(), soft_glyphs)
    assert got == expected
    assert calls, "arbitration must evaluate some evidence"
    assert len(set(calls)) == len(calls), (
        "each (glyph, char) pair must be evaluated exactly once: "
        f"{len(calls)} calls, {len(set(calls))} unique"
    )
    # The crop's visible text shares characters with ~94 terms; without the
    # cache the same evidence would be re-evaluated once per term per
    # position (~3900 scans). The memoized pass evaluates exactly
    # glyphs x union-of-term-chars pairs (1169 for this crop).
    assert len(calls) < 2000
