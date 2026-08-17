"""Probabilistic decoder v2 -- domain prior tests.

Covers the "domain 测试" checklist:

* clear visual characters are not overridden by a weak lexicon prior;
* visual-equivalent characters (Latin ``T`` / Cyrillic ``Т``) can be
  disambiguated by ``allowed_chars`` inside the equivalence class;
* a non-unique domain prior keeps the result ambiguous;
* open-text mode does not depend on a lexicon.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fixedfontocr.domain_prior import (
    AllowedCharsPrior,
    LexiconPrior,
    LexiconPriorBuilder,
    OpenTextPrior,
    UnigramPrior,
    domain_prior_from_config,
)
from fixedfontocr.prob_decoder import ProbDecoderConfig, ProbabilisticDecoder
from fixedfontocr.model import load_model

from prob_helpers import (
    decoder_block,
    manual_candidate,
    v2_cnn_digits_model,
    v2_game_model,
)


def _decoder(model_path):
    model = load_model(model_path)
    dec, warn = ProbabilisticDecoder.try_build(model)
    assert dec is not None, warn
    return dec


def _manual_lattice(candidates, comps=None):
    from fixedfontocr.types import VisualLattice

    return VisualLattice(
        components=tuple(comps or ()),
        candidates=tuple(candidates),
        width=1,
        height=1,
    )


def test_open_text_prior_is_uniform():
    prior = OpenTextPrior(tuple("ABC"))
    assert prior.log_prior(0) == pytest.approx(math.log(1 / 3))
    assert prior.log_prior(1) == pytest.approx(math.log(1 / 3))
    prior2 = OpenTextPrior(tuple("ABC"), allowed_ids=frozenset({0, 1}))
    assert prior2.log_prior(0) == pytest.approx(math.log(0.5))
    assert prior2.log_prior(2) == -np.inf


def test_allowed_chars_prior_restricts_and_is_uniform():
    prior = AllowedCharsPrior(frozenset({1, 3}), tuple("0123"))
    assert prior.log_prior(1) == pytest.approx(math.log(0.5))
    assert prior.log_prior(3) == pytest.approx(math.log(0.5))
    assert prior.log_prior(0) == -np.inf
    assert prior.log_prior(2) == -np.inf


def test_lexicon_prior_never_logs_heuristic_match_score():
    """The lexicon prior uses *term support counts*, not the 0..1 matcher."""

    prior = LexiconPrior(counts={0: 10.0, 1: 5.0}, max_count=10.0)
    assert prior.log_prior(0) > prior.log_prior(1)
    assert prior.log_prior(0) <= 0.0
    # Characters absent from the lexicon keep the floor (unknown text OK).
    assert prior.log_prior(7) == pytest.approx(prior.floor_log_p)


def test_lexicon_prior_builder_counts_terms():
    builder = LexiconPriorBuilder(lexicon_name="ships", terms=("ABC", "ABD", "A"))
    prior = builder.build(list("ABCDE"))
    assert prior.counts[0] == 3.0  # A
    assert prior.counts[1] == 2.0  # B
    assert prior.max_count == 3.0


def test_clear_visual_not_overridden_by_weak_lexicon(v2_game_model):
    """End to end: a clear 潜乙 render stays 潜乙 with the ships lexicon."""

    from fixedfontocr import FixedFontOCR
    from fixedfontocr.defaults import resolve_font
    from PIL import Image, ImageDraw, ImageFont

    font_path = resolve_font(
        __import__("fixedfontocr.defaults", fromlist=["FONT_PATH"]).FONT_PATH
    )
    font = ImageFont.truetype(str(font_path), 32)
    img = Image.new("RGB", (96, 48), (0, 0, 0))
    ImageDraw.Draw(img).text((8, 4), "潜乙", font=font, fill=(255, 255, 255))
    arr = np.asarray(img, dtype=np.uint8)

    ocr = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    result = ocr.recognize(arr, lexicon="ships", lexicon_mode="prefer")
    # v2 (like v1, Goal 15) must never rewrite clear 潜乙 into 潜甲.
    assert result.text == "潜乙"
    assert result.matched_term != "潜甲"


def test_visual_equivalents_disambiguated_by_allowed_chars(v2_game_model):
    """T/Т are pixel-identical in the registered font; allowed_chars must
    pick the allowed member of the equivalence class."""

    from fixedfontocr import FixedFontOCR
    from fixedfontocr.oracle import render_glyph_mask

    mask = render_glyph_mask("T", _game_font(), font_size=32)
    # Paste the glyph on a canvas like a single-char line.
    canvas = np.zeros((max(32, mask.shape[0] + 4), max(32, mask.shape[1] + 4), 3),
                      dtype=np.uint8)
    canvas[2 : 2 + mask.shape[0], 2 : 2 + mask.shape[1]] = np.where(
        mask[:, :, None], 255, 0
    )
    ocr = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    latin = ocr.recognize(canvas, allowed_chars="T-23")
    cyr = ocr.recognize(canvas, allowed_chars="Т-23")
    # Both spellings are readable; the allowed set decides the member.
    assert latin.text[0] in ("T", "Т") and cyr.text[0] in ("T", "Т")
    assert set(latin.text) <= set("T-23")
    assert set(cyr.text) <= set("Т-23")


def _game_font():
    from fixedfontocr.defaults import FONT_PATH, resolve_font

    if not FONT_PATH.exists():
        pytest.skip("bundled font not present")
    return resolve_font(FONT_PATH)


def test_non_unique_domain_prior_stays_ambiguous(tmp_path_factory):
    """Near-tied domain support -> near-tied paths -> alternatives kept."""

    from prob_helpers import build_v2_cnn_digits_model

    model_path = build_v2_cnn_digits_model(
        tmp_path_factory.mktemp("v2_flat"), "cnn_flat"
    )
    dec = _decoder(model_path)
    # Two options with identical visual evidence; lexicon supports both.
    cand = manual_candidate(0, 1, [0, 1], [0.0, 0.0], template_scores=[0.5, 0.5])
    lattice = _manual_lattice([cand])
    prior = LexiconPrior(counts={0: 1.0, 1: 1.0}, max_count=1.0)
    path = dec.decode_lattice(lattice, domain=prior)
    assert len(path.alternatives) >= 1
    # The runner-up margin is ~0 (both chars equally supported).
    diag = path.prob_diagnostics
    assert diag["runner_up_margin"] < 1e-6
    assert diag["reject_reason"] == ""
    # The same lattice with a decisive prior picks one side.
    prior2 = LexiconPrior(counts={0: 10.0, 1: 1.0}, max_count=10.0)
    path2 = dec.decode_lattice(lattice, domain=prior2)
    assert path2.text in ("0", "1")


def test_open_text_does_not_depend_on_lexicon(v2_cnn_digits_model):
    """With the open-text prior, an irrelevant lexicon changes nothing."""

    dec = _decoder(v2_cnn_digits_model)
    cand = manual_candidate(0, 1, [0, 1], [3.0, 1.0])
    lattice = _manual_lattice([cand])
    p_open = dec.decode_lattice(lattice, domain=OpenTextPrior(tuple("01")))
    p_lex = dec.decode_lattice(
        lattice,
        domain=LexiconPrior(counts={0: 1.0}, max_count=1.0, floor_log_p=-6.0),
    )
    assert p_open.text == p_lex.text == "0"
    assert p_open.prob_diagnostics["decomposition"]["domain"] == pytest.approx(
        math.log(0.5)
    )
    assert p_open.char_ids == p_lex.char_ids


def test_domain_prior_from_config_overrides():
    prior = domain_prior_from_config(
        {"type": "open_text"},
        list("012"),
        allowed_ids={0, 1},
        lexicon=None,
    )
    assert isinstance(prior, AllowedCharsPrior)
    assert prior.log_prior(0) == pytest.approx(math.log(0.5))
    assert prior.log_prior(2) == -np.inf
    open_prior = domain_prior_from_config(None, list("012"))
    assert isinstance(open_prior, OpenTextPrior)


def test_lexicon_prior_calibration_applied(v2_cnn_digits_model):
    """The stored monotone log-calibration is applied to support counts."""

    from fixedfontocr.prob_math import calibrate_log_monotone

    cal = [(-2.0, -1.5), (0.0, -0.2)]
    raw = math.log(0.5)
    calibrated = float(calibrate_log_monotone(np.asarray([raw]), cal)[0])
    assert calibrated > raw  # monotone map lifts mid support
    prior = LexiconPrior(
        counts={0: 5.0, 1: 5.0}, max_count=10.0, calibration=tuple(cal)
    )
    assert prior.log_prior(0) == pytest.approx(calibrated)
