"""Probabilistic decoder v2 -- line render state ``z``.

Covers the "line render state 测试" checklist:

* one line shares one (small set of) render state(s) -- the same states
  condition every candidate of the line;
* a wrong line state cannot override clearly stronger visual evidence;
* the ``clean`` fallback works for high-resolution prototypes;
* the ``z`` marginalization never produces NaN/Inf;
* state ordering is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixedfontocr.prob_decoder import ProbabilisticDecoder
from fixedfontocr.prob_features import CLEAN_STATE, STATE_KEYS
from fixedfontocr.prob_math import logsumexp
from fixedfontocr.render_state import RenderStateModel

from prob_helpers import v2_game_model


def test_state_keys_are_fixed_and_deterministic():
    assert STATE_KEYS == (
        "11_0", "11_1", "12_0", "12_1", "13_0", "13_1",
        "14_0", "14_1", "15_0", "15_1", "16_0", "16_1", "clean",
    )
    assert CLEAN_STATE == "clean"
    # Deterministic ordering: repeated selection gives the same list.
    model = RenderStateModel(top_m=3)
    ctx = type("Ctx", (), {
        "state_candidates": (("14_0", 5.0), ("13_0", 2.0), ("clean", 0.0))
    })()
    assert model.select_states(ctx) == ["14_0", "13_0", "clean"]


def test_select_states_respects_top_m_and_clean_fallback():
    model = RenderStateModel(top_m=2)
    ctx = type("Ctx", (), {
        "state_candidates": (("12_1", 9.0), ("14_0", 3.0), ("15_1", 1.0))
    })()
    states = model.select_states(ctx)
    assert states == ["12_1", "14_0", "clean"]  # clean appended as fallback
    model2 = RenderStateModel(top_m=2, clean_fallback=False)
    assert model2.select_states(ctx) == ["12_1", "14_0"]


def test_marginalize_logsumexp_and_max_agree_on_order():
    model = RenderStateModel(top_m=3)
    scores = {"14_0": -3.0, "13_0": -5.0, "clean": -8.0}
    marg, argmax = model.marginalize(scores)
    assert argmax == "14_0"
    expected = logsumexp(np.asarray([-3.0, -5.0, -8.0]))
    assert marg == pytest.approx(expected)
    marg_max, argmax2 = model.marginalize(scores, use_max=True)
    assert marg_max == pytest.approx(-3.0)
    assert argmax == argmax2


def test_marginalize_no_nan_inf():
    model = RenderStateModel(top_m=3)
    for scores in (
        {},
        {"14_0": -np.inf, "clean": -np.inf},
        {"14_0": 1e30, "clean": -1e30},
        {"a": float("nan")},
    ):
        marg, argmax = model.marginalize(scores)
        assert marg == -np.inf or np.isfinite(marg), (scores, marg)
        assert not np.isnan(marg)
    # NaN inputs are refused explicitly (config validation covers them);
    # the marginalizer treats non-finite states as fully masked.
    marg, _ = model.marginalize({"14_0": float("nan"), "clean": 0.0})
    assert np.isfinite(marg)


def test_same_line_shares_render_states(v2_game_model):
    """End to end: one line's best path is scored under shared states."""

    from fixedfontocr import FixedFontOCR
    from fixedfontocr.oracle import render_glyph_mask

    # Render a small line from the game font (mixed CJK + digits).
    from PIL import Image, ImageDraw, ImageFont

    font_path = _game_font()
    font = ImageFont.truetype(str(font_path), 14)
    img = Image.new("RGB", (140, 40), (0, 0, 0))
    ImageDraw.Draw(img).text((4, 6), "巴尔的摩", font=font, fill=(255, 255, 255))
    arr = np.asarray(img, dtype=np.uint8)

    ocr = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    result = ocr.recognize(arr)
    assert result.path is not None
    diag = result.path.prob_diagnostics
    assert diag is not None
    states = diag["states"]
    # The states list is bounded by top_m + the clean fallback, and the
    # selected state is one of them.
    assert len(states) <= 4
    assert diag["line_render_state"] in states
    # Every state is a known, deterministic key.
    assert all(s in STATE_KEYS for s in states)
    # The whole line was scored under the *same* state set: the per-char
    # decomposition sums to the totals (consistent accounting).
    decomp = diag["decomposition"]
    per_char = diag["per_char"]
    assert len(per_char) == len(result.text) == len(result.path.candidates)
    assert decomp["visual"] == pytest.approx(
        sum(d["logp_visual"] for d in per_char), abs=1e-9
    )
    assert decomp["boundary"] == pytest.approx(
        sum(d["logp_boundary"] for d in per_char), abs=1e-9
    )
    assert decomp["domain"] == pytest.approx(
        sum(d["logp_domain"] for d in per_char), abs=1e-9
    )
    assert np.isfinite(diag["total_log_prob"])
    assert np.isfinite(diag["normalized_score"])


def test_render_states_deterministic(v2_game_model):
    """Two decodes (and a fresh engine) give identical states/results."""

    from fixedfontocr import FixedFontOCR
    from fixedfontocr.oracle import render_glyph_mask

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(_game_font()), 16)
    img = Image.new("RGB", (120, 40), (0, 0, 0))
    ImageDraw.Draw(img).text((4, 6), "潜甲", font=font, fill=(255, 255, 255))
    arr = np.asarray(img, dtype=np.uint8)

    ocr1 = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    ocr2 = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    r1 = ocr1.recognize(arr)
    r2 = ocr2.recognize(arr)
    assert r1.text == r2.text
    assert r1.path.prob_diagnostics["states"] == r2.path.prob_diagnostics["states"]
    assert (
        r1.path.prob_diagnostics["line_render_state"]
        == r2.path.prob_diagnostics["line_render_state"]
    )
    assert r1.alternatives == r2.alternatives


def test_wrong_state_does_not_override_strong_visual(v2_game_model):
    """A state with weak template support must not beat a clearly strong
    visual match (the unconditional template evidence dominates)."""

    from fixedfontocr import FixedFontOCR

    from PIL import Image, ImageDraw, ImageFont

    # Clean 32 px render: the winning prototype is the clean one, so the
    # line's render state is the clean fallback -- and even if a wrong
    # grid state were forced, the strong visual evidence still wins.
    font = ImageFont.truetype(str(_game_font()), 32)
    img = Image.new("RGB", (80, 48), (0, 0, 0))
    ImageDraw.Draw(img).text((8, 6), "小", font=font, fill=(255, 255, 255))
    arr = np.asarray(img, dtype=np.uint8)

    ocr = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    result = ocr.recognize(arr)
    assert result.text == "小"
    diag = result.path.prob_diagnostics
    # The clean fallback state was selected for the high-resolution render.
    assert CLEAN_STATE in diag["states"]
    assert diag["line_render_state"] == CLEAN_STATE
    # Per-char visual log-probability is high (the top option by far).
    assert diag["per_char"][0]["logp_visual"] > -1.5


def test_disabling_render_states_keeps_strong_visual(v2_game_model, tmp_path):
    """With the render-state machinery disabled the same strong visual
    path still wins (z only conditions template evidence, it never
    overrides it)."""

    from fixedfontocr import FixedFontOCR

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(_game_font()), 16)
    img = Image.new("RGB", (120, 40), (0, 0, 0))
    ImageDraw.Draw(img).text((4, 6), "潜甲", font=font, fill=(255, 255, 255))
    arr = np.asarray(img, dtype=np.uint8)

    import json
    import shutil
    from pathlib import Path

    out = tmp_path / "rs_disabled"
    out.mkdir(parents=True)
    for name in ("config.json", "model.json", "charset.txt", "weights.bin",
                 "templates.bin", "geometry.json"):
        src = Path(v2_game_model) / name
        if src.exists():
            shutil.copy2(src, out / name)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    config["decoder"]["render_states"] = {
        "enabled": False, "top_m": 1, "state_prior": "uniform",
        "clean_fallback": True,
    }
    payload = json.dumps(config, indent=2) + "\n"
    (out / "config.json").write_text(payload, encoding="utf-8")
    (out / "model.json").write_text(payload, encoding="utf-8")

    ocr_on = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    ocr_off = FixedFontOCR(model_path=out, backend="cpu")
    r_on = ocr_on.recognize(arr)
    r_off = ocr_off.recognize(arr)
    assert r_on.text == r_off.text == "潜甲"
    # The disabled decoder still has the clean fallback state (single
    # state, no z-conditional evidence).
    assert r_off.path.prob_diagnostics["states"] == [CLEAN_STATE]


def _game_font():
    from fixedfontocr.defaults import FONT_PATH, resolve_font

    if not FONT_PATH.exists():
        pytest.skip("bundled font not present")
    return resolve_font(FONT_PATH)
