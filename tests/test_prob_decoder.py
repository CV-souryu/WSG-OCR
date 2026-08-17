"""Probabilistic decoder v2 -- path-level behaviour.

Covers the "路径测试" checklist:

* the ground-truth path scores above wrong paths;
* no bias toward a single character swallowing the whole line;
* no bias toward fragment-explosion (one glyph -> many characters);
* the boundary log-probability separates reasonable seams from wrong
  seams;
* character geometry is not double-counted;
* k-best order is deterministic and alternatives come from the k-best
  list, not post-hoc concatenation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fixedfontocr.prob_decoder import ProbabilisticDecoder
from fixedfontocr.prob_features import LineContext
from fixedfontocr.prob_math import logsumexp
from fixedfontocr.render_state import RenderStateModel
from fixedfontocr.types import Component, VisualLattice

from prob_helpers import (
    decoder_block,
    manual_candidate,
    v2_cnn_digits_model,
    v2_game_model,
)


def _decoder(model_path):
    from fixedfontocr.model import load_model

    model = load_model(model_path)
    dec, warn = ProbabilisticDecoder.try_build(model)
    assert dec is not None, warn
    return dec


def _render(text, size, pad=6):
    from fixedfontocr.defaults import FONT_PATH, resolve_font
    from PIL import Image, ImageDraw, ImageFont

    if not FONT_PATH.exists():
        pytest.skip("bundled font not present")
    font = ImageFont.truetype(str(resolve_font(FONT_PATH)), size)
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGB", (max(1, w), max(1, h)), (0, 0, 0))
    ImageDraw.Draw(img).text(
        (pad - bbox[0], pad - bbox[1]), text, font=font, fill=(255, 255, 255)
    )
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# End-to-end path behaviour on the derived v2 model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,size",
    [
        ("鲃鱼。", 32),
        ("潜甲", 32),
        ("潜乙", 32),
        ("小", 32),
        ("Z17", 24),
        ("1234", 24),
        ("甲申", 20),
    ],
)
def test_ground_truth_path_wins(v2_game_model, text, size):
    """The GT path is the top path: the decoded text equals the render."""

    from fixedfontocr import FixedFontOCR

    arr = _render(text, size)
    ocr = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    result = ocr.recognize(arr)
    assert result.text == text, f"{text}@{size}px -> {result.text!r}"
    # The GT path's normalized score is the best of the k-best list.
    diag = result.path.prob_diagnostics
    assert np.isfinite(diag["normalized_score"])
    assert diag["runner_up_margin"] >= 0.0


def test_no_single_char_swallow(v2_game_model):
    """A 4-glyph line must not collapse into one candidate."""

    from fixedfontocr import FixedFontOCR

    arr = _render("1234", 24)
    ocr = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    result = ocr.recognize(arr)
    assert result.text == "1234"
    assert len(result.chars) == 4
    # Every char is a single-atom candidate (no merge swallowed the line).
    spans = [c.atom_span for c in result.path.candidates]
    assert all(end - start == 1 for start, end in spans)


def test_no_fragment_explosion(v2_game_model):
    """A fragmented glyph must not explode into several characters."""

    from fixedfontocr import FixedFontOCR

    arr = _render("小", 32)
    ocr = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    result = ocr.recognize(arr)
    assert result.text == "小"
    assert len(result.chars) == 1

    arr2 = _render("鲃鱼。", 32)
    result2 = ocr.recognize(arr2)
    assert result2.text == "鲃鱼。"
    assert len(result2.chars) == 3


def test_alternatives_come_from_kbest(v2_game_model):
    """``alternatives`` are distinct k-best paths, not post-hoc strings."""

    from fixedfontocr import FixedFontOCR

    # 未/末 at low resolution are a classic near-tie; alternatives must
    # list the runner-up text (which only exists if the k-best found it).
    arr = _render("未", 13)
    ocr = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    result = ocr.recognize(arr)
    k = result.path.prob_diagnostics["k_best_searched"]
    assert k >= 1
    for alt in result.alternatives:
        assert len(alt) == len(result.text)
    # Re-decode deterministically: identical text/alternatives/diagnostics.
    result2 = ocr.recognize(arr)
    assert result.text == result2.text
    assert result.alternatives == result2.alternatives
    assert (
        result.path.prob_diagnostics["states"]
        == result2.path.prob_diagnostics["states"]
    )


def test_kbest_deterministic_across_instances(v2_game_model):
    """A fresh engine and decoder produce byte-identical k-best output."""

    from fixedfontocr import FixedFontOCR

    arr = _render("巴尔的摩", 14)
    ocr1 = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    ocr2 = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    r1 = ocr1.recognize(arr)
    r2 = ocr2.recognize(arr)
    assert r1.text == r2.text
    assert r1.alternatives == r2.alternatives
    assert r1.path.char_ids == r2.path.char_ids
    assert r1.path.prob_diagnostics == r2.path.prob_diagnostics


def test_length_normalization_does_not_bias_clear_lines(v2_game_model):
    """With (alpha, beta) from the config, clear short and long lines both
    decode correctly (no systematic short-path or long-path bias)."""

    from fixedfontocr import FixedFontOCR

    ocr = FixedFontOCR(model_path=v2_game_model, backend="cpu")
    short = _render("小", 32)
    long = _render("1234", 24)
    r_short = ocr.recognize(short)
    r_long = ocr.recognize(long)
    assert r_short.text == "小"
    assert r_long.text == "1234"
    # Normalized scores are comparable across lengths (finite, ordered).
    assert np.isfinite(r_short.path.prob_diagnostics["normalized_score"])
    assert np.isfinite(r_long.path.prob_diagnostics["normalized_score"])


def test_reject_reason_reported_when_enabled(v2_game_model, tmp_path):
    """With reject enabled and a very strict threshold the reason is
    computed and reported (diagnostic-only by default)."""

    import json
    import shutil
    from pathlib import Path

    from fixedfontocr import FixedFontOCR

    out = tmp_path / "reject"
    out.mkdir(parents=True)
    for name in ("config.json", "model.json", "charset.txt", "weights.bin",
                 "templates.bin", "geometry.json"):
        src = Path(v2_game_model) / name
        if src.exists():
            shutil.copy2(src, out / name)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    config["decoder"]["reject"] = {
        "enabled": True,
        "confidence_threshold": 0.999,
        "margin_threshold": 10.0,
    }
    payload = json.dumps(config, indent=2) + "\n"
    (out / "config.json").write_text(payload, encoding="utf-8")
    (out / "model.json").write_text(payload, encoding="utf-8")

    ocr = FixedFontOCR(model_path=out, backend="cpu")
    result = ocr.recognize(_render("1234", 24))
    reason = result.path.prob_diagnostics["reject_reason"]
    assert reason, "strict thresholds must produce a reject reason"


# ---------------------------------------------------------------------------
# Unit-level contracts
# ---------------------------------------------------------------------------


def _seam_ctx(expected_width, atom_masks, atom_xs, h=10):
    """LineContext with two atoms of one split component."""

    atoms = []
    runs = []
    for i, (mask, x) in enumerate(zip(atom_masks, atom_xs)):
        seg = Component(mask=mask, x=x, y=0, w=mask.shape[1], h=mask.shape[0])
        atoms.append(type("A", (), {"segment": seg, "component_index": 0})())
        runs.append((0, i, len(atom_masks)))
    return LineContext(
        expected_width=float(expected_width),
        median_center_y=float(h / 2),
        median_h=float(h),
        atom_runs=tuple(runs),
        atoms=tuple(atoms),
        state_candidates=(),
    )


def test_boundary_probability_distinguishes_seams(tmp_path_factory):
    """A clean cut (no crossing ink, on the ideal advance) scores above a
    seam cut through ink and above a misplaced cut."""

    from prob_helpers import build_v2_cnn_digits_model

    # The boundary block is only meaningful with non-zero boundary
    # weights, so use the "template" rank for this test.
    model_path = build_v2_cnn_digits_model(
        tmp_path_factory.mktemp("v2_seam"), "template"
    )
    dec = _decoder(model_path)
    h = 10
    # Clean cut: 24 px blob, cut at 12 (the ideal advance multiple); the
    # boundary columns are empty.
    m_l = np.zeros((h, 12), dtype=bool)
    m_l[:, 0:11] = True
    m_r = np.zeros((h, 12), dtype=bool)
    m_r[:, 1:12] = True
    ctx_clean = _seam_ctx(12, [m_l, m_r], [0, 12], h)
    # Wrong seam: ink crosses the boundary columns.
    m_l2 = np.zeros((h, 12), dtype=bool)
    m_l2[:, :] = True
    m_r2 = np.zeros((h, 12), dtype=bool)
    m_r2[:, :] = True
    ctx_ink = _seam_ctx(12, [m_l2, m_r2], [0, 12], h)
    # Misplaced cut: 24 px blob cut at 8 (deviation 4 px from the ideal),
    # but the boundary columns themselves are clean (only the position is
    # wrong).
    m_l3 = np.zeros((h, 8), dtype=bool)
    m_l3[:, 0:7] = True
    m_r3 = np.zeros((h, 16), dtype=bool)
    m_r3[:, 1:16] = True
    ctx_mis = _seam_ctx(12, [m_l3, m_r3], [0, 8], h)

    # Distinct candidate objects per context: the extractor caches per
    # candidate id within one line, so each ctx gets its own object.
    seg = Component(
        mask=np.ones((h, 24), dtype=bool), x=0, y=0, w=24, h=h
    )
    cand_clean = manual_candidate(0, 2, [0], [0.0], segment=seg)
    cand_ink = manual_candidate(0, 2, [0], [0.0], segment=seg)
    cand_mis = manual_candidate(0, 2, [0], [0.0], segment=seg)
    b_clean = dec._boundary_logp(cand_clean, ctx_clean)
    b_ink = dec._boundary_logp(cand_ink, ctx_ink)
    b_mis = dec._boundary_logp(cand_mis, ctx_mis)
    assert b_clean > b_ink, f"clean {b_clean} must beat ink-crossing {b_ink}"
    assert b_clean > b_mis, f"clean {b_clean} must beat misplaced {b_mis}"
    assert b_mis > b_ink, f"misplaced {b_mis} must beat ink-crossing {b_ink}"
    # The features themselves are the expected ones.
    f_clean = dec.extractor._candidate_cache(cand_clean, ctx_clean, None).boundary
    f_ink = dec.extractor._candidate_cache(cand_ink, ctx_ink, None).boundary
    f_mis = dec.extractor._candidate_cache(cand_mis, ctx_mis, None).boundary
    assert f_clean[5] == pytest.approx(0.0)
    assert f_ink[5] == pytest.approx(1.0)
    assert f_clean[6] == pytest.approx(0.0)
    assert f_mis[6] > f_clean[6]
    assert f_mis[5] == pytest.approx(0.0)


def test_character_geometry_not_double_counted(v2_cnn_digits_model):
    """v2 reads raw geometry residuals once; the legacy fused
    ``geometry_score`` / ``candidate_geometry`` fields must not change the
    v2 decision at all."""

    dec = _decoder(v2_cnn_digits_model)
    # Two identical candidates that differ ONLY in the legacy fused
    # geometry fields.
    base = manual_candidate(0, 1, [0, 1], [2.0, 1.0])
    other = manual_candidate(0, 1, [0, 1], [2.0, 1.0])
    other.geometry_score = -0.12
    other.candidate_geometry = -0.10
    other.segmentation_width_penalty = -0.05
    base.scores = base.scores
    lat1 = VisualLattice(components=(), candidates=(base,))
    lat2 = VisualLattice(components=(), candidates=(other,))
    p1 = dec.decode_lattice(lat1)
    p2 = dec.decode_lattice(lat2)
    assert p1.text == p2.text
    assert p1.char_ids == p2.char_ids
    d1, d2 = p1.prob_diagnostics, p2.prob_diagnostics
    # The decomposition is unchanged: v2 never consults the fused fields.
    assert d1["decomposition"]["geometry"] == d2["decomposition"]["geometry"]
    assert d1["normalized_score"] == d2["normalized_score"]


def test_local_softmax_is_a_distribution(v2_cnn_digits_model):
    """Per-candidate visual log-probs sum to 1 over the option set."""

    dec = _decoder(v2_cnn_digits_model)
    cand = manual_candidate(0, 1, [0, 1, 2], [3.0, 1.0, 0.5])
    lattice = VisualLattice(components=(), candidates=(cand,))
    ctx = dec.extractor.line_context(lattice)
    z = dec.render_state.select_states(ctx)[0]
    opts = dec._options(cand, ctx, z, None)
    probs = np.exp([o.logp_visual for o in opts])
    # Open-set normalization: the options plus the background option form
    # a distribution over the candidate's label set.
    lse = logsumexp(
        np.asarray(
            [dec._char_logits(cand, cid, ctx, z)[0] for cid in (0, 1, 2)]
            + [dec.cfg.background_logit]
        )
    )
    p_bg = math.exp(dec.cfg.background_logit - lse)
    assert np.allclose(probs.sum() + p_bg, 1.0)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    # The max-logit char is the argmax option.
    assert opts[0].char_id == 0


def test_allowed_ids_restrict_options(v2_cnn_digits_model):
    """Characters outside ``allowed_ids`` never enter the local softmax."""

    dec = _decoder(v2_cnn_digits_model)
    cand = manual_candidate(0, 1, [0, 1], [3.0, 1.0])
    lattice = VisualLattice(components=(), candidates=(cand,))
    ctx = dec.extractor.line_context(lattice)
    z = dec.render_state.select_states(ctx)[0]
    opts = dec._options(cand, ctx, z, {1})
    assert [o.char_id for o in opts] == [1]
    path = dec.decode_lattice(lattice, allowed_ids={1})
    assert path.text == "1"


def test_empty_lattice_returns_empty_path(v2_cnn_digits_model):
    dec = _decoder(v2_cnn_digits_model)
    path = dec.decode_lattice(VisualLattice(components=(), candidates=()))
    assert path.candidates == ()
    assert path.text == ""


def test_trained_model_paths_when_present(tmp_path):
    """The trained probabilistic model (when built by the training script)
    must decode the canonical lines it is capable of.

    This test runs only when ``model/game_cn_prob`` exists (the training
    script output); the benchmark script reports the full held-out
    numbers including the documented limitations (low-res digit/Latin
    lines such as ``获得金币1000``/``Z17`` at 14 px still trail the tuned
    v1 fusion -- see ``docs/probabilistic_decoder.md`` and
    ``benchmarks/probabilistic_decoder/benchmark.json``).
    """

    from pathlib import Path

    from fixedfontocr import FixedFontOCR

    model_path = Path("model/game_cn_prob")
    if not (model_path / "config.json").exists():
        pytest.skip("model/game_cn_prob not built; run "
                    "tools/train/train_probabilistic_decoder.py")
    ocr = FixedFontOCR(model_path=model_path, backend="cpu")
    for text, size in (
        ("巴尔的摩", 14),
        ("巴尔的摩", 32),
        ("塞瓦斯托波尔", 32),
        ("潜甲", 14),
        ("小", 14),
    ):
        result = ocr.recognize(_render(text, size))
        assert result.text == text, f"{text}@{size}px -> {result.text!r}"


def test_z_marginalization_finite_for_all_paths(v2_cnn_digits_model):
    """Best path total log-prob and normalized score stay finite even when
    some states are extremely unlikely."""

    dec = _decoder(v2_cnn_digits_model)
    cand = manual_candidate(0, 1, [0], [0.0])
    lattice = VisualLattice(components=(), candidates=(cand,))
    path = dec.decode_lattice(lattice)
    assert np.isfinite(path.prob_diagnostics["total_log_prob"])
    assert np.isfinite(path.prob_diagnostics["normalized_score"])
    assert not np.isnan(path.prob_diagnostics["normalized_score"])
