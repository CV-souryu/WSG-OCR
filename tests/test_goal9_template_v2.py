"""Goal 9 acceptance: Template V2 multi-prototype matching.

The Goal 9 contract from ``fonts/goal``:

* one character no longer has a single template -- a prototype grid covers
  the real UI sizes 11..16 px, different sub-pixel phases and
  bilinear/area-like downsample degradation;
* matching keeps the geometry prefilter + XOR/popcount cascade;
* the matcher returns Top-K, best score, second score and margin without
  making the final character decision by itself (the decoder owns that);
* the low-resolution regressions (Z17 / 巴尔的摩) stay correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.dataset.generate_synthetic_samples import render  # noqa: E402

from fixedfontocr import FixedFontOCR, write_hybrid_model  # noqa: E402
from fixedfontocr.classifier import (  # noqa: E402
    TemplateClassifier,
    TemplateV2Classifier,
)
from fixedfontocr.fontgen import (  # noqa: E402
    DOWNSAMPLE_AREA,
    DOWNSAMPLE_BILINEAR,
    DOWNSAMPLE_CLEAN,
    TEMPLATE_V2_DOWNSAMPLE_MODES,
    TEMPLATE_V2_HIGH_RES_SIZE,
    TEMPLATE_V2_SIZES,
    TEMPLATE_V2_SUBPIXEL_PHASES,
    build_template_v2,
    build_templates,
    render_glyph,
    render_prototype,
    write_template_v2_model,
)
from fixedfontocr.model import load_model  # noqa: E402
from fixedfontocr.preprocess import (  # noqa: E402
    Component,
    compute_normalize_spec,
    glyph_normalize_geometry,
    normalize,
)
from fixedfontocr.scorer import SegmentScorer  # noqa: E402

from conftest import render_text  # noqa: E402


CHARSET = list("Z172小巴尔的摩")
REGRESSION_TEXT = ("Z17", "巴尔的摩")
QUERY_PHASES = ((0.0, 0.0), (0.3, 0.2), (0.7, 0.6))


def _normalize(mask, spec, target=24):
    h, w = mask.shape
    cand = Component(mask=mask, x=0, y=0, w=w, h=h)
    baseline_offset, scale = glyph_normalize_geometry(cand, spec, target)
    return normalize(
        mask,
        target,
        baseline_offset=baseline_offset,
        scale=scale,
        baseline_row=spec.baseline_row,
    )


def _build_v2(font_path, charset=None, **kwargs):
    charset = list(charset or CHARSET)
    chars, data = build_template_v2(font_path, charset, **kwargs)
    spec = compute_normalize_spec(font_path, chars, 24, TEMPLATE_V2_HIGH_RES_SIZE)
    return chars, data, TemplateV2Classifier(data, chars, normalize_spec=spec)


def _clean_queries(font_path, charset):
    spec = compute_normalize_spec(font_path, charset, 24, 32)
    glyphs, labels = [], []
    for ch in charset:
        mask = render_glyph(font_path, ch, 32)
        glyphs.append(_normalize(mask, spec))
        labels.append(ch)
    return np.stack(glyphs), labels, spec


def test_prototype_grid_covers_goal9_spec(font_path):
    chars, data, _ = _build_v2(font_path)

    expected = (
        len(TEMPLATE_V2_SIZES)
        * len(TEMPLATE_V2_SUBPIXEL_PHASES)
        * len(TEMPLATE_V2_DOWNSAMPLE_MODES)
        + 1  # clean high-resolution prototype
    )
    assert data.prototypes_per_char == expected
    assert set(TEMPLATE_V2_SIZES) == set(range(11, 17))
    assert set(TEMPLATE_V2_SUBPIXEL_PHASES) == {
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
    }
    assert TEMPLATE_V2_DOWNSAMPLE_MODES == ("bilinear", "area")

    sizes = {int(v) for v in data.render_sizes.reshape(-1)}
    assert sizes == set(TEMPLATE_V2_SIZES) | {TEMPLATE_V2_HIGH_RES_SIZE}
    assert set(int(v) for v in data.dx.reshape(-1)) == {0, 4}
    assert set(int(v) for v in data.dy.reshape(-1)) == {0, 4}
    assert set(int(v) for v in data.downsample_modes.reshape(-1)) == {
        DOWNSAMPLE_BILINEAR,
        DOWNSAMPLE_AREA,
        DOWNSAMPLE_CLEAN,
    }
    assert data.bits.shape == (
        len(chars),
        expected,
        (24 * 24 + 7) // 8,
    )


def test_prototypes_for_one_char_are_not_identical(font_path):
    chars, data, _ = _build_v2(font_path)
    for row in data.bits:
        unique = len(np.unique(row, axis=0))
        assert unique >= 5, f"character has only {unique} distinct prototypes"


def test_v2_model_roundtrip_and_v1_backward_compat(font_path, tmp_path):
    chars, data, _ = _build_v2(font_path)
    out = tmp_path / "v2"
    write_template_v2_model(out, chars, data, font_path=font_path)
    model = load_model(out)

    assert model.templates_v2 is not None
    assert model.templates is None
    assert model.config["template_version"] == 2
    assert model.classifier == "template"
    assert np.array_equal(model.templates_v2.bits, data.bits)
    assert np.array_equal(model.templates_v2.render_sizes, data.render_sizes)
    assert (out / "geometry.json").exists()

    # Legacy V1 files keep loading through the same loader.
    chars1, tpl1 = build_templates(font_path, chars, render_size=32)
    out1 = tmp_path / "v1"
    from fixedfontocr.fontgen import write_model

    write_model(out1, chars1, tpl1, font_path=font_path)
    model1 = load_model(out1)
    assert model1.templates is not None
    assert model1.templates_v2 is None
    assert model1.config.get("template_version", 1) == 1


def test_match_batch_returns_top_k_best_second_margin(font_path):
    chars, data, clf = _build_v2(font_path)
    glyphs, labels, _ = _clean_queries(font_path, chars)
    tb = clf.match_batch(glyphs, top_k=5)

    assert tb.top_k_ids.shape == (len(chars), 5)
    assert tb.top_k_scores.shape == (len(chars), 5)
    for i, label in enumerate(labels):
        assert chars[int(tb.ids[i])] == label
        assert int(tb.top_k_ids[i, 0]) == int(tb.ids[i])
        assert np.all(np.diff(tb.top_k_scores[i]) <= 1e-6)
        assert tb.top_k_scores[i, 0] == pytest.approx(float(tb.scores[i]), abs=1e-6)
        assert tb.top_k_scores[i, 1] == pytest.approx(
            float(tb.second_scores[i]), abs=1e-6
        )
        assert tb.margins[i] == pytest.approx(
            float(tb.scores[i] - tb.second_scores[i]), abs=1e-6
        )
        assert tb.margins[i] >= 0.0

    # The winning prototype's metadata must agree with the template set.
    for i in range(len(chars)):
        p = int(tb.best_prototypes[i])
        assert p >= 0
        assert tb.prototype_render_sizes[i] == data.render_sizes.reshape(-1)[p]
        assert tb.prototype_dx[i] == data.dx.reshape(-1)[p]
        assert tb.prototype_dy[i] == data.dy.reshape(-1)[p]
        assert (
            tb.prototype_downsample_modes[i]
            == data.downsample_modes.reshape(-1)[p]
        )


def test_geometry_prefilter_is_exact_for_top_k(font_path):
    chars, data, _ = _build_v2(font_path)
    filtered = TemplateV2Classifier(data, chars, candidate_filter=True)
    full = TemplateV2Classifier(data, chars, candidate_filter=False)
    glyphs, _, spec = _clean_queries(font_path, chars)
    # Make one query genuinely low-res (sub-pixel shifted 13 px) so the
    # prefilter has to work, not just score clean 32 px glyphs.
    low = render_prototype(font_path, "小", 13, 0.3, 0.2, "area")
    glyphs = np.concatenate([glyphs, _normalize(low, spec)[None]], axis=0)

    a = filtered.match_batch(glyphs, top_k=5)
    b = full.match_batch(glyphs, top_k=5)
    assert np.array_equal(a.ids, b.ids)
    assert np.array_equal(a.top_k_ids, b.top_k_ids)
    assert np.allclose(a.top_k_scores, b.top_k_scores, atol=0)
    assert np.allclose(a.margins, b.margins, atol=0)
    assert filtered.last_candidates is not None


def test_low_res_v2_beats_v1_and_selects_matching_size_prototype(font_path):
    # 未/末 is the classic low-res confusable pair: a single 32 px template
    # cannot separate their 11..13 px renders, but matching-size prototypes
    # can (Goal 9's core motivation).
    charset = list("Z172小巴尔的摩未末")
    chars, data, clf2 = _build_v2(font_path, charset=charset)
    spec = compute_normalize_spec(font_path, chars, 24, 32)
    chars1, tpl1 = build_templates(font_path, chars, render_size=32)
    clf1 = TemplateClassifier(tpl1, chars1, normalize_spec=spec)

    v2_ok = 0
    v2_top2 = 0
    v1_ok = 0
    total = 0
    size_offsets = []
    low_res_winners = 0
    correct_winners = 0
    for size in TEMPLATE_V2_SIZES:
        for ch in chars:
            for dx, dy in QUERY_PHASES:
                mask = render_prototype(font_path, ch, size, dx, dy, "bilinear")
                glyph = _normalize(mask, spec)
                r1 = clf1.match_batch(glyph[None])
                r2 = clf2.match_batch(glyph[None])
                v1_ok += int(chars1[int(r1.ids[0])] == ch)
                v2_ok += int(chars[int(r2.ids[0])] == ch)
                total += 1
                if chars[int(r2.ids[0])] == ch:
                    correct_winners += 1
                    if int(r2.prototype_render_sizes[0]) <= TEMPLATE_V2_SIZES[-1]:
                        low_res_winners += 1
                    size_offsets.append(
                        abs(int(r2.prototype_render_sizes[0]) - size)
                    )
                if ch in {chars[int(c)] for c in r2.top_k_ids[0, :2]}:
                    v2_top2 += 1

    # The matcher is not the final decision maker (Goal 9): an occasional
    # 11 px ambiguity is acceptable as long as the correct character stays
    # in Top-2/Top-K, which the decoder can still resolve.
    assert v2_top2 == total, f"V2 lost the label from Top-2 {total - v2_top2} times"
    assert v2_ok >= 0.95 * total, f"V2 top-1 accuracy too low: {v2_ok}/{total}"
    assert v1_ok < v2_ok, "V2 should beat a single 32 px template at low res"
    assert low_res_winners >= 0.9 * correct_winners
    assert np.median(size_offsets) <= 2


def test_template_only_v2_recognizes_low_res_regressions(font_path, tmp_path):
    charset = list("Z17巴尔的摩")
    chars, data = build_template_v2(font_path, charset)
    out = tmp_path / "v2_model"
    write_template_v2_model(out, chars, data, font_path=font_path)
    ocr = FixedFontOCR(model_path=out, backend="cpu")

    for text in REGRESSION_TEXT:
        # 11 px single-glyph matching is covered by the test above; full
        # lines need the Goal 4 segmentation lattice, so 12..16 px is the
        # template-only line acceptance here.
        for size in (12, 14, 16):
            image = render(font_path, text, size)
            result = ocr.recognize(image)
            assert result.text == text, (
                f"{text!r} at {size}px -> {result.text!r}"
            )


def test_hybrid_model_accepts_v2_templates(font_path, tmp_path):
    rng = np.random.default_rng(0)
    charset = list("0123456789")
    chars, data = build_template_v2(font_path, charset, render_sizes=(11, 16))
    weights = {
        "conv1.weight": rng.standard_normal((8, 1, 3, 3), dtype=np.float32),
        "conv1.bias": rng.standard_normal(8, dtype=np.float32),
        "dw1.weight": rng.standard_normal((8, 3, 3), dtype=np.float32),
        "dw1.bias": rng.standard_normal(8, dtype=np.float32),
        "pw1.weight": rng.standard_normal((16, 8), dtype=np.float32),
        "pw1.bias": rng.standard_normal(16, dtype=np.float32),
        "dw2.weight": rng.standard_normal((16, 3, 3), dtype=np.float32),
        "dw2.bias": rng.standard_normal(16, dtype=np.float32),
        "pw2.weight": rng.standard_normal((32, 16), dtype=np.float32),
        "pw2.bias": rng.standard_normal(32, dtype=np.float32),
        "fc.weight": rng.standard_normal((len(chars), 32), dtype=np.float32),
        "fc.bias": rng.standard_normal(len(chars), dtype=np.float32),
    }
    out = tmp_path / "hybrid_v2"
    write_hybrid_model(
        out,
        chars,
        templates=None,
        weights=weights,
        templates_v2=data,
        font_path=font_path,
    )
    model = load_model(out)
    assert model.classifier == "hybrid"
    assert model.templates_v2 is not None
    assert model.templates is None
    assert model.config["template_version"] == 2

    ocr = FixedFontOCR(model_path=out, backend="cpu")
    result = ocr.recognize(render_text("12345", font_path))
    assert result.text == "12345"


def test_exact_low_res_prototype_tie_goes_to_cnn(font_path, tmp_path):
    """'.' and '*' rasterize to the same blob at 11 px (Goal 7 domain).

    Template V2 must report the tie as two 1.0 scores with a zero margin
    (Top-K, no final decision), and the hybrid scorer must route such an
    exact-but-ambiguous low-res match to the CNN instead of trusting it.
    """

    charset = [".", "*"]
    chars, data = build_template_v2(font_path, charset, render_sizes=(11,))
    spec = compute_normalize_spec(font_path, chars, 24, 32)
    clf = TemplateV2Classifier(data, chars, normalize_spec=spec)

    # Find a low-res '*' prototype that is pixel-identical to a '.' one.
    identical = None
    for j in range(data.prototypes_per_char):
        for k in range(data.prototypes_per_char):
            if int(data.render_sizes[0, j]) == 11 and np.array_equal(
                data.bits[0, j], data.bits[1, k]
            ):
                identical = (j, k)
                break
        if identical:
            break
    assert identical is not None, "expected an identical 11px ./* prototype"

    # Reconstruct the query from the winning prototype's render parameters.
    _, k = identical
    mask = render_prototype(
        font_path,
        "*",
        11,
        float(data.dx[1, k]) / 8.0,
        float(data.dy[1, k]) / 8.0,
        "bilinear" if int(data.downsample_modes[1, k]) == 0 else "area",
    )
    glyph = _normalize(mask, spec)
    tb = clf.match_batch(glyph[None], top_k=2)
    assert set(int(c) for c in tb.top_k_ids[0]) == {0, 1}
    assert np.all(tb.top_k_scores[0] == 1.0)
    assert tb.margins[0] == 0.0

    # Hybrid scorer: random CNN weights still get the call because the
    # exact low-res tie is not trusted as a final template decision.
    rng = np.random.default_rng(0)
    weights = {
        "conv1.weight": rng.standard_normal((8, 1, 3, 3), dtype=np.float32),
        "conv1.bias": rng.standard_normal(8, dtype=np.float32),
        "dw1.weight": rng.standard_normal((8, 3, 3), dtype=np.float32),
        "dw1.bias": rng.standard_normal(8, dtype=np.float32),
        "pw1.weight": rng.standard_normal((16, 8), dtype=np.float32),
        "pw1.bias": rng.standard_normal(16, dtype=np.float32),
        "dw2.weight": rng.standard_normal((16, 3, 3), dtype=np.float32),
        "dw2.bias": rng.standard_normal(16, dtype=np.float32),
        "pw2.weight": rng.standard_normal((32, 16), dtype=np.float32),
        "pw2.bias": rng.standard_normal(32, dtype=np.float32),
        "fc.weight": rng.standard_normal((len(chars), 32), dtype=np.float32),
        "fc.bias": rng.standard_normal(len(chars), dtype=np.float32),
    }
    out = tmp_path / "hybrid_tie"
    write_hybrid_model(
        out,
        chars,
        templates=None,
        weights=weights,
        templates_v2=data,
        font_path=font_path,
    )
    scorer = SegmentScorer(load_model(out))
    h, w = mask.shape
    seg = Component(mask=mask, x=0, y=0, w=w, h=h)
    scores = scorer.score([seg])
    assert len(scores) == 1
    assert scores[0].score_type == "cnn"


def test_bundled_template_model_is_goal9_v2():
    model_dir = Path("model/game_cn_template")
    if not (model_dir / "config.json").exists():
        pytest.skip("model/game_cn_template not built; run tools/train/build_model.py")
    model = load_model(model_dir)
    assert model.templates_v2 is not None
    assert model.templates is None
    assert model.config.get("template_version") == 2
    assert model.templates_v2.prototypes_per_char >= 6 * 4 * 2
    sizes = {int(s) for s in model.templates_v2.render_sizes.reshape(-1)}
    assert set(range(11, 17)) <= sizes
