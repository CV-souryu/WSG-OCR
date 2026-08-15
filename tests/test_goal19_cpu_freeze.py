"""Goal 19: CPU Freeze gate.

``fonts/goal`` Goal 19 certifies the numpy CPU pipeline as the canonical
implementation before any formal WGPU work. This file pins every freeze
condition so the freeze cannot silently regress:

* segmentation lattice stability (fragmented glyphs are recoverable);
* 鲃/小, low-res Z17 and 巴尔的摩, mixed charset;
* the Top-K API (ranked Top-K, never a bare Top-1);
* the lexicon decoder and partial-word output;
* the complete CPU benchmark JSON;
* the complete regression dataset manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fixedfontocr import FixedFontOCR
from fixedfontocr.backends import Backend
from fixedfontocr.cnn import TinyCNNClassifier, forward
from fixedfontocr.defaults import (
    CPU_FREEZE,
    CPU_FREEZE_CONDITIONS,
    CPU_FREEZE_DATE,
    CPU_FREEZE_VERSION,
    MODEL_PATH,
)
from fixedfontocr.frontend import extract_frontend
from fixedfontocr.model import load_model
from fixedfontocr.postprocess import topk
from fixedfontocr.preprocess import find_lines
from fixedfontocr.scorer import SegmentScorer
from fixedfontocr.segmentation import build_candidates, connected_components
from fixedfontocr.types import Component, profile_from_dict


ROOT = Path(__file__).resolve().parents[1]
GAME_SAMPLES = ROOT / "tests" / "game_samples"
MANIFEST = GAME_SAMPLES / "manifest.json"
BENCHMARK = ROOT / "benchmarks" / "cpu_benchmark.json"

# The ten Goal 19 checkboxes, one per condition name in defaults.py.
EXPECTED_FREEZE_CONDITIONS = {
    "segmentation_lattice_stable",
    "fragmented_glyphs_solved",
    "low_res_z17_stable",
    "baltimore_stable",
    "mixed_charset_stable",
    "topk_api_stable",
    "lexicon_decoder_stable",
    "partial_word_stable",
    "cpu_benchmark_complete",
    "regression_dataset_complete",
}

REQUIRED_CATEGORIES = {
    "synthetic/normal-chinese",
    "synthetic/digits",
    "synthetic/mixed-cn-ascii",
    "synthetic/punctuation",
    "synthetic/fragment-glyphs",
    "synthetic/confusable",
    "synthetic/light-background",
    "synthetic/dark-background",
    "synthetic/antialiasing",
    "synthetic/font-size",
    "synthetic/goal14-fragments",
    "synthetic/goal14-lowres",
    "synthetic/goal14-mixed",
    "synthetic/goal14-digits",
    "synthetic/goal14-words",
    "synthetic/goal14-partial",
    "real-game/level",
    "real-game/ship-name",
}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sample(rel: str) -> dict:
    for sample in _manifest()["samples"]:
        if sample["file"] == rel:
            return sample
    raise AssertionError(f"sample {rel!r} is not in the regression manifest")


def _image(rel: str) -> np.ndarray:
    return np.asarray(Image.open(GAME_SAMPLES / rel).convert("RGB"), dtype=np.uint8)


def _profile(sample: dict):
    base = dict(_manifest()["default_profile"])
    base.update(sample.get("profile", {}))
    return profile_from_dict(base)


@pytest.fixture(scope="module")
def make_ocr():
    if not (MODEL_PATH / "config.json").exists():
        pytest.skip("model/game_cn not built")

    def _make(sample: dict) -> FixedFontOCR:
        ocr = FixedFontOCR(model_path=MODEL_PATH, backend="cpu")
        ocr.profile = _profile(sample)
        return ocr

    return _make


def test_freeze_marker_matches_goal19_checklist():
    assert CPU_FREEZE is True
    assert CPU_FREEZE_VERSION == "1.0"
    assert CPU_FREEZE_DATE == "2026-08-15"
    assert set(CPU_FREEZE_CONDITIONS) == EXPECTED_FREEZE_CONDITIONS


def test_segmentation_lattice_stable():
    """Original components survive and merged hypotheses are added, so no
    irreversible merge decision exists in the lattice (Goal 4/19)."""

    for rel, min_components in (
        ("synthetic/frag_xiao_14.png", 3),
        ("synthetic/frag_ba_yu_14.png", 5),
    ):
        sample = _sample(rel)
        frontend = extract_frontend(_image(rel), _profile(sample))
        line = find_lines(frontend.binary_mask, _profile(sample))[0]
        comps = connected_components(line)
        assert len(comps) >= min_components, rel
        candidates = build_candidates(
            comps, _profile(sample), max_merge_components=4, geometry=None
        )
        spans = {c.atom_span for c in candidates}
        n = len(comps)
        for i in range(n):
            assert (i, i + 1) in spans, f"{rel}: component {i} not preserved"
        longest = max(end - start for start, end in spans)
        assert longest >= 3, f"{rel}: no merge hypothesis for a fragmented glyph"


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("synthetic/frag_ba_yu_14.png", "鲃鱼"),
        ("synthetic/frag_ba_yu_32.png", "鲃鱼"),
        ("synthetic/frag_xiao_14.png", "小"),
        ("synthetic/frag_xiao_32.png", "小"),
    ],
)
def test_fragmented_glyphs_solved(make_ocr, rel, expected):
    """鲃 must not decode as $E and 小 must not fragment into several chars."""

    result = make_ocr(_sample(rel)).recognize(_image(rel))
    assert result.text == expected, f"{rel}: got {result.text!r}"
    assert len(result.chars) == len(expected), f"{rel}: fragmented decode"


def test_low_res_z17_stable(make_ocr):
    result = make_ocr(_sample("synthetic/z17_14.png")).recognize(
        _image("synthetic/z17_14.png")
    )
    assert result.text == "Z17"


def test_baltimore_stable(make_ocr):
    sample = _sample("synthetic/baltimore_14.png")
    result = make_ocr(sample).recognize(
        _image(sample["file"]),
        lexicon=sample["lexicon"],
        lexicon_mode="prefer",
    )
    assert result.text == "巴尔的摩"


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("synthetic/mixed_16.png", "舰船Lv.99"),
        ("synthetic/digits_16.png", "1234567890"),
        ("synthetic/punct_32.png", "。，！？"),
    ],
)
def test_mixed_charset_stable(make_ocr, rel, expected):
    result = make_ocr(_sample(rel)).recognize(_image(rel))
    assert result.text == expected, f"{rel}: got {result.text!r}"


def test_topk_api_stable():
    """Top-K is ranked via argpartition and never loses alternatives."""

    model = load_model(MODEL_PATH)
    classifier = TinyCNNClassifier(model.weights, model.charset)
    rng = np.random.default_rng(19)
    glyphs = (rng.random((8, 24, 24)) > 0.35).astype(np.uint8) * 255
    x = glyphs.astype(np.float32)[:, None, :, :] / 255.0
    logits = forward(x, model.weights)
    for k in (1, 2, 3, 5):
        reference = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
        ids, values = topk(logits, k)
        assert np.array_equal(ids, reference), f"k={k} ranked ids mismatch"
        assert np.array_equal(
            values, np.take_along_axis(logits, reference, axis=-1)
        ), f"k={k} ranked logits mismatch"
        batch = classifier.classify_batch(glyphs, top_k=k)
        assert np.array_equal(batch.ids, reference[:, 0]), f"k={k} top-1 mismatch"
        if k > 2:
            assert batch.topk_ids is not None
            assert np.array_equal(batch.topk_ids, reference), f"k={k} Top-K mismatch"


def test_decoded_candidates_expose_topk(make_ocr):
    """The public decode path keeps ranked Top-K on every candidate."""

    rel = "synthetic/mixed_16.png"
    result = make_ocr(_sample(rel)).recognize(_image(rel))
    assert result.text == "舰船Lv.99"
    assert result.path is not None
    scored = [c.scores for c in result.path.candidates]
    assert scored and all(
        scores is not None and len(scores.top_k) >= 2 for scores in scored
    )


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ("synthetic/qianjia_14.png", "潜甲"),
        ("synthetic/qianyi_14.png", "潜乙"),
        ("synthetic/long_word_16.png", "塞瓦斯托波尔"),
    ],
)
def test_lexicon_decoder_stable(make_ocr, rel, expected):
    sample = _sample(rel)
    result = make_ocr(sample).recognize(
        _image(rel),
        lexicon=sample["lexicon"],
        lexicon_mode=sample.get("lexicon_mode", "prefer"),
    )
    assert result.text == expected, f"{rel}: got {result.text!r}"


def test_partial_word_stable(make_ocr):
    sample = _sample("synthetic/partial_word_14.png")
    result = make_ocr(sample).recognize(
        _image(sample["file"]),
        lexicon=sample["lexicon"],
        lexicon_mode="prefer",
    )
    assert result.text == "尔的摩"
    assert result.matched_term == "巴尔的摩"
    assert result.matched_span == (1, 4)


def test_cpu_benchmark_complete():
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert set(data["ocr_stages"]) == {
        "foreground",
        "cc",
        "lattice_generation",
        "normalize",
        "template",
        "tinycnn",
        "decoder",
        "total",
    }
    for stage in data["ocr_stages"].values():
        assert stage["median"] > 0
        assert stage["p95"] >= stage["median"]
    assert set(data["cnn_batch"]) == {"1", "8", "16", "32", "64", "128"}
    assert set(data["charsets"]) == {"10", "100", "1894"}
    assert data["optimized_vs_reference"]["argmax_match"] is True
    assert data["optimized_vs_reference"]["max_error"] < 1e-5


def test_regression_dataset_complete():
    manifest = _manifest()
    samples = manifest["samples"]
    categories = {s["category"] for s in samples}
    missing = REQUIRED_CATEGORIES - categories
    assert not missing, f"game samples missing categories: {sorted(missing)}"
    real = [s for s in samples if s["category"].startswith("real-game")]
    assert len(real) >= 100, "Goal 18 requires at least 100 real game samples"
    assert len(samples) >= 250
    for sample in samples:
        assert sample.get("expected"), sample["file"]
        assert (GAME_SAMPLES / sample["file"]).is_file(), sample["file"]
        if sample.get("known_failure"):
            assert sample.get("known_failure_note"), sample["file"]
    known = [s for s in samples if s.get("known_failure")]
    assert known, "corpus must keep honest known-failure entries"


def test_recognize_classifies_lattice_once_and_never_reclassifies(make_ocr, monkeypatch):
    """P0-1: classification happens for all lattice candidates, not again
    for the few candidates selected by the decoder."""

    sample = _sample("synthetic/mixed_16.png")
    ocr = make_ocr(sample)
    calls: list[int] = []
    original = SegmentScorer.score

    def spy(self, segments, *args, **kwargs):
        calls.append(len(segments))
        return original(self, segments, *args, **kwargs)

    monkeypatch.setattr(SegmentScorer, "score", spy)
    result = ocr.recognize(_image(sample["file"]))
    assert result.text == sample["expected"]
    # One line -> exactly one batched scoring pass (over the full lattice).
    # The pre-P0-1 pipeline made a second pass here over just the chosen
    # path candidates.
    assert len(calls) == 1
    assert calls[0] > len(result.path.candidates)


def test_segment_scorer_uses_injected_cnn_backend(monkeypatch):
    """P0-4: SegmentScorer never reaches around its backend to cnn.forward."""

    model_dir = ROOT / "tests" / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        pytest.skip("cnn_digits fixture not generated")
    model = load_model(model_dir)

    class DummyBackend(Backend):
        def __init__(self):
            self.calls = 0

        def classify(self, glyphs):
            raise AssertionError("SegmentScorer must not use backend.classify")

        def forward_logits(self, glyphs):
            self.calls += 1
            return np.zeros((glyphs.shape[0], 10), dtype=np.float32)

    backend = DummyBackend()
    scorer = SegmentScorer(model, cnn_backend=backend)
    glyph = Component(
        mask=np.ones((12, 8), dtype=bool),
        x=0,
        y=0,
        w=8,
        h=12,
    )
    scores = scorer.score([glyph])
    assert len(scores) == 1
    assert backend.calls == 1
