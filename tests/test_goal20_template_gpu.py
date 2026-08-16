"""Goal 20 G2: GPU Template V2 matcher vs the CPU reference — exact parity.

The GPU matcher replicates ``TemplateV2Classifier`` semantics exactly
(same tolerances, integer XOR+popcount distances, ink-band fallback scan,
``(dist, char_id)`` Top-K ordering and np.argmin winner ties), so every
``TemplateBatch`` field must match the CPU reference EXACTLY, not within a
tolerance. Tests skip when ``wgpu`` or a GPU adapter is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.backends import (
    AutoTemplateMatcher,
    WGPUBackend,
    WGPUTemplateMatcher,
)
from fixedfontocr.classifier import (
    TemplateV2Classifier,
    TemplateV2Data,
)


def _random_weights(num_classes: int = 10, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
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
        "fc.weight": rng.standard_normal((num_classes, 32), dtype=np.float32),
        "fc.bias": rng.standard_normal(num_classes, dtype=np.float32),
    }


def _make_data(num_classes: int = 8, p: int = 6, seed: int = 1) -> TemplateV2Data:
    rng = np.random.default_rng(seed)
    bits = (rng.random((num_classes, p, 72)) < 0.5).astype(np.uint8)
    bits[bits.sum(axis=2) == 0, 0] = 1  # TemplateV2Data forbids empty prototypes
    rs = rng.integers(11, 17, (num_classes, p)).astype(np.uint8)
    dx = rng.integers(0, 9, (num_classes, p)).astype(np.uint8)
    dy = rng.integers(0, 9, (num_classes, p)).astype(np.uint8)
    modes = rng.integers(0, 3, (num_classes, p)).astype(np.uint8)
    return TemplateV2Data(
        bits=bits, render_sizes=rs, dx=dx, dy=dy, downsample_modes=modes
    )


@pytest.fixture(scope="module")
def weights() -> dict[str, np.ndarray]:
    return _random_weights(10, 11)


@pytest.fixture(scope="module")
def wgpu_backend(weights) -> WGPUBackend:
    pytest.importorskip("wgpu")
    try:
        return WGPUBackend(weights)
    except Exception as exc:  # no adapter / driver problem
        pytest.skip(f"WGPU adapter unavailable: {exc}")


@pytest.fixture(scope="module")
def data() -> TemplateV2Data:
    return _make_data()


@pytest.fixture(scope="module")
def charset(data) -> list[str]:
    return [f"c{i}" for i in range(data.num_classes)]


@pytest.fixture(scope="module")
def cpu_matcher(data, charset) -> TemplateV2Classifier:
    return TemplateV2Classifier(data, charset, 24)


@pytest.fixture(scope="module")
def gpu_matcher(wgpu_backend, data, charset) -> WGPUTemplateMatcher:
    return WGPUTemplateMatcher(wgpu_backend, data, charset, 24)


def _assert_batch_equal(got, ref) -> None:
    for field in (
        "ids",
        "scores",
        "second_scores",
        "margins",
        "dists",
        "second_dists",
        "second_ids",
        "top_k_ids",
        "top_k_scores",
        "best_prototypes",
        "prototype_render_sizes",
        "prototype_dx",
        "prototype_dy",
        "prototype_downsample_modes",
    ):
        g = getattr(got, field)
        r = getattr(ref, field)
        if r is None:
            assert g is None, field
        else:
            assert g is not None, field
            assert np.array_equal(g, r), (
                f"{field}: got {g!r}, ref {r!r}"
            )


def _random_glyphs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, 24, 24)) > 0.5).astype(np.uint8) * 255


def test_gpu_template_parity_all_fields(cpu_matcher, gpu_matcher) -> None:
    glyphs = _random_glyphs(5, seed=3)
    _assert_batch_equal(gpu_matcher.match_batch(glyphs), cpu_matcher.match_batch(glyphs))


def test_gpu_template_parity_batch_sizes(cpu_matcher, gpu_matcher) -> None:
    for n in (1, 2, 7, 13):
        glyphs = _random_glyphs(n, seed=100 + n)
        _assert_batch_equal(
            gpu_matcher.match_batch(glyphs), cpu_matcher.match_batch(glyphs)
        )


def test_gpu_template_parity_allowed_subsets(cpu_matcher, gpu_matcher) -> None:
    glyphs = _random_glyphs(4, seed=21)
    for allowed in (None, {0, 3, 7}, {3}, {0, 1, 2, 3, 4, 5, 6, 7}):
        _assert_batch_equal(
            gpu_matcher.match_batch(glyphs, allowed),
            cpu_matcher.match_batch(glyphs, allowed),
        )


def test_gpu_template_parity_empty_allowed(cpu_matcher, gpu_matcher) -> None:
    glyphs = _random_glyphs(2, seed=4)
    _assert_batch_equal(
        gpu_matcher.match_batch(glyphs, set()),
        cpu_matcher.match_batch(glyphs, set()),
    )


def test_gpu_template_parity_empty_batch(cpu_matcher, gpu_matcher) -> None:
    glyphs = np.empty((0, 24, 24), dtype=np.uint8)
    _assert_batch_equal(gpu_matcher.match_batch(glyphs), cpu_matcher.match_batch(glyphs))


def test_gpu_template_single_dispatch(gpu_matcher) -> None:
    gpu_matcher.match_batch(_random_glyphs(3, seed=8))
    assert gpu_matcher.last_dispatch_count == 1


def test_gpu_template_topk_limited(cpu_matcher, gpu_matcher) -> None:
    glyphs = _random_glyphs(3, seed=9)
    with pytest.raises(ValueError):
        gpu_matcher.match_batch(glyphs, top_k=3)


def test_gpu_template_auto_wrapper_parity(cpu_matcher, gpu_matcher) -> None:
    """The per-batch auto wrapper delegates to a matcher with exact output."""
    auto = AutoTemplateMatcher(
        cpu_matcher, gpu_matcher, batch_sizes=(1, 4, 8), repeat=2, iters=2
    )
    glyphs = _random_glyphs(6, seed=55)
    for allowed in (None, {0, 3, 7}, {3}):
        _assert_batch_equal(
            auto.match_batch(glyphs, allowed),
            cpu_matcher.match_batch(glyphs, allowed),
        )




# ---------------------------------------------------------------------
# Real-game crops_items corpus (dict mode) — the production scenario
# ---------------------------------------------------------------------

CROPS = (
    Path(__file__).resolve().parents[1]
    / "fonts" / "SourceHanSansSC" / "crops" / "crops_items"
)
BANK = CROPS.parent / "crops_items_recognition.csv.realglyphs.npz"
PINNED_CROPS = (
    "0_y112_y140_item1.png",  # Z17
    "1_y646_y674_item2.png",  # Z28
    "1_y646_y674_item6.png",  # Z1
    "0_y151_y179_item5.png",  # 47工程
    "0_y290_y318_item0.png",  # 初雪
    "1_y400_y428_item1.png",  # 乌戈里尼
)


def _require_crops() -> None:
    if not all((CROPS / name).is_file() for name in PINNED_CROPS) or not BANK.is_file():
        pytest.skip("crops_items corpus/bank not present (local dataset)")


def _load_crop(name: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(CROPS / name).convert("RGB"), dtype=np.uint8)


def _game_cn_ocr(backend: str):
    model_dir = Path("model/game_cn")
    if not (model_dir / "config.json").exists():
        pytest.skip("model/game_cn not present")
    pytest.importorskip("wgpu")
    try:
        return FixedFontOCR(model_path=model_dir, backend=backend)
    except Exception as exc:
        pytest.skip(f"WGPU adapter unavailable: {exc}")


def test_gpu_template_end_to_end_crops_dict_mode() -> None:
    """wgpu vs cpu on the real crops in dict mode (+ real-glyph bank)."""
    _require_crops()
    from fixedfontocr.backends import WGPUTemplateMatcher as WTM

    gpu_ocr = _game_cn_ocr("wgpu")
    assert isinstance(gpu_ocr._scorer.template, WTM)
    cpu_ocr = _game_cn_ocr("cpu")
    for name in PINNED_CROPS:
        image = _load_crop(name)
        got = gpu_ocr.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        ref = cpu_ocr.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        assert got.text == ref.text, (name, got.text, ref.text)
        assert got.matched_term == ref.matched_term, (
            name,
            got.matched_term,
            ref.matched_term,
        )


def test_gpu_template_auto_backend_crops_dict_mode() -> None:
    """backend="auto" stays in parity with CPU on the real crops (dict)."""
    _require_crops()
    from fixedfontocr.backends import AutoTemplateMatcher as ATM
    from fixedfontocr.backends import WGPUTemplateMatcher as WTM

    auto_ocr = _game_cn_ocr("auto")
    assert isinstance(auto_ocr._scorer.template, ATM)
    assert isinstance(auto_ocr._scorer.template.gpu, WTM)
    assert auto_ocr._scorer.template.crossover is not None
    cpu_ocr = _game_cn_ocr("cpu")
    for name in PINNED_CROPS:
        image = _load_crop(name)
        got = auto_ocr.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        ref = cpu_ocr.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        assert got.text == ref.text, (name, got.text, ref.text)
        assert got.matched_term == ref.matched_term, (
            name,
            got.matched_term,
            ref.matched_term,
        )

