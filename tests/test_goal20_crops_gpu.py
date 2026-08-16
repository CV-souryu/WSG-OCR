"""Goal 20 G3 GPU preprocessing — validated on the crops_items dict scenario.

The crops_items corpus (fonts/SourceHanSansSC/crops) is the production
target: these tests pin, on real game crops:

- ``WGPUBackend.preprocess_glyphs`` byte-exact against the CPU soft batch
  (per-crop segments taken from the CPU decoder's own candidates);
- ``WGPUBackend.forward_logits_from_image`` (preprocess + mega in ONE
  submit) against the CPU reference logits on the real model;
- dict-mode end-to-end parity over the whole labeled corpus:
  backend="wgpu" vs backend="cpu" must produce the same visible text and
  matched_term for every crop.

All tests skip when the corpus or a GPU adapter is unavailable.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.backends import CPUBackend, WGPUBackend
from fixedfontocr.scorer import _soft_glyph_batch

ROOT = Path(__file__).resolve().parents[1]
CROPS = ROOT / "fonts" / "SourceHanSansSC" / "crops" / "crops_items"
CSV = CROPS.parent / "crops_items_recognition.csv"
BANK = CSV.with_suffix(CSV.suffix + ".realglyphs.npz")
PINNED_CROPS = (
    "0_y112_y140_item1.png",  # Z17
    "1_y646_y674_item2.png",  # Z28
    "1_y646_y674_item6.png",  # Z1
    "0_y151_y179_item5.png",  # 47工程
    "0_y290_y318_item0.png",  # 初雪
    "1_y400_y428_item1.png",  # 乌戈里尼
)


def _require_crops() -> None:
    if not all((CROPS / name).is_file() for name in PINNED_CROPS):
        pytest.skip("crops_items corpus not present (local dataset)")


def _require_crops_and_bank() -> None:
    _require_crops()
    if not BANK.is_file():
        pytest.skip("crops_items real-glyph bank not present (local dataset)")


def _load_crop(name: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(CROPS / name).convert("RGB"), dtype=np.uint8)


def _labeled_crops() -> list[str]:
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    return [
        row[1]
        for row in rows[1:]
        if row[1] != "" and row[3] != "" and (CROPS / row[1]).is_file()
    ]


@pytest.fixture(scope="module")
def game_ocrs():
    model_dir = ROOT / "model" / "game_cn"
    if not (model_dir / "config.json").exists():
        pytest.skip("model/game_cn not present")
    pytest.importorskip("wgpu")
    try:
        gpu = FixedFontOCR(model_path=model_dir, backend="wgpu")
    except Exception as exc:
        pytest.skip(f"WGPU adapter unavailable: {exc}")
    cpu = FixedFontOCR(model_path=model_dir, backend="cpu")
    return cpu, gpu


@pytest.fixture(scope="module")
def game_model():
    from fixedfontocr.model import load_model

    model_dir = ROOT / "model" / "game_cn"
    if not (model_dir / "config.json").exists():
        pytest.skip("model/game_cn not present")
    return load_model(model_dir)


def _candidate_segments(cpu_ocr, image):
    """The decoder's own candidate segments for one crop (CPU reference)."""
    result = cpu_ocr.recognize(image)
    path = result.path
    assert path is not None and path.candidates
    return [cand.segment for cand in path.candidates]


def test_crops_gpu_preprocess_byte_exact(game_ocrs) -> None:
    """GPU soft batch == CPU soft batch, byte for byte, on real crops."""
    _require_crops()
    cpu_ocr, gpu_ocr = game_ocrs
    gpu_backend = gpu_ocr._scorer.cnn_backend
    assert isinstance(gpu_backend, WGPUBackend)
    for name in PINNED_CROPS:
        image = _load_crop(name)
        segments = _candidate_segments(cpu_ocr, image)
        got = gpu_backend.preprocess_glyphs(image, segments, None)
        soft = cpu_ocr.profile.soft_foreground(image)
        ref = _soft_glyph_batch(
            segments, soft, 24, [None] * len(segments), None
        )
        assert np.array_equal(got, ref), (
            f"{name}: GPU soft batch differs from the CPU reference"
        )


def test_crops_gpu_fused_logits_parity(game_ocrs, game_model) -> None:
    """preprocess+mega in ONE submit matches CPU forward_logits on crops."""
    _require_crops()
    cpu_ocr, gpu_ocr = game_ocrs
    gpu_backend = gpu_ocr._scorer.cnn_backend
    assert isinstance(gpu_backend, WGPUBackend)
    cpu_backend = CPUBackend(game_model.weights, game_model.input_size)
    for name in PINNED_CROPS:
        image = _load_crop(name)
        segments = _candidate_segments(cpu_ocr, image)
        soft = cpu_ocr.profile.soft_foreground(image)
        ref_batch = _soft_glyph_batch(
            segments, soft, 24, [None] * len(segments), None
        )
        ref = cpu_backend.forward_logits(ref_batch)
        got = gpu_backend.forward_logits_from_image(image, segments, None)
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() < 1e-4, name
    assert gpu_backend.last_dispatch_count == 2  # preprocess + mega, 1 submit


def test_crops_gpu_stage_parity(game_ocrs) -> None:
    """One-submit scoring stage == the two separate GPU calls, exactly."""
    _require_crops()
    cpu_ocr, gpu_ocr = game_ocrs
    from fixedfontocr.backends import WGPUScoringStage
    from fixedfontocr.preprocess import normalize

    assert isinstance(gpu_ocr._scorer.gpu_stage, WGPUScoringStage)
    stage = gpu_ocr._scorer.gpu_stage
    matcher = gpu_ocr._scorer.template
    backend = gpu_ocr._scorer.cnn_backend
    for name in PINNED_CROPS:
        image = _load_crop(name)
        segments = _candidate_segments(cpu_ocr, image)
        glyphs = np.stack([normalize(s.mask, 24) for s in segments])
        tb_stage, logits_stage = stage.score_line(glyphs, None)
        tb_sep = matcher.match_batch(glyphs, None)
        logits_sep = backend.forward_logits(glyphs)
        assert np.array_equal(tb_stage.top_k_ids, tb_sep.top_k_ids)
        assert np.array_equal(tb_stage.top_k_scores, tb_sep.top_k_scores)
        assert np.array_equal(tb_stage.best_prototypes, tb_sep.best_prototypes)
        assert np.array_equal(logits_stage, logits_sep), name
    assert stage.last_submit_count == 1


def test_crops_gpu_stage_single_submit_per_line(game_ocrs) -> None:
    """The production wgpu path closes the scoring chain into one submit."""
    _require_crops()
    _cpu_ocr, gpu_ocr = game_ocrs
    stage = gpu_ocr._scorer.gpu_stage
    assert stage is not None
    for name in PINNED_CROPS:
        image = _load_crop(name)
        gpu_ocr.recognize(image)
        assert stage.last_submit_count == 1


def test_crops_gpu_dict_mode_parity_pinned(game_ocrs) -> None:
    """wgpu vs cpu in dict mode (+ bank): same text and matched_term."""
    _require_crops_and_bank()
    cpu_ocr, gpu_ocr = game_ocrs
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


def test_crops_gpu_dict_mode_parity_full_corpus(game_ocrs) -> None:
    """Every labeled crop: wgpu and cpu agree in dict mode (+ bank)."""
    _require_crops_and_bank()
    cpu_ocr, gpu_ocr = game_ocrs
    mismatches: list[tuple[str, str, str]] = []
    for name in _labeled_crops():
        image = _load_crop(name)
        got = gpu_ocr.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        ref = cpu_ocr.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        if got.text != ref.text or got.matched_term != ref.matched_term:
            mismatches.append(
                (name, f"{got.text}|{got.matched_term}", f"{ref.text}|{ref.matched_term}")
            )
    assert not mismatches, f"{len(mismatches)} crops disagree: {mismatches[:5]}"
