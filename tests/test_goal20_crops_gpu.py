"""Goal 20 G3/G4 GPU preprocessing & single-submit scoring — crops_items dict.

The crops_items corpus (fonts/SourceHanSansSC/crops) is the production
target: these tests pin, on real game crops:

- ``WGPUBackend.preprocess_glyphs`` byte-exact against the CPU soft batch
  (per-crop segments taken from the CPU decoder's own candidates);
- ``WGPUBackend.forward_logits_from_image`` (preprocess + mega in ONE
  submit) against the CPU reference logits on the real model;
- ``WGPUScoringStage.score_line_from_image`` — GPU preprocess + template
  match + TinyCNN logits in ONE submit — against the CPU references;
- dict-mode end-to-end parity over the whole labeled corpus:
  backend="wgpu" vs backend="cpu" must produce the same visible text and
  matched_term for every crop, the answer key being
  ``crops_items_recognition.csv`` (文本值 = OCR text, 预期值 = dict term);
- the whole ``recognize()`` must be ONE GPU submit per line: every other
  GPU entry point is trip-wired so only the stage submission may run.

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


def _scorer_inputs(cpu_ocr, segments):
    """Replicate SegmentScorer.score's glyph/geometry inputs verbatim.

    Returns ``(glyphs, geoms)``: the binary-normalized ``uint8 [N, 24, 24]``
    batch the template matcher consumes and the per-segment geometry the
    preprocess path (CPU and GPU alike) uses for baseline placement.
    """
    from fixedfontocr.preprocess import glyph_normalize_geometry, normalize
    from fixedfontocr.types import Component

    model = cpu_ocr.model
    spec = model.normalize_spec
    if spec is None:
        return np.stack([normalize(s.mask, model.input_size) for s in segments]), None
    geoms = [
        glyph_normalize_geometry(
            Component(mask=s.mask, x=s.x, y=s.y, w=s.w, h=s.h),
            spec,
            model.input_size,
        )
        for s in segments
    ]
    glyphs = np.stack(
        [
            normalize(
                s.mask,
                model.input_size,
                baseline_offset=g[0],
                scale=g[1],
                baseline_row=spec.baseline_row,
            )
            for s, g in zip(segments, geoms)
        ]
    )
    return glyphs, geoms


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
    """One-submit scoring stage == the two separate GPU calls, exactly.

    Runs the stage with ``sparse=False`` (full-logits readback) so the
    exact full-matrix parity is pinned; the T2 sparse readback has its own
    contract test below.
    """
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
        tb_stage, logits_stage = stage.score_line(glyphs, None, sparse=False)
        tb_sep = matcher.match_batch(glyphs, None)
        logits_sep = backend.forward_logits(glyphs)
        assert np.array_equal(tb_stage.top_k_ids, tb_sep.top_k_ids)
        assert np.array_equal(tb_stage.top_k_scores, tb_sep.top_k_scores)
        assert np.array_equal(tb_stage.best_prototypes, tb_sep.best_prototypes)
        assert np.array_equal(logits_stage, logits_sep), name
    assert stage.last_submit_count == 1


def test_crops_gpu_stage_topk_sparse_contract(game_ocrs) -> None:
    """T2 sparse readback: exact on the supported ids, -inf elsewhere.

    The sparse stage returns a ``[N, C]`` matrix whose real values sit
    exactly on the union of the CNN top-7 ids and the template top-K ids
    (the only positions the hybrid fusion reads), byte-identical to the
    full readback there, and ``-inf`` everywhere else.
    """
    _require_crops()
    cpu_ocr, gpu_ocr = game_ocrs
    from fixedfontocr.backends import WGPUScoringStage
    from fixedfontocr.postprocess import topk as np_topk
    from fixedfontocr.preprocess import normalize

    assert isinstance(gpu_ocr._scorer.gpu_stage, WGPUScoringStage)
    stage = gpu_ocr._scorer.gpu_stage
    matcher = gpu_ocr._scorer.template
    backend = gpu_ocr._scorer.cnn_backend
    for name in PINNED_CROPS:
        image = _load_crop(name)
        segments = _candidate_segments(cpu_ocr, image)
        glyphs = np.stack([normalize(s.mask, 24) for s in segments])
        tb_stage, logits_stage = stage.score_line(glyphs, None)  # sparse=True
        assert stage.last_submit_count == 1, name
        full = backend.forward_logits(glyphs)
        tb_sep = matcher.match_batch(glyphs, None)
        n = len(segments)
        row = np.arange(n)[:, None]
        # supported positions: CNN top-7 ids + template top-K ids
        cnn_ids, _ = np_topk(full, 7)
        tpl_ids = np.maximum(tb_sep.top_k_ids, 0)
        supported = np.zeros_like(full, dtype=bool)
        supported[row, np.maximum(cnn_ids, 0)] = True
        supported[row, tpl_ids] = True
        # every supported position is exact
        assert np.array_equal(logits_stage[supported], full[supported]), name
        # everything else is -inf
        assert np.all(np.isneginf(logits_stage[~supported])), name
        # the supported set is complete for fusion: CNN top-5 + template ids
        assert np.all(logits_stage[row, tpl_ids] == full[row, tpl_ids]), name


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


def test_crops_gpu_stage_from_image_parity(game_ocrs, game_model) -> None:
    """preprocess+template+mega in ONE submit == the CPU references.

    ``score_line_from_image`` is the G2+G3+G4 fused encoder for soft-input
    hybrids: the preprocess dispatch writes the soft batch straight into the
    mega input buffer, the template dispatch runs in the same submission,
    and one readback returns template records + full logits. On real crops
    its output must equal the separate CPU references: the CPU template
    matcher over the binary glyphs and CPU ``forward_logits`` over the CPU
    soft batch.
    """
    _require_crops()
    cpu_ocr, gpu_ocr = game_ocrs
    from fixedfontocr.backends import WGPUScoringStage
    from fixedfontocr.scorer import _soft_glyph_batch

    assert isinstance(gpu_ocr._scorer.gpu_stage, WGPUScoringStage)
    stage = gpu_ocr._scorer.gpu_stage
    cpu_backend = CPUBackend(game_model.weights, game_model.input_size)
    cpu_template = cpu_ocr._scorer.template
    for name in PINNED_CROPS:
        image = _load_crop(name)
        segments = _candidate_segments(cpu_ocr, image)
        glyphs, geoms = _scorer_inputs(cpu_ocr, segments)
        spec = cpu_ocr.model.normalize_spec
        tb, logits = stage.score_line_from_image(
            image, segments, glyphs, None, spec, geoms, sparse=False
        )
        ref_tb = cpu_template.match_batch(glyphs, None)
        soft = cpu_ocr.profile.soft_foreground(image)
        soft_batch = _soft_glyph_batch(segments, soft, cpu_ocr.model.input_size, geoms, spec)
        ref_logits = cpu_backend.forward_logits(soft_batch)
        assert np.array_equal(tb.top_k_ids, ref_tb.top_k_ids), name
        assert np.array_equal(tb.top_k_scores, ref_tb.top_k_scores), name
        assert np.array_equal(tb.best_prototypes, ref_tb.best_prototypes), name
        assert np.abs(logits - ref_logits).max() < 1e-4, name
    assert stage.last_submit_count == 1
    assert stage.last_dispatch_count == 3  # preprocess + template + mega


def test_crops_gpu_recognize_single_submit_ocr_result(game_ocrs) -> None:
    """The whole dict-mode recognize() is exactly ONE GPU submit per line.

    Every other GPU entry point is trip-wired: the standalone template
    matcher, the CNN logits paths and the standalone preprocess must NOT run
    during a production recognize — all scoring evidence has to come out of
    the stage's single submission, and the OCR result must match CPU.
    """
    _require_crops_and_bank()
    cpu_ocr, gpu_ocr = game_ocrs
    stage = gpu_ocr._scorer.gpu_stage
    assert stage is not None
    matcher = gpu_ocr._scorer.template
    backend = gpu_ocr._scorer.cnn_backend
    assert isinstance(backend, WGPUBackend)

    def _tripwire(name: str):
        def _fail(*_args, **_kwargs):
            raise AssertionError(
                f"{name} must not run: the whole line must be scored in the "
                "single gpu_stage submit"
            )

        return _fail

    setattr(matcher, "match_batch", _tripwire("template.match_batch"))
    for attr in (
        "classify",
        "forward_logits",
        "forward_logits_from_image",
        "preprocess_glyphs",
    ):
        setattr(backend, attr, _tripwire(f"backend.{attr}"))

    for name in PINNED_CROPS:
        image = _load_crop(name)
        stage.last_submit_count = 0
        got = gpu_ocr.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        assert stage.last_submit_count == 1, name
        ref = cpu_ocr.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        assert got.text == ref.text, (name, got.text, ref.text)
        assert got.matched_term == ref.matched_term, (
            name,
            got.matched_term,
            ref.matched_term,
        )


def test_crops_gpu_soft_hybrid_recognize_single_submit(game_ocrs, tmp_path) -> None:
    """Soft-input hybrid: preprocess+template+mega in one submit, == CPU.

    game_cn is binary-trained; a soft copy of the same model exercises the
    G2+G3+G4 fused encoder end to end: recognize() must run exactly ONE GPU
    submit (three dispatches inside it) and match ``backend="cpu"`` on every
    pinned crop. The other GPU entry points are trip-wired like in the
    binary test, so the OCR result can only come out of the stage submit.
    """
    _require_crops_and_bank()
    cpu_ocr, gpu_ocr = game_ocrs  # noqa: F841 — presence guard for model/wgpu
    import json
    import shutil

    soft_dir = tmp_path / "game_cn_soft"
    shutil.copytree(ROOT / "model" / "game_cn", soft_dir)
    cfg = json.loads((soft_dir / "config.json").read_text(encoding="utf-8"))
    cfg["input_mode"] = "soft"
    (soft_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False), encoding="utf-8"
    )
    soft_cpu = FixedFontOCR(model_path=soft_dir, backend="cpu")
    soft_gpu = FixedFontOCR(model_path=soft_dir, backend="wgpu")
    stage = soft_gpu._scorer.gpu_stage
    assert stage is not None
    matcher = soft_gpu._scorer.template
    backend = soft_gpu._scorer.cnn_backend
    assert isinstance(backend, WGPUBackend)

    def _tripwire(name: str):
        def _fail(*_args, **_kwargs):
            raise AssertionError(
                f"{name} must not run: the whole line must be scored in the "
                "single gpu_stage submit"
            )

        return _fail

    setattr(matcher, "match_batch", _tripwire("template.match_batch"))
    for attr in (
        "classify",
        "forward_logits",
        "forward_logits_from_image",
        "preprocess_glyphs",
    ):
        setattr(backend, attr, _tripwire(f"backend.{attr}"))

    for name in PINNED_CROPS:
        image = _load_crop(name)
        stage.last_submit_count = 0
        got = soft_gpu.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        ref = soft_cpu.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        assert stage.last_submit_count == 1, name
        assert stage.last_dispatch_count == 3, name  # preprocess+template+mega
        assert got.text == ref.text, (name, got.text, ref.text)
        assert got.matched_term == ref.matched_term, (
            name,
            got.matched_term,
            ref.matched_term,
        )


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


def test_crops_gpu_dict_mode_csv_answers_single_submit(game_ocrs) -> None:
    """Every CSV-labeled crop: wgpu == cpu, one submit, answers match the CSV.

    The answer key is ``crops_items_recognition.csv``: 文本值 is the OCR
    text recorded for the crop and 预期值 is the dict-mode ground truth the
    engine must surface as ``matched_term``. For every labeled crop the GPU
    result must equal the CPU result (text and matched_term), the whole
    recognize() must be one GPU submit, and — for every crop the current
    CPU engine can read at all — the answer must equal the CSV row.

    Two crops return '' on the CPU engine today (their CSV rows were
    recorded by an earlier engine state), so they are excluded from the
    CSV-match assertion but still asserted wgpu == cpu.
    """
    _require_crops_and_bank()
    cpu_ocr, gpu_ocr = game_ocrs
    stage = gpu_ocr._scorer.gpu_stage
    assert stage is not None
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    # Crops the current CPU engine cannot read (returns ''); the CSV answer
    # for these rows is not reproducible by either backend.
    KNOWN_CPU_EMPTY = {"0_y226_y254_item2.png", "1_y400_y428_item1.png"}
    parity_bad: list[tuple[str, str, str]] = []
    csv_bad: list[tuple[str, str, str]] = []
    submit_bad: list[str] = []
    checked = 0
    for row in rows[1:]:
        name = row[1]
        if name == "" or row[3] == "" or not (CROPS / name).is_file():
            continue
        checked += 1
        image = _load_crop(name)
        stage.last_submit_count = 0
        got = gpu_ocr.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        ref = cpu_ocr.recognize(
            image, lexicon="ship_names", lexicon_mode="dict", real_glyph_bank=BANK
        )
        if got.text != ref.text or got.matched_term != ref.matched_term:
            parity_bad.append(
                (name, f"{got.text}|{got.matched_term}", f"{ref.text}|{ref.matched_term}")
            )
        if stage.last_submit_count != 1:
            submit_bad.append(name)
        if name not in KNOWN_CPU_EMPTY:
            if got.text != row[2] or got.matched_term != row[3]:
                csv_bad.append(
                    (name, f"{got.text}|{got.matched_term}", f"{row[2]}|{row[3]}")
                )
    assert checked >= 200, f"corpus shrank? only {checked} labeled crops"
    assert not parity_bad, (
        f"{len(parity_bad)} crops disagree with CPU: {parity_bad[:5]}"
    )
    assert not submit_bad, f"crops with !=1 GPU submit: {submit_bad[:5]}"
    assert not csv_bad, (
        f"{len(csv_bad)} crops miss the CSV answer: {csv_bad[:5]}"
    )
