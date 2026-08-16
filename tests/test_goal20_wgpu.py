"""Goal 20 (WGPU phase 2) contract tests.

Pins the acceptance criteria defined in ``docs/goal20.md``:

- the "now" tests already pass on phase 1 and pin the invariants that must
  never regress while WGPU work happens against the current repo CPU
  implementation as the parity reference (NOT the Goal 19 version-1.0
  snapshot; see ``docs/goal20.md`` section 1);
- the ``xfail`` tests pin the four phase-2 tasks (T1 shader fusion, T2
  DP Top-K 回灌, T3 GPU preprocessing, T4 staged readback). Each one flips
  to a plain test when its task lands — see ``docs/goal20.md`` section 3.

Per-layer numeric parity, exported-vector parity and auto-backend selection
live in ``tests/test_wgpu.py``, ``tests/test_wgpu_vectors.py`` and
``tests/test_auto_backend.py`` (28 passed on Apple M4 / Metal / wgpu 0.32).
All tests skip when ``wgpu`` or a GPU adapter is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.backends import CPUBackend, WGPUBackend
from fixedfontocr.model import write_cnn_model
from fixedfontocr.types import Component, default_profile

from conftest import render_text

T1 = "Goal 20 T1 (shader fusion) pending: docs/goal20.md"
T2 = "Goal 20 T2 (DP Top-K 回灌) pending: docs/goal20.md"
T3 = "Goal 20 T3 (GPU preprocessing) pending: docs/goal20.md"
T4 = "Goal 20 T4 (staged readback) pending: docs/goal20.md"


def _random_weights(num_classes: int, seed: int = 0) -> dict[str, np.ndarray]:
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


@pytest.fixture(scope="module")
def weights() -> dict[str, np.ndarray]:
    return _random_weights(10, seed=11)


@pytest.fixture(scope="module")
def wgpu(weights) -> WGPUBackend:
    pytest.importorskip("wgpu")
    try:
        return WGPUBackend(weights)
    except Exception as exc:  # no adapter / driver problem
        pytest.skip(f"WGPU adapter unavailable: {exc}")


@pytest.fixture(scope="module")
def random_glyphs() -> np.ndarray:
    rng = np.random.default_rng(3)
    return (rng.random((5, 24, 24)) > 0.5).astype(np.uint8) * 255


def _cpu_masked_topk(
    logits: np.ndarray, allowed: np.ndarray, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy reference for ``classify_topk``: masked, logit-descending Top-K."""
    masked = np.where(allowed[None, :], logits, -np.inf)
    order = np.argsort(-masked, axis=1)[:, :k]
    return order, np.take_along_axis(masked, order, axis=1)


def _cpu_soft_batch(
    image: np.ndarray, segments: list[Component], spec
) -> np.ndarray:
    """CPU reference for ``preprocess_glyphs`` (default profile soft path)."""
    from fixedfontocr.preprocess import _soft_glyph_batch

    profile = default_profile()
    soft = profile.soft_foreground(image)
    return _soft_glyph_batch(
        segments, soft, profile.target_size, [None] * len(segments), spec
    )


# ---------------------------------------------------------------------
# Now-passing phase-1 invariants (must never regress)
# ---------------------------------------------------------------------


def test_goal20_cpu_baseline_current_repo_golden(weights, random_glyphs) -> None:
    """The parity baseline is the current repo CPU implementation (HEAD).

    Goal 19's version-1.0 snapshot is NOT the reference: the CPU has moved
    on since the freeze (backend boundary, touching-glyph splits, dict
    arbitration, streaming/perf commits). These golden values were recorded
    from :class:`CPUBackend` at HEAD ``2f46da7`` on the module fixtures
    (random weights seed 11, glyphs seed 3). A deliberate CPU change must
    update these values first and then re-check WGPU parity against the new
    baseline; an accidental CPU drift fails here.
    """

    cpu = CPUBackend(weights)
    result = cpu.classify(random_glyphs)
    assert result.char_ids.tolist() == [2, 2, 2, 2, 2]
    assert np.allclose(
        result.scores,
        [56.957630, 53.742603, 85.849075, 66.846016, 80.057144],
        atol=1e-5,
        rtol=0.0,
    )
    logits = cpu.forward_logits(random_glyphs)
    assert np.allclose(
        logits[0],
        [
            -68.704765,
            35.210476,
            133.132095,
            76.174461,
            -37.672977,
            -169.487106,
            73.998726,
            -150.574036,
            36.311676,
            -7.022514,
        ],
        atol=1e-5,
        rtol=0.0,
    )


def test_goal20_production_dp_runs_on_injected_backend(tmp_path, weights) -> None:
    """The DP scorer receives the selected backend from the API boundary."""
    out = tmp_path / "cnn"
    write_cnn_model(out, list("0123456789"), weights)
    try:
        gpu_ocr = FixedFontOCR(model_path=out, backend="wgpu")
    except Exception as exc:
        pytest.skip(f"WGPU adapter unavailable: {exc}")
    cpu_ocr = FixedFontOCR(model_path=out, backend="cpu")
    assert isinstance(gpu_ocr._scorer.cnn_backend, WGPUBackend)
    assert isinstance(cpu_ocr._scorer.cnn_backend, CPUBackend)


def test_goal20_end_to_end_wgpu_cpu_parity(font_path) -> None:
    """Same rendered image through CPU and WGPU must produce the same text."""
    model_dir = Path(__file__).parent / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        pytest.skip("cnn_digits fixture not generated; run scripts/train_tinycnn.py")
    pytest.importorskip("wgpu")
    try:
        gpu_ocr = FixedFontOCR(model_path=model_dir, backend="wgpu")
    except Exception as exc:
        pytest.skip(f"WGPU adapter unavailable: {exc}")
    cpu_ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    for text in ("1234567890", "42"):
        image = render_text(text, font_path)
        got = gpu_ocr.recognize(image)
        ref = cpu_ocr.recognize(image)
        assert got.text == ref.text == text
        assert [c.char for c in got.chars] == list(text)


# ---------------------------------------------------------------------
# T1: shader fusion (8 -> 4 dispatches)
# ---------------------------------------------------------------------


def test_goal20_fused_pipeline_dispatch_count(wgpu, random_glyphs) -> None:
    """ONE dispatch runs the whole TinyCNN (mega shader), zero intermediate
    copy-backs: classify() and forward_logits() each submit exactly one
    compute dispatch per call."""
    wgpu.classify(random_glyphs)
    assert wgpu.last_dispatch_count == 1
    wgpu.forward_logits(random_glyphs)
    assert wgpu.last_dispatch_count == 1


# ---------------------------------------------------------------------
# T2: DP Top-K 回灌 (classify_topk + logits_for)
# ---------------------------------------------------------------------


@pytest.mark.xfail(reason=T2, strict=False)
def test_goal20_classify_topk_parity(wgpu, weights, random_glyphs) -> None:
    """classify_topk(glyphs, allowed_mask) matches the CPU masked Top-K."""
    cpu = CPUBackend(weights)
    logits = cpu.forward_logits(random_glyphs)
    num_classes = weights["fc.weight"].shape[0]
    allowed = np.zeros(num_classes, dtype=bool)
    allowed[2:8] = True
    got_ids, got_logits = wgpu.classify_topk(random_glyphs, allowed)
    ref_ids, ref_logits = _cpu_masked_topk(logits, allowed)
    assert got_ids.shape == ref_ids.shape
    assert got_logits.shape == ref_logits.shape
    assert np.array_equal(got_ids, ref_ids)
    assert np.abs(got_logits - ref_logits).max() < 1e-4


@pytest.mark.xfail(reason=T2, strict=False)
def test_goal20_logits_for_gather_parity(wgpu, weights, random_glyphs) -> None:
    """logits_for(glyphs, char_ids) matches the CPU logit gather (hybrid fusion)."""
    cpu = CPUBackend(weights)
    logits = cpu.forward_logits(random_glyphs)
    requested = np.array(
        [[0, 3, 9], [1, 1, 7], [2, 5, 0], [4, 6, 8], [9, 3, 2]], dtype=np.int32
    )
    got = wgpu.logits_for(random_glyphs, requested)
    ref = logits[np.arange(logits.shape[0])[:, None], requested]
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-4


# ---------------------------------------------------------------------
# T3: GPU preprocessing (RGB once -> ROI/gray/resize/normalize in WGSL)
# ---------------------------------------------------------------------


@pytest.mark.xfail(reason=T3, strict=False)
def test_goal20_gpu_preprocess_parity(wgpu) -> None:
    """GPU preprocess must be byte-identical to the CPU soft batch."""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (48, 96, 3), dtype=np.uint8)
    segments = [
        Component(mask=np.ones((20, 18), dtype=bool), x=4, y=6, w=18, h=20),
        Component(mask=np.ones((16, 10), dtype=bool), x=60, y=10, w=10, h=16),
    ]
    got = wgpu.preprocess_glyphs(image, segments, None)
    ref = _cpu_soft_batch(image, segments, None)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


# ---------------------------------------------------------------------
# T4: persistent/staged readback (amortize the map_sync floor)
# ---------------------------------------------------------------------


@pytest.mark.xfail(reason=T4, strict=False)
def test_goal20_staged_readback_parity(wgpu, random_glyphs) -> None:
    """staged=True readback returns identical results to the sync path."""
    base = wgpu.classify(random_glyphs)
    staged = wgpu.classify(random_glyphs, staged=True)
    assert np.array_equal(base.char_ids, staged.char_ids)
    assert np.abs(base.scores - staged.scores).max() < 1e-4
    logits = wgpu.forward_logits(random_glyphs)
    staged_logits = wgpu.forward_logits(random_glyphs, staged=True)
    assert np.abs(logits - staged_logits).max() < 1e-4
