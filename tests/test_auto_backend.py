from __future__ import annotations

from pathlib import Path

import numpy as np

from fixedfontocr import FixedFontOCR
from fixedfontocr.backends import AutoBackend, CPUBackend, WGPUBackend

from conftest import render_text


def _measurements() -> dict[int, dict[str, float]]:
    # CPU wins up to batch 16, GPU wins from 32 onward (typical crossover).
    return {
        1: {"cpu": 1.0, "gpu": 10.0},
        8: {"cpu": 2.0, "gpu": 10.0},
        16: {"cpu": 3.0, "gpu": 10.0},
        32: {"cpu": 10.0, "gpu": 4.0},
        64: {"cpu": 20.0, "gpu": 4.0},
    }


def test_auto_backend_picks_by_measured_crossover():
    cpu = object()
    gpu = object()
    auto = AutoBackend(cpu, gpu, _measurements())  # type: ignore[arg-type]
    assert auto.pick(1) is cpu
    assert auto.pick(16) is cpu
    assert auto.pick(32) is gpu
    assert auto.pick(64) is gpu
    assert auto.pick(1000) is gpu
    assert auto.crossover == (32, "wgpu")


def test_auto_backend_falls_back_to_cpu_when_gpu_is_unavailable():
    cpu = object()
    auto = AutoBackend(cpu, None, _measurements())  # type: ignore[arg-type]
    assert auto.pick(64) is cpu
    assert auto.crossover is None


def test_auto_backend_single_measured_batch():
    cpu = object()
    gpu = object()
    auto = AutoBackend(cpu, gpu, {32: {"cpu": 1.0, "gpu": 2.0}})  # type: ignore[arg-type]
    assert auto.pick(1) is cpu
    assert auto.pick(1000) is cpu
    assert auto.crossover is None


def test_auto_backend_engine_matches_cpu(font_path):
    model_dir = Path(__file__).parent / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        import pytest

        pytest.skip("cnn_digits fixture not generated")
    ocr = FixedFontOCR(model_path=model_dir, backend="auto")
    ref = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("1234567890 4242", font_path)
    assert ocr.recognize(image).text == ref.recognize(image).text
    assert ocr.resolved_backend in ("auto", "cpu")
    if isinstance(ocr._backend, AutoBackend):
        assert ocr.backend_benchmark is not None
        assert set(ocr.backend_benchmark) == {1, 8, 16, 32, 64, 128}
    else:
        assert isinstance(ocr._backend, CPUBackend)
        assert ocr.backend_benchmark_error is not None


def test_auto_backend_on_template_model(font_path, model_dir):
    ocr = FixedFontOCR(model_path=model_dir, backend="auto")
    image = render_text("1234567890", font_path)
    assert ocr.recognize(image).text == "1234567890"
    assert ocr.resolved_backend == "cpu"


def test_benchmark_backends_reports_both_columns(font_path):
    from fixedfontocr.backends import benchmark_backends
    from fixedfontocr.model import load_model

    model_dir = Path(__file__).parent / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        import pytest

        pytest.skip("cnn_digits fixture not generated")
    model = load_model(model_dir)
    cpu = CPUBackend(model.weights, model.input_size)
    try:
        gpu = WGPUBackend(model.weights, model.input_size)
    except Exception:
        import pytest

        pytest.skip("WGPU adapter unavailable")
    table = benchmark_backends(cpu, gpu, batch_sizes=(1, 8), repeat=1, iters=1)
    assert set(table) == {1, 8}
    for m in table.values():
        assert m["cpu"] > 0
        assert m["gpu"] > 0


def test_auto_backend_glyph_batch_parity(font_path):
    """AutoBackend.classify must be bit-identical to CPUBackend."""
    model_dir = Path(__file__).parent / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        import pytest

        pytest.skip("cnn_digits fixture not generated")
    ocr = FixedFontOCR(model_path=model_dir, backend="auto")
    rng = np.random.default_rng(7)
    glyphs = (rng.random((4, 24, 24)) > 0.5).astype(np.uint8) * 255
    got = ocr._backend.classify(glyphs)
    ref = CPUBackend(ocr.model.weights, ocr.model.input_size).classify(glyphs)
    assert np.array_equal(got.char_ids, ref.char_ids)
    assert np.abs(got.scores - ref.scores).max() < 1e-5
