"""CPU/WGPU numeric consistency tests for the TinyCNN backend.

The WGPU path is FP32 and every layer is verified separately against the
numpy reference (max abs error < 1e-4), and the final argmax/top-1 ids must
match exactly. The tests skip when ``wgpu`` or a GPU adapter is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.backends import CPUBackend, WGPUBackend
from fixedfontocr.cnn import (
    conv3x3,
    dwconv3x3,
    gap,
    linear,
    pointwise,
    relu,
)
from fixedfontocr.model import write_cnn_model

from conftest import render_text


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
def wgpu_weights() -> dict[str, np.ndarray]:
    return _random_weights(10, seed=11)


@pytest.fixture(scope="module")
def wgpu_backend(wgpu_weights) -> WGPUBackend:
    pytest.importorskip("wgpu")
    try:
        return WGPUBackend(wgpu_weights)
    except Exception as exc:  # no adapter / driver problem
        pytest.skip(f"WGPU adapter unavailable: {exc}")


@pytest.fixture(scope="module")
def random_glyphs() -> np.ndarray:
    rng = np.random.default_rng(3)
    return (rng.random((5, 24, 24)) > 0.5).astype(np.uint8) * 255


def _nhwc(a: np.ndarray) -> np.ndarray:
    """NCHW -> NHWC."""
    return np.transpose(a, (0, 2, 3, 1))


def _nchw(a: np.ndarray) -> np.ndarray:
    """NHWC -> NCHW."""
    return np.transpose(a, (0, 3, 1, 2))


def _cpu_reference_layers(weights, glyphs):
    x = glyphs.astype(np.float32)[:, None, :, :] * (1.0 / 255.0)
    c1 = relu(conv3x3(x, weights["conv1.weight"], weights["conv1.bias"], stride=2))
    d1 = relu(dwconv3x3(c1, weights["dw1.weight"], weights["dw1.bias"]))
    p1 = relu(pointwise(d1, weights["pw1.weight"], weights["pw1.bias"], stride=2))
    d2 = relu(dwconv3x3(p1, weights["dw2.weight"], weights["dw2.bias"]))
    p2 = relu(pointwise(d2, weights["pw2.weight"], weights["pw2.bias"], stride=2))
    pooled = gap(p2)
    logits = linear(pooled, weights["fc.weight"], weights["fc.bias"])
    return {
        "norm": np.pad(_nhwc(x), ((0, 0), (0, 0), (0, 0), (0, 3))),
        "c1": _nhwc(c1),
        "d1": _nhwc(d1),
        "p1": _nhwc(p1),
        "d2": _nhwc(d2),
        "p2": _nhwc(p2),
        "gap": pooled,
        "linear": logits,
    }


def _assert_close(got: np.ndarray, ref: np.ndarray, what: str) -> None:
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"{what}: shape {got.shape} != {ref.shape}"
    assert np.abs(got - ref).max() < 1e-4, f"{what}: max diff {np.abs(got - ref).max()}"


def test_wgpu_normalize_matches_cpu(wgpu_backend, random_glyphs, wgpu_weights):
    ref = _cpu_reference_layers(wgpu_weights, random_glyphs)["norm"]
    got = wgpu_backend.normalize(random_glyphs).reshape(ref.shape)
    _assert_close(got, ref, "normalize")


def test_wgpu_conv1_matches_cpu(wgpu_backend, random_glyphs, wgpu_weights):
    ref = _cpu_reference_layers(wgpu_weights, random_glyphs)["c1"]
    norm = wgpu_backend.normalize(random_glyphs).reshape(
        random_glyphs.shape[0], 24, 24, 4
    )
    got = wgpu_backend.conv1(norm).reshape(ref.shape)
    _assert_close(got, ref, "conv1")


def test_wgpu_dw1_matches_cpu(wgpu_backend, random_glyphs, wgpu_weights):
    layers = _cpu_reference_layers(wgpu_weights, random_glyphs)
    norm = wgpu_backend.normalize(random_glyphs).reshape(
        random_glyphs.shape[0], 24, 24, 4
    )
    c1 = wgpu_backend.conv1(norm).reshape(layers["c1"].shape)
    got = wgpu_backend.dw1(c1).reshape(layers["d1"].shape)
    _assert_close(got, layers["d1"], "dw1")


def test_wgpu_pw1_matches_cpu(wgpu_backend, random_glyphs, wgpu_weights):
    layers = _cpu_reference_layers(wgpu_weights, random_glyphs)
    norm = wgpu_backend.normalize(random_glyphs).reshape(
        random_glyphs.shape[0], 24, 24, 4
    )
    c1 = wgpu_backend.conv1(norm).reshape(layers["c1"].shape)
    d1 = wgpu_backend.dw1(c1).reshape(layers["d1"].shape)
    got = wgpu_backend.pw1(d1).reshape(layers["p1"].shape)
    _assert_close(got, layers["p1"], "pw1")


def test_wgpu_dw2_matches_cpu(wgpu_backend, random_glyphs, wgpu_weights):
    layers = _cpu_reference_layers(wgpu_weights, random_glyphs)
    norm = wgpu_backend.normalize(random_glyphs).reshape(
        random_glyphs.shape[0], 24, 24, 4
    )
    c1 = wgpu_backend.conv1(norm).reshape(layers["c1"].shape)
    d1 = wgpu_backend.dw1(c1).reshape(layers["d1"].shape)
    p1 = wgpu_backend.pw1(d1).reshape(layers["p1"].shape)
    got = wgpu_backend.dw2(p1).reshape(layers["d2"].shape)
    _assert_close(got, layers["d2"], "dw2")


def test_wgpu_pw2_matches_cpu(wgpu_backend, random_glyphs, wgpu_weights):
    layers = _cpu_reference_layers(wgpu_weights, random_glyphs)
    norm = wgpu_backend.normalize(random_glyphs).reshape(
        random_glyphs.shape[0], 24, 24, 4
    )
    c1 = wgpu_backend.conv1(norm).reshape(layers["c1"].shape)
    d1 = wgpu_backend.dw1(c1).reshape(layers["d1"].shape)
    p1 = wgpu_backend.pw1(d1).reshape(layers["p1"].shape)
    d2 = wgpu_backend.dw2(p1).reshape(layers["d2"].shape)
    got = wgpu_backend.pw2(d2).reshape(layers["p2"].shape)
    _assert_close(got, layers["p2"], "pw2")


def test_wgpu_gap_matches_cpu(wgpu_backend, random_glyphs, wgpu_weights):
    layers = _cpu_reference_layers(wgpu_weights, random_glyphs)
    norm = wgpu_backend.normalize(random_glyphs).reshape(
        random_glyphs.shape[0], 24, 24, 4
    )
    c1 = wgpu_backend.conv1(norm).reshape(layers["c1"].shape)
    d1 = wgpu_backend.dw1(c1).reshape(layers["d1"].shape)
    p1 = wgpu_backend.pw1(d1).reshape(layers["p1"].shape)
    d2 = wgpu_backend.dw2(p1).reshape(layers["d2"].shape)
    p2 = wgpu_backend.pw2(d2).reshape(layers["p2"].shape)
    got = wgpu_backend.gap(p2).reshape(layers["gap"].shape)
    _assert_close(got, layers["gap"], "gap")


def test_wgpu_linear_matches_cpu(wgpu_backend, random_glyphs, wgpu_weights):
    layers = _cpu_reference_layers(wgpu_weights, random_glyphs)
    got = wgpu_backend.linear(layers["gap"]).reshape(layers["linear"].shape)
    _assert_close(got, layers["linear"], "linear")


def test_wgpu_argmax_matches_cpu(wgpu_backend):
    rng = np.random.default_rng(9)
    for classes in (1, 2, 5, 10, 63, 64, 65, 3000, 7000):
        logits = rng.standard_normal((6, classes)).astype(np.float32)
        g_ids, g_scores = wgpu_backend.argmax(logits)
        order = np.argsort(-logits, axis=1, kind="stable")
        c_ids = order[:, 0]
        second = (
            -np.inf
            if classes == 1
            else logits[np.arange(6), order[:, 1]]
        )
        c_scores = logits[np.arange(6), c_ids] - second
        assert np.array_equal(g_ids, c_ids), f"argmax ids differ at C={classes}"
        if classes == 1:
            assert np.all(np.isinf(g_scores)) and np.all(np.isinf(c_scores))
        else:
            assert np.abs(g_scores - c_scores).max() < 1e-4, f"argmax scores at C={classes}"


def test_wgpu_fused_linear_argmax_matches_separate(wgpu_backend, random_glyphs, wgpu_weights):
    layers = _cpu_reference_layers(wgpu_weights, random_glyphs)
    fused_ids, fused_scores = wgpu_backend.fused(layers["gap"])
    lin = wgpu_backend.linear(layers["gap"]).reshape(layers["linear"].shape)
    lin_ids, lin_scores = wgpu_backend.argmax(lin)
    assert np.array_equal(fused_ids, lin_ids)
    assert np.abs(fused_scores - lin_scores).max() < 1e-4


def test_wgpu_classify_matches_cpu(wgpu_backend, random_glyphs):
    cpu = CPUBackend(wgpu_backend.weights)
    got = wgpu_backend.classify(random_glyphs)
    ref = cpu.classify(random_glyphs)
    assert np.array_equal(got.char_ids, ref.char_ids)
    assert np.abs(got.scores - ref.scores).max() < 1e-4


def test_wgpu_classify_various_batch_sizes(wgpu_backend, wgpu_weights):
    cpu = CPUBackend(wgpu_weights)
    rng = np.random.default_rng(17)
    for n in (1, 2, 8, 64, 65, 200):
        glyphs = (rng.random((n, 24, 24)) > 0.5).astype(np.uint8) * 255
        got = wgpu_backend.classify(glyphs)
        ref = cpu.classify(glyphs)
        assert len(got.char_ids) == n
        assert np.array_equal(got.char_ids, ref.char_ids), f"batch {n}"
        assert np.abs(got.scores - ref.scores).max() < 1e-4, f"batch {n}"


def test_wgpu_empty_batch(wgpu_backend):
    result = wgpu_backend.classify(np.empty((0, 24, 24), dtype=np.uint8))
    assert result.char_ids.shape == (0,)
    assert result.scores.shape == (0,)


def test_wgpu_recognize_end_to_end(font_path):
    model_dir = Path(__file__).parent / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        pytest.skip("cnn_digits fixture not generated; run scripts/train_tinycnn.py")
    pytest.importorskip("wgpu")
    try:
        gpu_ocr = FixedFontOCR(model_path=model_dir, backend="wgpu")
    except Exception as exc:
        pytest.skip(f"WGPU adapter unavailable: {exc}")
    cpu_ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    for text in ("1234567890", "42", "9876543210", "555555"):
        image = render_text(text, font_path)
        got = gpu_ocr.recognize(image)
        ref = cpu_ocr.recognize(image)
        assert got.text == ref.text == text
        assert [c.char for c in got.chars] == list(text)


def test_wgpu_recognize_cjk_end_to_end(font_path):
    """Same rendered image through CPU and WGPU must produce the same text."""
    model_dir = Path(__file__).parent / "fixtures" / "cnn_cjk"
    if not (model_dir / "config.json").exists():
        pytest.skip("cnn_cjk fixture not generated; run scripts/train_tinycnn.py")
    pytest.importorskip("wgpu")
    try:
        gpu_ocr = FixedFontOCR(model_path=model_dir, backend="wgpu")
    except Exception as exc:
        pytest.skip(f"WGPU adapter unavailable: {exc}")
    cpu_ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    for text in ("获得金币1000", "金币", "1000", "获得"):
        image = render_text(text, font_path)
        got = gpu_ocr.recognize(image)
        ref = cpu_ocr.recognize(image)
        assert got.text == ref.text == text


def test_wgpu_rejects_wrong_input_size(wgpu_backend):
    with pytest.raises(ValueError):
        wgpu_backend.classify(np.zeros((2, 10, 10), dtype=np.uint8))


def test_wgpu_random_model_matches_cpu(tmp_path, wgpu_weights):
    """A freshly written random-weight model must work through the public API."""
    pytest.importorskip("wgpu")
    charset = list("0123456789")
    out = tmp_path / "cnn"
    write_cnn_model(out, charset, wgpu_weights)
    try:
        gpu_ocr = FixedFontOCR(model_path=out, backend="wgpu")
    except Exception as exc:
        pytest.skip(f"WGPU adapter unavailable: {exc}")
    cpu_ocr = FixedFontOCR(model_path=out, backend="cpu")
    rng = np.random.default_rng(5)
    glyphs = (rng.random((4, 24, 24)) > 0.5).astype(np.uint8) * 255

    # Feed glyphs directly through both backends' public API.
    got = gpu_ocr._backend.classify(glyphs)
    ref = cpu_ocr._backend.classify(glyphs)
    assert np.array_equal(got.char_ids, ref.char_ids)
    assert np.abs(got.scores - ref.scores).max() < 1e-4


def test_wgpu_forward_logits_matches_cpu(wgpu_backend, random_glyphs, wgpu_weights):
    """The lattice scorer uses raw logits; WGPU must match the CPU reference."""
    cpu = CPUBackend(wgpu_weights)
    cpu_logits = cpu.forward_logits(random_glyphs)
    gpu_logits = wgpu_backend.forward_logits(random_glyphs)
    assert gpu_logits.shape == cpu_logits.shape
    assert np.abs(gpu_logits - cpu_logits).max() < 1e-4
