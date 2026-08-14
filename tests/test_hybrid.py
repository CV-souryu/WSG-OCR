from __future__ import annotations

import numpy as np

from fixedfontocr import FixedFontOCR, write_hybrid_model
from fixedfontocr.fontgen import build_templates

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


def _hybrid_model(tmp_path, font_path, template_threshold=0.95, cnn_threshold=-1e9):
    charset = "0123456789"
    chars, templates = build_templates(font_path, list(charset), render_size=32)
    weights = _random_weights(len(chars))
    out = tmp_path / "hybrid"
    write_hybrid_model(
        out,
        chars,
        templates,
        weights,
        template_threshold=template_threshold,
        cnn_threshold=cnn_threshold,
        font_path=font_path,
    )
    return out


def test_hybrid_template_level_returns_directly(tmp_path, font_path):
    model_dir = _hybrid_model(tmp_path, font_path)
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("1234567890", font_path)
    result = ocr.recognize(image)
    assert result.text == "1234567890"
    assert all(c.confidence >= 0.95 for c in result.chars)


def test_hybrid_cnn_fallback_runs(tmp_path, font_path):
    model_dir = _hybrid_model(
        tmp_path, font_path, template_threshold=1.01, cnn_threshold=-1e9
    )
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("1234567890", font_path)
    result = ocr.recognize(image)
    assert len(result.text) == 10
    assert all(ch in "0123456789" for ch in result.text)


def test_hybrid_unknown_level(tmp_path, font_path):
    model_dir = _hybrid_model(
        tmp_path, font_path, template_threshold=1.01, cnn_threshold=1e9
    )
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("1234567890", font_path)
    result = ocr.recognize(image)
    assert result.text == "??????????"


def test_hybrid_allowed_chars(tmp_path, font_path):
    model_dir = _hybrid_model(tmp_path, font_path)
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("13579", font_path)
    result = ocr.recognize(image, allowed_chars="13579")
    assert result.text == "13579"


def test_hybrid_model_loads_both_weights(tmp_path, font_path):
    from fixedfontocr.model import load_model

    model_dir = _hybrid_model(tmp_path, font_path)
    model = load_model(model_dir)
    assert model.classifier == "hybrid"
    assert model.templates is not None
    assert model.weights is not None


def test_hybrid_auto_backend_matches_cpu(tmp_path, font_path):
    model_dir = _hybrid_model(
        tmp_path, font_path, template_threshold=1.01, cnn_threshold=-1e9
    )
    cpu = FixedFontOCR(model_path=model_dir, backend="cpu")
    auto = FixedFontOCR(model_path=model_dir, backend="auto")
    image = render_text("1234567890", font_path)
    assert auto.recognize(image).text == cpu.recognize(image).text


def test_hybrid_wgpu_backend_matches_cpu(tmp_path, font_path):
    import pytest

    pytest.importorskip("wgpu")
    model_dir = _hybrid_model(
        tmp_path, font_path, template_threshold=1.01, cnn_threshold=-1e9
    )
    try:
        wgpu_ocr = FixedFontOCR(model_path=model_dir, backend="wgpu")
    except Exception as exc:
        pytest.skip(f"WGPU adapter unavailable: {exc}")
    cpu_ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    image = render_text("1234567890", font_path)
    assert wgpu_ocr.recognize(image).text == cpu_ocr.recognize(image).text
