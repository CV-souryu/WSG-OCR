from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.cnn import forward, forward_with_activations
from fixedfontocr.model import load_model, write_cnn_model
from fixedfontocr.reference_cnn import (
    reference_conv3x3,
    reference_dwconv3x3,
    reference_forward,
    reference_pointwise,
)

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


def test_cnn_model_roundtrip(tmp_path):
    charset = list("0123456789")
    weights = _random_weights(len(charset))
    out = tmp_path / "cnn"
    write_cnn_model(out, charset, weights)

    model = load_model(out)
    assert model.classifier == "tinycnn"
    assert model.templates is None
    assert model.charset == charset
    for name, arr in weights.items():
        assert np.allclose(model.weights[name], arr, atol=1e-6), name


def test_forward_matches_torch_reference():
    torch = pytest.importorskip("torch")
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
            self.d1 = nn.Conv2d(8, 8, 3, padding=1, groups=8)
            self.p1 = nn.Conv2d(8, 16, 1, stride=2)
            self.d2 = nn.Conv2d(16, 16, 3, padding=1, groups=16)
            self.p2 = nn.Conv2d(16, 32, 1, stride=2)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = torch.relu(self.c1(x))
            x = torch.relu(self.d1(x))
            x = torch.relu(self.p1(x))
            x = torch.relu(self.d2(x))
            x = torch.relu(self.p2(x))
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    torch.manual_seed(7)
    model = Tiny()
    weights = _random_weights(10, seed=7)
    state = {
        "conv1.weight": model.c1.weight,
        "conv1.bias": model.c1.bias,
        "dw1.weight": model.d1.weight,
        "dw1.bias": model.d1.bias,
        "pw1.weight": model.p1.weight,
        "pw1.bias": model.p1.bias,
        "dw2.weight": model.d2.weight,
        "dw2.bias": model.d2.bias,
        "pw2.weight": model.p2.weight,
        "pw2.bias": model.p2.bias,
        "fc.weight": model.fc.weight,
        "fc.bias": model.fc.bias,
    }
    for name, tensor in state.items():
        weights[name][:] = tensor.detach().numpy().reshape(weights[name].shape)

    x = np.random.default_rng(1).random((3, 1, 24, 24), dtype=np.float32)
    ref = model(torch.from_numpy(x)).detach().numpy()
    got = forward(x, weights)
    assert np.abs(ref - got).max() < 1e-5
    assert np.array_equal(ref.argmax(1), got.argmax(1))


def test_cnn_recognize_digits(font_path):
    model_dir = Path(__file__).parent / "fixtures" / "cnn_digits"
    if not (model_dir / "config.json").exists():
        pytest.skip("cnn_digits fixture not generated; run scripts/train_tinycnn.py")
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    for text in ("1234567890", "42", "9876543210", "555555", "3141592653"):
        result = ocr.recognize(render_text(text, font_path))
        assert result.text == text
        assert result.confidence > 0.8


def test_cnn_recognize_chinese(font_path):
    model_dir = Path(__file__).parent / "fixtures" / "cnn_cjk"
    if not (model_dir / "config.json").exists():
        pytest.skip("cnn_cjk fixture not generated; run scripts/train_tinycnn.py")
    ocr = FixedFontOCR(model_path=model_dir, backend="cpu")
    for text in ("获得金币1000", "金币", "1000", "获得"):
        result = ocr.recognize(render_text(text, font_path))
        assert result.text == text


def test_cnn_rejects_bad_weight_size(tmp_path):
    charset = list("0123456789")
    weights = _random_weights(len(charset))
    out = tmp_path / "cnn"
    write_cnn_model(out, charset, weights)
    # Corrupt: drop one f32 value.
    data = (out / "weights.bin").read_bytes()
    (out / "weights.bin").write_bytes(data[:-4])
    with pytest.raises(ValueError):
        load_model(out)


def test_stride2_optimization_matches_reference():
    """Optimized stride-2 conv/pw must match the full-then-slice reference."""

    weights = _random_weights(64, seed=17)
    rng = np.random.default_rng(23)
    for n in (1, 4, 16):
        x = rng.random((n, 1, 24, 24), dtype=np.float32)
        ref = reference_forward(x, weights)
        got = forward(x, weights)
        assert np.abs(ref - got).max() < 1e-5, n
        assert np.array_equal(ref.argmax(axis=1), got.argmax(axis=1)), n


def test_stride2_optimization_layers_match_reference():
    weights = _random_weights(10, seed=5)
    x = np.random.default_rng(9).random((3, 1, 24, 24), dtype=np.float32)
    acts = forward_with_activations(x, weights)
    ref = {
        "conv1": np.maximum(
            reference_conv3x3(
                x, weights["conv1.weight"], weights["conv1.bias"], stride=2
            ),
            0.0,
        ),
        "pw1": np.maximum(
            reference_pointwise(
                np.maximum(
                    reference_dwconv3x3(
                        acts["conv1"], weights["dw1.weight"], weights["dw1.bias"]
                    ),
                    0.0,
                ),
                weights["pw1.weight"],
                weights["pw1.bias"],
                stride=2,
            ),
            0.0,
        ),
        "pw2": np.maximum(
            reference_pointwise(
                np.maximum(
                    reference_dwconv3x3(
                        acts["pw1"], weights["dw2.weight"], weights["dw2.bias"]
                    ),
                    0.0,
                ),
                weights["pw2.weight"],
                weights["pw2.bias"],
                stride=2,
            ),
            0.0,
        ),
    }
    for name in ("conv1", "pw1", "pw2"):
        assert np.abs(acts[name] - ref[name]).max() < 1e-5, name
