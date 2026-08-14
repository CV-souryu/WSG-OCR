"""Goal 5 acceptance: TinyCNN V1 is frozen.

The Goal 5 contract from ``fonts/goal``:

* the network topology is fixed: 24x24 input, Conv3x3 (1->8, stride 2),
  DWConv, Pointwise (8->16, stride 2), DWConv, Pointwise (16->32, stride 2),
  GAP, Linear (32 -> charset);
* the only allowed input variations are one channel (binary or soft) and
  the experimental two-channel ``soft + binary`` stack;
* Transformer / LSTM / Attention / complex normalization must not appear
  in the TinyCNN path.

``fixedfontocr.cnn`` owns the frozen spec. Model load/export, the numpy
forward, the training mirror and the WGPU layer map all have to agree with
it; anything else is a drift that must fail loudly.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixedfontocr import write_hybrid_model  # noqa: E402
from fixedfontocr.backends import WGPUBackend  # noqa: E402
from fixedfontocr.cnn import (  # noqa: E402
    TINYCNN_V1_HEAD,
    TINYCNN_V1_INPUT_CHANNELS,
    TINYCNN_V1_INPUT_SIZE,
    TINYCNN_V1_LAYERS,
    TINYCNN_V1_NAME,
    TINYCNN_V1_TAIL,
    TinyCNNClassifier,
    cnn_tensor_shapes,
    forward,
    forward_with_activations,
    prepare_weights,
    validate_v1_weights,
)
from fixedfontocr.fontgen import build_templates  # noqa: E402
from fixedfontocr.model import (  # noqa: E402
    load_model,
    write_cnn_model,
    write_cnn_weights,
)


def _weights(num_classes: int = 10, input_channels: int = 1, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "conv1.weight": rng.standard_normal(
            (8, input_channels, 3, 3), dtype=np.float32
        ),
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


def test_frozen_v1_spec():
    """The architecture table is the Goal 5 document, verbatim."""

    assert TINYCNN_V1_NAME == "tinycnn_v1"
    assert TINYCNN_V1_INPUT_SIZE == 24
    assert TINYCNN_V1_INPUT_CHANNELS == (1, 2)
    assert TINYCNN_V1_HEAD == ("gap", 32)
    assert TINYCNN_V1_TAIL == ("linear", 32)
    assert TINYCNN_V1_LAYERS == (
        ("conv1", "conv3x3", 3, 2, (1, 2), 8, 1, "relu"),
        ("dw1", "dwconv3x3", 3, 1, 8, 8, 8, "relu"),
        ("pw1", "pointwise", 1, 2, 8, 16, 1, "relu"),
        ("dw2", "dwconv3x3", 3, 1, 16, 16, 16, "relu"),
        ("pw2", "pointwise", 1, 2, 16, 32, 1, "relu"),
    )


def test_tensor_shapes_match_frozen_spec():
    expected = [
        ("conv1.weight", (8, 1, 3, 3)),
        ("conv1.bias", (8,)),
        ("dw1.weight", (8, 3, 3)),
        ("dw1.bias", (8,)),
        ("pw1.weight", (16, 8)),
        ("pw1.bias", (16,)),
        ("dw2.weight", (16, 3, 3)),
        ("dw2.bias", (16,)),
        ("pw2.weight", (32, 16)),
        ("pw2.bias", (32,)),
        ("fc.weight", (1894, 32)),
        ("fc.bias", (1894,)),
    ]
    assert cnn_tensor_shapes(1894) == expected
    assert cnn_tensor_shapes(10, input_channels=2)[0] == (
        "conv1.weight",
        (8, 2, 3, 3),
    )
    with pytest.raises(ValueError, match="input_channels"):
        cnn_tensor_shapes(10, input_channels=3)


@pytest.mark.parametrize("input_channels", [1, 2])
def test_validate_accepts_v1_weights(input_channels):
    weights = _weights(input_channels=input_channels)
    shapes = validate_v1_weights(weights, num_classes=10)
    assert shapes == dict(cnn_tensor_shapes(10, input_channels))


def test_validate_rejects_extra_or_missing_tensors():
    weights = _weights()
    weights["bn1.weight"] = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="extra"):
        validate_v1_weights(weights)

    weights = _weights()
    del weights["dw2.bias"]
    with pytest.raises(ValueError, match="missing"):
        validate_v1_weights(weights)

    weights = _weights()
    weights["fc.weight"] = np.zeros((10, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        validate_v1_weights(weights)

    weights = _weights(input_channels=3)
    with pytest.raises(ValueError, match="input channels"):
        validate_v1_weights(weights)


def test_forward_rejects_drift_outside_frozen_v1():
    w1 = _weights(input_channels=1)
    assert forward(np.zeros((1, 1, 24, 24), dtype=np.float32), w1).shape == (1, 10)

    # Channel counts outside the frozen range are rejected, not truncated.
    with pytest.raises(ValueError, match="channels"):
        forward(np.zeros((1, 3, 24, 24), dtype=np.float32), w1)
    with pytest.raises(ValueError, match="expects"):
        forward(np.zeros((1, 2, 24, 24), dtype=np.float32), w1)

    # The input size is part of the freeze.
    with pytest.raises(ValueError, match="24x24"):
        forward(np.zeros((1, 1, 16, 16), dtype=np.float32), w1)

    # The two-channel soft+binary experiment runs through the same frozen
    # topology with conv1 widened to 2 input channels.
    w2 = _weights(input_channels=2)
    assert forward(np.zeros((1, 2, 24, 24), dtype=np.float32), w2).shape == (1, 10)


def test_forward_stages_are_exactly_the_frozen_set():
    weights = _weights()
    acts = forward_with_activations(
        np.zeros((2, 1, 24, 24), dtype=np.float32), weights
    )
    assert set(acts) == {"conv1", "dw1", "pw1", "dw2", "pw2", "gap", "logits"}


def test_runtime_has_no_forbidden_layers():
    """The numpy forward calls only the frozen Conv/DWConv/PW/GAP/Linear set."""

    src = (
        Path(__file__).resolve().parents[1]
        / "src" / "fixedfontocr" / "cnn.py"
    ).read_text(encoding="utf-8")
    calls = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id.lower())
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr.lower())
    forbidden = {
        "transformer",
        "lstm",
        "attention",
        "multihead",
        "embedding",
        "layernorm",
        "instancenorm",
        "groupnorm",
        "batchnorm",
    }
    assert not (forbidden & calls), forbidden & calls


def test_forward_parity_with_torch_is_frozen():
    torch = pytest.importorskip("torch")
    from torch import nn

    class FrozenTiny(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
            self.dw1 = nn.Conv2d(8, 8, 3, padding=1, groups=8)
            self.pw1 = nn.Conv2d(8, 16, 1, stride=2)
            self.dw2 = nn.Conv2d(16, 16, 3, padding=1, groups=16)
            self.pw2 = nn.Conv2d(16, 32, 1, stride=2)
            self.fc = nn.Linear(32, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.dw1(x))
            x = torch.relu(self.pw1(x))
            x = torch.relu(self.dw2(x))
            x = torch.relu(self.pw2(x))
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    torch.manual_seed(11)
    model = FrozenTiny(10)
    weights = _weights(10, seed=11)
    state = {
        "conv1.weight": model.conv1.weight,
        "conv1.bias": model.conv1.bias,
        "dw1.weight": model.dw1.weight,
        "dw1.bias": model.dw1.bias,
        "pw1.weight": model.pw1.weight,
        "pw1.bias": model.pw1.bias,
        "dw2.weight": model.dw2.weight,
        "dw2.bias": model.dw2.bias,
        "pw2.weight": model.pw2.weight,
        "pw2.bias": model.pw2.bias,
        "fc.weight": model.fc.weight,
        "fc.bias": model.fc.bias,
    }
    for name, tensor in state.items():
        weights[name][:] = tensor.detach().numpy().reshape(weights[name].shape)

    x = np.random.default_rng(13).random((4, 1, 24, 24), dtype=np.float32)
    ref = model(torch.from_numpy(x)).detach().numpy()
    got = forward(x, weights)
    assert np.abs(ref - got).max() < 1e-5
    assert np.array_equal(ref.argmax(1), got.argmax(1))


def test_prepare_weights_validates_frozen_set():
    prepared = prepare_weights(_weights())
    assert all(
        arr.dtype == np.float32 and arr.flags.c_contiguous
        for arr in prepared.values()
    )
    bad = _weights()
    bad["attention.weight"] = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="extra"):
        prepare_weights(bad)


def test_classifier_supports_single_channel_soft_and_binary():
    classifier = TinyCNNClassifier(_weights(10), list("0123456789"))
    rng = np.random.default_rng(5)
    binary = (rng.random((4, 24, 24)) > 0.6).astype(np.uint8) * 255
    soft = np.where(
        binary == 255,
        rng.integers(180, 256, binary.shape),
        rng.integers(0, 80, binary.shape),
    ).astype(np.uint8)
    b = classifier.classify_batch(binary)
    s = classifier.classify_batch(soft)
    assert b.ids.shape == (4,) and s.ids.shape == (4,)
    assert np.all(np.isfinite(b.margins)) and np.all(np.isfinite(s.margins))
    # The Top-K knob stays available even though V1 reports the first two.
    assert classifier.classify_batch(binary, top_k=5).ids.shape == (4,)


def test_model_config_marks_frozen_v1(tmp_path):
    charset = list("0123456789")
    out = tmp_path / "cnn"
    write_cnn_model(out, charset, _weights(len(charset)))
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["architecture"] == TINYCNN_V1_NAME
    assert load_model(out).classifier == "tinycnn"

    # An unknown architecture is rejected by the loader (freeze gate).
    config["architecture"] = "tinycnn_v2"
    (out / "config.json").write_text(
        json.dumps(config, indent=4) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="frozen"):
        load_model(out)

    # Legacy configs without the key default to the only frozen V1.
    config.pop("architecture")
    (out / "config.json").write_text(
        json.dumps(config, indent=4) + "\n", encoding="utf-8"
    )
    assert load_model(out).classifier == "tinycnn"


def test_hybrid_model_config_marks_frozen_v1(tmp_path, font_path):
    charset = list("0123456789")
    chars, templates = build_templates(font_path, charset, render_size=32)
    out = tmp_path / "hybrid"
    write_hybrid_model(out, chars, templates, _weights(len(charset)), font_path=font_path)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["architecture"] == TINYCNN_V1_NAME
    assert load_model(out).classifier == "hybrid"


def test_write_weights_rejects_non_frozen_tensor_set(tmp_path):
    weights = _weights(10)
    weights["bn1.weight"] = np.zeros(8, dtype=np.float32)
    with pytest.raises(ValueError, match="extra"):
        write_cnn_weights(tmp_path / "weights.bin", weights, 10)

    # The on-disk model format is frozen to the single-channel variant;
    # two-channel soft+binary is an in-memory experiment only.
    with pytest.raises(ValueError, match="frozen to 1 input channel"):
        write_cnn_weights(
            tmp_path / "weights2.bin", _weights(input_channels=2), 10
        )


def test_training_arch_matches_frozen_v1():
    torch = pytest.importorskip("torch")
    from tools.train import tinycnn_arch

    model = tinycnn_arch.TinyCNNBN(10)
    assert model.conv1.in_channels == 1
    assert model.conv1.out_channels == 8
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (2, 2)
    assert model.conv1.padding == (1, 1)
    assert model.dw1.in_channels == model.dw1.out_channels == model.dw1.groups == 8
    assert model.pw1.out_channels == 16 and model.pw1.stride == (2, 2)
    assert model.dw2.in_channels == model.dw2.out_channels == model.dw2.groups == 16
    assert model.pw2.out_channels == 32 and model.pw2.stride == (2, 2)
    assert model.fc.in_features == 32 and model.fc.out_features == 10

    weights = {
        name: arr.detach().numpy()
        for name, arr in tinycnn_arch.export_folded_weights(model).items()
    }
    validate_v1_weights(weights, num_classes=10)


def test_wgpu_layer_map_matches_frozen_v1():
    """The WGPU backend hard-codes the same frozen sizes and layer order."""

    assert (WGPUBackend._H, WGPUBackend._W) == (
        TINYCNN_V1_INPUT_SIZE,
        TINYCNN_V1_INPUT_SIZE,
    )
    assert (WGPUBackend._C1, WGPUBackend._C2, WGPUBackend._C3) == (8, 16, 32)
    frozen_names = [layer[0] for layer in TINYCNN_V1_LAYERS]
    mapped = [
        name
        for name in WGPUBackend._SHADER_FILES
        if name in set(frozen_names)
    ]
    assert mapped == frozen_names
