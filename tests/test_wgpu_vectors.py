"""WGPU layer tests driven by exported test_vectors.npz.

``tools/train/export_model.py`` emits ``test_vectors.npz`` containing the input,
every layer's activation, the final logits and the final char_id. This test
uses those vectors as the source of truth for the WGPU backend, exactly the
workflow described in the export plan: CPU/WGPU unit tests consume the same
golden file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from fixedfontocr.backends import WGPUBackend  # noqa: E402
from tools.train import tinycnn_arch  # noqa: E402
from tools.train.export_model import make_test_vectors  # noqa: E402


def _nhwc(a: np.ndarray) -> np.ndarray:
    return np.transpose(a, (0, 2, 3, 1))


def _nchw(a: np.ndarray) -> np.ndarray:
    return np.transpose(a, (0, 3, 1, 2))


def _pad_nhwc(a: np.ndarray, channels: int) -> np.ndarray:
    """Pad channels to a vec4 multiple (NHWC storage layout)."""
    out = np.zeros((*a.shape[:3], channels), dtype=np.float32)
    out[..., : a.shape[-1]] = a
    return out


@pytest.fixture(scope="module")
def vectors_and_weights():
    torch.manual_seed(21)
    charset = list("0123456789")
    model = tinycnn_arch.TinyCNNBN(len(charset))
    model.eval()
    weights = {
        name: arr.detach().numpy()
        for name, arr in tinycnn_arch.export_folded_weights(model).items()
    }
    out_dir = Path("/tmp") / "ffocr_wgpu_vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_test_vectors(weights, out_dir, None, num_samples=4, seed=21)
    return np.load(out_dir / "test_vectors.npz"), weights


@pytest.fixture(scope="module")
def wgpu(vectors_and_weights):
    pytest.importorskip("wgpu")
    _, weights = vectors_and_weights
    try:
        return WGPUBackend(weights)
    except Exception as exc:
        pytest.skip(f"WGPU adapter unavailable: {exc}")


def test_vectors_normalize(wgpu, vectors_and_weights):
    vec, _ = vectors_and_weights
    x = vec["input"]
    ref = np.pad(
        _nhwc(x.astype(np.float32)[:, None] / 255.0),
        ((0, 0), (0, 0), (0, 0), (0, 3)),
    )
    got = wgpu.normalize(x).reshape(ref.shape)
    assert np.abs(got - ref).max() < 1e-5


def test_vectors_layers_match_wgpu(wgpu, vectors_and_weights):
    vec, _ = vectors_and_weights
    x = vec["input"]
    norm = wgpu.normalize(x).reshape(4, 24, 24, 4)
    c1 = wgpu.conv1(norm).reshape(4, 12, 12, 8)
    d1 = wgpu.dw1(c1).reshape(4, 12, 12, 8)
    p1 = wgpu.pw1(d1).reshape(4, 6, 6, 16)
    d2 = wgpu.dw2(p1).reshape(4, 6, 6, 16)
    p2 = wgpu.pw2(d2).reshape(4, 3, 3, 32)
    pooled = wgpu.gap(p2).reshape(4, 32)
    for name, got, ref in (
        ("conv1", c1, _pad_nhwc(_nhwc(vec["conv1"]), 8)),
        ("dw1", d1, _pad_nhwc(_nhwc(vec["dw1"]), 8)),
        ("pw1", p1, _pad_nhwc(_nhwc(vec["pw1"]), 16)),
        ("dw2", d2, _pad_nhwc(_nhwc(vec["dw2"]), 16)),
        ("pw2", p2, _pad_nhwc(_nhwc(vec["pw2"]), 32)),
        ("gap", pooled, vec["gap"]),
    ):
        assert got.shape == ref.shape, name
        assert np.abs(got - ref).max() < 1e-4, name


def test_vectors_final_char_id_matches_wgpu(wgpu, vectors_and_weights):
    vec, _ = vectors_and_weights
    norm = wgpu.normalize(vec["input"]).reshape(4, 24, 24, 4)
    c1 = wgpu.conv1(norm).reshape(4, 12, 12, 8)
    d1 = wgpu.dw1(c1).reshape(4, 12, 12, 8)
    p1 = wgpu.pw1(d1).reshape(4, 6, 6, 16)
    d2 = wgpu.dw2(p1).reshape(4, 6, 6, 16)
    p2 = wgpu.pw2(d2).reshape(4, 3, 3, 32)
    pooled = wgpu.gap(p2).reshape(4, 32)
    ids, _ = wgpu.fused(pooled)
    assert np.array_equal(ids, vec["char_id"])
