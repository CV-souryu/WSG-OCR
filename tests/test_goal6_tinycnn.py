"""Goal 6 acceptance: optimized NumPy TinyCNN V1.

The Goal 6 contract from ``fonts/goal``:

* stride-2 layers compute only their target output positions (no full
  feature map followed by ``::2``);
* the hot forward path avoids building an activation dict or forcing
  same-dtype copies;
* weights are converted to contiguous float32 once at construction;
* inference is batched;
* Top-K uses ``argpartition`` and the runtime never calls a full
  ``argsort`` over the charset.

The NumPy vs PyTorch max-error/argmax gates are exercised by
``tests/test_cnn.py`` and ``tests/test_goal5_tinycnn.py``; this file
verifies the optimization checklist itself.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixedfontocr import cnn as cnn_module  # noqa: E402
from fixedfontocr.cnn import (  # noqa: E402
    TinyCNNClassifier,
    conv3x3,
    dwconv3x3,
    forward,
    prepare_weights,
)
from fixedfontocr.postprocess import top2, topk  # noqa: E402
from fixedfontocr.reference_cnn import (  # noqa: E402
    reference_conv3x3,
    reference_dwconv3x3,
)


def _weights(num_classes: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "conv1.weight": rng.standard_normal(
            (8, 1, 3, 3), dtype=np.float32
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


def test_stride2_uses_direct_strided_positions_not_full_then_slice():
    src = (
        ROOT / "src" / "fixedfontocr" / "cnn.py"
    ).read_text(encoding="utf-8")
    # The stride-2 im2col helper advances through the padded input with
    # doubled spatial strides, so only output positions are materialized.
    assert "_im2col_stride2" in src
    assert "xp.strides[2] * 2" in src
    # The old full-feature-map-then-slice pattern must be gone.
    assert "full[:, :, ::2, ::2]" not in src


def test_conv_and_dwconv_stride2_match_full_then_slice_reference():
    rng = np.random.default_rng(21)
    x = rng.random((3, 4, 12, 12), dtype=np.float32)
    w = rng.standard_normal((8, 4, 3, 3), dtype=np.float32)
    b = rng.standard_normal(8, dtype=np.float32)
    got = conv3x3(x, w, b, stride=2)
    ref = reference_conv3x3(x, w, b, stride=2)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-5

    dw_w = rng.standard_normal((4, 3, 3), dtype=np.float32)
    dw_b = rng.standard_normal(4, dtype=np.float32)
    got = dwconv3x3(x[:, :4], dw_w, dw_b, stride=2)
    ref = reference_dwconv3x3(x[:, :4], dw_w, dw_b, stride=2)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-5


def test_hot_forward_does_not_build_activation_dict(monkeypatch):
    """``forward()`` must not route through the activation-dict helper."""

    def _boom(*args, **kwargs):
        raise AssertionError("hot forward built an activation dict")

    monkeypatch.setattr(cnn_module, "forward_with_activations", _boom)
    weights = _weights(seed=4)
    x = np.random.default_rng(8).random((4, 1, 24, 24), dtype=np.float32)
    logits = forward(x, weights)
    assert logits.shape == (4, 10)
    assert logits.dtype == np.float32


def test_prepare_weights_keeps_contiguous_f32_without_copy():
    weights = _weights(seed=1)
    prepared = prepare_weights(weights)
    for name, arr in weights.items():
        assert prepared[name] is arr
        assert prepared[name].dtype == np.float32
        assert prepared[name].flags.c_contiguous

    # Non-contiguous input is converted once into a C-contiguous f32 copy.
    weights["conv1.weight"] = np.asfortranarray(weights["conv1.weight"])
    prepared = prepare_weights(weights)
    assert prepared["conv1.weight"] is not weights["conv1.weight"]
    assert prepared["conv1.weight"].flags.c_contiguous
    assert prepared["conv1.weight"].dtype == np.float32


def test_batch_inference_matches_single_glyph_inference():
    weights = _weights(seed=9)
    x = np.random.default_rng(3).random((8, 1, 24, 24), dtype=np.float32)
    batch_logits = forward(x, weights)
    for i in range(x.shape[0]):
        single = forward(x[i : i + 1], weights)[0]
        assert np.abs(single - batch_logits[i]).max() < 1e-4


def test_topk_uses_partition_and_matches_stable_full_ranking():
    rng = np.random.default_rng(17)
    logits = rng.standard_normal((6, 37), dtype=np.float32)
    order = np.argsort(-logits, axis=-1, kind="stable")
    for k in (1, 2, 3, 5, 8, 37):
        ids, values = topk(logits, k)
        assert ids.shape == (6, k)
        assert values.shape == (6, k)
        assert np.array_equal(ids, order[:, :k])
        assert np.array_equal(
            values, logits[np.arange(6)[:, None], order[:, :k]]
        )

    # Ties keep first-occurrence (stable) ordering just like argmax.
    tied = np.array(
        [
            [0.5, 0.5, 0.1, 0.3],
            [1.0, -1.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ids, values = topk(tied, 4)
    assert np.array_equal(ids, [[0, 1, 3, 2], [0, 3, 1, 2]])
    assert np.allclose(
        values,
        [[0.5, 0.5, 0.3, 0.1], [1.0, 0.0, -1.0, -1.0]],
        atol=0.0,
    )


def test_classifier_topk_returns_ranked_arrays_when_requested():
    charset = list("0123456789")
    clf = TinyCNNClassifier(_weights(10, seed=6), charset)
    rng = np.random.default_rng(11)
    glyphs = (rng.random((5, 24, 24)) > 0.5).astype(np.uint8) * 255

    batch2 = clf.classify_batch(glyphs)
    assert batch2.topk_ids is None
    assert batch2.topk_logits is None

    batch5 = clf.classify_batch(glyphs, top_k=5)
    assert batch5.topk_ids.shape == (5, 5)
    assert batch5.topk_logits.shape == (5, 5)
    assert np.array_equal(batch5.ids, batch2.ids)
    assert np.allclose(batch5.top1, batch2.top1, atol=0.0)
    assert np.allclose(batch5.top2, batch2.top2, atol=0.0)
    assert np.array_equal(batch5.second_ids, batch2.second_ids)

    x = glyphs.astype(np.float32)[:, None, :, :] * (1.0 / 255.0)
    ref_ids, ref_values = topk(forward(x, clf.weights), 5)
    assert np.array_equal(batch5.topk_ids, ref_ids)
    assert np.allclose(batch5.topk_logits, ref_values, atol=0.0)


def test_runtime_topk_never_calls_full_argsort():
    src = (
        ROOT / "src" / "fixedfontocr" / "postprocess.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert "argsort" not in calls
    assert "argpartition" in calls


def test_top2_still_matches_partition_reference():
    rng = np.random.default_rng(23)
    logits = rng.standard_normal((12, 29), dtype=np.float32)
    ids, top1, top2v, margins = top2(logits)
    order = np.argsort(-logits, axis=-1, kind="stable")
    assert np.array_equal(ids, order[:, 0])
    assert np.allclose(top1, logits[np.arange(12), order[:, 0]])
    assert np.allclose(top2v, logits[np.arange(12), order[:, 1]])
    assert np.allclose(margins, top1 - top2v)
