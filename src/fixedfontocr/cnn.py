"""TinyCNN CPU forward pass.

The network is fixed per the design doc and uses only the primitives a WGPU
backend can implement trivially:

    Input           1 x 24 x 24
    Conv 3x3        1 -> 8   stride 2, ReLU
    DWConv 3x3      8 -> 8   stride 1, ReLU
    Pointwise 1x1   8 -> 16  stride 2, ReLU
    DWConv 3x3      16 -> 16 stride 1, ReLU
    Pointwise 1x1   16 -> 32 stride 2, ReLU
    GlobalAvgPool   32
    Linear          32 -> num_chars
    Argmax

All convolutions use SAME (zero) padding and channels-last-free numpy
vectorization via einsum - no per-pixel Python loops.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .classifier import Classifier
from .preprocess import normalize
from .postprocess import top2
from .types import ClassificationBatch, Profile


def prepare_weights(
    weights: dict[str, NDArray[np.float32]],
) -> dict[str, NDArray[np.float32]]:
    """Return inference-ready weights: contiguous float32 copies.

    Conversion is done once at construction (classifier/backend init) so the
    hot forward path never pays for ``astype``/``ascontiguousarray``.
    """

    prepared: dict[str, NDArray[np.float32]] = {}
    for name, arr in weights.items():
        arr = np.asarray(arr)
        if arr.dtype != np.float32 or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        prepared[name] = arr
    return prepared


def _im2col(
    x: NDArray[np.float32], kh: int, kw: int, pad: int
) -> NDArray[np.float32]:
    """Extract ``kh x kw`` patches from ``(N, C, H, W)`` with zero padding."""
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = h + 2 * pad - kh + 1
    ow = w + 2 * pad - kw + 1
    shape = (n, c, oh, ow, kh, kw)
    strides = xp.strides[:2] + (xp.strides[2], xp.strides[3]) + (
        xp.strides[2],
        xp.strides[3],
    )
    return np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)


def _im2col_stride2(
    x: NDArray[np.float32], kh: int, kw: int, pad: int
) -> NDArray[np.float32]:
    """Strided ``kh x kw`` patches for a stride-2 SAME convolution.

    Only the output positions of the strided convolution are materialized
    (as a view), so the useless intermediate feature map of the old
    ``conv full -> slice`` implementation never exists.
    """

    n, c, h, w = x.shape
    oh = (h + 2 * pad - kh) // 2 + 1
    ow = (w + 2 * pad - kw) // 2 + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    shape = (n, c, oh, ow, kh, kw)
    strides = xp.strides[:2] + (xp.strides[2] * 2, xp.strides[3] * 2) + (
        xp.strides[2],
        xp.strides[3],
    )
    return np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)


def conv3x3(
    x: NDArray[np.float32],
    w: NDArray[np.float32],
    b: NDArray[np.float32],
    stride: int = 1,
) -> NDArray[np.float32]:
    """3x3 convolution, SAME padding. ``w`` is ``(OC, IC, 3, 3)``."""
    if stride == 2:
        patches = _im2col_stride2(x, 3, 3, pad=1)
        out = np.einsum("nihwab,o iab->nohw", patches, w, optimize=True)
        out += b.reshape(1, -1, 1, 1)
        return out.astype(np.float32)
    patches = _im2col(x, 3, 3, pad=1)
    out = np.einsum("nihwab,o iab->nohw", patches, w, optimize=True)
    out += b.reshape(1, -1, 1, 1)
    return out.astype(np.float32)


def dwconv3x3(
    x: NDArray[np.float32],
    w: NDArray[np.float32],
    b: NDArray[np.float32],
    stride: int = 1,
) -> NDArray[np.float32]:
    """Depthwise 3x3 convolution, SAME padding. ``w`` is ``(C, 3, 3)``."""
    if stride == 2:
        full = dwconv3x3(x, w, b, stride=1)
        return full[:, :, ::2, ::2]
    patches = _im2col(x, 3, 3, pad=1)
    out = np.einsum("nchwab,cab->nchw", patches, w, optimize=True)
    out += b.reshape(1, -1, 1, 1)
    return out.astype(np.float32)


def pointwise(
    x: NDArray[np.float32],
    w: NDArray[np.float32],
    b: NDArray[np.float32],
    stride: int = 1,
) -> NDArray[np.float32]:
    """1x1 convolution. ``w`` is ``(OC, IC)``."""
    if stride == 2:
        # Slice before the 1x1 matmul: the skipped positions are never
        # touched, which removes 3/4 of the pointwise arithmetic.
        x = x[:, :, ::2, ::2]
        out = np.einsum("nchw,oc->nohw", x, w, optimize=True)
        out += b.reshape(1, -1, 1, 1)
        return out.astype(np.float32)
    out = np.einsum("nchw,oc->nohw", x, w, optimize=True)
    out += b.reshape(1, -1, 1, 1)
    return out.astype(np.float32)


def relu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.maximum(x, 0.0).astype(np.float32)


def gap(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Global average pool: ``(N, C, H, W) -> (N, C)``."""
    return np.mean(x, axis=(2, 3)).astype(np.float32)


def linear(
    x: NDArray[np.float32], w: NDArray[np.float32], b: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Dense layer. ``x`` is ``(N, C)``, ``w`` is ``(OC, C)``."""
    return (x @ w.T + b).astype(np.float32)


def softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)


def forward_with_activations(
    x: NDArray[np.float32],
    weights: dict[str, NDArray[np.float32]],
) -> dict[str, NDArray[np.float32]]:
    """Run the fixed TinyCNN and return every stage's activations.

    ``weights`` must contain: conv1.weight/bias, dw1.weight/bias,
    pw1.weight/bias, dw2.weight/bias, pw2.weight/bias, fc.weight/bias.

    Returns a dict with ``conv1``, ``dw1``, ``pw1``, ``dw2``, ``pw2``,
    ``gap`` and ``logits``; all activations are in NCHW except ``gap``
    (``(N, C)``) and ``logits`` (``(N, classes)``). This is the reference
    used by ``tools/train/export_model.py`` to emit per-layer test vectors.
    """

    if any(
        arr.dtype != np.float32 or not arr.flags.c_contiguous
        for arr in weights.values()
    ):
        weights = prepare_weights(weights)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x[None, None, :, :]
    if x.ndim == 3:
        x = x[None, :, :, :]
    if x.shape[1] != 1:
        x = x[:, :1, :, :]

    c1 = relu(conv3x3(x, weights["conv1.weight"], weights["conv1.bias"], stride=2))
    d1 = relu(dwconv3x3(c1, weights["dw1.weight"], weights["dw1.bias"]))
    p1 = relu(pointwise(d1, weights["pw1.weight"], weights["pw1.bias"], stride=2))
    d2 = relu(dwconv3x3(p1, weights["dw2.weight"], weights["dw2.bias"]))
    p2 = relu(pointwise(d2, weights["pw2.weight"], weights["pw2.bias"], stride=2))
    v = gap(p2)
    return {
        "conv1": c1,
        "dw1": d1,
        "pw1": p1,
        "dw2": d2,
        "pw2": p2,
        "gap": v,
        "logits": linear(v, weights["fc.weight"], weights["fc.bias"]),
    }


def forward(
    x: NDArray[np.float32],
    weights: dict[str, NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Run the fixed TinyCNN and return logits ``(N, num_chars)``."""
    return forward_with_activations(x, weights)["logits"]


class TinyCNNClassifier(Classifier):
    """Classifier interface adapter around :func:`forward`."""

    def __init__(
        self,
        weights: dict[str, NDArray[np.float32]],
        charset: list[str],
        input_size: int = 24,
    ):
        self.weights = weights
        self.charset = list(charset)
        self.input_size = input_size
        if weights["fc.weight"].shape[0] != len(charset):
            raise ValueError("fc output size must match charset length")
        self.weights = prepare_weights(weights)

    def logits(self, mask: NDArray[np.bool_]) -> NDArray[np.float32]:
        glyph = normalize(mask, self.input_size).astype(np.float32) / 255.0
        return forward(glyph, self.weights)[0]

    def classify_batch(
        self,
        glyphs: NDArray[np.uint8],
        allowed_ids: set[int] | None = None,
        top_k: int = 2,
    ) -> ClassificationBatch:
        """Classify an ``uint8 [N, H, W]`` batch and return Top-K info.

        ``top_k`` is the public API knob for future dictionary decoding
        (e.g. ``top_k=5``); V1 returns the first two candidates in
        :class:`ClassificationBatch`. ``allowed_ids`` restricts the argmax
        to a UI-limited subset.
        """

        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        glyphs = np.asarray(glyphs, dtype=np.uint8)
        if glyphs.ndim != 3:
            raise ValueError(f"glyphs must be [N, H, W], got {glyphs.shape}")
        if glyphs.shape[1:] != (self.input_size, self.input_size):
            raise ValueError(
                f"expected {self.input_size}x{self.input_size} glyphs, "
                f"got {glyphs.shape[1:]}"
            )
        n = glyphs.shape[0]
        if n == 0:
            return ClassificationBatch(
                ids=np.empty(0, dtype=np.int32),
                top1=np.empty(0, dtype=np.float32),
                top2=np.empty(0, dtype=np.float32),
                margins=np.empty(0, dtype=np.float32),
            )
        if allowed_ids is not None and not allowed_ids:
            return ClassificationBatch(
                ids=np.full(n, -1, dtype=np.int32),
                top1=np.full(n, -np.inf, dtype=np.float32),
                top2=np.full(n, -np.inf, dtype=np.float32),
                margins=np.full(n, 0.0, dtype=np.float32),
            )
        x = glyphs.astype(np.float32)[:, None, :, :] * (1.0 / 255.0)
        logits = forward(x, self.weights)
        if allowed_ids is not None:
            masked = np.full_like(logits, -np.inf)
            idx = np.fromiter(sorted(allowed_ids), dtype=np.int64)
            masked[:, idx] = logits[:, idx]
            logits = masked
        ids, top1, top2v, margins = top2(logits)
        return ClassificationBatch(ids=ids, top1=top1, top2=top2v, margins=margins)

    def __call__(
        self,
        mask: NDArray[np.bool_],
        profile: Profile,
        allowed_ids: set[int] | None = None,
    ) -> tuple[str, float]:
        glyph = normalize(mask, self.input_size)[None, :, :]
        batch = self.classify_batch(glyph, allowed_ids=allowed_ids)
        i = int(batch.ids[0])
        if i < 0:
            return "?", 0.0
        return self.charset[i], float(batch.margins[0])
