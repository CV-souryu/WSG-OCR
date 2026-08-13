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
from .types import Profile


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


def conv3x3(
    x: NDArray[np.float32],
    w: NDArray[np.float32],
    b: NDArray[np.float32],
    stride: int = 1,
) -> NDArray[np.float32]:
    """3x3 convolution, SAME padding. ``w`` is ``(OC, IC, 3, 3)``."""
    if stride == 2:
        full = conv3x3(x, w, b, stride=1)
        return full[:, :, ::2, ::2]
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
    out = np.einsum("nchw,oc->nohw", x, w, optimize=True)
    out += b.reshape(1, -1, 1, 1)
    if stride == 2:
        out = out[:, :, ::2, ::2]
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


def forward(
    x: NDArray[np.float32],
    weights: dict[str, NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Run the fixed TinyCNN and return logits ``(N, num_chars)``.

    ``weights`` must contain: conv1.weight/bias, dw1.weight/bias,
    pw1.weight/bias, dw2.weight/bias, pw2.weight/bias, fc.weight/bias.
    """

    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x[None, None, :, :]
    if x.ndim == 3:
        x = x[None, :, :, :]
    if x.shape[1] != 1:
        x = x[:, :1, :, :]

    a = relu(conv3x3(x, weights["conv1.weight"], weights["conv1.bias"], stride=2))
    a = relu(dwconv3x3(a, weights["dw1.weight"], weights["dw1.bias"]))
    a = relu(pointwise(a, weights["pw1.weight"], weights["pw1.bias"], stride=2))
    a = relu(dwconv3x3(a, weights["dw2.weight"], weights["dw2.bias"]))
    a = relu(pointwise(a, weights["pw2.weight"], weights["pw2.bias"], stride=2))
    v = gap(a)
    return linear(v, weights["fc.weight"], weights["fc.bias"])


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

    def logits(self, mask: NDArray[np.bool_]) -> NDArray[np.float32]:
        glyph = normalize(mask, self.input_size).astype(np.float32) / 255.0
        return forward(glyph, self.weights)[0]

    def __call__(self, mask: NDArray[np.bool_], profile: Profile) -> tuple[str, float]:
        logits = self.logits(mask)
        probs = softmax(logits[None, :])[0]
        best = int(np.argmax(probs))
        return self.charset[best], float(probs[best])
