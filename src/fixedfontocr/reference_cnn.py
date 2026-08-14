"""Full-then-slice reference TinyCNN forward (verification only).

This is the pre-optimization NumPy implementation: stride-2 convolutions
compute the complete feature map and slice every other pixel afterwards.
The optimized :mod:`fixedfontocr.cnn` must match it within float noise
(``max error < 1e-5``, identical argmax), which is the P2 acceptance gate.
"""

from __future__ import annotations

import numpy as np


def reference_conv3x3(x, w, b, stride=1):
    n, c, h, ww = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    oh, ow = h + 2 - 3 + 1, ww + 2 - 3 + 1
    shape = (n, c, oh, ow, 3, 3)
    strides = xp.strides[:2] + (xp.strides[2], xp.strides[3]) + (
        xp.strides[2],
        xp.strides[3],
    )
    patches = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    out = np.einsum("nihwab,o iab->nohw", patches, w, optimize=True)
    out += b.reshape(1, -1, 1, 1)
    if stride == 2:
        out = out[:, :, ::2, ::2]
    return out.astype(np.float32)


def reference_dwconv3x3(x, w, b, stride=1):
    n, c, h, ww = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    oh, ow = h + 2 - 3 + 1, ww + 2 - 3 + 1
    shape = (n, c, oh, ow, 3, 3)
    strides = xp.strides[:2] + (xp.strides[2], xp.strides[3]) + (
        xp.strides[2],
        xp.strides[3],
    )
    patches = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    out = np.einsum("nchwab,cab->nchw", patches, w, optimize=True)
    out += b.reshape(1, -1, 1, 1)
    if stride == 2:
        out = out[:, :, ::2, ::2]
    return out.astype(np.float32)


def reference_pointwise(x, w, b, stride=1):
    out = np.einsum("nchw,oc->nohw", x, w, optimize=True)
    out += b.reshape(1, -1, 1, 1)
    if stride == 2:
        out = out[:, :, ::2, ::2]
    return out.astype(np.float32)


def reference_forward(x, weights):
    c1 = np.maximum(
        reference_conv3x3(
            x, weights["conv1.weight"], weights["conv1.bias"], stride=2
        ),
        0.0,
    )
    d1 = np.maximum(
        reference_dwconv3x3(c1, weights["dw1.weight"], weights["dw1.bias"]),
        0.0,
    )
    p1 = np.maximum(
        reference_pointwise(
            d1, weights["pw1.weight"], weights["pw1.bias"], stride=2
        ),
        0.0,
    )
    d2 = np.maximum(
        reference_dwconv3x3(p1, weights["dw2.weight"], weights["dw2.bias"]),
        0.0,
    )
    p2 = np.maximum(
        reference_pointwise(
            d2, weights["pw2.weight"], weights["pw2.bias"], stride=2
        ),
        0.0,
    )
    v = np.mean(p2, axis=(2, 3)).astype(np.float32)
    return (v @ weights["fc.weight"].T + weights["fc.bias"]).astype(np.float32)
