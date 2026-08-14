"""TinyCNN V1 CPU forward pass (Goal 5: frozen architecture).

The network is frozen per ``fonts/goal`` and uses only the primitives a
WGPU backend can implement trivially:

    Input           1 x 24 x 24
    Conv 3x3        1 -> 8   stride 2, ReLU
    DWConv 3x3      8 -> 8   stride 1, ReLU
    Pointwise 1x1   8 -> 16  stride 2, ReLU
    DWConv 3x3      16 -> 16 stride 1, ReLU
    Pointwise 1x1   16 -> 32 stride 2, ReLU
    GlobalAvgPool   32
    Linear          32 -> num_chars
    Argmax

The only allowed input variation is the number of channels fed into
``conv1``: one channel (binary or soft) is the frozen model format, and the
two-channel ``soft + binary`` stack is an explicitly allowed experiment
(``TINYCNN_V1_INPUT_CHANNELS``). Nothing else may grow: no Transformer,
LSTM, attention, or normalization layers beyond the ReLUs below.

All convolutions use SAME (zero) padding and channels-last-free numpy
vectorization via einsum - no per-pixel Python loops.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .classifier import Classifier
from .preprocess import normalize
from .postprocess import second_ids as topk_second_ids
from .postprocess import top2
from .types import ClassificationBatch, Profile


TINYCNN_V1_NAME = "tinycnn_v1"
"""Metadata value that marks a frozen V1 model in ``config.json``."""

TINYCNN_V1_INPUT_SIZE = 24
"""Frozen spatial input size (the design doc fixes the input at 24x24)."""

TINYCNN_V1_INPUT_CHANNELS = (1, 2)
"""Allowed ``conv1`` input channel counts.

1 = single-channel binary or soft foreground (the frozen model format);
2 = the explicitly allowed ``soft + binary`` experiment.
"""

# (name, kind, kernel, stride, in_channels, out_channels, groups, activation)
# conv1's in_channels is fixed to the value in TINYCNN_V1_INPUT_CHANNELS,
# which is why the table records the allowed range instead of a single int.
TINYCNN_V1_LAYERS: tuple[
    tuple[str, str, int, int, int | tuple[int, ...], int, int, str],
    ...,
] = (
    ("conv1", "conv3x3", 3, 2, TINYCNN_V1_INPUT_CHANNELS, 8, 1, "relu"),
    ("dw1", "dwconv3x3", 3, 1, 8, 8, 8, "relu"),
    ("pw1", "pointwise", 1, 2, 8, 16, 1, "relu"),
    ("dw2", "dwconv3x3", 3, 1, 16, 16, 16, "relu"),
    ("pw2", "pointwise", 1, 2, 16, 32, 1, "relu"),
)

TINYCNN_V1_HEAD = ("gap", 32)
"""Fixed head: global average pool over the 32-channel feature map."""

TINYCNN_V1_TAIL = ("linear", 32)
"""Fixed tail: linear projection from 32 features to the charset."""


def cnn_tensor_shapes(
    num_classes: int,
    input_channels: int = 1,
) -> list[tuple[str, tuple[int, ...]]]:
    """Canonical V1 weight shapes for ``weights.bin`` serialization.

    This is the single source of truth shared by the model loader/exporter
    and the validation gate. ``input_channels`` must be one of the frozen
    allowed values (1 or 2); the on-disk model format is frozen to 1.
    """

    if input_channels not in TINYCNN_V1_INPUT_CHANNELS:
        raise ValueError(
            f"TinyCNN V1 allows input_channels in {TINYCNN_V1_INPUT_CHANNELS}, "
            f"got {input_channels}"
        )
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    return [
        ("conv1.weight", (8, input_channels, 3, 3)),
        ("conv1.bias", (8,)),
        ("dw1.weight", (8, 3, 3)),
        ("dw1.bias", (8,)),
        ("pw1.weight", (16, 8)),
        ("pw1.bias", (16,)),
        ("dw2.weight", (16, 3, 3)),
        ("dw2.bias", (16,)),
        ("pw2.weight", (32, 16)),
        ("pw2.bias", (32,)),
        ("fc.weight", (num_classes, 32)),
        ("fc.bias", (num_classes,)),
    ]


def validate_v1_weights(
    weights: dict[str, NDArray[np.float32]],
    num_classes: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Reject any weight set that is not the frozen TinyCNN V1.

    The exact tensor set and every tensor shape must match the V1 spec:
    conv1 -> dw1 -> pw1 -> dw2 -> pw2 -> fc, with no extra tensors (e.g.
    BatchNorm, attention, or embedding weights) and no missing tensors.
    ``conv1`` may have 1 or 2 input channels per the frozen experiment
    range. Returns the validated name -> shape mapping.
    """

    if not isinstance(weights, dict) or not weights:
        raise ValueError("TinyCNN V1 weights must be a non-empty dict")
    arrays = {name: np.asarray(arr) for name, arr in weights.items()}
    if "fc.weight" not in arrays:
        raise ValueError("missing fc.weight: TinyCNN V1 requires the 12 fixed tensors")
    fc_out = int(arrays["fc.weight"].shape[0])
    if num_classes is None:
        num_classes = fc_out
    if num_classes != fc_out:
        raise ValueError(
            f"num_classes={num_classes} does not match fc.weight rows {fc_out}"
        )
    if "conv1.weight" not in arrays:
        raise ValueError("missing conv1.weight: TinyCNN V1 requires the 12 fixed tensors")
    input_channels = int(arrays["conv1.weight"].shape[1])
    if input_channels not in TINYCNN_V1_INPUT_CHANNELS:
        raise ValueError(
            f"conv1 input channels {input_channels} are outside the frozen "
            f"V1 experiment range {TINYCNN_V1_INPUT_CHANNELS}"
        )
    expected = dict(cnn_tensor_shapes(num_classes, input_channels))
    missing = sorted(set(expected) - set(arrays))
    extra = sorted(set(arrays) - set(expected))
    if missing or extra:
        raise ValueError(
            "TinyCNN V1 weight set is frozen to exactly "
            f"{list(expected)}; missing={missing}, extra={extra}"
        )
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"tensor {name} has shape {arrays[name].shape}, "
                f"but frozen TinyCNN V1 requires {shape}"
            )
    return expected


def prepare_weights(
    weights: dict[str, NDArray[np.float32]],
) -> dict[str, NDArray[np.float32]]:
    """Return inference-ready weights: contiguous float32 copies.

    Conversion is done once at construction (classifier/backend init) so the
    hot forward path never pays for ``astype``/``ascontiguousarray``.
    """

    validate_v1_weights(weights)
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
    """Run the frozen TinyCNN V1 and return every stage's activations.

    ``weights`` must contain: conv1.weight/bias, dw1.weight/bias,
    pw1.weight/bias, dw2.weight/bias, pw2.weight/bias, fc.weight/bias.
    The input must be ``[N, C, 24, 24]`` with ``C`` in the frozen V1 range
    (1 or 2) and must match ``conv1.weight``'s in-channel count; other
    spatial sizes, channel counts, or weight layouts are rejected.

    Returns a dict with ``conv1``, ``dw1``, ``pw1``, ``dw2``, ``pw2``,
    ``gap`` and ``logits``; all activations are in NCHW except ``gap``
    (``(N, C)``) and ``logits`` (``(N, classes)``). This is the reference
    used by ``tools/train/export_model.py`` to emit per-layer test vectors.
    """

    validate_v1_weights(weights)
    if any(
        arr.dtype != np.float32 or not arr.flags.c_contiguous
        for arr in weights.values()
    ):
        weights = prepare_weights(weights)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x[None, None, :, :]
    elif x.ndim == 3:
        x = x[None, :, :, :]
    elif x.ndim != 4:
        raise ValueError(
            f"input must be [N, C, H, W], got {x.ndim} dimensions"
        )
    if x.shape[2:] != (TINYCNN_V1_INPUT_SIZE, TINYCNN_V1_INPUT_SIZE):
        raise ValueError(
            f"TinyCNN V1 input is frozen to {TINYCNN_V1_INPUT_SIZE}x"
            f"{TINYCNN_V1_INPUT_SIZE}, got {x.shape[2:]}"
        )
    if x.shape[1] not in TINYCNN_V1_INPUT_CHANNELS:
        raise ValueError(
            f"TinyCNN V1 input channels must be in "
            f"{TINYCNN_V1_INPUT_CHANNELS}, got {x.shape[1]}"
        )
    if x.shape[1] != weights["conv1.weight"].shape[1]:
        raise ValueError(
            f"input has {x.shape[1]} channels but conv1.weight expects "
            f"{weights['conv1.weight'].shape[1]}"
        )

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
        if input_size != TINYCNN_V1_INPUT_SIZE:
            raise ValueError(
                f"TinyCNN V1 input is frozen to {TINYCNN_V1_INPUT_SIZE}x"
                f"{TINYCNN_V1_INPUT_SIZE}, got {input_size}"
            )
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
                second_ids=np.empty(0, dtype=np.int32),
            )
        if allowed_ids is not None and not allowed_ids:
            return ClassificationBatch(
                ids=np.full(n, -1, dtype=np.int32),
                top1=np.full(n, -np.inf, dtype=np.float32),
                top2=np.full(n, -np.inf, dtype=np.float32),
                margins=np.full(n, 0.0, dtype=np.float32),
                second_ids=np.full(n, -1, dtype=np.int32),
            )
        x = glyphs.astype(np.float32)[:, None, :, :] * (1.0 / 255.0)
        logits = forward(x, self.weights)
        if allowed_ids is not None:
            masked = np.full_like(logits, -np.inf)
            idx = np.fromiter(sorted(allowed_ids), dtype=np.int64)
            masked[:, idx] = logits[:, idx]
            logits = masked
        ids, top1, top2v, margins = top2(logits)
        return ClassificationBatch(
            ids=ids,
            top1=top1,
            top2=top2v,
            margins=margins,
            second_ids=topk_second_ids(logits, ids),
        )

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
