"""Batch classifier backends: numpy CPU and WGPU compute.

The OCR pipeline segments and normalizes on the CPU, then feeds an
``[N, 24, 24]`` uint8 glyph batch to a backend. Both backends share the same
``Backend.classify(glyphs)`` contract and return the same ``BackendResult``:

    char_ids:  int32 ``[N]``  index into the model charset
    scores:    f32   ``[N]``  confidence = top1_logit - top2_logit

The GPU backend uses NHWC storage with channels padded to multiples of 4
(``vec4<f32>`` per pixel) and a fixed TinyCNN pipeline:

    normalize -> conv3x3(stride 2) -> dwconv3x3 -> pointwise(stride 2)
              -> dwconv3x3 -> pointwise(stride 2) -> GAP
              -> fused linear + argmax

Every WGSL shader lives in ``src/fixedfontocr/shaders/`` so each layer can be
verified independently against the numpy reference (see ``tests/test_wgpu.py``).
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import resources

import numpy as np
from numpy.typing import NDArray

from .classifier import TemplateBatch, TemplateV2Classifier
from .cnn import forward, prepare_weights, validate_v1_weights
from .postprocess import top2


@dataclass(frozen=True)
class BackendResult:
    """Batch classification result.

    ``scores`` is the top-1/top-2 logit margin, so no softmax is needed and
    the WGPU path only has to read back ``3 x 4`` bytes per glyph.
    """

    char_ids: NDArray[np.int32]
    scores: NDArray[np.float32]


class Backend(ABC):
    """Unified batch classifier interface used by the OCR pipeline.

    ``classify`` is the cheap public Top-1 contract. The lattice scorer
    additionally needs raw ``[N, C]`` logits so it can build ranked Top-K
    evidence and apply ``allowed_chars`` masks in one algorithm regardless
    of backend; backends therefore also implement ``forward_logits``.
    """

    @abstractmethod
    def classify(self, glyphs: NDArray[np.uint8]) -> BackendResult:
        """Classify a batch of normalized glyphs.

        Parameters
        ----------
        glyphs:
            ``uint8 [N, H, W]``, values 0/255 (already normalized by the CPU).

        Returns
        -------
        BackendResult with ``char_ids`` (int32 ``[N]``) and ``scores``
        (f32 ``[N]``, top1-top2 logit margin).
        """

    @abstractmethod
    def forward_logits(self, glyphs: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Return raw ``[N, num_classes]`` logits for a glyph batch.

        The caller owns Top-K generation and allowed-class masking. Keeping
        the full logits at this boundary is what lets ``SegmentScorer`` run
        the identical hybrid fusion algorithm on CPUBackend and WGPUBackend.
        """


def _check_glyphs(glyphs: NDArray[np.uint8]) -> tuple[np.ndarray, int, int, int]:
    arr = np.asarray(glyphs)
    if arr.ndim != 3:
        raise ValueError(f"glyphs must be [N, H, W], got shape {arr.shape}")
    if arr.size and arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    n, h, w = arr.shape
    return np.ascontiguousarray(arr), n, h, w


def _margin_from_logits(
    logits: NDArray[np.float32],
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Top-1 id and top1-top2 margin from an ``[N, C]`` logit matrix."""
    ids, _, _, margins = top2(logits)
    # Single-class rows have second = -inf; the margin becomes +inf.
    return ids, margins.astype(np.float32)


class CPUBackend(Backend):
    """Numpy TinyCNN backend; the reference every GPU layer is checked against."""

    def __init__(self, weights: dict[str, NDArray[np.float32]], input_size: int = 24):
        self.weights = prepare_weights(weights)  # also validates frozen V1
        self.input_size = input_size
        self.num_classes = int(self.weights["fc.weight"].shape[0])

    def forward_logits(self, glyphs: NDArray[np.uint8]) -> NDArray[np.float32]:
        arr, n, _h, _w = _check_glyphs(glyphs)
        if _h != self.input_size or _w != self.input_size:
            raise ValueError(
                f"expected {self.input_size}x{self.input_size} glyphs, got {_h}x{_w}"
            )
        if n == 0:
            return np.empty((0, self.num_classes), dtype=np.float32)
        x = arr.astype(np.float32)[:, None, :, :] * (1.0 / 255.0)
        return np.ascontiguousarray(forward(x, self.weights), dtype=np.float32)

    def classify(self, glyphs: NDArray[np.uint8]) -> BackendResult:
        arr, n, _h, _w = _check_glyphs(glyphs)
        if _h != self.input_size or _w != self.input_size:
            raise ValueError(
                f"expected {self.input_size}x{self.input_size} glyphs, got {_h}x{_w}"
            )
        if n == 0:
            return BackendResult(
                char_ids=np.empty(0, dtype=np.int32),
                scores=np.empty(0, dtype=np.float32),
            )
        logits = self.forward_logits(arr)
        char_ids, scores = _margin_from_logits(logits)
        return BackendResult(char_ids=char_ids, scores=scores)



def _glyph_frame_params(segment, geometry, target: int, spec) -> tuple[int, ...]:
    """Per-glyph placement frame replicating ``normalize_grayscale``.

    The frame math runs on the host in float64 (python round-half-even)
    because WGSL has no f64; the per-pixel nearest-neighbor truncation is
    bit-exact in f32 (fractional parts are >= 1/24).
    Returns (x, y, w, h, new_h, new_w, y0, x0); new_h==0 marks an empty ROI.
    """
    h, w = segment.h, segment.w
    if h == 0 or w == 0:
        return (segment.x, segment.y, w, h, 0, 0, 0, 0)
    if geometry is None:
        scale = min(target * 0.8 / h, target * 0.8 / w)
        baseline_offset = None
    else:
        baseline_offset, scale = geometry
        scale = min(float(scale), target / h, target / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    if baseline_offset is None:
        y0 = (target - new_h) // 2
    else:
        row = spec.baseline_row if spec is not None else 18.0
        y0 = int(round(row - baseline_offset * scale))
        y0 = min(max(y0, 0), target - new_h)
    x0 = (target - new_w) // 2
    return (segment.x, segment.y, w, h, new_h, new_w, y0, x0)


class WGPUBackend(Backend):
    """WGPU TinyCNN backend.

    All network storage is NHWC with channel counts padded to multiples of
    four, so every pixel is one or more ``vec4<f32>``. The final
    ``linear_argmax`` shader runs the dense layer and the top-1/top-2
    reduction in one dispatch and reads back only ``12 bytes`` per glyph.
    """

    # Fixed TinyCNN spatial sizes.
    _H = 24
    _W = 24
    _C1 = 8
    _C2 = 16
    _C3 = 32
    _SHADER_FILES = {
        "normalize": "normalize",
        "conv1": "conv3x3",
        "dw1": "dwconv3x3",
        "pw1": "pointwise",
        "dw2": "dwconv3x3",
        "pw2": "pointwise",
        "gap": "gap",
        "linear": "linear",
        "argmax": "argmax",
        "fused": "linear_argmax",
        "mega": "mega",
        "preprocess_soft": "preprocess_soft",
    }

    def __init__(
        self,
        weights: dict[str, NDArray[np.float32]],
        input_size: int = 24,
        device=None,
    ):
        if input_size != self._H:
            raise ValueError(f"WGPU backend supports {self._H}x{self._H} glyphs")
        validate_v1_weights(weights)
        if weights["fc.weight"].shape[1] != self._C3:
            raise ValueError("WGPU backend requires the fixed 32-feature TinyCNN")
        import wgpu

        self._wgpu = wgpu
        self.weights = weights
        self.input_size = input_size
        self.num_classes = int(weights["fc.weight"].shape[0])

        if device is None:
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
            device = adapter.request_device_sync()
        self.device = device

        self._weight_bufs = self._upload_weights()
        self._pipelines: dict[str, object] = {}
        self._layouts: dict[str, object] = {}
        self._build_pipelines()
        self._cap = 0
        self._bufs: dict[str, object] = {}
        self._uniform_bufs: dict[tuple, object] = {}
        self._bg_cache: dict[tuple, object] = {}
        self._readback = None
        self._logits_readback = None
        self.last_timing: dict[str, float] | None = None
        self.last_dispatch_count: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self, glyphs: NDArray[np.uint8], *, profile: bool = False
    ) -> BackendResult:
        arr, n, h, w = _check_glyphs(glyphs)
        if h != self._H or w != self._W:
            raise ValueError(f"expected {self._H}x{self._W} glyphs, got {h}x{w}")
        if n == 0:
            if profile:
                self.last_timing = {
                    "upload": 0.0,
                    "compute": 0.0,
                    "readback": 0.0,
                    "total": 0.0,
                }
            return BackendResult(
                char_ids=np.empty(0, dtype=np.int32),
                scores=np.empty(0, dtype=np.float32),
            )
        self._ensure_buffers(n)
        raw, timing = self._mega_run(arr, mode=self._MEGA_CLASSIFY)
        if profile:
            self.last_timing = timing
        char_ids, scores = self._parse_results(raw)
        return BackendResult(char_ids=char_ids, scores=scores)

    # Mega-dispatch modes (mega.wgsl).
    _MEGA_CLASSIFY = 0
    _MEGA_LOGITS = 1

    def forward_logits(self, glyphs: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Raw ``[N, C]`` logits through the single-dispatch mega pipeline.

        One submission runs the entire TinyCNN per glyph workgroup; the full
        logits row of every glyph is read back once (the DP lattice scorer
        consumes it). Zero intermediate copy-backs.
        """

        arr, n, h, w = _check_glyphs(glyphs)
        if h != self._H or w != self._W:
            raise ValueError(f"expected {self._H}x{self._W} glyphs, got {h}x{w}")
        if n == 0:
            return np.empty((0, self.num_classes), dtype=np.float32)
        raw, _timing = self._mega_run(arr, mode=self._MEGA_LOGITS)
        return np.ascontiguousarray(
            raw.view("<f4").reshape(n, self.num_classes), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Per-layer entry points (used by the CPU/GPU consistency tests)
    # ------------------------------------------------------------------

    def normalize(self, glyphs: NDArray[np.uint8]) -> NDArray[np.float32]:
        arr, n, h, w = _check_glyphs(glyphs)
        buf_in = self._device_create_buffer(data=arr.tobytes(), usage=self._U_STORAGE)
        buf_out = self._device_create_buffer(
            size=n * h * w * 4 * 4, usage=self._U_STORAGE | self._U_COPY_SRC
        )
        self._run_one("normalize", self._bind("normalize", [buf_in, buf_out], n), n * h * w)
        return self._read(buf_out, n * h * w * 4 * 4).view("<f4")

    def conv1(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._run_conv_layer("conv1", x, 12, 12, self._C1 // 4)

    def dw1(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._run_dw_layer("dw1", x, 12, 12, self._C1 // 4)

    def pw1(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._run_pw_layer("pw1", x, 6, 6, self._C2 // 4)

    def dw2(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._run_dw_layer("dw2", x, 6, 6, self._C2 // 4)

    def pw2(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._run_pw_layer("pw2", x, 3, 3, self._C3 // 4)

    def gap(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        n = x.shape[0]
        vec4_count = x.shape[-1] // 4
        buf_in = self._device_create_buffer(data=x.astype(np.float32).tobytes(), usage=self._U_STORAGE)
        buf_out = self._device_create_buffer(
            size=n * vec4_count * 4 * 4, usage=self._U_STORAGE | self._U_COPY_SRC
        )
        self._run_one("gap", self._bind("gap", [buf_in, buf_out], n), n * vec4_count)
        return self._read(buf_out, n * vec4_count * 4 * 4).view("<f4")

    def linear(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        n, in_c = features.shape
        buf_in = self._device_create_buffer(
            data=features.astype(np.float32).tobytes(), usage=self._U_STORAGE
        )
        buf_out = self._device_create_buffer(
            size=n * self.num_classes * 4, usage=self._U_STORAGE | self._U_COPY_SRC
        )
        self._run_one(
            "linear",
            self._bind(
                "linear",
                [buf_in, self._weight_bufs["fc_w"], self._weight_bufs["fc_b"], buf_out],
                n,
            ),
            n * self.num_classes,
        )
        return self._read(buf_out, n * self.num_classes * 4).view("<f4")

    def argmax(self, logits: NDArray[np.float32]) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        n, c = logits.shape
        buf_in = self._device_create_buffer(
            data=logits.astype(np.float32).tobytes(), usage=self._U_STORAGE
        )
        buf_out = self._device_create_buffer(
            size=n * 12, usage=self._U_STORAGE | self._U_COPY_SRC
        )
        self._run_one(
            "argmax", self._bind("argmax", [buf_in, buf_out], n, out_c=c), n,
            workgroups=n,
        )
        return self._parse_results(self._read(buf_out, n * 12))

    def fused(self, features: NDArray[np.float32]) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        n, in_c = features.shape
        buf_in = self._device_create_buffer(
            data=features.astype(np.float32).tobytes(), usage=self._U_STORAGE
        )
        buf_out = self._device_create_buffer(
            size=n * 12, usage=self._U_STORAGE | self._U_COPY_SRC
        )
        self._run_one(
            "fused",
            self._bind(
                "fused",
                [buf_in, self._weight_bufs["fc_w"], self._weight_bufs["fc_b"], buf_out],
                n,
            ),
            n,
            workgroups=n,
        )
        return self._parse_results(self._read(buf_out, n * 12))

    # ------------------------------------------------------------------
    # GPU preprocessing (Goal 20 G3)
    # ------------------------------------------------------------------

    def _image_words(self, image: NDArray[np.uint8]) -> NDArray[np.uint32]:
        """RGB image bytes padded to u32 words (little-endian)."""
        nbytes = image.shape[0] * image.shape[1] * 3
        buf = np.zeros(nbytes + (-nbytes) % 4, dtype=np.uint8)
        buf[:nbytes] = image.reshape(-1)
        return buf.view(np.uint32)

    def _pp_params(
        self,
        segments: list,
        spec,
        geometries: list[tuple[float, float] | None] | None,
    ) -> NDArray[np.uint32]:
        n = len(segments)
        geoms = geometries if geometries is not None else [None] * n
        params = np.empty((n, 8), dtype=np.uint32)
        for i, seg in enumerate(segments):
            params[i] = _glyph_frame_params(seg, geoms[i], self._H, spec)
        return params

    def _pp_ensure(self, n: int) -> None:
        if n <= getattr(self, "_pp_cap", 0):
            return
        self._pp_cap = n
        u = self._U_STORAGE
        self._pp_bufs = {
            "params": self._device_create_buffer(
                size=n * 8 * 4, usage=u | self._U_COPY_DST
            ),
            "out": self._device_create_buffer(
                size=n * self._H * self._W, usage=u | self._U_COPY_SRC
            ),
        }
        self._pp_readback = self._device_create_buffer(
            size=n * self._H * self._W, usage=self._U_MAP_READ | self._U_COPY_DST
        )

    def _pp_bind_group(
        self,
        n: int,
        H: int,
        W: int,
        params: NDArray[np.uint32],
        img_words: NDArray[np.uint32],
        out_buf: object,
    ) -> object:
        """Upload preprocess params + RGB image words and build the bind group.

        ``out_buf`` receives the packed ``uint8 [N, 24, 24]`` soft batch; it
        may be the standalone ``_pp_bufs["out"]`` or — in a fused encoder —
        the mega input buffer, so the preprocess dispatch can hand its
        output straight to the mega dispatch in the same submit.
        """
        self._pp_ensure(n)
        self.device.queue.write_buffer(
            self._pp_bufs["params"], 0, params, 0, params.nbytes
        )
        img_buf = self._device_create_buffer(
            data=img_words.tobytes(), usage=self._U_STORAGE
        )
        uniform = self._device_create_buffer(
            data=self._uniform(n, H, W, 0).tobytes(),
            usage=self._wgpu.BufferUsage.UNIFORM,
        )
        entries = [
            {"binding": 0, "resource": {"buffer": uniform}},
            {
                "binding": 1,
                "resource": {"buffer": img_buf, "offset": 0, "size": img_buf.size},
            },
            {
                "binding": 2,
                "resource": {
                    "buffer": self._pp_bufs["params"],
                    "offset": 0,
                    "size": self._pp_bufs["params"].size,
                },
            },
            {
                "binding": 3,
                "resource": {"buffer": out_buf, "offset": 0, "size": out_buf.size},
            },
        ]
        return self.device.create_bind_group(
            layout=self._layouts["preprocess_soft"], entries=entries
        )

    def preprocess_glyphs(
        self,
        image: NDArray[np.uint8],
        segments: list,
        spec=None,
        geometries: list[tuple[float, float] | None] | None = None,
    ) -> NDArray[np.uint8]:
        """GPU soft glyph batch: RGB -> ROI gray + nearest-neighbor normalize.

        Byte-exact with the CPU ``_soft_glyph_batch`` path (default
        grayscale profile formula); one dispatch, one readback of the
        ``uint8 [N, 24, 24]`` batch.
        """
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"expected RGB image with shape (H, W, 3), got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")
        n = len(segments)
        if n == 0:
            return np.empty((0, self._H, self._W), dtype=np.uint8)
        H, W = image.shape[:2]
        params = self._pp_params(segments, spec, geometries)
        words = self._image_words(image)
        self._pp_ensure(n)
        bg = self._pp_bind_group(n, H, W, params, words, self._pp_bufs["out"])
        enc = self.device.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(self._pipelines["preprocess_soft"])
        p.set_bind_group(0, bg, [], 0, 99)
        p.dispatch_workgroups((n * 144 + 63) // 64, 1, 1)
        p.end()
        size = n * self._H * self._W
        enc.copy_buffer_to_buffer(self._pp_bufs["out"], 0, self._pp_readback, 0, size)
        self.device.queue.submit([enc.finish()])
        self._pp_readback.map_sync(self._wgpu.MapMode.READ)
        raw = np.frombuffer(self._pp_readback.read_mapped(size=size), dtype=np.uint8).copy()
        self._pp_readback.unmap()
        self.last_dispatch_count = 1
        return raw.reshape(n, self._H, self._W)

    def forward_logits_from_image(
        self,
        image: NDArray[np.uint8],
        segments: list,
        spec=None,
        geometries: list[tuple[float, float] | None] | None = None,
    ) -> NDArray[np.float32]:
        """Fused preprocess + TinyCNN in ONE submit (two dispatches).

        The preprocess shader writes the packed soft glyph batch straight
        into the mega input buffer, the mega dispatch follows in the same
        command encoder, and only the final ``[N, C]`` logits are read back.
        """
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"expected RGB image with shape (H, W, 3), got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")
        n = len(segments)
        if n == 0:
            return np.empty((0, self.num_classes), dtype=np.float32)
        H, W = image.shape[:2]
        params = self._pp_params(segments, spec, geometries)
        words = self._image_words(image)
        self._ensure_buffers(n)
        bg = self._pp_bind_group(n, H, W, params, words, self._bufs["input"])
        enc = self.device.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(self._pipelines["preprocess_soft"])
        p.set_bind_group(0, bg, [], 0, 99)
        p.dispatch_workgroups((n * 144 + 63) // 64, 1, 1)
        p.end()
        p2 = enc.begin_compute_pass()
        p2.set_pipeline(self._pipelines["mega"])
        p2.set_bind_group(0, self._mega_bind(self._MEGA_LOGITS), [], 0, 99)
        p2.dispatch_workgroups(n, 1, 1)
        p2.end()
        size = n * self.num_classes * 4
        enc.copy_buffer_to_buffer(
            self._bufs["logits"], 0, self._logits_readback, 0, size
        )
        self.device.queue.submit([enc.finish()])
        self._logits_readback.map_sync(self._wgpu.MapMode.READ)
        raw = np.frombuffer(
            self._logits_readback.read_mapped(size=size), dtype=np.uint8
        ).copy()
        self._logits_readback.unmap()
        self.last_dispatch_count = 2
        return np.ascontiguousarray(
            raw.view("<f4").reshape(n, self.num_classes), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # WGPU plumbing
    # ------------------------------------------------------------------

    @property
    def _U_STORAGE(self):
        return self._wgpu.BufferUsage.STORAGE

    @property
    def _U_COPY_SRC(self):
        return self._wgpu.BufferUsage.COPY_SRC

    @property
    def _U_COPY_DST(self):
        return self._wgpu.BufferUsage.COPY_DST

    @property
    def _U_MAP_READ(self):
        return self._wgpu.BufferUsage.MAP_READ

    def _device_create_buffer(self, size=None, data=None, usage=None):
        if data is not None:
            return self.device.create_buffer_with_data(data=data, usage=usage)
        return self.device.create_buffer(size=size, usage=usage)

    def _shader(self, name: str) -> str:
        return (
            resources.files("fixedfontocr")
            .joinpath("shaders", f"{name}.wgsl")
            .read_text(encoding="utf-8")
        )

    def _upload_weights(self) -> dict[str, object]:
        w = self.weights
        tensors = {
            # conv1 is specialized to the single real input channel: [oc][3][3].
            "conv1": w["conv1.weight"].reshape(-1),
            "dw1": w["dw1.weight"].reshape(-1),
            "pw1": w["pw1.weight"].reshape(-1),
            "dw2": w["dw2.weight"].reshape(-1),
            "pw2": w["pw2.weight"].reshape(-1),
            "fc": w["fc.weight"].reshape(-1),
        }
        biases = {
            "conv1": w["conv1.bias"],
            "dw1": w["dw1.bias"],
            "pw1": w["pw1.bias"],
            "dw2": w["dw2.bias"],
            "pw2": w["pw2.bias"],
            "fc": w["fc.bias"],
        }
        bufs: dict[str, object] = {}
        for name, arr in tensors.items():
            bufs[f"{name}_w"] = self._device_create_buffer(
                data=arr.astype(np.float32).tobytes(), usage=self._U_STORAGE
            )
        for name, arr in biases.items():
            bufs[f"{name}_b"] = self._device_create_buffer(
                data=arr.astype(np.float32).tobytes(), usage=self._U_STORAGE
            )
        # One concatenated weight buffer for the mega shader (layout offsets
        # are hard-coded in ``shaders/mega.wgsl``).
        fused = np.concatenate(
            [
                w["conv1.weight"].reshape(-1),
                w["conv1.bias"],
                w["dw1.weight"].reshape(-1),
                w["dw1.bias"],
                w["pw1.weight"].reshape(-1),
                w["pw1.bias"],
                w["dw2.weight"].reshape(-1),
                w["dw2.bias"],
                w["pw2.weight"].reshape(-1),
                w["pw2.bias"],
                w["fc.weight"].reshape(-1),
                w["fc.bias"],
            ]
        ).astype(np.float32)
        self._fused_w = self._device_create_buffer(
            data=fused.tobytes(), usage=self._U_STORAGE
        )
        return bufs

    def _uniform(self, *values: int) -> NDArray[np.uint8]:
        n_vec = math.ceil(len(values) / 4)
        buf = np.zeros(n_vec * 16, dtype=np.uint8)
        for i, v in enumerate(values):
            buf[i * 4 : i * 4 + 4] = np.asarray([v], dtype="<u4").view(np.uint8)
        return buf

    def _build_pipelines(self) -> None:
        w = self._wgpu
        stage = w.ShaderStage.COMPUTE

        def uniform_buf() -> dict:
            return {"type": "uniform"}

        def storage_ro() -> dict:
            return {"type": "read-only-storage"}

        def storage_rw() -> dict:
            return {"type": "storage"}

        specs: dict[str, list[dict]] = {
            "normalize": [uniform_buf(), storage_ro(), storage_rw()],
            "conv1": [uniform_buf(), storage_ro(), storage_ro(), storage_ro(), storage_rw()],
            "dw1": [uniform_buf(), storage_ro(), storage_ro(), storage_ro(), storage_rw()],
            "pw1": [uniform_buf(), storage_ro(), storage_ro(), storage_ro(), storage_rw()],
            "dw2": [uniform_buf(), storage_ro(), storage_ro(), storage_ro(), storage_rw()],
            "pw2": [uniform_buf(), storage_ro(), storage_ro(), storage_ro(), storage_rw()],
            "gap": [uniform_buf(), storage_ro(), storage_rw()],
            "linear": [uniform_buf(), storage_ro(), storage_ro(), storage_ro(), storage_rw()],
            "argmax": [uniform_buf(), storage_ro(), storage_rw()],
            "fused": [uniform_buf(), storage_ro(), storage_ro(), storage_ro(), storage_rw()],
            "mega": [uniform_buf(), storage_ro(), storage_ro(), storage_rw()],
            "preprocess_soft": [uniform_buf(), storage_ro(), storage_ro(), storage_rw()],
        }
        for name, entries in specs.items():
            bgl_entries = [
                {"binding": i, "visibility": stage, "buffer": b}
                for i, b in enumerate(entries)
            ]
            layout = self.device.create_bind_group_layout(entries=bgl_entries)
            pl = self.device.create_pipeline_layout(bind_group_layouts=[layout])
            shader = self.device.create_shader_module(
                code=self._shader(self._SHADER_FILES[name])
            )
            self._pipelines[name] = self.device.create_compute_pipeline(
                layout=pl, compute={"module": shader, "entry_point": "main"}
            )
            self._layouts[name] = layout

    def _bind(
        self, name: str, buffers: list[object], n: int, out_c: int | None = None
    ) -> object:
        key = (name, n, out_c)
        uniform = self._uniform_bufs.get(key)
        if uniform is None:
            uniform = self._device_create_buffer(
                data=self._uniform(*self._layer_params(name, n, out_c=out_c)).tobytes(),
                usage=self._wgpu.BufferUsage.UNIFORM,
            )
            self._uniform_bufs[key] = uniform
        entries = [{"binding": 0, "resource": {"buffer": uniform}}]
        for i, b in enumerate(buffers, start=1):
            entries.append(
                {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
            )
        return self.device.create_bind_group(layout=self._layouts[name], entries=entries)

    def _classify_bind(self, name: str, buffers: list[object], n: int) -> object:
        """Bind group for the persistent classify buffers, cached per capacity."""
        key = ("classify", name, self._cap, n)
        bg = self._bg_cache.get(key)
        if bg is None:
            bg = self._bind(name, buffers, n)
            self._bg_cache[key] = bg
        return bg

    def _dispatch(
        self, enc, name: str, bind_group: object, total: int, workgroups: int | None = None
    ) -> None:
        p = enc.begin_compute_pass()
        p.set_pipeline(self._pipelines[name])
        p.set_bind_group(0, bind_group, [], 0, 99)
        wgs = workgroups if workgroups is not None else (total + 63) // 64
        p.dispatch_workgroups(wgs, 1, 1)
        p.end()

    def _run_one(
        self, name: str, bind_group: object, total: int, workgroups: int | None = None
    ) -> None:
        enc = self.device.create_command_encoder()
        self._dispatch(enc, name, bind_group, total, workgroups=workgroups)
        self.device.queue.submit([enc.finish()])

    def _read(self, buf: object, size: int) -> NDArray[np.uint8]:
        staging = self._device_create_buffer(
            size=size, usage=self._U_MAP_READ | self._U_COPY_DST
        )
        enc = self.device.create_command_encoder()
        enc.copy_buffer_to_buffer(buf, 0, staging, 0, size)
        self.device.queue.submit([enc.finish()])
        staging.map_sync(self._wgpu.MapMode.READ)
        out = np.frombuffer(staging.read_mapped(), dtype=np.uint8).copy()
        staging.unmap()
        return out

    def _parse_results(
        self, raw: NDArray[np.uint8]
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        recs = raw.view(
            np.dtype([("best_id", "<u4"), ("best_score", "<f4"), ("second_score", "<f4")])
        )
        ids = recs["best_id"].astype(np.int32)
        # The shaders use -FLT_MAX as the "no second class" sentinel (C == 1);
        # normalize it to +inf so the margin matches the numpy reference.
        second = recs["second_score"]
        scores = np.where(
            second <= -3.4e38,
            np.inf,
            recs["best_score"] - second,
        ).astype(np.float32)
        return ids, scores

    # ------------------------------------------------------------------
    # Persistent per-call buffers
    # ------------------------------------------------------------------

    def _ensure_buffers(self, n: int) -> None:
        """Persistent mega-dispatch buffers (input + final records + staging).

        The mega shader keeps every intermediate tensor in workgroup shared
        memory, so there are no per-layer buffers to allocate at all: input,
        result/logits records and the two staging buffers are the whole set.
        """
        if n <= self._cap:
            return
        self._cap = n
        self._bufs = {
            "input": self._device_create_buffer(
                size=n * self._H * self._W,
                usage=self._U_STORAGE | self._U_COPY_DST,
            ),
            "result": self._device_create_buffer(
                size=n * 12, usage=self._U_STORAGE | self._U_COPY_SRC
            ),
            "logits": self._device_create_buffer(
                size=n * self.num_classes * 4,
                usage=self._U_STORAGE | self._U_COPY_SRC,
            ),
        }
        self._readback = self._device_create_buffer(
            size=n * 12, usage=self._U_MAP_READ | self._U_COPY_DST
        )
        self._logits_readback = self._device_create_buffer(
            size=n * self.num_classes * 4, usage=self._U_MAP_READ | self._U_COPY_DST
        )

    def _mega_uniform(self, mode: int) -> object:
        key = ("mega", self._cap, mode)
        buf = self._uniform_bufs.get(key)
        if buf is None:
            buf = self._device_create_buffer(
                data=self._uniform(
                    self._cap, self.num_classes, mode, 0
                ).tobytes(),
                usage=self._wgpu.BufferUsage.UNIFORM,
            )
            self._uniform_bufs[key] = buf
        return buf

    def _mega_bind(self, mode: int) -> object:
        key = ("mega", self._cap, mode)
        bg = self._bg_cache.get(key)
        if bg is None:
            out = (
                self._bufs["result"]
                if mode == self._MEGA_CLASSIFY
                else self._bufs["logits"]
            )
            entries = [
                {"binding": 0, "resource": {"buffer": self._mega_uniform(mode)}},
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self._bufs["input"],
                        "offset": 0,
                        "size": self._bufs["input"].size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self._fused_w,
                        "offset": 0,
                        "size": self._fused_w.size,
                    },
                },
                {
                    "binding": 3,
                    "resource": {"buffer": out, "offset": 0, "size": out.size},
                },
            ]
            bg = self.device.create_bind_group(
                layout=self._layouts["mega"], entries=entries
            )
            self._bg_cache[key] = bg
        return bg

    def _mega_run(
        self, arr: NDArray[np.uint8], mode: int
    ) -> tuple[NDArray[np.uint8], dict[str, float]]:
        """Run the whole TinyCNN in ONE dispatch and read back once.

        One workgroup per glyph; every intermediate tensor lives in
        workgroup shared memory, so the host performs no intermediate
        copy-backs and only the final record/logits are staged out.
        """
        n = arr.shape[0]
        self._ensure_buffers(n)
        t0 = time.perf_counter()
        self.device.queue.write_buffer(
            self._bufs["input"], 0, arr, 0, arr.nbytes
        )
        t1 = time.perf_counter()

        enc = self.device.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(self._pipelines["mega"])
        p.set_bind_group(0, self._mega_bind(mode), [], 0, 99)
        p.dispatch_workgroups(n, 1, 1)
        p.end()
        if mode == self._MEGA_CLASSIFY:
            size = n * 12
            enc.copy_buffer_to_buffer(self._bufs["result"], 0, self._readback, 0, size)
            staging = self._readback
        else:
            size = n * self.num_classes * 4
            enc.copy_buffer_to_buffer(
                self._bufs["logits"], 0, self._logits_readback, 0, size
            )
            staging = self._logits_readback
        self.device.queue.submit([enc.finish()])
        t2 = time.perf_counter()
        staging.map_sync(self._wgpu.MapMode.READ)
        raw = np.frombuffer(staging.read_mapped(size=size), dtype=np.uint8).copy()
        staging.unmap()
        t3 = time.perf_counter()
        self.last_dispatch_count = 1
        return raw, {
            "upload": t1 - t0,
            "compute": t2 - t1,
            "readback": t3 - t2,
            "total": t3 - t0,
        }

    def _layer_params(
        self, name: str, n: int, out_c: int | None = None
    ) -> tuple[int, ...]:
        if name == "normalize":
            return (n, self._H, self._W, 0)
        if name in ("conv1", "pw1", "pw2"):
            if name == "conv1":
                ih = iw = 24
                oh = ow = 12
                stride = 2
                iv = 1
                ov = self._C1 // 4
            elif name == "pw1":
                ih = iw = 12
                oh = ow = 6
                stride = 2
                iv = self._C1 // 4
                ov = self._C2 // 4
            else:
                ih = iw = 6
                oh = ow = 3
                stride = 2
                iv = self._C2 // 4
                ov = self._C3 // 4
            return (n, ih, iw, oh, ow, stride, iv, ov)
        if name in ("dw1", "dw2"):
            if name == "dw1":
                ih = iw = 12
                oh = ow = 12
                stride = 1
                v = self._C1 // 4
            else:
                ih = iw = 6
                oh = ow = 6
                stride = 1
                v = self._C2 // 4
            return (n, ih, iw, oh, ow, stride, v, 0)
        if name == "gap":
            return (n, 3, 3, self._C3 // 4, 0, 0, 0, 0)
        if name == "linear":
            return (n, self._C3, self.num_classes, 0)
        if name == "argmax":
            return (n, self.num_classes if out_c is None else out_c, 0, 0)
        if name == "fused":
            return (n, self._C3, self.num_classes, 0)
        raise KeyError(name)

    def _run_conv_layer(
        self, name: str, x: NDArray[np.float32], oh: int, ow: int, ov: int
    ) -> NDArray[np.float32]:
        n = x.shape[0]
        buf_in = self._device_create_buffer(
            data=x.astype(np.float32).tobytes(), usage=self._U_STORAGE
        )
        buf_out = self._device_create_buffer(
            size=n * oh * ow * ov * 4 * 4, usage=self._U_STORAGE | self._U_COPY_SRC
        )
        self._run_one(
            name,
            self._bind(
                name,
                [buf_in, self._weight_bufs[f"{name}_w"], self._weight_bufs[f"{name}_b"], buf_out],
                n,
            ),
            n * oh * ow * ov,
        )
        return self._read(buf_out, n * oh * ow * ov * 16).view("<f4")

    def _run_dw_layer(
        self, name: str, x: NDArray[np.float32], oh: int, ow: int, v: int
    ) -> NDArray[np.float32]:
        n = x.shape[0]
        buf_in = self._device_create_buffer(
            data=x.astype(np.float32).tobytes(), usage=self._U_STORAGE
        )
        buf_out = self._device_create_buffer(
            size=n * oh * ow * v * 4 * 4, usage=self._U_STORAGE | self._U_COPY_SRC
        )
        self._run_one(
            name,
            self._bind(
                name,
                [buf_in, self._weight_bufs[f"{name}_w"], self._weight_bufs[f"{name}_b"], buf_out],
                n,
            ),
            n * oh * ow * v,
        )
        return self._read(buf_out, n * oh * ow * v * 16).view("<f4")

    def _run_pw_layer(
        self, name: str, x: NDArray[np.float32], oh: int, ow: int, ov: int
    ) -> NDArray[np.float32]:
        n = x.shape[0]
        buf_in = self._device_create_buffer(
            data=x.astype(np.float32).tobytes(), usage=self._U_STORAGE
        )
        buf_out = self._device_create_buffer(
            size=n * oh * ow * ov * 4 * 4, usage=self._U_STORAGE | self._U_COPY_SRC
        )
        self._run_one(
            name,
            self._bind(
                name,
                [buf_in, self._weight_bufs[f"{name}_w"], self._weight_bufs[f"{name}_b"], buf_out],
                n,
            ),
            n * oh * ow * ov,
        )
        return self._read(buf_out, n * oh * ow * ov * 16).view("<f4")


def median_time(fn, repeat: int, iters: int) -> float:
    """Median per-call wall time in seconds (warm-up + repeated runs)."""
    fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter() - t0) / iters)
    samples.sort()
    return samples[len(samples) // 2]


def benchmark_backends(
    cpu: CPUBackend,
    gpu: WGPUBackend,
    batch_sizes: tuple[int, ...] = (1, 8, 16, 32, 64, 128),
    repeat: int = 3,
    iters: int = 5,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Time both backends on synthetic glyph batches.

    Returns ``{batch: {"cpu": seconds, "gpu": seconds}}``. The GPU's
    per-call synchronization floor usually makes it slower for small lines,
    so the crossover recorded here is what ``backend="auto"`` uses at
    runtime instead of guessing.
    """

    rng = np.random.default_rng(seed)
    measurements: dict[int, dict[str, float]] = {}
    for batch in batch_sizes:
        glyphs = (rng.random((batch, cpu.input_size, cpu.input_size)) > 0.5).astype(
            np.uint8
        ) * 255
        tc = median_time(lambda: cpu.classify(glyphs), repeat, max(1, iters))
        tg = median_time(lambda: gpu.classify(glyphs), repeat, max(1, iters))
        measurements[int(batch)] = {"cpu": tc, "gpu": tg}
    return measurements


class AutoBackend(Backend):
    """Runtime backend selection from a measured CPU/GPU crossover table.

    ``backend="auto"`` runs :func:`benchmark_backends` once during engine
    construction, then every ``classify()`` call picks the faster measured
    backend for the actual batch size. Unmeasured sizes use linear
    interpolation inside the table and the last segment's marginal cost for
    extrapolation.
    """

    def __init__(
        self,
        cpu: CPUBackend,
        gpu: WGPUBackend,
        measurements: dict[int, dict[str, float]],
    ):
        self.cpu = cpu
        self.gpu = gpu
        self.measurements = dict(measurements)
        self.batch_sizes = tuple(sorted(self.measurements))

    @property
    def crossover(self) -> tuple[int, str] | None:
        """Smallest batch where the GPU is faster (if any)."""
        if self.gpu is None:
            return None
        for batch in self.batch_sizes:
            m = self.measurements[batch]
            if m["gpu"] < m["cpu"]:
                return batch, "wgpu"
        return None

    def _estimate(self, table: dict[int, float], n: int) -> float:
        sizes = self.batch_sizes
        if len(sizes) == 1:
            return table[sizes[0]]
        if n <= sizes[0]:
            return table[sizes[0]]
        if n >= sizes[-1]:
            a, b = sizes[-2], sizes[-1]
            marginal = (table[b] - table[a]) / (b - a)
            return max(0.0, table[b] + marginal * (n - b))
        for a, b in zip(sizes, sizes[1:]):
            if a <= n <= b:
                t = (n - a) / (b - a)
                return table[a] + t * (table[b] - table[a])
        raise AssertionError("unreachable")

    def pick(self, n: int) -> Backend:
        if n <= 0 or self.gpu is None:
            return self.cpu
        cpu_t = self._estimate(
            {b: m["cpu"] for b, m in self.measurements.items()}, n
        )
        gpu_t = self._estimate(
            {b: m["gpu"] for b, m in self.measurements.items()}, n
        )
        return self.gpu if gpu_t < cpu_t else self.cpu

    def classify(self, glyphs: NDArray[np.uint8]) -> BackendResult:
        arr, n, _, _ = _check_glyphs(glyphs)
        return self.pick(n).classify(arr)

    def forward_logits(self, glyphs: NDArray[np.uint8]) -> NDArray[np.float32]:
        arr, n, _, _ = _check_glyphs(glyphs)
        return self.pick(n).forward_logits(arr)


# ----------------------------------------------------------------------
# GPU Template V2 matcher (Goal 20 G2: template matching in one dispatch)
# ----------------------------------------------------------------------

_GPU_TEMPLATE_K = 5
_GPU_TEMPLATE_C_MAX = 7000
_GPU_TEMPLATE_P_MAX = 256
_GPU_TEMPLATE_RECORD = 15  # u32 words per glyph result record


def _allowed_mask(
    num_classes: int, allowed_ids: set[int] | None, words: int
) -> NDArray[np.uint32]:
    """Bitmask of allowed char ids (all-ones when ``allowed_ids`` is None).

    Shared by the GPU template matcher and the fused scoring stage so the
    two one-submit paths encode the identical mask.
    """
    if allowed_ids is None:
        mask = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
        rem = num_classes % 32
        if rem:
            mask[-1] &= np.uint32((1 << rem) - 1)
    else:
        mask = np.zeros(words, dtype=np.uint32)
        for cid in sorted(allowed_ids):
            mask[cid // 32] |= np.uint32(1 << (cid % 32))
    return mask


class WGPUTemplateMatcher:
    """Template V2 matching on the GPU: one workgroup per glyph, one dispatch.

    Replicates :class:`fixedfontocr.classifier.TemplateV2Classifier`
    exactly: same coarse-feature tolerances, the same XOR+popcount distances
    (exact integers), the same ink-band fallback scan and the same
    ``(dist, char_id)`` Top-K ordering, so ``match_batch`` output is
    identical to the CPU reference — not just within a tolerance. The host
    reads back one 60-byte record per glyph.
    """

    def __init__(
        self,
        gpu: "WGPUBackend",
        data: object,
        charset: list[str],
        input_size: int = 24,
        default_top_k: int = 5,
        normalize_spec=None,
    ):
        expected = (input_size * input_size + 7) // 8
        if data.bytes_per_template != expected:
            raise ValueError(
                "TemplateV2Data bytes_per_template does not match input_size"
            )
        if input_size != 24:
            raise ValueError("GPU template matcher supports 24x24 glyphs")
        num_classes = int(data.num_classes)
        if num_classes > _GPU_TEMPLATE_C_MAX:
            raise ValueError(
                f"GPU template matcher supports up to {_GPU_TEMPLATE_C_MAX} "
                f"classes, got {num_classes}"
            )
        p = int(data.prototypes_per_char)
        if p > _GPU_TEMPLATE_P_MAX:
            raise ValueError(
                f"GPU template matcher supports up to {_GPU_TEMPLATE_P_MAX} "
                f"prototypes per char, got {p}"
            )
        if len(charset) != num_classes:
            raise ValueError("charset and TemplateV2Data must have the same length")

        self._gpu = gpu
        self._wgpu = gpu._wgpu
        self.device = gpu.device
        self.data = data
        self.charset = list(charset)
        self.input_size = input_size
        self.num_classes = num_classes
        self.prototypes_per_char = p
        self.default_top_k = max(2, int(default_top_k))
        self.last_candidates: int | None = None
        self.last_dispatch_count: int | None = None
        self._allowed_words = (num_classes + 31) // 32

        # Coarse features come from the CPU builder (byte-identical arrays);
        # the CPU reference also serves the single-glyph lexicon entry.
        ref = TemplateV2Classifier(
            data, charset, input_size, normalize_spec=normalize_spec
        )
        self._cpu_ref = ref
        feats = ref._features

        self._tbits = gpu._device_create_buffer(
            data=data.bits.reshape(-1).view(np.uint32).tobytes(),
            usage=gpu._U_STORAGE,
        )
        ink = feats["ink"].astype(np.uint32)
        w0 = (
            ink
            | (feats["h"].astype(np.uint32) << 16)
            | (feats["w"].astype(np.uint32) << 24)
        )
        w1 = (
            feats["top"].astype(np.uint32)
            | (feats["left"].astype(np.uint32) << 8)
            | (feats["bottom"].astype(np.uint32) << 16)
            | (feats["right"].astype(np.uint32) << 24)
        )
        fused = np.stack([w0, w1], axis=1).reshape(-1).astype(np.uint32)
        self._feats = gpu._device_create_buffer(
            data=fused.tobytes(), usage=gpu._U_STORAGE
        )
        rs = data.render_sizes.reshape(-1).astype(np.uint32)
        meta = (
            rs
            | (data.dx.reshape(-1).astype(np.uint32) << 8)
            | (data.dy.reshape(-1).astype(np.uint32) << 16)
            | (data.downsample_modes.reshape(-1).astype(np.uint32) << 24)
        )
        self._meta = gpu._device_create_buffer(
            data=meta.tobytes(), usage=gpu._U_STORAGE
        )

        w = self._wgpu
        stage = w.ShaderStage.COMPUTE
        bgl_entries = [
            {"binding": 0, "visibility": stage, "buffer": {"type": "uniform"}},
            {"binding": 1, "visibility": stage, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": stage, "buffer": {"type": "read-only-storage"}},
            {"binding": 3, "visibility": stage, "buffer": {"type": "read-only-storage"}},
            {"binding": 4, "visibility": stage, "buffer": {"type": "read-only-storage"}},
            {"binding": 5, "visibility": stage, "buffer": {"type": "read-only-storage"}},
            {"binding": 6, "visibility": stage, "buffer": {"type": "storage"}},
        ]
        layout = self.device.create_bind_group_layout(entries=bgl_entries)
        pl = self.device.create_pipeline_layout(bind_group_layouts=[layout])
        shader = self.device.create_shader_module(code=gpu._shader("template_match"))
        self._pipeline = self.device.create_compute_pipeline(
            layout=pl, compute={"module": shader, "entry_point": "main"}
        )
        self._layout = layout

        self._cap = 0
        self._glyph_buf = None
        self._allowed_buf = None
        self._result_buf = None
        self._readback = None
        self._bind_group = None

    def _allowed_prototypes(self, allowed_ids: set[int] | None) -> NDArray[np.int64]:
        """Global prototype indices for every (optionally restricted) char."""
        if allowed_ids is None:
            return np.arange(
                self.num_classes * self.prototypes_per_char, dtype=np.int64
            )
        chars = np.fromiter(sorted(allowed_ids), dtype=np.int64)
        if chars.size == 0:
            return np.empty(0, dtype=np.int64)
        p = self.prototypes_per_char
        base = np.repeat(chars * p, p)
        off = np.tile(np.arange(p, dtype=np.int64), chars.size)
        return base + off

    def _ensure_buffers(self, n: int) -> None:
        if n <= self._cap:
            return
        self._cap = n
        g = self._gpu
        u = g._U_STORAGE
        self._glyph_buf = g._device_create_buffer(
            size=n * 18 * 4, usage=u | g._U_COPY_DST
        )
        self._allowed_buf = g._device_create_buffer(
            size=self._allowed_words * 4, usage=u | g._U_COPY_DST
        )
        self._result_buf = g._device_create_buffer(
            size=n * _GPU_TEMPLATE_RECORD * 4, usage=u | g._U_COPY_SRC
        )
        self._readback = g._device_create_buffer(
            size=n * _GPU_TEMPLATE_RECORD * 4,
            usage=g._U_MAP_READ | g._U_COPY_DST,
        )
        area = self.input_size * self.input_size
        uniform = g._device_create_buffer(
            data=g._uniform(
                n,
                self.num_classes,
                self.prototypes_per_char,
                0,
                area,
                area + 1,
                self._allowed_words,
                0,
            ).tobytes(),
            usage=self._wgpu.BufferUsage.UNIFORM,
        )
        entries = [
            {"binding": 0, "resource": {"buffer": uniform}},
            {
                "binding": 1,
                "resource": {
                    "buffer": self._glyph_buf,
                    "offset": 0,
                    "size": self._glyph_buf.size,
                },
            },
            {
                "binding": 2,
                "resource": {"buffer": self._tbits, "offset": 0, "size": self._tbits.size},
            },
            {
                "binding": 3,
                "resource": {"buffer": self._feats, "offset": 0, "size": self._feats.size},
            },
            {
                "binding": 4,
                "resource": {"buffer": self._meta, "offset": 0, "size": self._meta.size},
            },
            {
                "binding": 5,
                "resource": {
                    "buffer": self._allowed_buf,
                    "offset": 0,
                    "size": self._allowed_buf.size,
                },
            },
            {
                "binding": 6,
                "resource": {
                    "buffer": self._result_buf,
                    "offset": 0,
                    "size": self._result_buf.size,
                },
            },
        ]
        self._bind_group = self.device.create_bind_group(
            layout=self._layout, entries=entries
        )

    def _parse_results(
        self, raw: NDArray[np.uint8], n: int, area: int
    ) -> TemplateBatch:
        rec = raw.view("<u4").reshape(n, _GPU_TEMPLATE_RECORD)
        ids = rec[:, :5].astype(np.int32)
        dists = rec[:, 5:10].astype(np.int32)
        valid = ids >= 0
        kk = valid.sum(axis=1)
        out_dists = dists[:, 0]
        second = np.where(kk > 1, dists[:, 1], area).astype(np.int32)
        second_ids = np.where(kk > 1, ids[:, 1], -1).astype(np.int32)
        # Float expressions mirror TemplateV2Classifier.match_batch exactly
        # (f32 division, f64 subtraction, f32 cast).
        out_scores = np.where(
            valid, 1.0 - dists.astype(np.float32) / area, 0.0
        ).astype(np.float32)
        scores = (1.0 - out_dists.astype(np.float32) / area).astype(np.float32)
        second_scores = (1.0 - second.astype(np.float32) / area).astype(np.float32)
        margins = ((second - out_dists).astype(np.float32) / area).astype(np.float32)
        best_proto = np.where(ids[:, 0] >= 0, rec[:, 10].astype(np.int32), -1)
        return TemplateBatch(
            ids=ids[:, 0].copy(),
            scores=scores,
            second_scores=second_scores,
            margins=margins,
            dists=out_dists,
            second_dists=second,
            second_ids=second_ids,
            top_k_ids=ids.copy(),
            top_k_scores=out_scores,
            best_prototypes=best_proto,
            prototype_render_sizes=rec[:, 11].astype(np.int32),
            prototype_dx=rec[:, 12].astype(np.int32),
            prototype_dy=rec[:, 13].astype(np.int32),
            prototype_downsample_modes=rec[:, 14].astype(np.int32),
        )

    def match(
        self,
        mask: NDArray[np.bool_],
        profile,
        allowed_ids: set[int] | None = None,
    ) -> tuple[str, float]:
        """Single-glyph entry used by the lexicon layer (CPU reference).

        Same contract as :meth:`TemplateV2Classifier.match`; the one-glyph
        normalize + match stays on the CPU reference so the result is
        identical to the canonical matcher.
        """
        return self._cpu_ref.match(mask, profile, allowed_ids)

    def match_batch(
        self,
        glyphs: NDArray[np.uint8],
        allowed_ids: set[int] | None = None,
        top_k: int | None = None,
    ) -> TemplateBatch:
        """Top-K template match for a ``uint8 [N, H, W]`` glyph batch (GPU)."""
        glyphs = np.asarray(glyphs, dtype=np.uint8)
        if glyphs.ndim != 3:
            raise ValueError(f"glyphs must be [N, H, W], got {glyphs.shape}")
        n = glyphs.shape[0]
        area = self.input_size * self.input_size
        k = self.default_top_k if top_k is None else int(top_k)
        if k != _GPU_TEMPLATE_K:
            raise ValueError(
                f"GPU template matcher supports top_k={_GPU_TEMPLATE_K}, got {k}"
            )
        if n == 0:
            return TemplateBatch(
                ids=np.empty(0, dtype=np.int32),
                scores=np.empty(0, dtype=np.float32),
                second_scores=np.empty(0, dtype=np.float32),
                margins=np.empty(0, dtype=np.float32),
                dists=np.empty(0, dtype=np.int32),
                second_dists=np.empty(0, dtype=np.int32),
                second_ids=np.empty(0, dtype=np.int32),
                top_k_ids=np.empty((0, k), dtype=np.int32),
                top_k_scores=np.empty((0, k), dtype=np.float32),
                best_prototypes=np.empty(0, dtype=np.int32),
                prototype_render_sizes=np.empty(0, dtype=np.int32),
                prototype_dx=np.empty(0, dtype=np.int32),
                prototype_dy=np.empty(0, dtype=np.int32),
                prototype_downsample_modes=np.empty(0, dtype=np.int32),
            )
        if glyphs.shape[1:] != (self.input_size, self.input_size):
            raise ValueError(
                f"expected {self.input_size}x{self.input_size} glyphs, "
                f"got {glyphs.shape[1:]}"
            )
        proto_ids = self._allowed_prototypes(allowed_ids)
        if proto_ids.size == 0:
            return TemplateBatch(
                ids=np.full(n, -1, dtype=np.int32),
                scores=np.zeros(n, dtype=np.float32),
                second_scores=np.zeros(n, dtype=np.float32),
                margins=np.zeros(n, dtype=np.float32),
                dists=np.full(n, area, dtype=np.int32),
                second_dists=np.full(n, area, dtype=np.int32),
                second_ids=np.full(n, -1, dtype=np.int32),
                top_k_ids=np.full((n, k), -1, dtype=np.int32),
                top_k_scores=np.zeros((n, k), dtype=np.float32),
                best_prototypes=np.full(n, -1, dtype=np.int32),
                prototype_render_sizes=np.full(n, -1, dtype=np.int32),
                prototype_dx=np.full(n, -1, dtype=np.int32),
                prototype_dy=np.full(n, -1, dtype=np.int32),
                prototype_downsample_modes=np.full(n, -1, dtype=np.int32),
            )

        bits = np.packbits(glyphs.reshape(n, -1), axis=1, bitorder="little")
        words = bits.view(np.uint32).reshape(-1)
        mask = _allowed_mask(self.num_classes, allowed_ids, self._allowed_words)

        self._ensure_buffers(n)
        self.device.queue.write_buffer(self._glyph_buf, 0, words, 0, words.nbytes)
        self.device.queue.write_buffer(
            self._allowed_buf, 0, mask, 0, mask.nbytes
        )
        enc = self.device.create_command_encoder()
        p = enc.begin_compute_pass()
        p.set_pipeline(self._pipeline)
        p.set_bind_group(0, self._bind_group, [], 0, 99)
        p.dispatch_workgroups(n, 1, 1)
        p.end()
        size = n * _GPU_TEMPLATE_RECORD * 4
        enc.copy_buffer_to_buffer(self._result_buf, 0, self._readback, 0, size)
        self.device.queue.submit([enc.finish()])
        self._readback.map_sync(self._wgpu.MapMode.READ)
        raw = np.frombuffer(self._readback.read_mapped(size=size), dtype=np.uint8).copy()
        self._readback.unmap()
        self.last_dispatch_count = 1
        return self._parse_results(raw, n, area)


class AutoTemplateMatcher:
    """Per-batch template backend selection (CPU vs GPU), AutoBackend style.

    The GPU template matcher wins from ~batch 4 on the game model but pays
    a fixed per-call sync floor, so this wrapper benchmarks both matchers at
    construction and delegates every ``match_batch`` call to the measured
    winner for the actual glyph count.
    """

    def __init__(
        self,
        cpu: "TemplateV2Classifier",
        gpu: "WGPUTemplateMatcher",
        batch_sizes: tuple[int, ...] = (1, 4, 8, 16, 32),
        repeat: int = 3,
        iters: int = 3,
    ):
        self.cpu = cpu
        self.gpu = gpu
        rng = np.random.default_rng(0)
        self.measurements: dict[int, dict[str, float]] = {}
        for batch in batch_sizes:
            glyphs = (rng.random((batch, 24, 24)) > 0.5).astype(np.uint8) * 255
            self.measurements[batch] = {
                "cpu": median_time(
                    lambda: cpu.match_batch(glyphs), repeat, iters
                ),
                "gpu": median_time(
                    lambda: gpu.match_batch(glyphs), repeat, iters
                ),
            }
        self.batch_sizes = tuple(sorted(self.measurements))

    @property
    def crossover(self) -> tuple[int, str] | None:
        """Smallest batch where the GPU matcher is faster (if any)."""
        for batch in self.batch_sizes:
            m = self.measurements[batch]
            if m["gpu"] < m["cpu"]:
                return batch, "wgpu"
        return None

    def _estimate(self, table: dict[int, float], n: int) -> float:
        sizes = self.batch_sizes
        if len(sizes) == 1:
            return table[sizes[0]]
        if n <= sizes[0]:
            return table[sizes[0]]
        if n >= sizes[-1]:
            a, b = sizes[-2], sizes[-1]
            marginal = (table[b] - table[a]) / (b - a)
            return max(0.0, table[b] + marginal * (n - b))
        for a, b in zip(sizes, sizes[1:]):
            if a <= n <= b:
                t = (n - a) / (b - a)
                return table[a] + t * (table[b] - table[a])
        raise AssertionError("unreachable")

    def pick(self, n: int) -> object:
        if n <= 0:
            return self.cpu
        cpu_t = self._estimate(
            {b: m["cpu"] for b, m in self.measurements.items()}, n
        )
        gpu_t = self._estimate(
            {b: m["gpu"] for b, m in self.measurements.items()}, n
        )
        return self.gpu if gpu_t < cpu_t else self.cpu

    def match(
        self,
        mask: NDArray[np.bool_],
        profile,
        allowed_ids: set[int] | None = None,
    ) -> tuple[str, float]:
        """Single-glyph lexicon entry: delegate to the canonical CPU matcher."""
        return self.cpu.match(mask, profile, allowed_ids)

    def match_batch(
        self,
        glyphs: NDArray[np.uint8],
        allowed_ids: set[int] | None = None,
        top_k: int | None = None,
    ) -> TemplateBatch:
        n = int(np.asarray(glyphs).shape[0])
        return self.pick(n).match_batch(glyphs, allowed_ids, top_k)


class WGPUScoringStage:
    """Template + TinyCNN scoring in ONE submit, ONE readback (Goal 20 G4).

    The visual DP itself stays on the CPU: ``decode_dp`` uses f64 arithmetic
    with a 1e-12 tie tolerance that f32 WGSL cannot reproduce, and the DP is
    microsecond-scale anyway. The GPU's job is the scoring chain, so this
    stage records the template dispatch and the mega-logits dispatch into a
    single command encoder and reads both results back with ONE ``map_sync``.
    """

    def __init__(
        self,
        gpu: "WGPUBackend",
        template_matcher: "WGPUTemplateMatcher",
    ):
        self.gpu = gpu
        self.template = template_matcher
        self.device = gpu.device
        self._wgpu = gpu._wgpu
        self.last_submit_count: int | None = None
        self.last_dispatch_count: int | None = None
        self._cap = 0
        self._staging = None

    def _ensure(self, n: int) -> None:
        if n <= self._cap:
            return
        self._cap = n
        size = n * (_GPU_TEMPLATE_RECORD * 4 + self.gpu.num_classes * 4)
        self._staging = self.gpu._device_create_buffer(
            size=size, usage=self.gpu._U_MAP_READ | self.gpu._U_COPY_DST
        )

    def score_line(
        self,
        glyphs: NDArray[np.uint8],
        allowed_ids: set[int] | None = None,
    ) -> tuple[TemplateBatch, NDArray[np.float32]]:
        """One submit: template match + full TinyCNN logits; one readback.

        ``glyphs`` is the packed-binary ``uint8 [N, 24, 24]`` batch (the
        same one the CPU template matcher and binary-mode CNN consume).
        """
        glyphs = np.asarray(glyphs, dtype=np.uint8)
        if glyphs.ndim != 3:
            raise ValueError(f"glyphs must be [N, H, W], got {glyphs.shape}")
        n = glyphs.shape[0]
        area = self.template.input_size * self.template.input_size
        if n == 0:
            return (
                self.template.match_batch(glyphs, allowed_ids),
                np.empty((0, self.gpu.num_classes), dtype=np.float32),
            )
        if glyphs.shape[1:] != (self.template.input_size,) * 2:
            raise ValueError(
                f"expected {self.template.input_size}x{self.template.input_size} "
                f"glyphs, got {glyphs.shape[1:]}"
            )
        proto_ids = self.template._allowed_prototypes(allowed_ids)
        if proto_ids.size == 0:
            return (
                self.template.match_batch(glyphs, allowed_ids),
                np.zeros((n, self.gpu.num_classes), dtype=np.float32),
            )

        self.template._ensure_buffers(n)
        self.gpu._ensure_buffers(n)
        self._ensure(n)

        bits = np.packbits(glyphs.reshape(n, -1), axis=1, bitorder="little")
        words = bits.view(np.uint32).reshape(-1)
        mask = _allowed_mask(self.template.num_classes, allowed_ids, self.template._allowed_words)
        self.device.queue.write_buffer(
            self.template._glyph_buf, 0, words, 0, words.nbytes
        )
        self.device.queue.write_buffer(
            self.template._allowed_buf, 0, mask, 0, mask.nbytes
        )
        self.device.queue.write_buffer(
            self.gpu._bufs["input"], 0, glyphs, 0, glyphs.nbytes
        )

        enc = self.device.create_command_encoder()
        p1 = enc.begin_compute_pass()
        p1.set_pipeline(self.template._pipeline)
        p1.set_bind_group(0, self.template._bind_group, [], 0, 99)
        p1.dispatch_workgroups(n, 1, 1)
        p1.end()
        p2 = enc.begin_compute_pass()
        p2.set_pipeline(self.gpu._pipelines["mega"])
        p2.set_bind_group(0, self.gpu._mega_bind(self.gpu._MEGA_LOGITS), [], 0, 99)
        p2.dispatch_workgroups(n, 1, 1)
        p2.end()
        tpl_bytes = n * _GPU_TEMPLATE_RECORD * 4
        logits_bytes = n * self.gpu.num_classes * 4
        enc.copy_buffer_to_buffer(
            self.template._result_buf, 0, self._staging, 0, tpl_bytes
        )
        enc.copy_buffer_to_buffer(
            self.gpu._bufs["logits"], 0, self._staging, tpl_bytes, logits_bytes
        )
        self.device.queue.submit([enc.finish()])
        self._staging.map_sync(self._wgpu.MapMode.READ)
        raw = np.frombuffer(
            self._staging.read_mapped(size=tpl_bytes + logits_bytes),
            dtype=np.uint8,
        ).copy()
        self._staging.unmap()
        self.last_submit_count = 1
        self.last_dispatch_count = 2  # template + mega
        tb = self.template._parse_results(raw[:tpl_bytes], n, area)
        logits = np.ascontiguousarray(
            raw[tpl_bytes:].view("<f4").reshape(n, self.gpu.num_classes),
            dtype=np.float32,
        )
        return tb, logits

    def score_line_from_image(
        self,
        image: NDArray[np.uint8],
        segments: list,
        glyphs: NDArray[np.uint8],
        allowed_ids: set[int] | None = None,
        spec=None,
        geometries: list[tuple[float, float] | None] | None = None,
    ) -> tuple[TemplateBatch, NDArray[np.float32]]:
        """One submit: GPU preprocess + template match + TinyCNN logits.

        The G3 preprocess dispatch (RGB -> packed soft glyph batch), the G2
        template dispatch and the G1 mega-logits dispatch are recorded into
        ONE command encoder: the preprocess shader writes its output straight
        into the mega input buffer, and a single staging readback returns the
        template records + full ``[N, C]`` logits. This closes the "one
        submit produces the OCR result" chain for soft-input hybrid models —
        the CPU only uploads the RGB image, the per-glyph frame params and
        the binary glyph words the template matcher consumes.

        ``glyphs`` is the binary-normalized ``uint8 [N, 24, 24]`` batch the
        CPU already computed for the template path (identical to
        :meth:`score_line`); the soft batch for the CNN is produced on the
        GPU from ``image`` + ``segments`` + ``geometries`` and is byte-exact
        with the CPU ``_soft_glyph_batch`` path.
        """
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"expected RGB image with shape (H, W, 3), got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")
        glyphs = np.asarray(glyphs, dtype=np.uint8)
        if glyphs.ndim != 3:
            raise ValueError(f"glyphs must be [N, H, W], got {glyphs.shape}")
        n = glyphs.shape[0]
        area = self.template.input_size * self.template.input_size
        if n == 0:
            return (
                self.template.match_batch(glyphs, allowed_ids),
                np.empty((0, self.gpu.num_classes), dtype=np.float32),
            )
        if glyphs.shape[1:] != (self.template.input_size,) * 2:
            raise ValueError(
                f"expected {self.template.input_size}x{self.template.input_size} "
                f"glyphs, got {glyphs.shape[1:]}"
            )
        if len(segments) != n:
            raise ValueError(
                f"expected {n} segments for {n} glyphs, got {len(segments)}"
            )
        proto_ids = self.template._allowed_prototypes(allowed_ids)
        if proto_ids.size == 0:
            return (
                self.template.match_batch(glyphs, allowed_ids),
                np.zeros((n, self.gpu.num_classes), dtype=np.float32),
            )

        H, W = image.shape[:2]
        params = self.gpu._pp_params(segments, spec, geometries)
        img_words = self.gpu._image_words(image)

        self.template._ensure_buffers(n)
        self.gpu._ensure_buffers(n)
        self._ensure(n)

        bits = np.packbits(glyphs.reshape(n, -1), axis=1, bitorder="little")
        words = bits.view(np.uint32).reshape(-1)
        mask = _allowed_mask(self.template.num_classes, allowed_ids, self.template._allowed_words)
        self.device.queue.write_buffer(
            self.template._glyph_buf, 0, words, 0, words.nbytes
        )
        self.device.queue.write_buffer(
            self.template._allowed_buf, 0, mask, 0, mask.nbytes
        )
        pp_bg = self.gpu._pp_bind_group(
            n, H, W, params, img_words, self.gpu._bufs["input"]
        )

        enc = self.device.create_command_encoder()
        p0 = enc.begin_compute_pass()
        p0.set_pipeline(self.gpu._pipelines["preprocess_soft"])
        p0.set_bind_group(0, pp_bg, [], 0, 99)
        p0.dispatch_workgroups((n * 144 + 63) // 64, 1, 1)
        p0.end()
        p1 = enc.begin_compute_pass()
        p1.set_pipeline(self.template._pipeline)
        p1.set_bind_group(0, self.template._bind_group, [], 0, 99)
        p1.dispatch_workgroups(n, 1, 1)
        p1.end()
        p2 = enc.begin_compute_pass()
        p2.set_pipeline(self.gpu._pipelines["mega"])
        p2.set_bind_group(0, self.gpu._mega_bind(self.gpu._MEGA_LOGITS), [], 0, 99)
        p2.dispatch_workgroups(n, 1, 1)
        p2.end()
        tpl_bytes = n * _GPU_TEMPLATE_RECORD * 4
        logits_bytes = n * self.gpu.num_classes * 4
        enc.copy_buffer_to_buffer(
            self.template._result_buf, 0, self._staging, 0, tpl_bytes
        )
        enc.copy_buffer_to_buffer(
            self.gpu._bufs["logits"], 0, self._staging, tpl_bytes, logits_bytes
        )
        self.device.queue.submit([enc.finish()])
        self._staging.map_sync(self._wgpu.MapMode.READ)
        raw = np.frombuffer(
            self._staging.read_mapped(size=tpl_bytes + logits_bytes),
            dtype=np.uint8,
        ).copy()
        self._staging.unmap()
        self.last_submit_count = 1
        self.last_dispatch_count = 3  # preprocess + template + mega
        tb = self.template._parse_results(raw[:tpl_bytes], n, area)
        logits = np.ascontiguousarray(
            raw[tpl_bytes:].view("<f4").reshape(n, self.gpu.num_classes),
            dtype=np.float32,
        )
        return tb, logits
