# WGPU backend (phase 1)

## Scope of phase 1

Only the classifier is on the GPU. The CPU still does:

- color/grayscale mask
- line finding
- run-length connected components
- candidate lattice + visual DP segmentation
- 24x24 resize + normalization

The result is an `uint8 [N, 24, 24]` glyph batch (0/255) that is handed to a
unified backend:

```python
class Backend:
    def classify(self, glyphs):  # [N, 24, 24] uint8
        ...  # -> BackendResult(char_ids int32 [N], scores f32 [N])
```

`CPUBackend` is the numpy reference. `WGPUBackend` implements the same
contract with WGSL compute shaders, so the OCR pipeline above never needs to
know which backend is active.

The production recognizer now segments with the candidate lattice + visual
DP (see `architecture.md`); the WGPU re-integration that feeds the
DP-selected candidates back into the WGSL classifier is the next phase and
is intentionally not part of this CPU freeze. The Goal 19 CPU freeze is
complete: the numpy CPU pipeline is the canonical implementation, and this
re-integration is the formal WGPU phase (Goal 20) that starts from that
freeze.

## Data layout

All GPU tensors are NHWC (`N x H x W x C`) with channel counts padded to
multiples of four:

| stage | channels | padded | per-pixel elements |
| --- | --- | --- | --- |
| input glyph | 1 | 4 | 1 x `vec4<f32>` |
| conv1 out | 8 | 8 | 2 x `vec4<f32>` |
| dw1 / pw1 out | 8 / 16 | 8 / 16 | 2 / 4 x `vec4<f32>` |
| dw2 / pw2 out | 16 / 32 | 16 / 32 | 4 / 8 x `vec4<f32>` |
| GAP out | 32 | 32 | 8 x `f32` |

The first layer is specialized: input is one `vec4<f32>` per pixel with the
real channel in component `x`, and conv1 broadcasts `x` to every output
channel. Depthwise and pointwise layers use all four components.

## Shaders

Each layer is its own WGSL file under `src/fixedfontocr/shaders/`, all FP32,
so it can be validated independently against the numpy layer:

```
normalize.wgsl        uint8 [N,H,W] -> NHWC f32 (0..1, channel 0 only)
conv3x3.wgsl          conv1: 3x3 stride 2, SAME pad, ReLU
dwconv3x3.wgsl        depthwise 3x3 stride 1, SAME pad, ReLU
pointwise.wgsl        1x1 stride 2, ReLU
gap.wgsl              global average pool
linear.wgsl           dense layer (for layer verification)
argmax.wgsl           top-1/top-2 reduction over logits (for verification)
linear_argmax.wgsl    fused dense + top-1/top-2 (for verification)
mega.wgsl             the WHOLE chain in one dispatch (production path)
```

Since the mega rewrite (Goal 20 phase 2) the production path no longer
dispatches layer by layer: `mega.wgsl` runs one 64-thread workgroup per
glyph and computes normalize -> conv1 -> dw1 -> pw1 -> dw2 -> pw2 -> gap ->
dense -> top-2 (classify mode) or full `[C]` logits (logits mode) with every
intermediate tensor in workgroup shared memory. `classify()` and
`forward_logits()` each submit exactly ONE dispatch and read back once —
zero intermediate copy-backs. The per-layer shaders above are kept as the
verified primitives for the parity tests (`tests/test_wgpu.py`).

The classify-mode record layout is the same 12-byte triple the per-layer
head used to write (best id + top-2 scores, now packed as three u32s):

```wgsl
struct Result {
    best_id: u32,
    best_score: f32,
    second_score: f32,
};
```

Confidence is `best_score - second_score` (no softmax). For a 50-glyph line
the readback is 600 bytes.

## Numeric consistency

`tests/test_wgpu.py` compares every layer against numpy on random inputs:

- normalize, conv1, dw1, pw1, dw2, pw2, gap, linear: max abs error < 1e-4
- argmax and fused linear+argmax: identical top-1 ids, scores < 1e-4
  (reduction is also verified with 7000 classes)
- `classify()` across batch sizes 1..200: identical ids, scores < 1e-4
- end-to-end `FixedFontOCR(backend="wgpu")` == `backend="cpu"` on digits
  and CJK (`获得金币1000`) rendered with the registered
  `fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf`

The tests skip when `wgpu` or a GPU adapter is unavailable.

Measured on the checked-in fixtures, the layer/logit errors are comfortably
below 1e-5:

| scenario | max layer error | max logit error | max score diff | argmax |
| --- | --- | --- | --- | --- |
| digits (10 classes) | 7.6e-6 (pw2) | 2.9e-6 | 1.7e-6 | 100% identical |
| CJK fixture (8 classes) | 3.8e-6 (pw1) | 2.1e-6 | 1.9e-6 | 100% identical |

For synthetic random-weight stress models (standard-normal weights, not a
trained distribution) the f32 accumulation order diverges a little more from
BLAS: 3000 classes -> max score diff ~2.4e-4, 7000 classes -> ~1.8e-4.
Top-1 ids still match 100%. The 1e-4 test tolerance is the conservative bound;
real trained models stay near 1e-5.

## Benchmark tool (`tools/benchmark/benchmark_wgpu.py`)

The phase-level benchmark measures steady-state `classify()` on this machine:

```bash
python tools/benchmark/benchmark_wgpu.py tests/fixtures/cnn_digits
python tools/benchmark/benchmark_wgpu.py model/game_cn
python tools/benchmark/benchmark_wgpu.py --classes 3000 --charset cjk
python tools/benchmark/benchmark_wgpu.py --classes 7000 --charset cjk
```

`--batch-sizes` defaults to `1,8,16,32,64,128`, `--repeat`/`--iters` control
sampling, and `--output` writes a JSON table. Every batch is verified for
CPU/WGPU parity (identical argmax ids + max score difference) before it is
timed.

Phase columns are host-observable wall time:

- `upload`: `queue.write_buffer` (host -> GPU copy)
- `compute`: command encoding + `queue.submit` (the actual GPU execution is
  overlapped with the readback sync)
- `readback`: copy-to-staging + `map_sync` + host read (includes the GPU sync
  floor)
- `gpu total`: upload + compute + readback

### Benchmark findings (Apple M4, Metal, wgpu 0.32, numpy 2.5)

`tests/fixtures/cnn_digits` (10 classes):

| batch | CPU | upload | compute | readback | GPU total | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 183 µs | 26 µs | 124 µs | 1.34 ms | 1.49 ms | 0.1x |
| 8 | 265 µs | 21 µs | 103 µs | 1.38 ms | 1.52 ms | 0.2x |
| 16 | 425 µs | 19 µs | 100 µs | 1.37 ms | 1.47 ms | 0.3x |
| 32 | 708 µs | 17 µs | 96 µs | 1.37 ms | 1.48 ms | 0.5x |
| 64 | 1.33 ms | 26 µs | 112 µs | 1.38 ms | 1.50 ms | 0.9x |
| 128 | 2.65 ms | 29 µs | 104 µs | 1.38 ms | 1.50 ms | 1.8x |

`model/game_cn` (real 1894-class CJK model):

| batch | CPU | upload | compute | readback | GPU total | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 210 µs | 24 µs | 116 µs | 1.33 ms | 1.47 ms | 0.1x |
| 8 | 593 µs | 22 µs | 111 µs | 1.39 ms | 1.53 ms | 0.4x |
| 16 | 1.28 ms | 21 µs | 100 µs | 1.37 ms | 1.48 ms | 0.9x |
| 32 | 2.54 ms | 25 µs | 122 µs | 1.31 ms | 1.46 ms | 1.7x |
| 64 | 5.15 ms | 28 µs | 133 µs | 1.32 ms | 1.48 ms | 3.5x |
| 128 | 10.73 ms | 23 µs | 99 µs | 1.31 ms | 1.44 ms | 7.5x |

Synthetic random-weight CJK models (same architecture, class-count stress):

| scenario | batch | CPU | GPU total | speedup |
| --- | --- | --- | --- | --- |
| 3000 classes | 8 | 1.02 ms | 1.44 ms | 0.7x |
| 3000 classes | 16 | 2.01 ms | 1.42 ms | 1.4x |
| 3000 classes | 64 | 8.16 ms | 1.44 ms | 5.7x |
| 3000 classes | 128 | 16.43 ms | 1.47 ms | 11.1x |
| 7000 classes | 8 | 2.42 ms | 2.74 ms | 0.9x |
| 7000 classes | 16 | 5.57 ms | 1.51 ms | 3.7x |
| 7000 classes | 64 | 19.87 ms | 2.71 ms | 7.3x |
| 7000 classes | 128 | 40.80 ms | 2.76 ms | 14.8x |

The GPU has a fixed ~1.3-1.5 ms per-call `map_sync` floor on this platform
(readback column), so the crossover depends mostly on CPU cost per glyph and
class count:

- digits (10 classes): GPU wins from batch 128
- real 1894-class CJK: GPU wins from batch 32
- synthetic 3000/7000-class CJK: GPU wins from batch 16

`backend="auto"` re-runs a startup benchmark on the actual model (default
batch sizes 1/8/16/32/64/128) and extrapolates, so the threshold is measured
per device rather than hard-coded.

`scripts/benchmark.py --repeat 3 --iters 20` measures the classify backend
with total latencies only:

| batch | CPU latency | GPU latency | speedup |
| --- | --- | --- | --- |
| 1 | ~190 µs | ~1.5 ms | 0.1x |
| 8 | ~270 µs | ~1.5 ms | 0.2x |
| 16 | ~410 µs | ~1.5 ms | 0.3x |
| 64 | ~1.3 ms | ~1.5 ms | 0.9x |
| 256 | ~5.4 ms | ~1.5 ms | 3.6x |

The fixed floor is almost all `map_sync`/GPU synchronization; command
encoding and submission are ~0.1 ms. Typical OCR lines (10-50 glyphs) stay on
CPU for small models, while CJK models with 1000+ classes cross over around
batch 16-32.

### Same result on the bundled game font

`scripts/benchmark_custom_font.py` repeats the comparison with the bundled
game font (`fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf`) and the hybrid
model `model/game_cn` (1894 classes from `charsets/sets/combined.txt`). All
batches produce identical top-1 ids on CPU and GPU (score difference stays
around 1e-5; measured up to 2.1e-5 on random glyph batches):

| batch | CPU latency | GPU latency | speedup |
| --- | --- | --- | --- |
| 1 | ~182 µs | ~1.47 ms | 0.1x |
| 8 | ~279 µs | ~1.51 ms | 0.2x |
| 16 | ~501 µs | ~1.50 ms | 0.3x |
| 32 | ~849 µs | ~1.49 ms | 0.6x |
| 64 | ~1.34 ms | ~1.47 ms | 0.9x |
| 128 | ~2.63 ms | ~1.46 ms | 1.8x |
| 256 | ~5.66 ms | ~1.48 ms | 3.8x |

End-to-end `recognize()` on this font (backend="cpu" vs backend="wgpu"):

| case | CPU | GPU | speedup |
| --- | --- | --- | --- |
| 6 chars | ~1.92 ms | ~3.07 ms | 0.6x |
| 8 chars | ~2.48 ms | ~3.71 ms | 0.7x |
| 22 chars | ~8.56 ms | ~9.09 ms | 0.9x |

Engine construction: template model 6.8 ms, tinycnn CPU 0.08 ms, tinycnn WGPU
2.9 ms (shader compilation). Per-glyph single call: template 17 µs, tinycnn
CPU 196 µs; the WGPU crossover for this model is around batch 32.

## Auto backend selection

`FixedFontOCR(backend="auto")` runs the same table at construction (batch
sizes 1/8/16/32/64/128) and records the crossover. At runtime the `AutoBackend`
classify() call picks the measured winner for the actual batch size, so a
machine where the GPU wins at N >= 16 uses WGPU for large lines and CPU for
small ones automatically; models with fewer classes shift that threshold
higher (e.g. 10-class digits only pays off at N >= 128).

## Exported test vectors

`tools/train/export_model.py` emits `runtime_model/test_vectors.npz` with the
input glyphs, every layer's activation (numpy reference), final logits and
final `char_id`. `tests/test_wgpu_vectors.py` feeds those vectors through
the WGPU backend layer by layer and compares the final ids, so a model
export is verified against the GPU without re-training or re-rendering.

## Next steps (phase 2)

The formal Goal 20 phase is prepared in [`goal20.md`](goal20.md) (design,
task breakdown T1-T4, acceptance criteria) with the contract skeleton
`tests/test_goal20_wgpu.py`. Parity baseline = the current repo CPU
implementation at HEAD, not the Goal 19 version-1.0 snapshot. Summary of
the four tasks:

- **T1 shader fusion**: DONE as a stronger form — the `mega.wgsl` single
  dispatch (one workgroup per glyph, shared-memory intermediates) replaced
  the eight-dispatch chain for both `classify()` and `forward_logits()`.
- **T2 DP Top-K 回灌**: `classify_topk(glyphs, allowed_mask)` +
  `logits_for(glyphs, char_ids)` so the lattice scorer reads back
  ~`N*(K*8+12)` bytes instead of the full `[N, C]` logits.
- **T3 GPU preprocessing**: upload the RGB image once and do ROI crop,
  grayscale, nearest-neighbor resize and normalize in WGSL; keep only
  bounding-box computation on CPU (soft path first, per-profile gated).
- **T4 persistent/staged readback**: `forward_logits` in one command
  encoder with a single `map_sync`, double-buffered `staged=True`
  readback to amortize the ~1.5 ms sync floor across frames.
