# FixedFontOCR

Deterministic fixed-font OCR designed around the "template fast path, CNN
for the hard cases" strategy: the same model files feed a numpy CPU backend
and a WGPU compute backend, and the engine benchmarks them at startup so
`backend="auto"` picks the faster one per batch size instead of guessing.

## Repository layout

```
.
├── src/fixedfontocr/    # the package
│   ├── api.py           # FixedFontOCR engine (auto backend, hybrid, allowed_chars)
│   ├── backends.py      # CPUBackend / WGPUBackend / AutoBackend + benchmark
│   ├── classifier.py    # template matcher (with coarse candidate filtering)
│   ├── cnn.py           # numpy TinyCNN reference (Conv + ReLU only)
│   ├── defaults.py      # bundled game font / charset / model paths
│   ├── postprocess.py   # allowed-chars restriction / confidence scoring
│   ├── model.py         # config.json + charset.txt + weights.bin I/O
│   ├── preprocess.py    # segmentation + 24x24 normalization
│   ├── shaders/         # WGSL layer shaders
│   └── types.py         # CharResult / OCRResult / Profile
├── charsets/            # charset assets (English names, no spaces)
│   ├── words/           #   generated word lists: one word per line
│   └── sets/            #   single-line OCR charsets (combined.txt default)
├── tools/               # dev-only pipeline (PyTorch + Pillow), grouped:
│   ├── charset/         #   export_names.py + extract_charset.py
│   ├── dataset/         #   generate_font_dataset.py + collect_real_samples.py
│   ├── train/           #   train.py + export_model.py + build_model.py
│   ├── benchmark/       #   benchmark.py + benchmark_wgpu.py + footprint.py
│   └── register_font.py #   fonts/registry.json updater
├── scripts/             # legacy entry points (generate/train/benchmark)
├── tests/               # pytest suite + checked-in CNN fixtures
├── docs/
├── fonts/               # registered local fonts (gitignored; registry.json tracked)
├── pyproject.toml
└── README.md
```

## Font Policy

Fonts are the only source of glyph shapes: every font used for rendering,
training, or template generation must live under `fonts/` and be registered
in `fonts/registry.json` by SHA256. System fonts, fallback fonts, and
font-family augmentation are forbidden; a missing glyph is an explicit
error. Model directories record the source font's SHA256 in
`config.json` (`font_sha256`). See [AGENTS.md](AGENTS.md).

## Quick start

```bash
# 1. Build the bundled hybrid recognizer (template + TinyCNN) from the
#    game font and the CN config charset
python tools/train/build_model.py

# 2. Recognize (auto benchmarks CPU vs WGPU once at construction)
```

```python
import numpy as np
from fixedfontocr import FixedFontOCR

ocr = FixedFontOCR(model_path="model/game_cn", backend="auto")
image: np.ndarray = ...  # shape (H, W, 3), dtype uint8

result = ocr.recognize(image)
print(result.text)        # "获得金币1000"
print(result.confidence)
for c in result.chars:
    print(c.char, c.x, c.y, c.w, c.h, c.confidence)
```

Three classifier types are supported in the same model directory format:

- **Template** (default): `fixedfontocr-generate` renders every character
  to a bitset; recognition is XOR + popcount over a coarse-feature-filtered
  candidate set.
- **TinyCNN**: the small fixed network (Conv3x3 stride-2, DWConv3x3,
  Pointwise1x1 stride-2, GAP, Linear). Runtime inference is pure numpy or
  WGSL; training uses PyTorch.
- **Hybrid**: template level-1 + TinyCNN level-2 + unknown. Only glyphs
  whose template confidence is low (the 未/末, 日/曰, 0/O cases) reach the
  CNN.

## Automatic backend selection (`backend="auto"`)

On construction the engine times `classify()` at batch sizes 1, 8, 16, 32,
64 and 128 on both backends, records the table and the crossover, and wraps
both in an `AutoBackend`:

```python
ocr = FixedFontOCR(model_path="model/", backend="auto")
ocr.backend_benchmark   # {1: {...}, 8: {...}, ..., 64: {...}}
ocr._backend.crossover  # e.g. (32, "wgpu") — GPU wins from batch 32
```

Each `recognize()` call then picks the faster measured backend for the
actual glyph count. If WGPU is unavailable (no adapter/driver) or the
benchmark fails, `auto` falls back to CPU and records why in
`ocr.backend_benchmark_error`. The standalone measurement tool is
`python tools/benchmark/benchmark.py model/`.

On this machine (Apple M4) the GPU has a ~1.5 ms per-call sync floor, so it
only wins for large batches; typical OCR lines stay on CPU. On a machine
where the crossover is smaller, `auto` uses the GPU for those batches
automatically.

## Hybrid three-level strategy

```text
template (coarse-filtered candidates)
  -> confidence >= template_threshold? -> return char
  -> else TinyCNN (CPU or WGPU)
       -> confidence >= cnn_threshold? -> return char
       -> else "?" (unknown)
```

Build a hybrid model by exporting a trained CNN together with template
models (same charset):

```bash
python tools/train/export_model.py model.pth --templates template_model/ \
    --output hybrid_model/ --template-threshold 0.90
```

`config.json` then declares `"classifier": "hybrid"` with
`template_threshold` / `cnn_threshold`, and `templates.bin` + `weights.bin`
hold both stages.

The bundled recognizer is built from `charsets/sets/combined.txt` and
`fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf`; `tools/train/build_model.py`
regenerates the whole thing in one command.

## UI-limited charsets (`allowed_chars`)

Fixed-font UI fields rarely use the full charset. Restrict the output
alphabet per call and the classifier can no longer confuse a coin counter
with letters:

```python
ocr.recognize(frame, allowed_chars="0123456789")   # coin count
ocr.recognize(frame, allowed_chars="Lv.0123456789")  # level
ocr.recognize(frame, allowed_chars=None)             # full charset
```

Characters outside the set are never emitted; positions with no surviving
candidate become `"?"` with confidence 0.

## Training and export pipeline

Development dependencies (`torch`, `pillow`, `fonttools`, `numpy`) live in
the `train` extra; runtime inference has no PyTorch dependency.

```bash
# 1. Synthetic data: font + random size/offset/outline/shadow/blur/
#    scale/alpha/background/threshold perturbations -> 24x24 glyphs
python tools/dataset/generate_font_dataset.py \
    fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf synth.npz \
    --charset charsets/sets/combined.txt --samples-per-char 300

# 2. Real screenshots: label-named directories ("0/", "金币/", "Lv.12/");
#    --font records the same registered font as the synthetic data
python tools/dataset/collect_real_samples.py real/ real.npz \
    --font fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf

# 3. Train with Conv+BatchNorm+ReLU (BN is training-only)
python tools/train/train.py --synthetic synth.npz --real real.npz \
    --output model.pth --device auto

# 4. Export hybrid: fold BN into conv weights, add template fast path
python tools/train/export_model.py model.pth \
    --templates model/game_cn_template --output model/game_cn
```

`tools/train/build_model.py` runs all four steps with the project defaults
in one command.

The exporter folds every BatchNorm into the preceding convolution's
weight/bias, so the runtime (numpy and WGSL) only needs Conv + ReLU:

```
runtime_model/
├── model.json      # alias of config.json
├── config.json     # canonical loader config
├── weights.bin     # f32 CNN tensors (BN already folded)
├── charset.txt
└── test_vectors.npz
    # input, every layer's activations, final logits, final char_id
```

`test_vectors.npz` is the golden file used by the WGPU unit tests
(`tests/test_wgpu_vectors.py`): each layer's output and the final char_id
are compared against the exported file, so a model change that breaks
CPU/WGPU parity is caught without needing the original training data.

## Storage and memory

`python tools/benchmark/footprint.py model/` prints the on-disk and in-memory
footprint of any model. Measured numbers for a 3000-char CJK template model:

| item | size |
| --- | --- |
| template weights on disk | ~211 KiB (`N × 72 B` bitsets + 8 B header) |
| charset.txt | ~9 KiB (3000 CJK chars) |
| template bits in RAM | ~211 KiB (shared with the loaded model, no copy) |
| coarse candidate features | ~23 KiB (`8 B/char`: uint16 ink + uint8 bbox/margins) |
| popcount table | 0 B with numpy ≥ 2.0 (`bitwise_count`), 64 KiB fallback |

A `C`-class TinyCNN adds `4032 + 33·C` f32 bytes (conv weights + linear
weights): ~391 KiB for 3000 classes. Hybrid models store both stages.

The WGPU backend keeps persistent per-call buffers of ~24.3 KiB per glyph
(input, normalized NHWC tensor, conv stages, GAP, result + readback), so a
64-glyph line costs ~1.5 MiB of GPU-side memory; buffers grow to the largest
batch seen and stay allocated. A 1920×1080 frame also needs a ~2 MiB bool
mask during segmentation.

`test_vectors.npz` is dev-only: export deployed models with
`--skip-test-vectors` to keep the runtime directory as small as possible.

## Pipeline

```
numpy RGB
  -> profile color/grayscale mask
  -> line finding
  -> connected components + merge/split
  -> 24x24 normalization (CPU)
  -> backend:
       template: coarse feature filter -> XOR + popcount
       tinycnn:  CPUBackend / WGPUBackend / AutoBackend
       hybrid:   template first, CNN fallback, unknown
  -> allowed_chars restriction (postprocess)
  -> charset -> string
```

## Model format

```
model/
├── config.json    # input_width, input_height, classes, version, dtype,
│                  # classifier (template|tinycnn|hybrid), thresholds,
│                  # font_sha256 (source font identity)
├── charset.txt
└── weights.bin    # template: uint32 count + packed bits;
                   # tinycnn/hybrid: f32 tensors in fixed order
                   # hybrid also has templates.bin
```

## Profiles

Per-UI-type tuning lives in
[`src/fixedfontocr/types.py`](src/fixedfontocr/types.py): text color /
grayscale threshold, expected character height/width, stroke width,
spacing, and normalized size. Pass a custom profile to
`FixedFontOCR(..., profile=...)`.

## Current status

- Template (with coarse candidate filtering), TinyCNN CPU and TinyCNN WGPU
  are implemented and tested on Latin and CJK.
- `backend="auto"` benchmarks CPU vs WGPU at construction and selects per
  batch size at runtime.
- Hybrid template→CNN→unknown models, `allowed_chars`, BN folding and
  `test_vectors.npz` exports are implemented and covered by tests.
- Template coarse features are stored in compact uint8/uint16 arrays
  (8 B/char) and numpy ≥ 2.0 uses `bitwise_count` instead of the 64 KiB
  popcount table; `tools/benchmark/footprint.py` reports storage/memory.
- Next milestones: shader fusion (fewer dispatches), GPU preprocessing,
  and real-screenshot collection at scale.
