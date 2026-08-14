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
│   ├── frontend.py      # Goal 2 Visual Frontend: binary mask + soft
│   │                    #   foreground from one RGB pass
│   ├── segmentation.py  # candidate lattice + visual DP decoder (P0)
│   ├── scorer.py        # unified template/CNN scoring (P4/P6)
│   ├── defaults.py      # bundled game font / charset / model paths
│   ├── postprocess.py   # allowed-chars restriction / confidence scoring
│   ├── model.py         # config.json + charset.txt + weights.bin I/O
│   ├── preprocess.py    # mask / lines / run-length CC + Goal 3 baseline-
│   │                    #   aligned binary/soft 24x24 normalization
│   ├── shaders/         # WGSL layer shaders
│   └── types.py         # Goal 1 core types: Component / VisualCandidate /
│                        #   VisualLattice / VisualScores / DecodePath /
│                        #   LexiconMatch + OCRResult / Profile
├── charsets/            # charset assets (English names, no spaces)
│   ├── words/           #   generated word lists: one word per line
│   └── sets/            #   single-line OCR charsets (combined.txt default)
├── tools/               # dev-only pipeline (PyTorch + Pillow), grouped:
│   ├── charset/         #   export_names.py + extract_charset.py
│   ├── dataset/         #   generate_font_dataset.py + collect_real_samples.py
│   │                     #   + extract_game_samples.py + synthetic samples
│   ├── train/           #   train.py + export_model.py + build_model.py
│   ├── benchmark/       #   cpu_benchmark.py + benchmark_wgpu.py + footprint.py
│   └── register_font.py #   fonts/registry.json updater
├── scripts/             # legacy entry points (generate/train/benchmark)
├── benchmarks/          # machine snapshots of the CPU benchmark suite
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
print(result.alternatives)  # Top-K single-substitution alternatives
print(result.matched_term)  # None until a lexicon is supplied
for c in result.chars:
    print(c.char, c.x, c.y, c.w, c.h, c.confidence)
```

`result.chars` is kept as a public compatibility projection. The internal
pipeline now works on `VisualCandidate`/`VisualLattice`/`DecodePath`, and
`OCRResult` explicitly separates the visible `text` from the
lexicon-inferred `matched_term`/`matched_span`.

Three classifier types are supported in the same model directory format:

- **Template** (default): `fixedfontocr-generate` renders every character
  to a bitset; recognition is XOR + popcount over a coarse-feature-filtered
  candidate set.
- **TinyCNN**: the small fixed network (Conv3x3 stride-2, DWConv3x3,
  Pointwise1x1 stride-2, GAP, Linear). Runtime inference is pure numpy or
  WGSL; training uses PyTorch.
- **Hybrid**: template level-1 + TinyCNN level-2 + unknown. Only glyphs
  whose template confidence *or* top-1/top-2 margin is low (the 未/末,
  日/曰, 0/O cases) reach the CNN — a high score with a tiny margin is
  treated as ambiguous (P6).

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
    --output hybrid_model/ --template-threshold 0.90 \
    --template-margin-threshold 0.04
```

`config.json` then declares `"classifier": "hybrid"` with
`template_threshold` / `template_margin_threshold` / `cnn_threshold`, and
`templates.bin` + `weights.bin` hold both stages.

The bundled recognizer is built from `charsets/sets/combined.txt` and
`fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf`; `tools/train/build_model.py`
regenerates the whole thing in one command.

## Segmentation: candidate lattice + visual DP

Segmentation never merges connected components irreversibly. Every line is
expanded into a candidate lattice: each original component, merges of up to
4 consecutive components, and `split(Cx)` atoms for wide components that
have a real vertical valley. Candidates are pruned by width/height,
component gap, vertical proximity, ink area and the expected font bbox
before scoring. Every surviving candidate is scored once in a batch, and a
dynamic program over the lattice picks the path with the highest mean
visual score. The lattice is exposed as `VisualLattice`, each hypothesis as
`VisualCandidate`, and the decoder output as `DecodePath`. This is what
fixes the old `stroke_width=2` special cases:

```text
鲃     -> 鲃      (fragmented into 3 components)
鲃鱼。 -> 鲃鱼。
小     -> 小      (3 components)
潜甲   -> 潜甲    (潜 has 4 components)
潜乙   -> 潜乙
巴尔的摩 -> 巴尔的摩
Z17    -> Z17
甲申   -> 甲申    (connected two-glyph blob split by split(Cx))
```

See `docs/architecture.md` for the scoring details (unified 0..1 visual
scores, margin-aware hybrid gate, O(C) Top-2) and the P2 stride-2 CNN
optimization.

## Visual Frontend (Goal 2)

Every RGB input is converted once into two representations:

```text
RGB
 ├─ binary mask     -> connected components / font geometry / Template
 └─ soft foreground -> TinyCNN
```

`fixedfontocr.extract_frontend(image, profile)` returns a
`VisualFrontend` holding both: `binary_mask` is the hard color/grayscale
decision used by segmentation and templates, and `soft_foreground` is a
0..255 foreground-strength map that keeps anti-aliasing, alpha, edge gray
and low-resolution intensity for the CNN.

The model config records which glyph representation the CNN was trained on
(`"input_mode": "binary"` for 0/255 masks or `"soft"` for 0..255 soft
glyphs). Soft-mode models are fed soft glyphs end to end; binary-mode
models keep their original 0/255 input so existing checkpoints are not
silently retrained by a frontend change. The bundled `model/game_cn`
hybrid is currently binary-mode; `tests/fixtures/cnn_digits` and
`cnn_cjk` are soft-mode fixtures that exercise the soft CNN path.

Generate a soft training dataset with
`tools/dataset/generate_font_dataset.py --soft`; `input_mode` is stored in
the npz, propagated through training/export, and written to the model
config.

## CPU benchmark suite and game regression set

```bash
# median + p95 per OCR stage, CNN batches 1..128, and 10/100/3000/7000
# charset template+CNN timings; ends with the optimized-vs-reference check
python tools/benchmark/cpu_benchmark.py --output benchmarks/cpu_benchmark.json

# the frozen regression corpus (real level badges + synthetic coverage)
python -m pytest tests/test_game_samples.py
```

`tests/game_samples/manifest.json` maps every image to its expected text,
category, profile overrides and the registered font; `slot1` level badges
are a documented limitation (`L`/`V` touch at one pixel and form one
connected component at both scales).

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
  -> connected components -> candidate lattice (merge + split(Cx))
  -> 24x24 normalization (CPU, Goal 3 baseline frame)
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
│                  # font_sha256 + normalize (Goal 3 baseline frame)
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

- Goal 3 normalization is implemented: glyphs keep their aspect ratio, are
  aligned to the font baseline (a fixed output row) and centered/padded into
  24×24. The same per-glyph frame (ink-fit scale + baseline offset) is used
  by template generation, training data and runtime candidates (binary and
  soft), and the frame parameters are stored per model in
  `config.json["normalize"]`.
- Goal 1 core data structures are defined and wired through segmentation
  and the public result: `Component`, `VisualCandidate`, `VisualLattice`,
  `VisualScores` (Top-K), `DecodePath`, `LexiconMatch` and the extended
  `OCRResult`.
- Goal 4 segmentation lattice is implemented: original components are kept,
  merge/split candidates are generated with geometric pruning, and a visual
  DP decodes the best path (`鲃`/`小`/`鲃鱼。` plus `潜甲`/`潜乙`/
  `巴尔的摩`/`Z17` regressions are covered).
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
