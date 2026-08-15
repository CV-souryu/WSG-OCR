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
│   ├── classifier.py    # template matcher V1/V2 (coarse filter + XOR/popcount)
│   ├── cnn.py           # numpy TinyCNN reference (Conv + ReLU only)
│   ├── decoder.py       # Goal 13 joint decoder: DP + beam search over the
│   │                    #   lattice with visual/geometry/lexicon/word-prior
│   │                    #   evidence, outputting the best path + alternatives
│   ├── frontend.py      # Goal 2 Visual Frontend: binary mask + soft
│   │                    #   foreground from one RGB pass
│   ├── geometry.py      # Goal 8 font geometry database: offline generation
│   │                    #   + runtime lookup (no fontTools at runtime)
│   ├── segmentation.py  # candidate lattice + visual DP decoder (P0), feeds
│   │                    #   the Goal 13 joint decoder when lexicon is used
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
├── data/                # raw game config dumps + generated training datasets
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
result = ocr.recognize(image, lexicon="ships", lexicon_mode="prefer")
print(result.matched_term)  # dictionary-inferred entity, e.g. "巴尔的摩"
print(result.matched_span)  # half-open range inside matched_term
for c in result.chars:
    print(c.char, c.x, c.y, c.w, c.h, c.confidence)
```

`result.chars` is kept as a public compatibility projection. The internal
pipeline now works on `VisualCandidate`/`VisualLattice`/`DecodePath`, and
`OCRResult` explicitly separates the visible `text` from the
lexicon-inferred `matched_term`/`matched_span`.

Three classifier types are supported in the same model directory format:

- **Template** (default): `fixedfontocr-generate` writes a Goal 9
  multi-prototype set -- every character owns 49 prototypes (11..16 px
  renders × sub-pixel phases × bilinear/area-like downsample, plus one
  clean high-res render). Recognition is XOR + popcount over a
  coarse-feature-filtered candidate set, aggregated per character into
  Top-K / best / second / margin.
- **TinyCNN**: the small fixed network (Conv3x3 stride-2, DWConv3x3,
  Pointwise1x1 stride-2, GAP, Linear). Runtime inference is pure numpy or
  WGSL; training uses PyTorch.
- **Hybrid**: template level-1 + TinyCNN level-2 + unknown. Only glyphs
  whose template confidence *or* top-1/top-2 margin is low (the 未/末,
  日/曰, 0/O cases) reach the CNN — a high score with a tiny margin is
  treated as ambiguous (P6). With Template V2 an exact match on a
  low-res prototype with a zero margin is also routed to the CNN: at
  11 px, '.' and '*' rasterize to the same blob, so the template reports
  the tie as Top-K instead of deciding.

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

## Joint decoder (Goal 13)

The decoder is the architecture's core: it consumes the scored
`VisualLattice` (every merge/split candidate with its Top-K visual scores),
the font `geometry.json` database and an optional lexicon, then picks the
best path with one unified score:

```text
total = visual + geometry + lexicon + word_prior - segmentation_penalty
```

* `visual` is the classifier-only fused score of the chosen character;
* `geometry` is the Goal 8 font-geometry agreement of the candidate;
* `lexicon` is the Goal 11/12 dictionary match of the complete visible
  string, scaled by the path's visual uncertainty so a confident screen
  glyph is never rewritten by a dictionary substring (Goal 15);
* `word_prior` is a per-character prior estimated from the lexicon's own
  terms, gated by each character's visual uncertainty;
* `segmentation_penalty` is a small per-extra-hypothesis cost that breaks
  near-ties toward fewer characters, so a glyph (`小`) is not fragmented
  into several look-alike characters.

`decode_dp` is the first-version exact dynamic program (its state keeps the
lexicon prefix trie node); `decode_beam` is the beam-search upgrade
(`beam_width` 8..32, default 16) that supplies `DecodePath.alternatives`.
The exact DP stays authoritative for the best path so the frozen CPU
segmentation regressions are never overturned by a fragmented look-alike
path; the beam's runner-up texts are ranked with the full formula.
`recognize(..., lexicon=...)` runs the joint decoder and the chosen
characters (`DecodePath.char_ids`) are authoritative, while `apply_lexicon`
still handles strict mode and the Goal 12 `matched_term`/`matched_span`
annotation.

## Font geometry database (Goal 8)

Before Goal 8 the lattice relied on a generic ``0.8 * median height``
expected-glyph width and a small hand-tuned geometry penalty. The project
now ships a per-font geometry database generated offline from ``fonts/``:

```text
fonts/ SourceHanSansSC-Bold.otf + charset
  -> geometry.json
       char_id, advance, bbox width/height, aspect ratio, ink count,
       component count, baseline (+ derived ink ratio)
```

Every template/CNN/hybrid model written by the package includes
`geometry.json`. The runtime loader keeps it as plain JSON + small lookup
tables (no fontTools/Pillow dependency), and `FixedFontOCR` uses it in
three places:

* candidate pruning uses the font's real narrow vs. full-width bbox
  ratios, so `1 I l i !` never share a full-width CJK prior;
* the segmentation geometry score compares each candidate's bbox, ink
  ratio, component count and baseline with the classifier's Top-1
  character (a 3-component `小` merge matches its database entry; a single
  fragment of it does not);
* split/merge decisions use the database-derived expected glyph width
  instead of the old height-only heuristic.

`tests/test_goal8_geometry.py` pins the database contract and the Goal 4
regressions still pass with the database enabled.

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

## Low-resolution training domain (Goal 7)

Training data is no longer dominated by a clean 32 px render. The dataset
generator now simulates the full game pipeline:

```text
fonts/ font
  -> supersampled render (2-3x)
  -> random 10..18 px final size
  -> bilinear / area-like downsampling
  -> sub-pixel x/y offset, scale variation, blur, alpha, brightness,
     background blend, outline and shadow
  -> Goal 3 24x24 normalized glyph
```

`tools/dataset/generate_font_dataset.py` records `render_sizes`,
`source_sizes`, `downsample_modes` and per-sample `augmentations` in the
npz, so a training run can prove it covered every Goal 7 size (10, 11,
12, 13, 14, 15, 16, 17, 18 px). `tools/train/build_model.py` and
`scripts/train_tinycnn.py` default to this low-res domain, and the
`Z17` / `巴尔的摩` characters are part of the regression charset
(`tests/test_goal7_low_res.py`).

## CPU benchmark suite and game regression set

```bash
# median + p95 per OCR stage, CNN batches 1..128, and 10/100/3000/7000
# charset template+CNN timings; ends with the optimized-vs-reference check
python tools/benchmark/cpu_benchmark.py --output benchmarks/cpu_benchmark.json

# the frozen regression corpus (real level badges + synthetic coverage)
python -m pytest tests/test_game_samples.py
```

`tests/game_samples/manifest.json` maps every image to its expected text,
category, profile overrides, optional lexicon/matched-term annotations and
the registered font. The Goal 14 battery adds small-size `鲃鱼`/`小`/
`潜甲`/`潜乙`/`Z17`/`巴尔的摩`, mixed Chinese+ASCII, digits, short/long
words and a partial-word crop to the corpus; `slot1` level badges remain a
documented limitation (`L`/`V` touch at one pixel and form one connected
component at both scales).

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

## Game corpus（游戏语料）

The raw game corpus is `data/cn/`, a set of Unity `JsonUtility` config dumps
(numeric dict keys are unquoted, so they are read with the tolerant parser in
`tools/charset/extract_charset.py`). Each file contributes one extracted word
list and one OCR charset:

| raw source (`data/cn/`) | field | extracted word list | runtime domain |
| --- | --- | --- | --- |
| `ship_h.json` | `title`（原版舰名） | `charsets/words/ship_names.txt` | `ships` |
| `ship.json` | `title`（和谐后舰名） | `charsets/words/ship_names_harmonized.txt` | `ships` |
| `equip.json` | `title`（装备名） | `charsets/words/equipment_names.txt` | `equipment` |
| `language.json` | `schinese`（UI 文案） | `charsets/words/ui_texts.txt` | `ui` |

`tools/charset/export_names.py` regenerates the word lists: it strips
rich-text color codes (`^C...`), deduplicates strings in first-seen order and
writes one word per line. `tools/charset/extract_charset.py` then derives the
single-line OCR charsets in `charsets/sets/` from those word lists, and
`src/fixedfontocr/lexicon.py` loads the word lists by domain at runtime
(`ships` merges the original and harmonized ship-name lists).

`data/cn/` also holds generated training artifacts (`.npz` synthetic/real
glyph datasets, `.pth` checkpoints, `game_cn` runtime models); those are
derived data, not the source corpus.

## Training and export pipeline

Development dependencies (`torch`, `pillow`, `fonttools`, `numpy`) live in
the `train` extra; runtime inference has no PyTorch dependency.

```bash
# 1. Synthetic data: Goal 7 low-res domain (10..18 px supersampled render,
#    bilinear/area-like downsample, sub-pixel offset, scale, blur, alpha,
#    brightness, background blend, outline/shadow) -> 24x24 glyphs
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
| template V2 weights on disk | ~10.3 MiB for 3000 chars (`N × 49 × 72 B` bitsets + per-prototype metadata; Goal 9) |
| charset.txt | ~9 KiB (3000 CJK chars) |
| geometry.json | ~0.9 MiB for 3000 chars (per-char Goal 8 metrics, ~300 B/char) |
| template bits in RAM | ~211 KiB (shared with the loaded model, no copy) |
| template V2 bits in RAM | ~10.3 MiB + ~0.6 MiB prototype metadata/coarse features |
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
       template: coarse feature filter -> XOR + popcount over 49
                 prototypes/char -> per-char Top-K / best / second / margin
       tinycnn:  CPUBackend / WGPUBackend / AutoBackend
       hybrid:   template first, CNN fallback, unknown
  -> allowed_chars restriction (postprocess)
  -> Goal 13 joint decoder (DP + beam alternatives, lexicon/word-prior
     evidence, segmentation penalty)
  -> apply_lexicon (none/prefer/strict annotation, Goal 11/12)
  -> charset -> string
```

## Model format

```
model/
├── config.json    # input_width, input_height, classes, version, dtype,
│                  # classifier (template|tinycnn|hybrid), thresholds,
│                  # architecture (tinycnn_v1), font_sha256 + normalize
│                  # (Goal 3 baseline frame)
├── geometry.json  # Goal 8 font geometry database (per-char metrics)
├── charset.txt
└── weights.bin    # template: uint32 count + packed bits;
                   # tinycnn/hybrid: f32 tensors in fixed order
                   # hybrid also has templates.bin

Template models written since Goal 9 use the V2 format instead:
``weights.bin`` (or ``templates.bin`` for hybrid) starts with the magic
``TPL2``, then ``uint32 count / prototypes_per_char / bytes_per_template``,
then a per-prototype metadata block (render size, sub-pixel phase in 1/8 px,
downsample mode) and finally the packed bitset payload. Legacy V1 files
are detected by the absence of the magic and keep loading unchanged.
``config.json`` records ``"template_version": 2``.
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
- Goal 5 TinyCNN V1 is frozen: the exact Conv3x3→DWConv→Pointwise→GAP→Linear
  topology and 24×24 input are enforced at model load/export and in the
  numpy/torch/WGPU layer maps. Only single-channel (binary or soft) and the
  experimental two-channel soft+binary input variants are allowed; no
  Transformer/LSTM/Attention/complex normalization may be added.
- Goal 6 CPU TinyCNN optimization is complete: stride-2 layers compute only
  target output positions, the hot forward skips activation-dict and
  same-dtype copies, weights are prepared as contiguous float32 once, batch
  inference is vectorized, and Top-K uses argpartition without a full
  argsort. NumPy vs PyTorch parity (`max error < 1e-5`, identical argmax)
  is covered by `tests/test_goal5_tinycnn.py` and
  `tests/test_goal6_tinycnn.py`.
- Goal 7 low-resolution training domain is implemented: synthetic data is
  generated from `fonts/` at 10..18 px through supersampled rendering +
  bilinear/area-like downsampling with sub-pixel offset, scale, blur, alpha,
  brightness, background blend, outline and shadow augmentation. The
  training entry points default to this domain and
  `tests/test_goal7_low_res.py` covers every size plus `Z17`/`巴尔的摩`.
- Goal 8 font geometry database is implemented: every model built from a
  registered font embeds `geometry.json` (advance, bbox, aspect, ink,
  component count, baseline per character), and the runtime uses it for
  candidate pruning, the geometry score and split/merge width priors.
  `tests/test_goal8_geometry.py` covers generation, round-trip, narrow vs.
  full-width priors and the regression set.
- Goal 9 Template V2 is implemented: `fixedfontocr-generate` defaults to a
  49-prototype-per-character grid (11..16 px × sub-pixel phases ×
  bilinear/area-like downsample + clean high-res render); matching keeps
  the geometry prefilter + XOR/popcount cascade and returns Top-K / best /
  second / margin with winning-prototype metadata. The bundled
  `model/game_cn_template` and hybrid `model/game_cn` are shipped in the
  V2 format, and `tests/test_goal9_template_v2.py` covers the grid,
  round-trip, prefilter exactness, low-res confusables (未/末, Z/2) and the
  '.'/ '*' low-res tie routing to the CNN.
- Goal 10 unified visual scoring is implemented: hybrid candidates keep
  `template_raw_score`, `cnn_logit`, `cnn_margin` and `geometry_score`
  separately, and the decoder consumes one weighted
  `visual_score = a*cnn_score + b*template_score + c*geometry_score`
  (weights + calibration live in `config.json`). The bundled hybrid model
  is tuned on `data/cn/real_game.npz`, public confidence is calibrated into
  `[0, 1]`, and `tools/train/tune_visual.py` re-fits the weights on any
  real-screenshot npz. `tests/test_goal10_unified_scoring.py` pins the
  evidence fields, formula, geometry integration and calibration.
- Goal 11 lexicon layer is implemented: `charsets/words/` is loaded by
  domain (`ships`, `equipment`, `ui`, plus `all` or explicit paths), and
  `recognize(..., lexicon=..., lexicon_mode=...)` supports
  `none`/`prefer`/`strict`. The matcher handles exact and full-term-in-text
  alignments; `prefer` only rewrites visually
  uncertain characters when the dictionary target is already in the
  candidate's Top-K, and `strict` rejects non-dictionary text.
  `tests/test_goal11_lexicon.py` covers loading, matching and the modes.
- Goal 12 partial-word support is implemented: a screen crop of a
  dictionary term is matched as `prefix_crop` / `suffix_crop` / `inner_crop`
  while internally missing characters are ranked as `gap_crop` with a
  higher penalty. The result always keeps the visible `text` and only
  annotates the inferred entity: `text="C2C3C4C5"`,
  `matched_term="C1C2C3C4C5C6"`, `matched_span=(1, 5)`; the visible text is
  never fabricated into the full term. `tests/test_goal12_partial_word.py`
  covers matcher conventions, penalty ordering and end-to-end crops
  (`巴尔的摩`, `塞瓦斯托波尔`).
- Goal 13 joint decoder is implemented: `decoder.py` consumes the visual
  lattice + Top-K visual scores + font geometry + lexicon and scores every
  path as `visual + geometry + lexicon + word_prior - segmentation_penalty`.
  `decode_dp` is the exact first-version DP (lexicon-prefix trie state),
  `decode_beam` is the beam-search upgrade (`beam_width` 8..32, default 16)
  that outputs the best path and `alternatives`; the public
  `recognize(..., lexicon=...)` pipeline uses the decoder, keeps the frozen
  CPU regressions intact and never lets a dictionary substring override
  confident visual evidence (Goal 15). `tests/test_goal13_decoder.py` pins
  the formula, DP/beam behavior, segmentation penalty, lexicon tie-breaking,
  geometry input and the end-to-end acceptance strings.
- Goal 14 typical-problem regression battery is established:
  `tests/test_goal14_regressions.py` pins the named cases at the project's
  small sizes (12/14/16 px) plus 32 px — `鲃鱼` never decodes as `$E鱼`,
  `小` never fragments into multiple characters, `潜甲`/`潜乙` never merge,
  `Z17` and `巴尔的摩` stay correct at small sizes, and the battery covers
  mixed Chinese+ASCII (`舰船Lv.99`), digits, punctuation, short/long words
  and partial-word crops. The fixes behind it: the hybrid scorer keeps a
  full Top-5 in `VisualScores` and lets low-res template evidence
  corroborate a low-margin CNN pick instead of discarding it, the font
  geometry score stops double-penalizing multi-component glyphs (小/鲃/潜)
  and rewards an exact component-count match, and the game-sample corpus
  grew to 37 images with per-sample lexicon annotations.
- Template (with coarse candidate filtering), TinyCNN CPU and TinyCNN WGPU
  are implemented and tested on Latin and CJK.
- `backend="auto"` benchmarks CPU vs WGPU at construction and selects per
  batch size at runtime.
- Hybrid template→CNN→unknown models, `allowed_chars`, BN folding and
  `test_vectors.npz` exports are implemented and covered by tests.
- Template coarse features are stored in compact uint8/uint16 arrays
  (8 B/char) and numpy ≥ 2.0 uses `bitwise_count` instead of the 64 KiB
  popcount table; `tools/benchmark/footprint.py` reports storage/memory.
- Next milestones: real-screenshot collection at scale, shader fusion
  (fewer dispatches) and GPU preprocessing.
