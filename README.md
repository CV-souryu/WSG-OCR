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
│   ├── tracker.py       # Goal 16 cross-frame tracker: ROI change detection,
│   │                    #   result caching, multi-frame logits fusion, voting
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
│   │                     #   + extract_portrait_card_samples.py (Goal 18)
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
`decode_beam` re-ranks every retained complete path with the full
crop-aware `score_path` formula, so a path whose complete partial-word
lexicon score is higher can overturn a path that only looked better from
DP-local prefix/term bonuses. `segment_line` invokes the beam when a
lexicon is supplied; without a lexicon it uses the exact DP path as the
frozen CPU segmentation reference.

`segment_line` classifies the whole candidate lattice once; `recognize`
then projects `DecodePath.char_ids` and the chosen `candidate.scores`
entries straight into `CharResult` -- there is no second classification of
the decoder-selected glyphs. `apply_lexicon` still handles strict mode and
the Goal 12 `matched_term`/`matched_span` annotation.

## Cross-frame tracking (Goal 16)

`recognize` stays a pure single-frame function. When the same UI crop is
seen repeatedly, an optional tracker layers temporal state on top without
mutating the OCR engine:

```python
tracker = ocr.tracker()

result = tracker.update(frame1)   # 巴尔的摩
result = tracker.update(frame2)   # 巴你的摩 (one-frame flicker)
result = tracker.update(frame3)   # 巴尔的摩 (stable)
```

Each update first computes a compact text-ROI signature. An unchanged ROI
is served from the tracker's result cache (no re-run of segmentation or
classification). When the ROI content does change, the recent per-character
Top-K evidence is aligned and summed, so one bad frame cannot overwhelm the
accumulated logits, and majority text voting suppresses flicker once enough
frames agree. The tracker owns all of this state; `ocr.recognize(image)` is
unchanged and temporal state never leaks into the baseline OCR tests.

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
# Goal 17: median + p95 for foreground / CC / lattice generation /
# normalize / template / TinyCNN / decoder / total, CNN batches 1..128,
# and 10/100/1894 charset template+CNN timings; ends with the
# optimized-vs-reference check
python tools/benchmark/cpu_benchmark.py --output benchmarks/cpu_benchmark.json

# the frozen regression corpus (real level badges + synthetic coverage)
python -m pytest tests/test_game_samples.py
```

`tests/game_samples/manifest.json` maps every image to its expected text,
category, profile overrides, optional lexicon/matched-term annotations and
the registered font. The Goal 14 battery adds small-size `鲃鱼`/`小`/
`潜甲`/`潜乙`/`Z17`/`巴尔的摩`, mixed Chinese+ASCII, digits, short/long
words and a partial-word crop to the corpus, and the Goal 15 battery pins
that the ships lexicon never rewrites clear `潜乙` into `潜甲`; `slot1`
level badges remain a documented limitation (`L`/`V` touch at one pixel
and form one connected component at both scales).

Goal 18 turns the corpus into a real game dataset: besides the 12 level
badges, `real-game/ship-name/` holds 244 crops taken directly from 19 real
WSG ship-list screenshots (validated ship names from the portrait-card
alignment). `tools/dataset/extract_portrait_card_samples.py` regenerates
them from the external `portrait-card-testset.zip`; samples the current
model does not read yet are kept in the manifest as `known_failure` entries
with the observed output in `known_failure_note`, so real regressions stay
visible instead of being dropped. The corpus currently contains 256 real
game samples (the Goal 18 100+ milestone); 500+ / 1000+ accumulation is the
next step as more real screenshots are labeled.

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
  `none`/`prefer`/`strict`/`topk`/`dict`. The matcher handles exact and
  full-term-in-text alignments; `prefer` only rewrites visually
  uncertain characters when the dictionary target is already in the
  candidate's Top-K, and `strict` rejects non-dictionary text.
  `tests/test_goal11_lexicon.py` covers loading, matching and the modes.
- Associative dictionary mode (`lexicon_mode="dict"`) is implemented:
  the decoded visible text is scanned position by position and every
  dictionary term that can explain it — exact, cropped at either edge,
  confusable (Top-K substitution, e.g. 14px `波`/`彼`), or partially
  damaged (forced completion of a unique prefix/suffix, e.g.
  `华盛蜂` → `华盛顿`) — competes as one word hypothesis scored from the
  per-position visual Top-K evidence. The visible `text` is never
  rewritten; the winning word is annotated as `matched_term` /
  `matched_span`, and a result without any surviving word hypothesis is
  rejected as empty output ("return nothing unless a word matches").
  Single-char terms need stronger evidence, uniqueness is enforced with
  a margin, and clear visual text is never rewritten (Goal 15, `潜乙`
  stays `潜乙`). A forced-completion hit is verified against the crop
  itself (every conflict position re-matched with the registered font's
  templates); when no hypothesis survives, an optional real-glyph bank
  (`recognize(..., real_glyph_bank=...)`, built by
  `tools/lexicon/build_real_glyph_bank.py` from labeled crops) runs the
  same alignment with NCC evidence from real game glyphs -- game renders
  vs game renders, no font-render domain gap (resolves the 初雪/白雪/
  夕雾 same-shape ties). NCC evidence is memoized per (glyph, character)
  pair, so crops whose visible characters are common (尔/维 → ~94 candidate
  terms) scan each prototype stack once instead of once per term
  (乌戈里尼 crop arbitration 115.8 → 21.1 ms). On the 244 real ship-name
  crops it annotates 235/244
  correctly with zero wrong associations (the rest are rejected as
  unreadable). `tests/test_dict_mode.py` pins the contract.
- Touching-game-glyph merges are vetoed for hybrid models: real-game
  `Z+1`, `Z+2`, `4+7` pairs glue into one wide blob that the CNN reads as
  one character (`灶`, `“`), and the mean-visual path score alone prefers
  that merge over the correct split (`Z17` crops decoded as `灶7`).
  `_drop_weak_merges` now down-weights a multi-atom candidate whose Top-1
  is not template-confirmed when the blob spans ≥ 1.5x the line's typical
  glyph width and either (a) its components are strictly side-by-side
  (glyph-internal radicals overlap horizontally; adjacent characters do
  not) and one full-size atom alone is already a better character than
  the blob, or (b) the blob has zero template support and ≥ 2 of its
  atoms are full-size confident glyphs. Real glyphs keep their template-
  confirmed whole (raw score ≥ 0.9 / template-chosen Top-1). On the
  labeled `fonts/SourceHanSansSC/crops/crops_items` corpus this fixes
  `Z17`/`Z28`/`Z1`/`47工程` in every lexicon mode with zero regressions
  against the CSV baseline, and dict+real-glyph-bank mode now resolves
  233/235 labeled items (remaining rejects: a `4+3` blob read as `“` and
  a crop missing half of `乌戈里尼·维瓦尔迪`).
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
  carries per-sample lexicon annotations for the Goal 14/15 battery.
- Goal 15 lexicon/visual priority is implemented: the lexicon may annotate
  or help only when the visual evidence is uncertain. Confident visible
  text is never rewritten, dictionary corrections require the target to be
  in the character's visual Top-K, a non-unique or near-tied dictionary
  match keeps the OCR text and its `alternatives`, and unknown text is
  still emitted normally. The decoder gates every lexicon/word-prior term
  by `1 - mean(visual)`, and `apply_lexicon` enforces the same rule at the
  result level with `prefer_threshold`/`correction_unique_margin`.
  `tests/test_goal15_lexicon_visual_priority.py` pins the contract,
  including the canonical case that clear `潜乙` is never rewritten to
  `潜甲` even though `潜甲` is in the ships lexicon.
- Goal 16 cross-frame tracking is implemented: `ocr.tracker()` / 
  `FrameTracker` layers ROI change detection, per-ROI result caching,
  multi-frame Top-K logits fusion and stable text voting on top of the
  pure single-frame API. `tests/test_goal16_cross_frame.py` pins caching,
  ROI change flags, the `巴尔的摩 / 巴你的摩 / 巴尔的摩` stabilization
  example and tracker reset.
- Goal 19 CPU freeze is complete: the numpy CPU pipeline is the canonical
  implementation (`src/fixedfontocr/defaults.py`: `CPU_FREEZE = True`,
  version 1.0, 2026-08-15). `tests/test_goal19_cpu_freeze.py` pins every
  freeze checkbox — segmentation lattice stability, 鲃/小, low-res
  `Z17`/`巴尔的摩`, mixed charset, Top-K API, lexicon decoder,
  partial-word, the complete CPU benchmark and the complete regression
  dataset — so WGPU work (Goal 20) starts from a stable reference.
- Template (with coarse candidate filtering), TinyCNN CPU and TinyCNN WGPU
  are implemented and tested on Latin and CJK.
- `backend="auto"` benchmarks CPU vs WGPU at construction and selects per
  batch size at runtime.
- Hybrid template→CNN→unknown models, `allowed_chars`, BN folding and
  `test_vectors.npz` exports are implemented and covered by tests.
- Template coarse features are stored in compact uint8/uint16 arrays
  (8 B/char) and numpy ≥ 2.0 uses `bitwise_count` instead of the 64 KiB
  popcount table; `tools/benchmark/footprint.py` reports storage/memory.
- Template matching streams its XOR/popcount pass in 2048-prototype
  chunks, the coarse-filter features are fetched once per batch instead
  of once per candidate, and the load-time prototype unpack runs one
  character at a time. Peak RSS for recognizing a full roster line
  dropped from ~258 MB to ~109 MB (model load from ~163 MB to ~59 MB).
- CPU speedups (byte-identical outputs; 628 tests + the 235-crop corpus
  regression hold): the FontGeometryDatabase narrow/full aggregate ratios
  are cached once instead of recomputed per access (lattice_generation
  13.2 → 0.36 ms), per-char minima use `np.minimum.reduceat` over the
  char-ordered prototype blocks (~0.17 → ~0.01 ms per call), and the
  coarse filter's feature unpack now runs along the byte axis — the
  historical prototype-axis unpack scrambled the positional features so
  real-game glyphs never passed the filter and every candidate fell
  through to the full ink-band exact scan. Recorded benchmark total
  168.3 → 95.3 ms (template 133.6 → 80.3 ms); real-game corpus: items
  14.6 → 10.2 ms/张, full lines 108.1 → 72.6 ms. The exact scan remains
  ink-band bound (~57k of 92.8k prototypes per candidate on game crops),
  so further CPU gains need parallelism or the WGPU path (Goal 20).
- Next milestones (after the Goal 19 CPU freeze): Goal 20 WGPU phase 2
  (DP-selected candidates back into the WGSL classifier, shader fusion,
  GPU preprocessing) — prepared in [`docs/goal20.md`](docs/goal20.md)
  (design + task breakdown T1-T4 + acceptance criteria) with the contract
  skeleton `tests/test_goal20_wgpu.py` — and real-screenshot corpus
  accumulation to 500+/1000+.
- Goal 20 G1 landed: `shaders/mega.wgsl` runs the ENTIRE TinyCNN in one
  dispatch (one 64-thread workgroup per glyph, all intermediates in
  workgroup shared memory). `WGPUBackend.classify()`/`forward_logits()`
  each submit exactly one dispatch with a single final readback — zero
  intermediate copy-backs. The 8-sync `forward_logits` wall
  (12-17 ms) is gone (1.4-2.7 ms).
- Goal 20 G2 landed: `shaders/template_match.wgsl` runs the whole
  Template V2 cascade (coarse filter, XOR+popcount, ink-band fallback,
  deterministic Top-K, winner prototype) in one dispatch per glyph batch,
  byte-exact against the CPU reference (`tests/test_goal20_template_gpu.py`).
  `backend="wgpu"` uses it always, `backend="auto"` picks per batch
  (crossover 4). End-to-end `recognize()` on `model/game_cn` is now
  **3.2-4.7x faster on GPU/auto** than CPU with identical text output.
- Goal 20 G3 landed: `shaders/preprocess_soft.wgsl` uploads the RGB image
  once and does ROI crop + grayscale + nearest-neighbor resize + baseline
  placement in one dispatch, byte-exact against the CPU soft batch;
  `WGPUBackend.forward_logits_from_image` runs preprocess + mega in ONE
  submit. Validated on the crops_items dict-mode corpus
  (`tests/test_goal20_crops_gpu.py`: byte parity, fused-logits parity,
  full-corpus wgpu/cpu parity; auto backend shows no regression on any
  crop and 1.86x on the big 乌戈里尼 crop).
- Goal 20 G4 landed: `WGPUScoringStage.score_line` closes the whole
  scoring chain (template match + TinyCNN logits) into ONE submit with
  ONE readback per line (2 syncs -> 1; e.g. 初雪 crop 26.4 -> 21.1 ms).
  `score_line_from_image` additionally folds the G3 preprocess dispatch
  into the same encoder for soft-input hybrids, so a full OCR line
  (G2+G3+G4) comes out of ONE submit / ONE `map_sync` — the whole
  `recognize()` on the crops_items dict corpus is verified to be a single
  GPU submission (trip-wired tests) with answers matching
  `crops_items_recognition.csv` and `backend="wgpu" == backend="cpu"`.
  The visual DP intentionally stays on CPU: `decode_dp` uses f64
  arithmetic with a 1e-12 tie tolerance that f32 WGSL cannot reproduce,
  and it is microsecond-scale — fonts/goal keeps the decoder on the CPU
  boundary.
- Goal 20 T2 landed: `Backend.classify_topk` + `logits_for` (GPU: new
  mega modes with numpy fallbacks on the ABC) and the stage's production
  path reads back the masked top-7 + template-id gather instead of the
  full `[N, C]` logits — **48.9x less readback** on game_cn (7,636 B ->
  156 B per glyph) with a tie-boundary fallback guard; the hybrid fusion
  algorithm is unchanged (`tests/test_goal20_crops_gpu.py` sparse
  contract + full-corpus CSV parity).
- Goal 20 T4 landed: `classify`/`forward_logits` accept `staged=True`
  (`map_async` on ping-pong staging buffers, byte-identical results).
  Measured on Metal M4 the per-frame floor is the submit→map round trip
  itself (1.65-1.71 ms for sync/staged/pipelined alike), so double
  buffering shows no measurable gain on this platform — documented as an
  honest negative result; the API is in place for heavier host/GPU
  scenarios.
- Goal 20 template scan tuned: `template_match.wgsl` runs 512 threads per
  workgroup with register-local per-char minima (28 KB `sm_min` shared
  array removed) — `match_batch` N=1/7: 9.1/11.7 ms -> 4.0/2.7 ms,
  byte-exact parity unchanged, template auto crossover batch 4 -> 2, and
  every pinned crops crop now reaches or beats CPU end-to-end (乌戈里尼
  2.3x).
- **Goal 20 is complete** (G1-G4 + T1-T4): `tests/test_goal20_wgpu.py` is
  fully green (no xfail), the full suite is 656 passed / 0 failed on the
  current HEAD CPU reference, and the design/task/benchmark writeup lives
  in [`docs/goal20.md`](docs/goal20.md) with the architecture in
  [`docs/wgpu.md`](docs/wgpu.md).
