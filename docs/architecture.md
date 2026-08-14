# Architecture

FixedFontOCR is a deterministic fixed-font OCR engine with a template fast
path and a TinyCNN fallback. Both a numpy CPU backend and a WGPU compute
backend consume the exact same model files and produce the same results;
`backend="auto"` benchmarks them once at construction and selects per batch
size at runtime. The model format and the public API are the two stable
contracts.

## Repository layout

```
.
├── src/fixedfontocr/    # the package
├── tools/               # dev-only pipeline (torch/pillow), grouped by task:
│   ├── charset/         #   word-list export + charset extraction
│   ├── dataset/         #   synthetic + real-sample data generation
│   ├── train/           #   training / model export / one-command build
│   ├── benchmark/       #   CPU/GPU benchmark + footprint analysis
│   └── register_font.py #   fonts/registry.json updater
├── scripts/             # legacy model generation & training entry points
├── tests/               # pytest suite; checked-in fixture models in tests/fixtures/
├── docs/                # this documentation
├── charsets/            # charset assets: words/ (one word per line) + sets/
├── fonts/               # registered font files (gitignored; registry.json tracked)
├── pyproject.toml
└── README.md
```

## Font Policy

Only fonts under `fonts/` and registered in `fonts/registry.json` (by
SHA256) may be rendered, trained, or embedded in model metadata. System
font discovery, fallback fonts, and font-family augmentation are
prohibited; missing glyphs are explicit errors. Every model config stores
the source font's SHA256 as `font_sha256`.

## Module map

| Module | Responsibility |
| --- | --- |
| `src/fixedfontocr/api.py` | Public `FixedFontOCR` engine. Loads the model, builds the selected backend (`cpu`/`wgpu`/`auto`), runs the template→CNN→unknown hybrid flow and applies `allowed_chars`. |
| `src/fixedfontocr/backends.py` | Unified batch classifier API (`Backend.classify(glyphs)`): `CPUBackend` (numpy), `WGPUBackend` (WGSL compute) and `AutoBackend` (measured per-batch selection). |
| `src/fixedfontocr/shaders/*.wgsl` | Phase-1 WGSL layers: normalize, conv3x3, dwconv3x3, pointwise, gap, linear, argmax, fused linear+argmax. |
| `src/fixedfontocr/types.py` | Goal 1 core dataclasses: `Component`, `VisualCandidate`, `VisualLattice`, `VisualScores` (Top-K), `DecodePath`, `LexiconMatch`, extended `OCRResult`; plus compatibility `CharResult`, `Profile`, `ClassificationBatch`, `CandidateScore`. |
| `src/fixedfontocr/preprocess.py` | Color/grayscale mask, line finding, run-length connected components, Goal 3 binary + soft baseline-aligned 24×24 normalization (`NormalizeSpec`, `glyph_normalize_geometry`). |
| `src/fixedfontocr/frontend.py` | Goal 2 Visual Frontend: one RGB pass extracts `binary_mask` (segmentation/template) and `soft_foreground` (TinyCNN), plus binary/soft glyph normalization helpers. |
| `src/fixedfontocr/segmentation.py` | Candidate lattice (every original component + merges up to 4 components + split(Cx) atoms) and the visual DP decoder. This is the production segmentation path. |
| `src/fixedfontocr/scorer.py` | `SegmentScorer`: batch template/CNN scoring with the margin-aware hybrid gate; converts raw scores to a shared 0..1 visual score. |
| `src/fixedfontocr/classifier.py` | `Classifier` interface plus `TemplateClassifier`: coarse-feature candidate filtering (ink count, bbox, margins) followed by XOR + popcount; `match_batch` for the lattice. |
| `src/fixedfontocr/cnn.py` | `TinyCNNClassifier` numpy forward pass (stride-2 optimized), `forward_with_activations` for exported test vectors and `classify_batch(glyphs, top_k=2)`. |
| `src/fixedfontocr/reference_cnn.py` | Full-then-slice reference forward used by tests to prove the optimized forward matches (P2 acceptance). |
| `src/fixedfontocr/postprocess.py` | `allowed_chars` → charset-index restriction and O(C) top-1/top-2 scoring (no full argsort). |
| `src/fixedfontocr/fontgen.py` | Renders font glyphs into normalized bitset templates; writes model directories. |
| `src/fixedfontocr/model.py` | Model format I/O: `config.json` + `charset.txt` + `weights.bin` (+ `templates.bin` for hybrid). |
| `src/fixedfontocr/cli.py` | `fixedfontocr-generate` command-line entry point. |
| `tools/register_font.py` | Adds a font under `fonts/` to `fonts/registry.json` with its SHA256. |

## Pipeline

```
numpy RGB
  -> Visual Frontend (one pass)
       binary mask   -> profile color/grayscale mask
       soft          -> 0..255 foreground strength (anti-aliasing/alpha/edges)
  -> line finding (binary mask)
  -> run-length connected components (original components are kept)
  -> candidate lattice (single components + merges of up to 4 + split(Cx),
     geometrically filtered)
  -> 24x24 normalization (Goal 3 baseline frame, binary + soft)
  -> batched scoring (TemplateClassifier.match_batch / TinyCNN forward)
       template: confidence AND top-1/top-2 margin must both be high
       cnn:      soft glyphs when model input_mode="soft", binary otherwise
                 sigmoid(top1-top2 margin) -> shared 0..1 visual score
       hybrid:   template gate first, CNN fallback for ambiguous glyphs
  -> visual DP (max mean visual score path, geometry penalties)
  -> allowed_chars restriction (postprocess)
  -> charset -> string
```

## Normalization (Goal 3)

Normalization is the point where the fixed font's vertical layout is
preserved. The old path took a tight glyph bbox and force-stretched/centered
it into 24×24; that destroys baseline information, so `1/I/l`, `Z/2/7`,
full-width CJK, narrow Latin and punctuation all ended up in unrelated
frames.

The Goal 3 frame is:

```
glyph
  -> keep aspect ratio
  -> font baseline alignment (fixed output row)
  -> horizontal centering / padding
  -> 24x24
```

`NormalizeSpec` (computed from the registered font + charset with
`compute_normalize_spec`, stored in `config.json["normalize"]`) holds the
font's script ratios: median full-width ink height/width and the CJK
baseline offset. Every candidate's geometry is then estimated once by
`glyph_normalize_geometry`:

* the scale stays the classic aspect-preserving ink-fit (`0.8 × 24` on the
  largest ink dimension), so glyph shapes are stable across font sizes;
* full-width (CJK) candidates are placed with their baseline ~0.087 em below
  their ink bottom, matching the font's true layout;
* narrow candidates (digits, caps, ascenders, `1/I/l`...) sit on the
  baseline, so `Z/2/7` share one bottom row while full-width CJK extends
  below it.

Critically, template generation (`fontgen.build_templates`), synthetic
dataset generation (`tools/dataset/generate_font_dataset.py`,
`scripts/train_tinycnn.py`) and runtime candidates
(`segmentation.segment_line` -> `SegmentScorer` -> `api.py`) all call the
same `glyph_normalize_geometry` + `normalize`/`normalize_grayscale` pair, so
a rendered glyph and its template land on the exact same bitmap (proven by
`tests/test_goal3_normalize.py`). Binary masks and soft foreground ROIs use
identical placement; the soft path keeps anti-aliasing/edge intensity.

Models without a `normalize` spec (pre-Goal-3 checkpoints) keep the legacy
centered normalization, so old checkpoints remain loadable.

## Visual Frontend (Goal 2)

The frontend is the single entry point from RGB pixels to the two internal
representations used downstream. `extract_frontend(image, profile)` runs
one pass over the `uint8 [H, W, 3]` input and returns:

* `binary_mask` — the hard `Profile.color_mask` decision. Connected
  components, font-geometry checks and the Template matcher consume this;
* `soft_foreground` — a 0..255 foreground-strength map with no hard cutoff.
  Anti-aliased edge pixels, alpha-blended intensities, edge gray and
  low-resolution sampling information survive here for the TinyCNN.

The model config records `input_mode` (`"binary"` or `"soft"`) so the
runtime feeds the CNN exactly the representation it was trained on.
`generate_font_dataset.py --soft` stores 0..255 glyphs in the training npz,
`tools/train/train.py` propagates the mode into the checkpoint, and
`tools/train/export_model.py` writes it into the runtime config. Binary
mode is the default, so pre-existing models keep their 0/255 training
domain; soft-mode models (e.g. the checked-in `cnn_digits`/`cnn_cjk`
fixtures) are scored with soft glyphs end to end.

TinyCNN models use a unified batch backend; both implementations return the
same `BackendResult(char_ids, scores)` where `scores` is the top1-top2 logit
margin. See [`docs/wgpu.md`](wgpu.md) for the WGPU architecture.

## Automatic backend selection

`FixedFontOCR(backend="auto")` times `classify()` on both backends at batch
sizes 1/8/16/32/64/128 during construction, stores the table
(`ocr.backend_benchmark`) and wraps the pair in `AutoBackend`. Each runtime
call picks the faster measured backend for the actual glyph count, using
linear interpolation inside the table and marginal-cost extrapolation beyond
it. If WGPU construction or the benchmark fails, auto falls back to CPU and
records the reason in `ocr.backend_benchmark_error`.

## Candidate filtering

The template matcher precomputes per-character coarse features from the
normalized bitmaps: ink pixel count, bbox height/width and top/left/bottom/
right margins. A glyph's features are compared with generous tolerances and
only the surviving candidates enter the XOR + popcount scan. For a 3000-char
font this is the "width filter → bitmap coarse feature → exact match"
cascade from the design plan.

The filter is exact, not heuristic: Hamming distance is always at least the
ink-count difference, so after the coarse scan the matcher re-scans every
candidate with `|ink_input - ink_template| <= best_dist`. That second pass
can never exclude a potential winner, so the result is identical to a full
scan while confident glyphs only pay for a few XOR+popcount candidates
(measured ~1.4x faster than a full 3000-char scan on a CJK font).

## Segmentation candidate lattice and visual DP

The old pipeline merged connected components irreversibly before
classification, so fragment-heavy glyphs such as `鲃` or `小` could never be
recovered without a `stroke_width` special case. The lattice
(`src/fixedfontocr/segmentation.py`) keeps every original component and
generates all consecutive candidate ranges: single components, merges of up
to 4 components, and `split(Cx)` atoms for wide components. A wide component
is split only when its width clearly exceeds the line's expected glyph bbox
*and* a vertical valley exists, so closed CJK glyphs (口/日/中) stay whole
while a low-resolution two-glyph blob becomes left/right alternatives plus
the original whole candidate. Candidate shapes are pruned before scoring by
maximum width/height, minimum height, component gap, vertical overlap/
proximity and ink area; then every surviving candidate is scored once in a
batch and the best path is decoded with dynamic programming.

Goal 1 names these objects explicitly: each original blob is a
`Component`, each merge/split hypothesis is a `VisualCandidate`, the full
non-destructive hypothesis set is a `VisualLattice`, and the DP output is a
`DecodePath`. Split candidates carry an atom span in addition to their
component range, so the decoder can cover a split component piece by piece
while the original component remains available as a whole-candidate merge.
The scorer stores a `VisualScores` Top-K object per candidate (template/CNN
second-ranked id included), so the decoder never has to commit to a Top-1
pick before it sees geometry, lexicon or path evidence.

The DP maximizes the *mean* visual score of the candidates on the path
(ties prefer fewer segments). Using the mean — not the sum — is essential:
summing would reward splitting one glyph into many fragments because each
fragment contributes positive score. Small geometry penalties (vertical
alignment, internal gaps, merged width vs. the line's typical glyph width)
break ties and keep clearly adjacent characters apart.

For CNN-only models (no template half) a conservative merge gate applies in
addition: a merged candidate must score >= `cnn_merge_threshold` (0.7), be
at least as good as every one of its parts, and its components must be in
glyph-internal proximity. This keeps CNN-only fixtures such as `1000` from
merging digits while the hybrid path (the recommended production model)
relies on the template margin gate instead.

Acceptance (verified by `tests/test_segmentation.py` and
`tests/test_goal4_segmentation.py`):

```text
鲃     -> 鲃
鲃鱼。 -> 鲃鱼。
小     -> 小
潜甲   -> 潜甲
潜乙   -> 潜乙
巴尔的摩 -> 巴尔的摩
Z17    -> Z17
```

No `stroke_width = 2` special case is needed.

## Unified scoring and Top-K

Template confidence is in `[0,1]` while the CNN reports an unbounded logit
margin; the two units are never averaged directly. `SegmentScorer` converts
every candidate to a `CandidateScore(visual_score, raw_score, score_type)`:

- template: `visual = confidence` (0..1) when `confidence >= template_threshold`
  AND (the match is exact — `confidence == 1.0` — OR the normalized
  top-1/top-2 distance margin is `>= template_margin_threshold`). Exact
  matches are always trusted because small punctuation glyphs can be
  pixel-perfect yet have a tiny normalized margin;
- CNN: `visual = sigmoid(top1 - top2)` — the binary softmax probability that
  the top-1 class beats the top-2 class.

The public `CharResult.confidence` is always mapped to 0..1 as well
(template confidence stays as-is; CNN margins go through the same sigmoid).
The CNN classifier API returns `ClassificationBatch(ids, top1, top2,
margins)` from `classify_batch(glyphs, top_k=2)`, so segmentation and a
future dictionary decoder get the raw top-2 information instead of a single
`char + confidence`. `TemplateBatch` now also carries `second_ids`, and both
classifier paths can be projected into `VisualScores(char_ids, logits,
top_k, margin, raw_score, score_type)`. Top-2 uses `argmax` + a single
`argpartition`, i.e. O(C) per glyph rather than a full `argsort`.

## Goal 1 data structures

The public contract of Goal 1 is:

```python
Component(        # one connected component: mask + bbox
VisualCandidate(  # one glyph hypothesis over components[start:end]
    start, end, bbox, glyph, geometry_score, scores
)
VisualLattice(    # all components + candidates, no irreversible decision
VisualScores(     # Top-K char_ids/logits, margin, raw_score, score_type
DecodePath(       # chosen candidates, mean score, text, alternatives
LexiconMatch(     # dictionary entity + span over visible text
OCRResult(        # visible text + confidence + matched_term/span +
                  # alternatives + lexicon_match + path
```

`OCRResult.text` is exactly the visible screen text; `matched_term` and
`lexicon_match` are dictionary-inferred entities and are kept separate by
design. `result.chars` remains as a compatibility projection.

## TinyCNN stride-2 optimization

The old forward computed the full feature map for the stride-2 convolutions
and sliced every other pixel. `cnn.py` now builds strided im2col views and
pointwise layers slice before the 1×1 matmul, so the skipped positions are
never computed. Weights are converted to contiguous float32 once at
classifier/backend construction. `reference_cnn.py` keeps the old
implementation; `tests/test_cnn.py` verifies `max error < 1e-5` and
identical argmax, and `tools/benchmark/cpu_benchmark.py` re-checks it on
every run (measured bit-identical: max error 0.0).

## Hybrid models

The model format adds `"classifier": "hybrid"`: `templates.bin` (bitset
templates, same payload as a template model's `weights.bin`) plus
`weights.bin` (f32 CNN tensors), both over one shared `charset.txt`.
`config.json` carries `template_threshold`, `template_margin_threshold` and
`cnn_threshold`. The template level runs on every candidate; a match is only
trusted when both its confidence and its top-1/top-2 margin are high (P6).
A template with `best=0.96 / second=0.95` is treated as ambiguous and falls
back to the CNN, while `best=0.92 / second=0.71` is trusted. Below
`cnn_threshold` a glyph is reported as `"?"`.

## allowed_chars

`recognize(image, allowed_chars=...)` maps the string to charset indices
before classification:

- template: candidates are restricted before the XOR/popcount scan;
- tinycnn:  the backend runs as usual, then any top-1 outside the set is
  re-scored over the allowed subset with the numpy forward;
- hybrid:   both levels respect the restriction.

Confidence for a restricted pick is the top1-top2 margin inside the allowed
subset (1.0 when only one class is allowed).

## Model format

```
model/
├── config.json    # input_width, input_height, classes, version, dtype,
│                  # classifier (template|tinycnn|hybrid) + thresholds
├── charset.txt    # one character per line / one string
└── weights.bin    # template: uint32 count + packed bits;
                   # tinycnn/hybrid: f32 tensors in fixed order
                   # hybrid also has templates.bin
```

Template entries are `input_width * input_height` bits packed little-endian
(72 bytes per 24×24 glyph). TinyCNN models store the f32 tensors in fixed
order `(conv1, dw1, pw1, dw2, pw2, fc)`. The same `weights.bin` is what a
WGPU backend would read, keeping both backends bit-identical.

## Training and export

`tools/` contains the dev-only pipeline:

1. `dataset/generate_font_dataset.py` renders a registered font with random
   size ±2 px, x/y offset, outline, shadow, blur, scale, alpha, background
   and threshold into 24×24 glyphs, recording the font SHA256 in the npz;
2. `dataset/collect_real_samples.py` converts label-named screenshot directories
   into glyph samples (single-character images or full lines segmented by
   the production pipeline); `--font` records the matching font SHA256;
3. `train/train.py` trains the Conv+BatchNorm+ReLU TinyCNN on the merged data
   and rejects datasets from different fonts;
4. `train/export_model.py` folds every BatchNorm into its convolution's
   weight/bias, writes `runtime_model/` and emits `test_vectors.npz`
   (input, per-layer activations, logits, char_id) used by
   `tests/test_wgpu_vectors.py`;
5. `benchmark/benchmark.py` prints the CPU/GPU table and crossover measured on the
   current machine;
6. `benchmark/footprint.py` reports on-disk sizes and runtime memory estimates.

## CPU benchmark suite (P7)

`tools/benchmark/cpu_benchmark.py` is the frozen-CPU measurement suite. It
records median and p95 latencies (never a single run) for every OCR stage
(mask, line detection, connected components, candidate generation,
normalize, classifier, decoder, end-to-end), for TinyCNN batch sizes
1/8/16/32/64/128, and for template/CNN classification over digit (~10),
small (~100), CJK (~3000) and CJK (~7000) charsets sampled from the
registered font's Unicode coverage. The last section re-runs the optimized
vs. reference forward check. `--fast` is the CI smoke mode.
`benchmarks/cpu_benchmark.json` snapshots one machine's numbers.

## Game regression set (P8)

`tests/game_samples/` holds the regression corpus: real game level badges
(`level/`, 1x and 4x) and deterministic synthetic renders (`synthetic/`)
covering normal Chinese, digits, mixed Chinese+ASCII, punctuation,
fragment-heavy glyphs (`鲃`, `小`), confusable pairs (`甲/申`, `未/末`),
light/dark backgrounds, anti-aliasing and multiple font sizes.
`manifest.json` records each image's expected text, category, profile
overrides and the registered font. `tests/test_game_samples.py` fails on any
segmentation/CNN change that breaks the set.

The level badges are also training data:
`tools/dataset/extract_game_samples.py` splits them into per-character crops
(by blank-column valleys, independent of the OCR classifier) and
`collect_real_samples.py` merges them into the fine-tuning dataset.
`data/cn/game_cn_finetuned.pth` is the fine-tuned checkpoint and
`model/game_cn` the exported runtime model.

`slot1` level badges are a documented known limitation: `L` and `V` touch
at one pixel, forming a single connected component at both scales, so they
are excluded from the passing set.

## CPU freeze checklist

The CPU route is the golden reference for future WGPU TinyCNN work,
dictionary decoding and beam search. Freeze criteria and where each is
verified:

| Criterion | Evidence |
| --- | --- |
| 鲃/小 no longer need morphology special-casing | `tests/test_segmentation.py::test_acceptance_does_not_need_stroke_width_special_case` |
| Adjacent Chinese characters are not merged | `tests/test_segmentation.py::test_adjacent_chinese_characters_not_merged` |
| Template + TinyCNN stable | margin-aware hybrid gate (`tests/test_scorer.py`), full suite green |
| 3K/7K charset usable | `tools/benchmark/cpu_benchmark.py` charsets 10/100/3000/7000 |
| Top-2 without full sort | `postprocess.top2` (argmax + argpartition), `tests/test_segmentation.py::test_top2_matches_argsort_reference` |
| Stride-2 forward has no useless work | optimized conv/pw + `tests/test_cnn.py` parity tests, benchmark max_error 0.0 |
| CPU benchmark fixed | `tools/benchmark/cpu_benchmark.py` + `benchmarks/cpu_benchmark.json` |
| Real game regression passes | `tests/test_game_samples.py` (23 samples) |
| Classifier outputs Top-K/raw score | `ClassificationBatch(ids, top1, top2, margins)` + `CandidateScore` |
| Segmentation supports candidate lattice | `src/fixedfontocr/segmentation.py` |

## Storage and memory footprint

Template storage is a packed bitset: `N × ((input_size² + 7) // 8)` bytes
(72 B per 24×24 glyph), so a 3000-char font is ~211 KiB on disk and the same
array is shared by the classifier at runtime (no copy). Coarse candidate
features use 8 B/char (uint16 ink + six uint8 bbox/margin values). With
numpy ≥ 2.0 the popcount fast path is `np.bitwise_count` (0 B table); older
numpy falls back to a 64 KiB 16-bit lookup table.

TinyCNN weights are `4032 + 33·C` f32 bytes (C = classes): ~391 KiB at
3000 classes; hybrid models store template + CNN. The WGPU backend keeps
persistent batch buffers of 24,920 B/glyph (input, NHWC-normalized tensor,
conv stages, GAP, result record and readback staging), e.g. ~1.5 MiB for a
64-glyph line; the buffers grow to the largest seen batch and are retained.

## Profiles

Per-UI-type tuning lives in `src/fixedfontocr/types.py`: text color /
grayscale threshold, expected character height/width, stroke width,
spacing, and normalized size. Pass a custom profile to
`FixedFontOCR(..., profile=...)`.

## Background robustness

The default profile assumes bright text on a dark background: it thresholds
luminance at `grayscale_threshold=140`. Measured with
`fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf`:

- Pure blue and dark backgrounds do not interfere: blue contributes only
  ~11% of the luminance, so black, navy, medium/bright blue, dark gray and
  blue-to-black gradients all recognize correctly with white text.
- Mixed backgrounds do interfere: any background region brighter than the
  threshold (white/light cells, bright bands, light gradients) is treated as
  ink. A white/blue checkerboard raised the foreground ratio from ~24% to
  62% and recognition collapsed. Light gradients below the threshold are
  fine; gradients crossing it are not.
- Dark text on a bright background needs a color-based profile
  (`use_grayscale=False`, `target_color=<text color>`, `tolerance=...`);
  merely flipping `bright_text` is not enough when the background is also
  dark in luminance (e.g. dark text on blue).

For UI screens with patterned or variable backgrounds, use the per-screen
profile's `target_color`/`tolerance` to select the exact text color, restrict
the OCR region of interest, or preprocess the frame to remove the background
before it reaches the pipeline.
