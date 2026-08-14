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
| `src/fixedfontocr/geometry.py` | Goal 8 font geometry database: offline generation from a registered font (`advance`, bbox, aspect, ink, component count, baseline) plus the runtime JSON lookup table used by pruning and geometry scoring. |
| `src/fixedfontocr/lexicon.py` | Goal 11 Lexicon Layer + Goal 12 partial-word support: loads `charsets/words/` by domain, normalizes whitespace, ranks exact/full/partial-word matches (prefix/suffix/inner crops, internal-gap penalty) and applies `none`/`prefer`/`strict` modes to the decoded result. |
| `src/fixedfontocr/segmentation.py` | Candidate lattice (every original component + merges up to 4 components + split(Cx) atoms) and the visual DP decoder. This is the production segmentation path. |
| `src/fixedfontocr/decoder.py` | Goal 13 joint decoder: exact DP (`decode_dp`, lexicon-prefix trie state) + beam-search upgrade (`decode_beam`, `beam_width` 8..32) scoring `visual + geometry + lexicon + word_prior - segmentation_penalty` and returning the best path + alternatives. |
| `src/fixedfontocr/scorer.py` | `SegmentScorer`: batch template/CNN scoring with the margin-aware hybrid gate; converts raw scores to a shared 0..1 visual score. |
| `src/fixedfontocr/classifier.py` | `Classifier` interface plus `TemplateClassifier` (V1 single-template) and `TemplateV2Classifier` (Goal 9 multi-prototype): coarse-feature candidate filtering (ink count, bbox, margins) followed by XOR + popcount; Top-K/best/second/margin + winning-prototype metadata for the lattice. |
| `src/fixedfontocr/cnn.py` | `TinyCNNClassifier` numpy forward pass (stride-2 optimized), `forward_with_activations` for exported test vectors and `classify_batch(glyphs, top_k=2)`. |
| `src/fixedfontocr/reference_cnn.py` | Full-then-slice reference forward used by tests to prove the optimized forward matches (P2 acceptance). |
| `src/fixedfontocr/postprocess.py` | `allowed_chars` → charset-index restriction and O(C) top-1/top-2 scoring (no full argsort). |
| `src/fixedfontocr/fontgen.py` | Renders font glyphs into normalized bitset templates (V1 and Goal 9 V2 prototype grids); writes model directories. |
| `src/fixedfontocr/model.py` | Model format I/O: `config.json` + `charset.txt` + `weights.bin` (+ `templates.bin` for hybrid); V1/V2 template detection and `template_version`. |
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
     geometrically filtered with the Goal 8 font geometry database)
  -> 24x24 normalization (Goal 3 baseline frame, binary + soft)
  -> batched scoring (TemplateClassifier.match_batch / TinyCNN forward)
       template: confidence AND top-1/top-2 margin must both be high
       cnn:      soft glyphs when model input_mode="soft", binary otherwise
                 sigmoid(top1-top2 margin) -> shared 0..1 visual score
       hybrid:   template gate first, CNN fallback for ambiguous glyphs
  -> Goal 13 joint decoder (DP best path + beam alternatives; local
     visual/geometry/word-prior/segmentation terms, crop-aware lexicon
     match gated by visual uncertainty)
  -> allowed_chars restriction (postprocess)
  -> lexicon layer (none / prefer / strict, Goal 11)
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

## Font geometry database (Goal 8)

The offline database is the replacement for the Goal 4 "expected font bbox
stand-in". It is generated once from the registered font:

```text
char_id
advance              (em units, from the font hmtx)
bbox width/height    (em units, from the outline bounds)
aspect ratio         (outline bbox width / height)
ink count            (pixels in the deterministic binary raster)
component count      (4-connected components of that raster)
baseline             (font baseline position in em / ink bbox ratio)
ink ratio            (ink / raster bbox area, for runtime comparison)
```

`fontgen.write_model`, `model.write_cnn_model` and
`model.write_hybrid_model` all write `geometry.json` when a font is
available. The runtime loader (`model.load_model`) parses it without
fontTools/Pillow and attaches it to `OCRModel.geometry`; models generated
before Goal 8 simply have `geometry=None` and keep the heuristic path.

At runtime the database is used in three places:

1. **Candidate pruning** (`_geometry_bbox_ok`): the line's expected glyph
   width is derived from the font's narrow/full-width ratios instead of
   `0.8 * median height`, and candidate bboxes that no charset glyph can
   occupy (e.g. a full-width blob 1.8x too wide) are filtered before
   scoring.
2. **Geometry score** (`geometry_score`): after the classifier picks a
   Top-1 character, the candidate's bbox width/height/aspect, ink ratio,
   component count and baseline are compared to that character's database
   entry. Penalties stay small (bounded at 0.12) so the visual score remains
   dominant, but a fragmented `小` no longer has the same prior as a merged
   `小`, and a full-width blob is never treated like `1/I/l/i/!`.
3. **Split/merge decisions** (`_estimate_expected_width`): narrow lines
   (`Z17`) and full-width lines (`巴尔的摩`) use their own font-derived
   expected widths, so a connected low-res two-glyph blob is split on the
   real glyph scale rather than a single global heuristic.

The Goal 4/Goal 7 regression strings are unchanged with the database
enabled (`tests/test_goal8_geometry.py`).

## Template V2 (Goal 9)

Goal 9 upgrades the template layer from one bitmap per character to a
multi-prototype grid generated offline from the registered font:

```text
Z
├─ 11 px   (4 sub-pixel phases × bilinear/area-like downsample)
├─ 12 px   (4 sub-pixel phases × bilinear/area-like downsample)
├─ 13 px   (4 sub-pixel phases × bilinear/area-like downsample)
├─ 14 px   (4 sub-pixel phases × bilinear/area-like downsample)
├─ 15 px   (4 sub-pixel phases × bilinear/area-like downsample)
├─ 16 px   (4 sub-pixel phases × bilinear/area-like downsample)
└─ 32 px clean render (the legacy V1 prototype, kept so V2 is never worse
                       on clean screenshots)
```

Each prototype is rendered on a supersampled canvas (`render_size × 3`),
shifted by a fractional sub-pixel phase, downsampled with bilinear or
area-like (BOX) resampling, binarized, and normalized into the same Goal 3
24×24 baseline frame used at runtime. Missing ink at the configured
threshold is retried at lower thresholds before an explicit error, and the
data layer rejects all-zero prototypes (Font Policy: missing glyphs are
hard errors).

`TemplateV2Classifier` keeps the V1 cascade -- geometry prefilter (ink,
bbox, margins) then XOR + popcount -- but runs it per prototype and
aggregates by character:

* best character = argmin over characters of their best-prototype distance;
* second score = the second-ranked character's best-prototype distance;
* margin = normalized best/second gap;
* Top-K = the top `K` characters, each with the winning prototype's render
  size, sub-pixel phase and downsample mode.

The prefilter is exact for the returned Top-K: after scanning the
geometrically close prototypes, every prototype whose ink-count delta is no
larger than the current K-th character distance is scanned too (Hamming
distance >= |ink delta|), so no contender can be missed. The matcher never
decides the final character; it only feeds the scorer/decoder with ranked
visual evidence.

The hybrid gate treats an exact low-res match with a zero margin as
ambiguous (e.g. '.' and '*' rasterize to the same blob at 11 px) and routes
it to the CNN, while an exact clean-prototype match keeps the V1 exact-match
semantics. `templates.bin`/`weights.bin` for V2 models start with the magic
`TPL2` and carry per-prototype metadata; V1 files are detected by the
absence of the magic and keep loading (`tests/test_goal9_template_v2.py`).

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

## Low-resolution training domain (Goal 7)

The synthetic generator renders every glyph through the low-resolution
game pipeline instead of a single clean 32 px raster:

```
fonts/ font
  -> supersampled render (2-3x)
  -> random final font size in 10..18 px
  -> bilinear / area-like downsampling (GPU-sampling / UI-scale
     degradation)
  -> sub-pixel x/y offset, different scale ratios, slight blur (pre- and
     post-downsample), alpha, brightness, background blending, outline and
     shadow changes
  -> Goal 3 baseline-aligned 24x24 glyph (binary or soft)
```

`tools/dataset/generate_font_dataset.py` stores `render_sizes`,
`source_sizes`, `downsample_modes` and per-sample `augmentation` tags in
the npz so the training provenance is explicit. `tools/train/build_model.py`
passes the 10..18 px supersampled range to the generator by default, and the
legacy `scripts/train_tinycnn.py` delegates to the same generator. The
`Z17` and `巴尔的摩` glyphs are exercised at every size by
`tests/test_goal7_low_res.py`.

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

## Joint decoder (Goal 13)

The decoder is the new architecture's core. Its inputs are the scored
`VisualLattice` (all merge/split candidates with per-character Top-K
`VisualScores`), the Goal 8 `FontGeometryDatabase`, and an optional
`Lexicon`; its output is the best `DecodePath` plus `alternatives`. Every
path is scored with the Goal 13 formula:

```text
total = visual + geometry + lexicon + word_prior - segmentation_penalty
```

* `visual` is the classifier-only fused score of the chosen character
  (template + CNN before the geometry term), so geometry is never
  double-counted;
* `geometry` is the candidate's Goal 8 agreement with the chosen
  character (bbox/aspect/ink/component/baseline plus the pipeline's
  merge/split penalties);
* `lexicon` is the Goal 11/12 match of the complete visible string
  (exact, term-in-text, prefix/suffix/inner crop, gap crop) scaled by
  `1 - mean(visual)` -- the Goal 15 gate that stops a dictionary substring
  such as "Z3" inside "ABCXYZ999" from overriding a confident pick;
* `word_prior` is a per-character unigram support estimated from the
  lexicon's terms, gated by each character's visual uncertainty;
* `segmentation_penalty` is a small per-extra-hypothesis cost (default
  0.005) that only breaks near-ties toward fewer characters. It is
  deliberately tiny: a genuine multi-character line must never be
  penalized for its length, and the substantive merge/split costs live in
  the geometry score.

`decode_dp` is the first-version exact dynamic program. Its DP state is
`(atom_end, lexicon_trie_node)`, so the current lexicon prefix participates
in the path while candidate terms are being explored (exact term
completions and lexicon prefixes receive a small look-ahead bonus). The
returned path is re-scored with the complete formula, including crop-aware
lexicon matches.

`decode_beam` is the beam-search upgrade (`beam_width` 8..32, default 16).
The exact DP stays authoritative for the best path -- ranking complete
paths by their whole-path mean would let a glyph split into several
high-scoring look-alike fragments (e.g. the CNN-only `获得金币1000` fixture
splitting `得` into two `得` pieces) overturn the frozen CPU reference. The
beam explores complete-path hypotheses and returns the runner-up texts as
`DecodePath.alternatives`, ranked with the full Goal 13 formula.

The public pipeline passes the lexicon into `segment_line`, which runs the
joint decoder, and the decoder's chosen characters (`DecodePath.char_ids`)
are authoritative in `OCRResult`. The model's unknown gate (CNN threshold /
empty allowed set) still wins, `apply_lexicon` keeps handling strict mode
and `matched_term`/`matched_span` annotation, and `tests/test_goal13_decoder.py`
pins the formula, DP/beam agreement, segmentation penalty, lexicon
tie-breaking, geometry input and the Goal 4/7/11/12 end-to-end regressions.

## Unified scoring and Top-K

Template confidence is in `[0,1]` while the CNN reports unbounded logits;
the two units are never averaged directly. Since Goal 10 every hybrid
candidate keeps the raw evidence separately and the decoder consumes one
weighted value:

```text
visual_score = a * cnn_score + b * template_score + c * geometry_score
```

* `template_raw_score` is the template's 0..1 confidence for the chosen
  character;
* `cnn_logit` is the raw top-class logit and `cnn_margin` the chosen
  character's logit margin (relative to the runner-up), normalized through
  the same sigmoid into `cnn_score`;
* `geometry_score` is the Goal 8 geometry penalty computed by the
  segmentation layer;
* `visual_score` is clipped to `[0,1]` and used by the visual DP.

The weights (`visual_weights`) and a monotone piecewise-linear calibration
(`visual_calibration`) are stored in `config.json`; the bundled model is
tuned on the real-screenshot set and `tools/train/tune_visual.py` re-fits
both from any `collect_real_samples.py` npz (the tuner uses
`SegmentScorer.score_fused`, the pure weighted fusion without the
compatibility trust gate). The public
`CharResult.confidence` is the calibrated value, always in `[0,1]`.

The compatibility trust gate from P6 is still used for *labelling* the
dominant source (`score_type`): a template match above
`template_threshold` with a non-trivial top-1/top-2 margin (or an exact
match) keeps the template character and can only be boosted by the CNN;
otherwise the CNN evidence chooses the character. Either way
`VisualScores` stores both classifiers' raw quantities, so no decision is
collapsed into a single scale before the decoder.

`ClassificationBatch` carries Top-K ids/logits plus the full logit matrix
for fusion, and Top-2/Top-K use `argmax` + `argpartition` (O(C) per glyph)
rather than a full `argsort`.

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

## TinyCNN V1 freeze (Goal 5)

Goal 5 freezes the network. `fixedfontocr/cnn.py` owns the canonical spec
(`TINYCNN_V1_LAYERS`, `TINYCNN_V1_INPUT_SIZE`, `TINYCNN_V1_INPUT_CHANNELS`,
`TINYCNN_V1_HEAD`, `TINYCNN_V1_TAIL`):

| Layer | Kind | Kernel | Stride | Channels | Groups | Activation |
| --- | --- | --- | --- | --- | --- | --- |
| conv1 | Conv3x3 SAME | 3 | 2 | 1 → 8 (or 2 → 8) | 1 | ReLU |
| dw1 | DWConv3x3 SAME | 3 | 1 | 8 → 8 | 8 | ReLU |
| pw1 | Pointwise | 1 | 2 | 8 → 16 | 1 | ReLU |
| dw2 | DWConv3x3 SAME | 3 | 1 | 16 → 16 | 16 | ReLU |
| pw2 | Pointwise | 1 | 2 | 16 → 32 | 1 | ReLU |
| gap | GlobalAvgPool | — | — | 32 → 32 | — | — |
| fc | Linear | — | — | 32 → charset | — | — |

Input is 24×24. The allowed input variations are exactly: one channel
(binary or soft, recorded by `config.json["input_mode"]`) and the
experimental two-channel `soft + binary` stack (`conv1` widened to 2 input
channels). The on-disk model format is frozen to the single-channel
variant. Transformer, LSTM, Attention and normalization layers beyond the
ReLUs are forbidden; BatchNorm exists only in the training mirror and is
folded into the convolution weights at export, so runtime stays Conv + ReLU.

Enforcement points:

* `validate_v1_weights` rejects any weight set that is not the exact 12
  tensor V1 set: extra tensors (e.g. BN/attention weights), missing
  tensors, wrong shapes, and `conv1` input channels outside `(1, 2)` all
  raise;
* `forward`/`forward_with_activations` reject non-24×24 inputs and channel
  counts outside `{1, 2}` instead of silently truncating;
* runtime model configs carry `"architecture": "tinycnn_v1"`; `load_model`
  rejects unknown architectures, while legacy configs without the key
  default to V1;
* `write_cnn_weights` rejects non-V1 tensor sets and refuses to serialize
  the experimental two-channel variant;
* `tests/test_goal5_tinycnn.py` pins the spec, torch parity, training-mirror
  shapes, the WGPU layer map, and every model-format gate above.

## Goal 6 CPU TinyCNN optimization

Goal 6 is the NumPy-runtime optimization pass over the frozen V1 network.
Every item from the goal file is enforced by `tests/test_goal6_tinycnn.py`:

* **Stride-2 computes target positions directly.** `_im2col_stride2` builds
  strided patches (the im2col view advances by two input pixels per output
  pixel) for `conv3x3`/`dwconv3x3`, and pointwise layers slice before the
  1×1 matmul. The old full-feature-map-then-`::2` pattern is gone.
* **No meaningless copies in the hot path.** Layer outputs use
  `np.asarray(..., dtype=np.float32)` instead of `astype`, so an already
  float32 result is returned without a copy; `forward()` runs through a
  shared private helper with `keep_activations=False`, so inference never
  builds the activation dict that `forward_with_activations` returns.
* **Weights are prepared once.** `prepare_weights` validates the frozen
  tensor set at construction and converts any non-f32/non-contiguous array
  into a C-contiguous float32 copy; already-prepared arrays are reused
  identity-wise.
* **Batched inference.** The forward pass is `[N, C, 24, 24]` end-to-end
  (no per-glyph Python loop), and the benchmark measures batch sizes
  1/8/16/32/64/128.
* **Top-K via partition.** `postprocess.topk` uses `argpartition` to select
  the top-K columns, then orders only those K elements (ties by original
  column index). The runtime never calls a full `argsort` over the charset;
  `top2` is the specialized two-column fast path. `classify_batch(top_k=k)`
  exposes ranked `topk_ids`/`topk_logits` arrays for `k > 2`.

The NumPy vs PyTorch gate stays the same: `tests/test_cnn.py` and
`tests/test_goal5_tinycnn.py` assert `max error < 1e-5` and identical
argmax, and `tools/benchmark/cpu_benchmark.py` re-checks the optimized
forward against the full-then-slice reference on every run.

## Hybrid models

The model format adds `"classifier": "hybrid"`: `templates.bin` (bitset
templates, V1 or Goal 9 V2) plus `weights.bin` (f32 CNN tensors), both over
one shared `charset.txt`.
`config.json` carries `template_threshold`, `template_margin_threshold` and
`cnn_threshold`. The template level runs on every candidate; a match is only
trusted when both its confidence and its top-1/top-2 margin are high (P6).
A template with `best=0.96 / second=0.95` is treated as ambiguous and falls
back to the CNN, while `best=0.92 / second=0.71` is trusted. Below
`cnn_threshold` a glyph is reported as `"?"`. With V2 templates, an exact
match (`best=1.0`) that comes from a low-res prototype with zero margin is
also treated as ambiguous and falls back to the CNN.

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
│                  # classifier (template|tinycnn|hybrid), architecture
│                  # (tinycnn_v1) + thresholds
├── charset.txt    # one character per line / one string
└── weights.bin    # template: uint32 count + packed bits;
                   # tinycnn/hybrid: f32 tensors in fixed order
                   # hybrid also has templates.bin
```

Template V1 entries are `input_width * input_height` bits packed
little-endian (72 bytes per 24×24 glyph). Goal 9 V2 files replace the
`uint32 count` header with the `TPL2` magic plus
`count / prototypes_per_char / bytes_per_template`, a per-prototype
metadata block (render size, sub-pixel phase in 1/8 px, downsample mode)
and the packed bitset payload. TinyCNN models store the f32 tensors in
fixed order `(conv1, dw1, pw1, dw2, pw2, fc)`. The same `weights.bin` is
what a WGPU backend would read, keeping both backends bit-identical.

## Training and export

`tools/` contains the dev-only pipeline:

1. `dataset/generate_font_dataset.py` renders a registered font through the
   Goal 7 low-res domain (10..18 px supersampled render, bilinear/area-like
   downsample, sub-pixel offset, scale, blur, alpha, brightness, background
   blend, outline and shadow) into 24×24 glyphs, recording the font SHA256,
   per-sample render sizes and augmentation tags in the npz;
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
| Goal 6 CPU TinyCNN optimization | `tests/test_goal6_tinycnn.py` (strided im2col, no activation-dict forward, prepared weights, batch, Top-K via partition) |
| Goal 7 low-res training domain | `tests/test_goal7_low_res.py` (10..18 px coverage, supersampled bilinear/area-like downsample, sub-pixel/scale/blur/alpha/brightness/background/outline augmentation, Z17/巴尔的摩) |
| Goal 8 font geometry database | `tests/test_goal8_geometry.py` (per-char advance/bbox/aspect/ink/component/baseline, JSON round-trip + model embedding, narrow vs. full-width priors, geometry score and pruning) |
| Goal 9 Template V2 | `tests/test_goal9_template_v2.py` (11..16 px × sub-pixel × downsample grid, V1/V2 round-trip, prefilter Top-K exactness, low-res confusables 未/末 & Z/2, exact low-res tie routing to CNN, bundled models in V2 format) |
| Goal 13 joint decoder | `src/fixedfontocr/decoder.py` + `tests/test_goal13_decoder.py` (DP + beam search, `visual + geometry + lexicon + word_prior - segmentation_penalty`, alternatives, Goal 15 visual-uncertainty gate, end-to-end `鲃`/`小`/`潜甲`/`潜乙`/`巴尔的摩`/`Z17` with lexicon) |
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

The Goal 8 font geometry database (`geometry.json`) adds ~300 B/char
(~0.9 MiB for 3000 classes) and is loaded once as plain JSON; the runtime
keeps only the small `char_id -> entry` dict plus the median
narrow/full-width ratios used by pruning.

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
