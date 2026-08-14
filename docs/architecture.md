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
| `src/fixedfontocr/types.py` | Shared dataclasses: `CharResult`, `OCRResult`, `Profile` (per-UI-type segmentation tuning). |
| `src/fixedfontocr/preprocess.py` | CPU segmentation pipeline: color/grayscale mask → line finding → connected components → merging / width splitting → 24×24 normalization. |
| `src/fixedfontocr/classifier.py` | `Classifier` interface plus `TemplateClassifier`: coarse-feature candidate filtering (ink count, bbox, margins) followed by XOR + popcount. |
| `src/fixedfontocr/cnn.py` | `TinyCNNClassifier` numpy forward pass + `forward_with_activations` for exported test vectors (pure numpy; only primitives a WGPU backend can implement). |
| `src/fixedfontocr/postprocess.py` | `allowed_chars` → charset-index restriction and confidence scoring (masked top1/top2 margin). |
| `src/fixedfontocr/fontgen.py` | Renders font glyphs into normalized bitset templates; writes model directories. |
| `src/fixedfontocr/model.py` | Model format I/O: `config.json` + `charset.txt` + `weights.bin` (+ `templates.bin` for hybrid). |
| `src/fixedfontocr/cli.py` | `fixedfontocr-generate` command-line entry point. |
| `tools/register_font.py` | Adds a font under `fonts/` to `fonts/registry.json` with its SHA256. |

## Pipeline

```
numpy RGB
  -> profile color/grayscale mask
  -> line finding
  -> connected components
  -> fragment merging / width splitting
  -> 24x24 normalization (CPU in both backends)
  -> classifier:
       template: coarse feature filter -> per-glyph XOR + popcount (CPU)
       tinycnn:  CPUBackend / WGPUBackend / AutoBackend.classify(glyphs)
       hybrid:   template first; low-confidence glyphs go to the CNN;
                 still-low confidence becomes "?"
  -> allowed_chars restriction (postprocess)
  -> charset -> string
```

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

## Hybrid models

The model format adds `"classifier": "hybrid"`: `templates.bin` (bitset
templates, same payload as a template model's `weights.bin`) plus
`weights.bin` (f32 CNN tensors), both over one shared `charset.txt`.
`config.json` carries `template_threshold` and `cnn_threshold`.

The engine runs the template level on every glyph. Glyphs with confidence
below `template_threshold` are batched into a second `classify()` call on
the selected backend; below `cnn_threshold` they are reported as `"?"`.

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
