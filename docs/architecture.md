# Architecture

FixedFontOCR is a deterministic CPU baseline for fixed-font OCR. A future WGPU
backend is designed to consume the exact same model files and produce the same
results, so the model format and the public API are the two stable contracts.

## Repository layout

```
.
├── src/fixedfontocr/    # the package
├── scripts/             # model generation & TinyCNN training utilities
├── tests/               # pytest suite; checked-in fixture models in tests/fixtures/
├── docs/                # this documentation
├── fonts/               # local font files (gitignored; drop your own fonts here)
├── pyproject.toml
└── README.md
```

## Module map

| Module | Responsibility |
| --- | --- |
| `src/fixedfontocr/api.py` | Public `FixedFontOCR` engine. Loads the model and picks the classifier (`template` vs `tinycnn`) from `config.json`. |
| `src/fixedfontocr/types.py` | Shared dataclasses: `CharResult`, `OCRResult`, `Profile` (per-UI-type segmentation tuning). |
| `src/fixedfontocr/preprocess.py` | CPU segmentation pipeline: color/grayscale mask → line finding → connected components → classifier-verified merging / width splitting → 24×24 normalization. |
| `src/fixedfontocr/classifier.py` | `Classifier` interface plus `TemplateClassifier` (XOR + popcount against packed bitset templates). |
| `src/fixedfontocr/cnn.py` | `TinyCNNClassifier` numpy forward pass (pure numpy; only primitives a WGPU backend can implement). |
| `src/fixedfontocr/fontgen.py` | Renders font glyphs into normalized bitset templates; writes model directories. |
| `src/fixedfontocr/model.py` | Model format I/O: `config.json` + `charset.txt` + `weights.bin` (both template and tinycnn variants). |
| `src/fixedfontocr/cli.py` | `fixedfontocr-generate` command-line entry point. |

## Pipeline

```
numpy RGB
  -> profile color/grayscale mask
  -> line finding
  -> connected components
  -> classifier-verified fragment merging / width splitting
  -> 24x24 normalization
  -> classifier (template XOR + popcount, or TinyCNN)
  -> charset -> string
```

## Model format

```
model/
├── config.json    # input_width, input_height, classes, version, dtype, classifier
├── charset.txt    # one character per line / one string
└── weights.bin    # template: uint32 count + packed bits;
                   # tinycnn: f32 tensors in fixed order
```

Template entries are `input_width * input_height` bits packed little-endian
(72 bytes per 24×24 glyph). TinyCNN models store the f32 tensors in fixed
order `(conv1, dw1, pw1, dw2, pw2, fc)`. The same `weights.bin` is what a WGPU
backend would read, keeping both backends bit-identical.

## Profiles

Per-UI-type tuning lives in `src/fixedfontocr/types.py`: text color /
grayscale threshold, expected character height/width, stroke width, spacing,
and normalized size. Pass a custom profile to `FixedFontOCR(..., profile=...)`.
