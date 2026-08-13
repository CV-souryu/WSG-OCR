# FixedFontOCR

Deterministic CPU baseline for fixed-font OCR, designed so a future WGPU
backend can consume the exact same model files and produce the same results.

## Repository layout

```
.
├── src/fixedfontocr/    # the package
├── scripts/             # model generation & TinyCNN training utilities
├── tests/               # pytest suite; fixture CNN models under tests/fixtures/
├── docs/                # design & architecture notes
├── fonts/               # local font files (gitignored — drop your own fonts here)
├── pyproject.toml
└── README.md
```

## Quick start

```bash
# 1. Generate a template model from your game's font file
python scripts/generate_model.py game_font.ttf model/ --charset charset.txt
# or, after `pip install -e .`:
fixedfontocr-generate game_font.ttf model/ --charset charset.txt

# 2. Recognize
```

```python
import numpy as np
from fixedfontocr import FixedFontOCR

ocr = FixedFontOCR(model_path="model/", backend="auto")
image: np.ndarray = ...  # shape (H, W, 3), dtype uint8

result = ocr.recognize(image)
print(result.text)        # "获得金币1000"
print(result.confidence)  # 0.997
for c in result.chars:
    print(c.char, c.x, c.y, c.w, c.h, c.confidence)
```

Two classifiers are supported in the same model directory format:

- **Template baseline** (default): `fixedfontocr-generate` renders every
  character to a 72-byte bitset; recognition is XOR + popcount.
- **TinyCNN**: the small fixed network from the design (Conv3x3 stride-2,
  DWConv3x3, Pointwise1x1 stride-2, GAP, Linear). Training uses PyTorch but
  runtime inference is pure numpy:

```bash
python scripts/train_tinycnn.py game_font.ttf model/ --charset charset.txt \
    --render-size-min 26 --render-size-max 32
```

The resulting `config.json` contains `"classifier": "tinycnn"` and the
`weights.bin` stores the f32 tensors in fixed order
(conv1, dw1, pw1, dw2, pw2, fc). `FixedFontOCR` picks the classifier from
the config automatically.

## Pipeline

```
numpy RGB
  -> profile color/grayscale mask
  -> line finding
  -> connected components
  -> classifier-verified fragment merging / width splitting
  -> 24x24 normalization
  -> template classifier (XOR + popcount)
  -> charset -> string
```

## Model format

```
model/
├── config.json    # input_width, input_height, classes, version, dtype
├── charset.txt    # one character per line / one string
└── weights.bin    # template: uint32 count + packed bits;
                   # tinycnn: f32 tensors in fixed order
```

Template entries are `input_width * input_height` bits packed little-endian
(72 bytes per 24x24 glyph). The same `weights.bin` is what a WGPU backend
would read, keeping both backends bit-identical.

## Profiles

Per-UI-type tuning lives in [`src/fixedfontocr/types.py`](src/fixedfontocr/types.py):
text color / grayscale threshold, expected character height/width, stroke
width, spacing, and normalized size. Pass a custom profile to
`FixedFontOCR(..., profile=...)`.

## Current status

- Template baseline and TinyCNN CPU are implemented and tested on Latin
  digits/letters and CJK. Fixture CNN models live in `tests/fixtures/`.
- The WGPU backend is the next milestone; the model format and API are laid
  out so it can slot in without changing the public interface.
