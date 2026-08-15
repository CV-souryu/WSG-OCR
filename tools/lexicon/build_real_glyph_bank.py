#!/usr/bin/env python3
"""Build a real-glyph prototype bank from a labeled crops corpus.

For every labeled row whose decoded path has exactly one candidate per
expected character, the soft foreground of each glyph (Goal 2, keeps the
game's real anti-aliasing) is normalized to 24x24 and stored as a
prototype of that character. Same characters from many crops become many
prototypes, so NCC matching at runtime compares game renders with game
renders -- no font-render domain gap.

Bank format (``<corpus>.realglyphs.npz``):

    labels         # numpy array of single characters (dtype '<U1')
    data           # float32 [N, 24, 24] soft glyph patches
    font_sha256    # 0-d unicode array with the registered font's SHA256
    source         # 0-d unicode array with the source CSV path

Only the registered font's OCR pipeline produced these crops, so the bank
records the same font identity (Font Policy) even though it stores pixels
instead of font renderings.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fixedfontocr import FixedFontOCR  # noqa: E402
from fixedfontocr.defaults import compute_font_sha256, resolve_font  # noqa: E402
from fixedfontocr.frontend import extract_frontend  # noqa: E402

FONT = "SourceHanSansSC/SourceHanSansSC-Bold.otf"


def glyph_patch(soft: np.ndarray, seg) -> np.ndarray:
    """24x24 normalized soft glyph patch (same shape for every sample)."""

    g = soft[seg.y:seg.y + seg.h, seg.x:seg.x + seg.w].astype(np.float32)
    g /= 255.0
    return np.asarray(
        Image.fromarray((g * 255).astype(np.uint8)).resize(
            (24, 24), Image.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="crops_items_recognition.csv")
    parser.add_argument("image_dir", type=Path, help="directory with the crops")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "model" / "game_cn"
    )
    parser.add_argument(
        "--font", type=Path, default=ROOT / "fonts" / FONT
    )
    args = parser.parse_args()

    ocr = FixedFontOCR(model_path=args.model, backend="cpu")
    font_path = resolve_font(args.font)

    labels: list[str] = []
    patches: list[np.ndarray] = []
    skipped = 0
    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in list(csv.reader(f))[1:] if r[3] != ""]
    for name, exp in ((r[1], r[3]) for r in rows):
        img = np.asarray(
            Image.open(args.image_dir / name).convert("RGB"), dtype=np.uint8
        )
        frontend = extract_frontend(img, ocr.profile)
        result = ocr.recognize(img)
        path = result.path
        if path is None or len(path.candidates) != len(exp):
            skipped += 1
            continue
        for cand, ch in zip(path.candidates, exp):
            labels.append(ch)
            patches.append(glyph_patch(frontend.soft_foreground, cand.segment))

    if not patches:
        raise SystemExit("no aligned glyphs collected")
    data = np.stack(patches).astype(np.float32)
    out = args.csv.with_suffix(args.csv.suffix + ".realglyphs.npz")
    np.savez(
        out,
        labels=np.asarray(labels, dtype="<U1"),
        data=data,
        font_sha256=np.asarray(compute_font_sha256(font_path), dtype="<U64"),
        source=np.asarray(str(args.csv), dtype="<U256"),
    )
    chars = len(set(labels))
    print(
        f"wrote {out} ({data.shape[0]} glyphs, {chars} unique chars, "
        f"skipped {skipped} misaligned rows)"
    )


if __name__ == "__main__":
    main()
