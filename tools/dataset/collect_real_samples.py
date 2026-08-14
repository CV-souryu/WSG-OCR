#!/usr/bin/env python3
"""Collect real-screenshot glyph samples into a training .npz.

Directory layout (each directory name is the label):

    real/
    ├── 0/          # single-character images of "0"
    │   ├── shot1.png
    │   └── shot2.png
    ├── 金币/       # full-line images whose text is exactly "金币"
    │   └── crop1.png
    └── Lv.12/
        └── crop1.png

For multi-character labels the line must contain exactly one glyph per label
character (the standard connected-component segmentation is used), otherwise
the image is skipped with a warning. Output matches
``tools/dataset/generate_font_dataset.py`` (x/y/chars/font_sha256) plus
``origins`` when ``--font`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fixedfontocr.defaults import (  # noqa: E402
    compute_font_sha256,
    resolve_font,
)
from fixedfontocr.preprocess import collect_glyphs, normalize  # noqa: E402
from fixedfontocr.types import default_profile  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_directory(
    root: str | Path,
    output: str | Path,
    profile=None,
    font_path: str | Path | None = None,
) -> None:
    from PIL import Image

    profile = profile or default_profile()
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    font_sha256 = None
    if font_path is not None:
        font_path = resolve_font(font_path)
        font_sha256 = compute_font_sha256(font_path)

    charset: list[str] = []
    label_id: dict[str, int] = {}
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    origins: list[str] = []
    skipped = 0

    labels_map: dict[str, str] = {}
    labels_path = root / "labels.json"
    if labels_path.is_file():
        labels_map = json.loads(labels_path.read_text(encoding="utf-8"))
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        label = labels_map.get(label_dir.name, label_dir.name)
        label_chars = list(label)
        if not label_chars:
            continue
        for ch in label_chars:
            if ch not in label_id:
                label_id[ch] = len(charset)
                charset.append(ch)
        files = sorted(
            p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS
        )
        for img_path in files:
            arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            if len(label_chars) == 1:
                mask = profile.color_mask(arr)
                ys, xs = np.where(mask)
                if ys.size == 0:
                    print(f"skip {img_path}: no ink found")
                    skipped += 1
                    continue
                tight = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                x_list.append(normalize(tight, profile.target_size))
                y_list.append(label_id[label_chars[0]])
                origins.append(str(img_path))
                continue

            segments, glyphs = collect_glyphs(arr, profile)
            if len(segments) != len(label_chars):
                print(
                    f"skip {img_path}: {len(segments)} glyphs != "
                    f"{len(label_chars)} label chars"
                )
                skipped += 1
                continue
            for ch, seg in zip(label_chars, segments):
                x_list.append(normalize(seg.mask, profile.target_size))
                y_list.append(label_id[ch])
                origins.append(f"{img_path}#{ch}")

    if not x_list:
        raise SystemExit(f"no samples collected from {root}")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": np.stack(x_list),
        "y": np.asarray(y_list, dtype=np.int64),
        "chars": np.asarray(charset, dtype="<U8"),
        "origins": np.asarray(origins, dtype="<U512"),
    }
    if font_sha256 is not None:
        payload["font_sha256"] = np.asarray([font_sha256], dtype="<U64")
    np.savez_compressed(out, **payload)
    print(
        f"wrote {len(x_list)} real samples ({len(charset)} classes) to {out} "
        f"({skipped} images skipped)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="directory of label-named subdirectories")
    parser.add_argument("output", help="output .npz dataset")
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="registered font under fonts/ this UI text comes from",
    )
    args = parser.parse_args()
    collect_directory(args.root, args.output, font_path=args.font)


if __name__ == "__main__":
    main()
