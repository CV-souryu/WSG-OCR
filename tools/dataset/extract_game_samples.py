#!/usr/bin/env python3
"""Extract per-character training crops from the game-sample regression set.

The level badges in ``tests/game_samples/real-game/level/`` are fixed
fixtures with known expected text. For each image this script splits the
line into one box per expected character (using the deepest blank-column
valleys, so it does not depend on the OCR classifier) and writes tight RGB
crops into ``out_dir/{label}/...png`` — the layout
:mod:`collect_real_samples` consumes.

Usage:
    python tools/dataset/extract_game_samples.py \
        tests/game_samples/manifest.json data/cn/real_game
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from fixedfontocr.preprocess import find_lines  # noqa: E402
from fixedfontocr.segmentation import connected_components  # noqa: E402
from fixedfontocr.types import profile_from_dict  # noqa: E402


def split_boxes(
    mask: np.ndarray,
    k: int,
    comps: list | None = None,
) -> list[tuple[int, int]]:
    """Split a line mask into ``k`` glyph boxes.

    Prefers the deepest blank-column valleys. When glyphs sit in adjacent
    columns with no blank run between them (e.g. slot1's ``L``/``V``), the
    connected-component boundaries are used instead.
    """

    proj = mask.any(axis=0)
    w = proj.size
    blanks: list[tuple[int, int]] = []
    x = 0
    while x < w:
        if not proj[x]:
            start = x
            while x < w and not proj[x]:
                x += 1
            blanks.append((start, x))
        else:
            x += 1
    if len(blanks) < k - 1:
        if comps is not None and len(comps) == k:
            cuts = [c.x for c in comps[1:]]
            boxes: list[tuple[int, int]] = []
            prev = 0
            for cut in cuts:
                boxes.append((prev, cut))
                prev = cut
            boxes.append((prev, w))
            return boxes
        raise ValueError(
            f"found {len(blanks)} blank runs but need {k - 1} cuts "
            f"for {k} glyphs"
        )
    cuts_blank = sorted(
        blanks, key=lambda r: (-(r[1] - r[0]), r[0])
    )[: k - 1]
    cuts = [sum(r) // 2 for r in cuts_blank]
    cuts.sort()
    boxes: list[tuple[int, int]] = []
    prev = 0
    for cut in cuts:
        boxes.append((prev, cut))
        prev = cut
    boxes.append((prev, w))
    return boxes


def extract(
    manifest_path: Path,
    out_dir: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = profile_from_dict(manifest.get("default_profile", {}))
    root = manifest_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for sample in manifest["samples"]:
        if sample.get("category") != "real-game/level":
            # Synthetic renders are full lines with fragment-heavy glyphs and
            # the Goal 18 ship-name crops mix Chinese/Latin/punctuation with
            # decorative glyphs; blank-valley splitting would misalign
            # per-character training crops. Only the level badges are
            # extracted as per-character training crops.
            continue
        path = root / sample["file"]
        expected = sample["expected"]
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        mask = profile.color_mask(arr)
        lines = find_lines(mask, profile)
        if len(lines) != 1:
            raise SystemExit(f"{path}: expected one line, found {len(lines)}")
        line = lines[0]
        y0, y1 = line.y, line.y + line.h
        comps = connected_components(line)
        try:
            boxes = split_boxes(line.mask, len(expected), comps=comps)
        except ValueError as exc:
            print(f"skip {path}: {exc}")
            continue
        for ch, (x0, x1) in zip(expected, boxes):
            part = line.mask[:, x0:x1]
            ys, xs = np.where(part)
            if ys.size == 0:
                raise SystemExit(f"{path}: no ink in box {ch!r} at {x0}:{x1}")
            img_x0 = line.x + x0
            crop = arr[
                y0 + ys.min() : y0 + ys.max() + 1,
                img_x0 + xs.min() : img_x0 + xs.max() + 1,
            ]
            # "." cannot be a directory name; the collector maps "dot" back
            # to "." via labels.json.
            dir_name = "dot" if ch == "." else ch
            label_dir = out_dir / dir_name
            label_dir.mkdir(parents=True, exist_ok=True)
            stem = path.stem
            idx = sum(1 for _ in label_dir.glob(f"{stem}_*.png"))
            img = Image.fromarray(crop)
            img.save(label_dir / f"{stem}_{idx}.png")
            written += 1
    (out_dir / "labels.json").write_text(
        json.dumps({"dot": "."}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {written} crops to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path, help="label-named crop directory")
    args = parser.parse_args()
    extract(args.manifest, args.output)


if __name__ == "__main__":
    main()
