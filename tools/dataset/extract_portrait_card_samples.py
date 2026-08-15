#!/usr/bin/env python3
"""Extract real game ship-name samples from portrait-card screenshots.

Goal 18 needs a corpus of *real* game screenshots, not just synthetic
renders. The portrait-card testset (19 real 1920x1080 screenshots of the
WSG ship list) carries one validated ship name per visible card. This tool
crops each name band to a tight RGB sample, writes it under
``tests/game_samples/real-game/ship-name/`` and updates ``manifest.json``.

The source layout (already extracted from ``portrait-card-testset.zip``)::

    <source>/
      alignment.json       # per-viewport observed names + text bands
      screens/viewport-*.png

Usage:
    python tools/dataset/extract_portrait_card_samples.py \
        --source /path/to/portrait-card-testset

``manifest.json`` entries are regenerated for the ``real-game/ship-name``
category (other categories are preserved). Every crop is classified with the
current runtime model; samples the model does not yet read are marked
``known_failure`` with the observed output in the note, so the corpus keeps
real regressions even when the algorithm is not perfect yet.
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

from fixedfontocr import FixedFontOCR  # noqa: E402
from fixedfontocr.types import profile_from_dict  # noqa: E402


COLUMN_X0 = 86
COLUMN_X1 = 278
COLUMN_PITCH = 211


def crop_name(
    screen: np.ndarray,
    band: tuple[int, int],
    column: int,
    threshold: int,
    margin: int,
) -> np.ndarray:
    """Return a tight RGB crop of one name band from a full screenshot."""

    y0, y1 = band
    x0 = COLUMN_X0 + column * COLUMN_PITCH
    region = screen[y0:y1, x0 : x0 + (COLUMN_X1 - COLUMN_X0)]
    gray = region.mean(axis=2)
    mask = gray > threshold
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError(f"no ink in band {band} column {column}")
    top = max(0, y0 + ys.min() - margin)
    bottom = min(screen.shape[0], y0 + ys.max() + 1 + margin)
    left = max(0, x0 + xs.min() - margin)
    right = min(screen.shape[1], x0 + xs.max() + 1 + margin)
    return screen[top:bottom, left:right]


def extract(
    source: Path,
    out_dir: Path,
    manifest_path: Path,
    model_path: Path,
    threshold: int,
    margin: int,
) -> None:
    alignment = json.loads((source / "alignment.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = profile_from_dict(manifest.get("default_profile", {}))
    ocr = FixedFontOCR(model_path=model_path, backend="cpu")
    ocr.profile = profile

    target_dir = out_dir / "real-game" / "ship-name"
    target_dir.mkdir(parents=True, exist_ok=True)

    new_samples: list[dict] = []
    written = 0
    failures = 0
    for viewport in alignment:
        v = viewport["viewport"]
        screen = np.asarray(
            Image.open(source / "screens" / f"viewport-{v:02d}.png").convert("RGB"),
            dtype=np.uint8,
        )
        observed = viewport["observed"]
        row_lengths = viewport["row_lengths"]
        row_start = 0
        for row, band in enumerate(viewport["bands"]):
            for column in range(row_lengths[row]):
                name = observed[row_start + column]
                if name is None:
                    continue
                try:
                    crop = crop_name(screen, tuple(band), column, threshold, margin)
                except ValueError as exc:
                    print(f"skip v{v} r{row} c{column} {name!r}: {exc}")
                    continue
                filename = f"vp{v:02d}_r{row}_c{column}.png"
                rel = Path("real-game") / "ship-name" / filename
                Image.fromarray(crop).save(target_dir / filename)
                written += 1

                result = ocr.recognize(crop, lexicon="ships", lexicon_mode="prefer")
                sample: dict = {
                    "file": rel.as_posix(),
                    "expected": name,
                    "category": "real-game/ship-name",
                    "lexicon": "ships",
                    "note": (
                        f"real ship-list card vp{v:02d} row {row} col {column}; "
                        f"source screens/viewport-{v:02d}.png"
                    ),
                }
                if result.text != name:
                    sample["known_failure"] = True
                    sample["known_failure_note"] = (
                        f"current model returns {result.text!r}"
                    )
                    failures += 1
                new_samples.append(sample)
            row_start += row_lengths[row]

    # Replace only the regenerated category; keep hand-picked fixtures.
    manifest["samples"] = [
        s for s in manifest["samples"] if s.get("category") != "real-game/ship-name"
    ] + sorted(
        new_samples,
        key=lambda s: s["file"],
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {written} real ship-name samples to {target_dir} "
        f"({failures} known failures) and updated {manifest_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="extracted portrait-card-testset root (alignment.json + screens/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "tests" / "game_samples",
        help="game_samples root (default: tests/game_samples)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "tests" / "game_samples" / "manifest.json",
        help="manifest path (default: tests/game_samples/manifest.json)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "model" / "game_cn",
        help="runtime model used for known-failure marking",
    )
    parser.add_argument("--threshold", type=int, default=140)
    parser.add_argument("--margin", type=int, default=2)
    args = parser.parse_args()
    extract(
        args.source,
        args.out,
        args.manifest,
        args.model,
        args.threshold,
        args.margin,
    )


if __name__ == "__main__":
    main()
