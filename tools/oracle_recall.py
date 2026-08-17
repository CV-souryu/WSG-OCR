#!/usr/bin/env python3
"""Oracle lattice recall report over a rendered corpus.

The oracle metric answers the first question when recognition fails:
was the correct segmentation even *in* the candidate lattice?  Scores are
ignored; only candidate geometry matters.  A missing oracle path is a
segmentation problem (no CNN can fix it); a present oracle path with a
wrong decode is a classification / decoding problem.

This runner renders a corpus with the registered font -- normal strings,
plus deliberately glued (fully-touching) pairs that reproduce the real-game
``LV`` / ``4+3`` / ``U-`` failure mode -- and prints, per sample:

* oracle recall (is the GT segmentation in the lattice?);
* attribution (ok / segmentation / decoding) when the model is available;
* for segmentation failures, the missing cut positions (straddled
  boundaries) and per-character best coverage.

Usage:
    python tools/oracle_recall.py [--font-size 32] [--no-model]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fixedfontocr import FixedFontOCR  # noqa: E402
from fixedfontocr.defaults import FONT_PATH, MODEL_PATH, resolve_font  # noqa: E402
from fixedfontocr.geometry import FontGeometryDatabase  # noqa: E402
from fixedfontocr.oracle import (  # noqa: E402
    glue_ground_truth,
    render_ground_truth,
)
from fixedfontocr.oracle import evaluate_oracle, format_report  # noqa: E402
from fixedfontocr.types import default_profile  # noqa: E402

# Normal strings plus the acceptance corpus; the glued pairs below exercise
# the fully-touching failure mode the valley heuristic cannot cut.
CORPUS = [
    "潜甲",
    "鲃鱼。",
    "小",
    "巴尔的摩",
    "Z17",
    "LV.40",
    "T-23",
    "U-",
    "4+3",
    "1000",
    "岛风",
    "甲申",
    "获得金币1000",
]

# (left, right, bridge) -- "full" glues the glyphs into one component with
# no vertical valley at the boundary (the hard case); "row" is a thin
# valley connector the current splitter handles.
GLUED = [
    ("甲", "申", "full"),
    ("4", "3", "full"),
    ("L", "V", "full"),
    ("U", "-", "full"),
    ("甲", "申", "row"),
    ("4", "3", "row"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-size", type=int, default=32)
    parser.add_argument("--no-model", action="store_true", help="skip decode attribution")
    args = parser.parse_args()

    font = resolve_font(FONT_PATH)
    profile = default_profile()

    lines = [
        render_ground_truth(text, font, font_size=args.font_size)
        for text in CORPUS
    ]
    for left, right, bridge in GLUED:
        la = render_ground_truth(left, font, font_size=args.font_size)
        lb = render_ground_truth(right, font, font_size=args.font_size)
        lines.append(glue_ground_truth(la, lb, bridge=bridge))

    geometry = None
    if (MODEL_PATH / "geometry.json").exists():
        geometry = FontGeometryDatabase.load(MODEL_PATH / "geometry.json")

    ocr = None
    if not args.no_model and (MODEL_PATH / "config.json").exists():
        ocr = FixedFontOCR(model_path=MODEL_PATH, backend="cpu")

    def decode_fn(gt):
        if ocr is None or gt.image is None:
            return gt.text
        return ocr.recognize(gt.image).text

    report = evaluate_oracle(
        lines,
        profile,
        geometry=geometry,
        decode_fn=decode_fn,
    )
    print(f"font: {font}  (font-size {args.font_size}px)")
    print(f"model: {'present' if ocr is not None else 'not built'}"
          f"  geometry: {'present' if geometry is not None else 'none'}")
    print()
    print(format_report(report))
    print()
    print("reading the table:")
    print("  oracle=True  + status=ok        -> lattice fine, decode fine")
    print("  oracle=True  + status=decoding  -> classifier/decoder problem")
    print("  oracle=False + reason=unassigned_atom -> no cut near the GT")
    print("    boundary: the valley heuristic missed it (straddled boundary")
    print("    reports the missing cut x); needs forced advance-based cuts")
    print("  oracle=False + reason=missing_candidate -> the needed merge/split")
    print("    candidate was pruned or never generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
