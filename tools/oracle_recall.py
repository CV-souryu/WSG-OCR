#!/usr/bin/env python3
"""Oracle lattice recall report over a rendered corpus.

The oracle metric answers the first question when recognition fails:
was the correct segmentation even *in* the candidate lattice?  Scores are
ignored; only candidate geometry matters.  A missing oracle path is a
segmentation problem (no CNN can fix it); a present oracle path with a
wrong decode is a classification / decoding problem.

Goal 21 candidate-generation upgrades under measurement:

* forced advance/seam cuts rescue valley-free fully-touching blobs
  (甲+申 zero-gap);
* small punctuation (``-``/``.``/``·``) bypasses the Goal 8 bbox envelope,
  so ``T-23`` / ``U-`` candidates reach the scorer;
* the expected-width estimate caps wide multi-glyph components.

Per sample the report shows oracle recall, attribution (ok / segmentation
/ decoding) and the candidate count; the summary adds the glued-subset
recall, total candidate counts and the CPU decode p95.

Usage:
    python tools/oracle_recall.py [--font-size 32] [--no-model] [--bench]
"""

from __future__ import annotations

import argparse
import sys
import time
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

# Normal strings plus the acceptance corpus.
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

# Glued (single connected component) pairs: "full" = zero gap (甲+申 is
# valley-free: the forced seam cut case); "row" = thin connector (a real
# valley: the classic low-res connector case).  L+V and U- never form one
# component in this font's clean raster (their outer ink rows do not
# overlap), so they stay normal-render samples.
GLUED = [
    ("甲", "申", "full"),
    ("甲", "申", "row"),
    ("4", "3", "row"),
]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))])


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
    glued_indices: list[int] = []
    for left, right, bridge in GLUED:
        la = render_ground_truth(left, font, font_size=args.font_size)
        lb = render_ground_truth(right, font, font_size=args.font_size)
        lines.append(glue_ground_truth(la, lb, bridge=bridge))
        glued_indices.append(len(lines) - 1)

    geometry = None
    if (MODEL_PATH / "geometry.json").exists():
        geometry = FontGeometryDatabase.load(MODEL_PATH / "geometry.json")

    ocr = None
    decode_times: list[float] = []
    if not args.no_model and (MODEL_PATH / "config.json").exists():
        ocr = FixedFontOCR(model_path=MODEL_PATH, backend="cpu")

    def decode_fn(gt):
        if ocr is None or gt.image is None:
            return gt.text
        t0 = time.perf_counter()
        text = ocr.recognize(gt.image).text
        decode_times.append((time.perf_counter() - t0) * 1000.0)
        return text

    report = evaluate_oracle(
        lines,
        profile,
        geometry=geometry,
        decode_fn=decode_fn,
    )
    samples = report.samples
    glued = [samples[i] for i in glued_indices]
    glued_recall = sum(1 for s in glued if s.oracle) / max(len(glued), 1)
    total_cands = sum(s.n_candidates for s in samples)
    glued_cands = sum(s.n_candidates for s in glued)

    print(f"font: {font}  (font-size {args.font_size}px)")
    print(f"model: {'present' if ocr is not None else 'not built'}"
          f"  geometry: {'present' if geometry is not None else 'none'}")
    print()
    print(format_report(report))
    print()
    print("summary:")
    print(f"  glued-pair subset oracle recall: {glued_recall:.1%} "
          f"({sum(1 for s in glued if s.oracle)}/{len(glued)}) "
          f"candidates {glued_cands}")
    print(f"  total candidates: {total_cands} over {report.total} lines "
          f"(avg {total_cands / max(report.total, 1):.1f}/line)")
    if decode_times:
        print(f"  CPU decode p95: {_p95(decode_times):.1f} ms "
          f"({len(decode_times)} decoded lines)")
    print()
    print("reading the table:")
    print("  oracle=True  + status=ok        -> lattice fine, decode fine")
    print("  oracle=True  + status=decoding  -> classifier/decoder problem")
    print("  oracle=False + reason=unassigned_atom -> no cut near the GT")
    print("    boundary (straddled boundary reports the missing cut x)")
    print("  oracle=False + reason=missing_candidate -> the needed merge/split")
    print("    candidate was pruned or never generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
