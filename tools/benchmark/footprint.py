#!/usr/bin/env python3
"""Print storage and runtime-memory footprints of a FixedFontOCR model.

Measured on a 3000-char CJK template model:

    disk:   216 KB weights.bin + ~9 KB charset.txt + ~0.9 MB
            geometry.json (Goal 8) (~1.1 MB total)
    memory: 216 KB template bits + 24 KB coarse features + 0 popcount table
            (numpy >= 2.0 vectorized bit_count; 64 KB fallback table on
            older numpy)

Goal 9 Template V2 models replace the single bitset per character with a
prototype grid (default 49 per character: 6 sizes × 4 sub-pixel phases × 2
downsample modes + 1 clean high-res render), so the same 3000-char model
stores ~10.3 MB of bitsets plus ~0.6 MB of metadata and coarse features.

A 3000-class TinyCNN stores 4032 bytes of conv weights + 33*C*4 bytes of
linear weights (C = classes), e.g. ~391 KB. Hybrid models store both.
The WGPU backend additionally allocates ~22.4 KB per glyph of persistent
batch buffers, so a 64-glyph line costs ~1.4 MB of GPU-side storage.

Usage:
    python tools/benchmark/footprint.py model/ [--batch-sizes 1,8,16,32,64,256]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fixedfontocr.model import load_model  # noqa: E402


def fmt_bytes(n: int) -> str:
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB ({n:,} B)"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB ({n:,} B)"
    return f"{n:,} B"


def wgpu_buffers_per_batch(batch: int, classes: int = 0) -> int:
    """Persistent buffers allocated by WGPUBackend for capacity N.

    The mega shader keeps every intermediate tensor in workgroup shared
    memory, so only the packed input, the result/logits records and the two
    staging buffers exist (no per-layer buffers at all).
    """
    return batch * (
        24 * 24  # input (packed uint8 glyphs)
        + 12  # result record (best id + top-2 scores)
        + classes * 4  # logits record (forward_logits)
        + 12  # result readback staging
        + classes * 4  # logits readback staging
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="template / tinycnn / hybrid model dir")
    parser.add_argument("--batch-sizes", default="1,8,16,32,64,256")
    parser.add_argument("--image", help="optional WxH image size for the mask estimate")
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        raise SystemExit(f"not a model directory: {model_dir}")
    model = load_model(model_dir)
    n = len(model.charset)

    print(f"model: {model_dir}  classifier={model.classifier}  classes={n}")
    print("\n== on-disk (runtime model dir) ==")
    disk_total = 0
    for name in (
        "config.json",
        "model.json",
        "charset.txt",
        "weights.bin",
        "templates.bin",
        "geometry.json",
        "test_vectors.npz",
    ):
        p = model_dir / name
        if p.exists():
            size = p.stat().st_size
            disk_total += size
            print(f"  {name:18s} {fmt_bytes(size)}")
    print(f"  {'total':18s} {fmt_bytes(disk_total)}")

    print("\n== in-memory (CPU) ==")
    memory = 0
    if model.templates is not None:
        template_bytes = n * ((model.input_size * model.input_size + 7) // 8)
        features_bytes = n * 8  # uint16 ink + six uint8 bbox/margins
        popcount_bytes = 0 if hasattr(np, "bitwise_count") else 65536
        print(f"  template bitset:  {fmt_bytes(template_bytes)}")
        print(f"  coarse features:  {fmt_bytes(features_bytes)}")
        print(f"  popcount table:   {fmt_bytes(popcount_bytes)} (0 with numpy>=2.0)")
        memory += template_bytes + features_bytes + popcount_bytes
    if model.templates_v2 is not None:
        p = model.templates_v2.prototypes_per_char
        bytes_per = model.templates_v2.bytes_per_template
        template_bytes = n * p * bytes_per
        meta_bytes = n * p * 4  # render size + dx/dy + downsample mode
        features_bytes = n * p * 8  # per-prototype coarse features
        popcount_bytes = 0 if hasattr(np, "bitwise_count") else 65536
        print(
            f"  template V2 bitset: {fmt_bytes(template_bytes)} "
            f"({n} chars × {p} prototypes)"
        )
        print(f"  prototype metadata: {fmt_bytes(meta_bytes)}")
        print(f"  coarse features:    {fmt_bytes(features_bytes)}")
        print(f"  popcount table:     {fmt_bytes(popcount_bytes)} (0 with numpy>=2.0)")
        memory += template_bytes + meta_bytes + features_bytes + popcount_bytes
    if model.weights is not None:
        cnn_bytes = sum(int(a.nbytes) for a in model.weights.values())
        print(f"  cnn weights:      {fmt_bytes(cnn_bytes)}")
        memory += cnn_bytes
    print(f"  {'total':18s} {fmt_bytes(memory)}")

    if args.image:
        try:
            w, h = (int(v) for v in args.image.lower().split("x"))
        except ValueError:
            raise SystemExit("--image must look like 1920x1080")
        print(
            f"\n  per-image mask:   {fmt_bytes(w * h)} "
            f"(bool mask for {args.image})"
        )

    print("\n== WGPU persistent batch buffers (per classify call) ==")
    batch_sizes = [int(v) for v in args.batch_sizes.split(",") if v.strip()]
    for batch in batch_sizes:
        print(
            f"  batch {batch:>4d}: {fmt_bytes(wgpu_buffers_per_batch(batch, classes=n))}"
        )

    print("\nnotes:")
    print("  - template bitsets are shared with the loaded model (no copy).")
    print("  - test_vectors.npz is dev-only; drop it from deployed models")
    print("    (tools/train/export_model.py --skip-test-vectors).")
    print("  - WGPU buffers grow to the largest seen batch and stay allocated.")


if __name__ == "__main__":
    main()
