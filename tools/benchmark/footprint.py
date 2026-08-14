#!/usr/bin/env python3
"""Print storage and runtime-memory footprints of a FixedFontOCR model.

Measured on a 3000-char CJK template model:

    disk:   216 KB weights.bin + ~9 KB charset.txt + ~0.9 MB
            geometry.json (Goal 8) (~1.1 MB total)
    memory: 216 KB template bits + 24 KB coarse features + 0 popcount table
             (numpy >= 2.0 vectorized bit_count; 64 KB fallback table on
             older numpy)

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


def wgpu_buffers_per_batch(batch: int) -> int:
    """Persistent classify buffers allocated by WGPUBackend for capacity N."""
    return batch * (
        24 * 24  # input
        + 24 * 24 * 4 * 4  # norm (NHWC padded to vec4)
        + 12 * 12 * 8 * 4  # c1
        + 12 * 12 * 8 * 4  # d1
        + 6 * 6 * 16 * 4  # p1
        + 6 * 6 * 16 * 4  # d2
        + 3 * 3 * 32 * 4  # p2
        + 32 * 4  # gap
        + 12  # result
        + 12  # readback
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
        print(f"  batch {batch:>4d}: {fmt_bytes(wgpu_buffers_per_batch(batch))}")

    print("\nnotes:")
    print("  - template bitsets are shared with the loaded model (no copy).")
    print("  - test_vectors.npz is dev-only; drop it from deployed models")
    print("    (tools/train/export_model.py --skip-test-vectors).")
    print("  - WGPU buffers grow to the largest seen batch and stay allocated.")


if __name__ == "__main__":
    main()
