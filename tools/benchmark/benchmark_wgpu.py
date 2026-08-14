#!/usr/bin/env python3
"""CPU vs WGPU TinyCNN benchmark with phase-level timing.

Measures the steady-state ``classify()`` path on this machine and reports:

    cpu       total numpy backend wall time
    upload    host -> GPU buffer copy (``queue.write_buffer``)
    compute   command encoding + queue submission (dispatches are recorded
              here; actual GPU execution is overlapped with the readback)
    readback  copy-to-staging + ``map_sync`` + host read (includes the GPU
              sync floor)
    gpu       upload + compute + readback (sum of the phases above)

Every batch is also checked for CPU/WGPU parity (identical argmax ids and
the max score difference) before timing, so the table is only produced for
inputs where the two backends agree.

Models can come from an existing model directory (a tinycnn/hybrid model,
e.g. ``tests/fixtures/cnn_digits``) or be synthesized on the fly for a given
class count, which is how the 3K/7K CJK scenarios are benchmarked without
requiring trained models:

    python tools/benchmark/benchmark_wgpu.py tests/fixtures/cnn_digits
    python tools/benchmark/benchmark_wgpu.py --classes 3000 --charset cjk
    python tools/benchmark/benchmark_wgpu.py --classes 7000 --charset cjk

Usage:
    python tools/benchmark/benchmark_wgpu.py [model_dir | --classes N] [options]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from fixedfontocr.backends import CPUBackend, WGPUBackend  # noqa: E402
from fixedfontocr.model import load_model  # noqa: E402


def random_weights(num_classes: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Random f32 weights for the fixed TinyCNN architecture."""
    rng = np.random.default_rng(seed)
    return {
        "conv1.weight": rng.standard_normal((8, 1, 3, 3), dtype=np.float32),
        "conv1.bias": rng.standard_normal(8, dtype=np.float32),
        "dw1.weight": rng.standard_normal((8, 3, 3), dtype=np.float32),
        "dw1.bias": rng.standard_normal(8, dtype=np.float32),
        "pw1.weight": rng.standard_normal((16, 8), dtype=np.float32),
        "pw1.bias": rng.standard_normal(16, dtype=np.float32),
        "dw2.weight": rng.standard_normal((16, 3, 3), dtype=np.float32),
        "dw2.bias": rng.standard_normal(16, dtype=np.float32),
        "pw2.weight": rng.standard_normal((32, 16), dtype=np.float32),
        "pw2.bias": rng.standard_normal(32, dtype=np.float32),
        "fc.weight": rng.standard_normal((num_classes, 32), dtype=np.float32),
        "fc.bias": rng.standard_normal(num_classes, dtype=np.float32),
    }


def median(values: list[float]) -> float:
    return float(sorted(values)[len(values) // 2])


def median_cpu_time(fn, repeat: int, iters: int) -> float:
    """Median per-call wall time of ``fn()`` (warm-up + ``repeat`` samples)."""
    fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter() - t0) / iters)
    return median(samples)


def median_gpu_phases(
    gpu: WGPUBackend, glyphs: np.ndarray, repeat: int
) -> dict[str, float]:
    """Median upload/compute/readback/total times over ``repeat`` calls."""
    gpu.classify(glyphs, profile=True)  # warm-up (also allocates buffers)
    phases: dict[str, list[float]] = {
        "upload": [],
        "compute": [],
        "readback": [],
        "total": [],
    }
    for _ in range(repeat):
        gpu.classify(glyphs, profile=True)
        t = gpu.last_timing
        if t is None:
            raise RuntimeError("profile=True did not record timing")
        for key in phases:
            phases[key].append(t[key])
    return {key: median(values) for key, values in phases.items()}


def fmt_us(seconds: float) -> str:
    us = seconds * 1e6
    return f"{us:9.1f}" if us < 1000 else f"{us / 1e3:8.2f}k"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, help="tinycnn/hybrid model dir")
    parser.add_argument("--classes", type=int, help="synthetic model class count")
    parser.add_argument("--charset", choices=("digits", "cjk"), default="digits")
    parser.add_argument("--batch-sizes", default="1,8,16,32,64,128")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="write the table as JSON")
    args = parser.parse_args()

    if (args.model is None) == (args.classes is None):
        parser.error("pass exactly one of MODEL or --classes")

    if args.model is not None:
        model = load_model(args.model)
        if model.classifier == "template":
            raise SystemExit("benchmark needs a tinycnn or hybrid model")
        weights = model.weights
        label = f"{args.model} ({len(model.charset)} classes)"
    else:
        weights = random_weights(args.classes, seed=args.seed)
        label = (
            f"synthetic {args.charset} ({args.classes} classes)"
        )

    cpu = CPUBackend(weights)
    try:
        gpu = WGPUBackend(weights)
    except Exception as exc:
        raise SystemExit(f"WGPU unavailable: {exc}")

    batch_sizes = tuple(
        int(v) for v in args.batch_sizes.split(",") if v.strip()
    )
    print(f"scenario: {label}")
    print(
        f"  {'batch':>6s} {'cpu':>10s} {'upload':>10s} {'compute':>10s} "
        f"{'readback':>10s} {'gpu total':>10s} {'speedup':>8s} "
        f"{'ids':>5s} {'score diff':>11s}"
    )
    table: dict[int, dict] = {}
    crossover: int | None = None
    for batch in batch_sizes:
        rng = np.random.default_rng(args.seed * 1000 + batch)
        glyphs = (rng.random((batch, 24, 24)) > 0.5).astype(np.uint8) * 255

        ref = cpu.classify(glyphs)
        got = gpu.classify(glyphs)
        ids_ok = bool(np.array_equal(ref.char_ids, got.char_ids))
        score_diff = float(np.abs(ref.scores - got.scores).max())

        tc = median_cpu_time(
            lambda: cpu.classify(glyphs), args.repeat, max(1, args.iters)
        )
        phases = median_gpu_phases(gpu, glyphs, args.repeat)
        tg = phases["total"]
        if crossover is None and tg < tc:
            crossover = batch
        table[batch] = {
            "cpu_total": tc,
            "gpu_upload": phases["upload"],
            "gpu_compute": phases["compute"],
            "gpu_readback": phases["readback"],
            "gpu_total": tg,
            "ids_match": ids_ok,
            "score_maxdiff": score_diff,
        }
        print(
            f"  {batch:>6d} {fmt_us(tc):>10s} {fmt_us(phases['upload']):>10s} "
            f"{fmt_us(phases['compute']):>10s} {fmt_us(phases['readback']):>10s} "
            f"{fmt_us(tg):>10s} {tc / tg:>7.1f}x {str(ids_ok):>5s} "
            f"{score_diff:>10.1e}"
        )

    print(
        "\nGPU wins from batch "
        + (str(crossover) if crossover is not None else "none (CPU always faster)")
    )
    if args.output:
        payload = {
            "scenario": label,
            "batch_sizes": list(batch_sizes),
            "crossover": crossover,
            "table": {str(b): m for b, m in table.items()},
        }
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
