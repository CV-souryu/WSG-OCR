#!/usr/bin/env python3
"""Benchmark CPU vs WGPU backends and print the auto-selection crossover.

This is the measurement step behind ``FixedFontOCR(backend="auto")``: it
times classify() at batch sizes 1/8/16/32/64/128 and reports where the GPU
becomes faster on this machine.

Usage:
    python tools/benchmark.py runtime_model/ [--repeat 5 --iters 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fixedfontocr.backends import (  # noqa: E402
    CPUBackend,
    WGPUBackend,
    benchmark_backends,
)
from fixedfontocr.model import load_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="tinycnn/hybrid model directory")
    parser.add_argument("--batch-sizes", default="1,8,16,32,64,128")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output", help="optional json file with the table")
    args = parser.parse_args()

    model = load_model(Path(args.model))
    if model.classifier == "template":
        raise SystemExit("benchmark needs a tinycnn or hybrid model")
    cpu = CPUBackend(model.weights, model.input_size)
    try:
        gpu = WGPUBackend(model.weights, model.input_size)
    except Exception as exc:
        print(f"WGPU unavailable: {exc}")
        gpu = None

    batch_sizes = tuple(int(v) for v in args.batch_sizes.split(",") if v.strip())
    if gpu is None:
        print("no GPU backend; auto-selection would use CPU for every batch")
        table = {b: {"cpu": 0.0, "gpu": None} for b in batch_sizes}
    else:
        table = benchmark_backends(
            cpu, gpu, batch_sizes=batch_sizes, repeat=args.repeat, iters=args.iters
        )
        print(f"\n{'batch':>8s} {'cpu':>12s} {'gpu':>12s} {'winner':>6s}")
        crossover = None
        for batch in batch_sizes:
            m = table[batch]
            winner = "gpu" if m["gpu"] < m["cpu"] else "cpu"
            if winner == "gpu" and crossover is None:
                crossover = batch
            print(
                f"{batch:>8d} {m['cpu'] * 1e6:>9.1f}µs "
                f"{m['gpu'] * 1e6:>9.1f}µs {winner:>6s}"
            )
        print(
            "\nGPU wins from batch "
            + (str(crossover) if crossover else "none (CPU always faster)")
        )

    if args.output:
        Path(args.output).write_text(
            json.dumps({"batch_sizes": list(batch_sizes), "table": table}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
