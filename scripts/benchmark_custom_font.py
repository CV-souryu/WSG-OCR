#!/usr/bin/env python3
"""CPU vs WGPU performance comparison for the bundled game font.

Trains nothing; it consumes an already-trained hybrid (or TinyCNN) model
directory built from the same font and charset:

    python tools/train/build_model.py ...

    python scripts/benchmark_custom_font.py \
        --font fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf \
        --model model/game_cn

It measures, all on glyphs rendered from the given font:

1. Engine construction (template / tinycnn CPU / tinycnn WGPU).
2. Per-glyph template and per-glyph TinyCNN CPU classification.
3. Batched TinyCNN classify(): numpy CPU vs WGPU, batch 1..256.
4. End-to-end recognize(): backend="cpu" vs backend="wgpu".

CPU and WGPU results are checked against each other (identical char ids,
score difference < 1e-4), so the comparison is apples-to-apples.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from fixedfontocr import FixedFontOCR  # noqa: E402
from fixedfontocr.backends import CPUBackend, WGPUBackend  # noqa: E402
from fixedfontocr.classifier import TemplateClassifier  # noqa: E402
from fixedfontocr.cnn import TinyCNNClassifier  # noqa: E402
from fixedfontocr import defaults  # noqa: E402
from fixedfontocr.defaults import resolve_font  # noqa: E402
from fixedfontocr.fontgen import build_templates, write_model  # noqa: E402
from fixedfontocr.model import load_model  # noqa: E402
from fixedfontocr.preprocess import normalize  # noqa: E402
from fixedfontocr.types import default_profile  # noqa: E402


def render_text(
    text: str, font_path: Path | str, font_size: int = 32
) -> np.ndarray:
    """Render white text on black; return RGB uint8."""
    font = ImageFont.truetype(str(font_path), font_size)
    bbox = font.getbbox(text)
    pad = 8
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGB", (max(1, w), max(1, h)), (0, 0, 0))
    ImageDraw.Draw(img).text(
        (pad - bbox[0], pad - bbox[1]), text, font=font, fill=(255, 255, 255)
    )
    return np.asarray(img, dtype=np.uint8)


def glyph_mask(font_path: Path | str, char: str, size: int = 32) -> np.ndarray:
    """Tight boolean ink mask for one character."""
    font = ImageFont.truetype(str(font_path), size)
    canvas = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(canvas).text((0, 0), char, font=font, fill=255)
    arr = np.asarray(canvas, dtype=np.uint8) >= 140
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    if ys.size == 0:
        raise ValueError(f"no ink rendered for {char!r}")
    return arr[int(ys[0]) : int(ys[-1]) + 1, int(xs[0]) : int(xs[-1]) + 1]


def median_time(fn, repeat: int, iters: int) -> float:
    fn()  # warm-up
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        samples.append((time.perf_counter() - t0) / iters)
    samples.sort()
    return samples[len(samples) // 2]


def fmt_us(seconds: float) -> str:
    us = seconds * 1e6
    return f"{us:9.1f} µs" if us < 1000 else f"{us / 1e3:9.2f} ms"


def fmt_rate(per_sec: float) -> str:
    if per_sec >= 1e6:
        return f"{per_sec / 1e6:8.1f} M/s"
    if per_sec >= 1e3:
        return f"{per_sec / 1e3:8.1f} K/s"
    return f"{per_sec:8.0f}/s"


def bench_construction(font: Path, tinycnn_model: Path, repeat: int) -> None:
    print("\n== engine construction ==")
    with tempfile.TemporaryDirectory() as tmp:
        model = load_model(tinycnn_model)
        chars, templates = build_templates(font, list(model.charset), render_size=32)
        write_model(Path(tmp) / "template", chars, templates, font_path=font)
        t_tpl = median_time(
            lambda: FixedFontOCR(model_path=Path(tmp) / "template", backend="cpu"),
            repeat,
            5,
        )
        t_cpu = median_time(
            lambda: FixedFontOCR(model_path=tinycnn_model, backend="cpu"), repeat, 10
        )
        try:
            t_gpu = median_time(
                lambda: FixedFontOCR(model_path=tinycnn_model, backend="wgpu"),
                repeat,
                10,
            )
        except Exception as exc:
            t_gpu = None
            print(f"  WGPU unavailable: {exc}")
        print(f"  template            {t_tpl * 1e3:7.2f} ms")
        print(f"  hybrid cpu          {t_cpu * 1e3:7.2f} ms")
        if t_gpu is not None:
            print(f"  hybrid wgpu         {t_gpu * 1e3:7.2f} ms")


def bench_per_glyph(font: Path, tinycnn_model: Path, repeat: int, iters: int) -> None:
    print("\n== per-glyph classification (single glyph at a time) ==")
    model = load_model(tinycnn_model)
    profile = default_profile()
    with tempfile.TemporaryDirectory() as tmp:
        chars, templates = build_templates(font, list(model.charset), render_size=32)
        write_model(Path(tmp) / "template", chars, templates, font_path=font)
        tpl = TemplateClassifier(templates, chars)
        masks = [glyph_mask(font, c) for c in model.charset]
        state = {"i": 0}

        def cycle_tpl() -> None:
            tpl(masks[state["i"] % len(masks)], profile)
            state["i"] += 1

        t_tpl = median_time(cycle_tpl, repeat, iters)

    clf = TinyCNNClassifier(
        weights=model.weights, charset=model.charset, input_size=model.input_size
    )
    state2 = {"i": 0}

    def cycle_cnn() -> None:
        clf(masks[state2["i"] % len(masks)], profile)
        state2["i"] += 1

    t_cnn = median_time(cycle_cnn, repeat, iters)
    print(f"  {'template cpu':24s} {fmt_us(t_tpl):>12s} {fmt_rate(1 / t_tpl):>12s}")
    print(f"  {'tinycnn cpu':24s} {fmt_us(t_cnn):>12s} {fmt_rate(1 / t_cnn):>12s}")


def bench_batched_classify(
    font: Path, tinycnn_model: Path, repeat: int, iters: int
) -> None:
    print("\n== TinyCNN batched classify(): CPU vs WGPU ==")
    model = load_model(tinycnn_model)
    cpu = CPUBackend(model.weights, model.input_size)
    try:
        gpu = WGPUBackend(model.weights, model.input_size)
    except Exception as exc:
        print(f"  WGPU unavailable: {exc}")
        return
    glyphs = [normalize(glyph_mask(font, c), model.input_size) for c in model.charset]
    print(
        f"  {'batch':>8s} {'cpu latency':>12s} {'gpu latency':>12s} "
        f"{'cpu rate':>12s} {'gpu rate':>12s} {'speedup':>8s} {'ids match':>10s}"
    )
    for batch in (1, 8, 16, 32, 64, 128, 256):
        g = np.stack([glyphs[i % len(glyphs)] for i in range(batch)])
        ref = cpu.classify(g)
        got = gpu.classify(g)
        ok = np.array_equal(ref.char_ids, got.char_ids)
        diff = float(np.abs(ref.scores - got.scores).max())
        tc = median_time(lambda: cpu.classify(g), repeat, max(1, iters // 4))
        tg = median_time(lambda: gpu.classify(g), repeat, max(1, iters // 4))
        print(
            f"  {batch:>8d} {fmt_us(tc):>12s} {fmt_us(tg):>12s} "
            f"{fmt_rate(batch / tc):>12s} {fmt_rate(batch / tg):>12s} "
            f"{tc / tg:>7.1f}x {str(ok):>10s} (Δ{diff:.1e})"
        )


def bench_end_to_end(
    font: Path, tinycnn_model: Path, repeat: int, iters: int
) -> None:
    print("\n== end-to-end recognize(): CPU vs WGPU ==")
    try:
        ocr_gpu = FixedFontOCR(model_path=tinycnn_model, backend="wgpu")
    except Exception as exc:
        print(f"  WGPU unavailable: {exc}")
        return
    ocr_cpu = FixedFontOCR(model_path=tinycnn_model, backend="cpu")
    cases = [
        ("ship   (4 chars)", "俾斯麦提尔比茨"),
        ("equip  (6 chars)", "大型单装炮"),
        ("copy  (10 chars)", "获得金币1000数量提示"),
    ]
    print(
        f"  {'case':20s} {'cpu latency':>12s} {'gpu latency':>12s} "
        f"{'cpu rate':>12s} {'gpu rate':>12s} {'speedup':>8s} {'text match':>10s}"
    )
    for label, text in cases:
        image = render_text(text, font)
        r_cpu = ocr_cpu.recognize(image)
        r_gpu = ocr_gpu.recognize(image)
        n = len(r_cpu.text)
        t_cpu = median_time(lambda: ocr_cpu.recognize(image), repeat, max(1, iters // 2))
        t_gpu = median_time(lambda: ocr_gpu.recognize(image), repeat, max(1, iters // 2))
        print(
            f"  {label:20s} {fmt_us(t_cpu):>12s} {fmt_us(t_gpu):>12s} "
            f"{fmt_rate(n / t_cpu):>12s} {fmt_rate(n / t_gpu):>12s} "
            f"{t_cpu / t_gpu:>7.1f}x "
            f"{str(r_cpu.text == r_gpu.text):>10s}  {r_cpu.text!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--font",
        type=Path,
        default=defaults.FONT_PATH,
        help="registered font under fonts/ (default: bundled game font)",
    )
    parser.add_argument("--model", type=Path, default=defaults.MODEL_PATH)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    args.font = resolve_font(args.font)
    if not (args.model / "config.json").exists():
        raise SystemExit(
            f"hybrid model not found: {args.model} — build it first with "
            f"tools/train/build_model.py"
        )

    print(
        f"machine: {platform.processor() or platform.machine()} | "
        f"cpu_count={os.cpu_count()} | numpy={np.__version__}"
    )
    print(f"font: {args.font} | model: {args.model}")

    bench_construction(args.font, args.model, args.repeat)
    bench_per_glyph(args.font, args.model, args.repeat, args.iters)
    bench_batched_classify(args.font, args.model, args.repeat, args.iters)
    bench_end_to_end(args.font, args.model, args.repeat, args.iters)


if __name__ == "__main__":
    main()
