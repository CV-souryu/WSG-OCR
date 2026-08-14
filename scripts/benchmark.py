#!/usr/bin/env python3
"""Micro-benchmarks for FixedFontOCR.

Measures, on this machine:

1. Model loading time (template and TinyCNN model directories).
2. Engine construction time (``FixedFontOCR(...)``; includes the template
   popcount table build).
3. Per-glyph classification latency and throughput for both classifiers,
   plus the vectorization gain of the batched TinyCNN numpy forward.
4. End-to-end ``recognize()`` on rendered text lines, with a
   segmentation-vs-classification breakdown.
5. TinyCNN CPU vs WGPU classify() latency/throughput, and end-to-end
   ``recognize()`` with ``backend="wgpu"`` (skips if no GPU adapter).

Results are medians of several timed runs and are only meaningful relative
to each other on this machine.

Usage:
    python scripts/benchmark.py [--repeat N] [--iters N]
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
from fixedfontocr import defaults  # noqa: E402
from fixedfontocr.backends import CPUBackend, WGPUBackend  # noqa: E402
from fixedfontocr.classifier import TemplateClassifier  # noqa: E402
from fixedfontocr.cnn import TinyCNNClassifier, forward  # noqa: E402
from fixedfontocr.fontgen import build_templates, write_model  # noqa: E402
from fixedfontocr.model import load_model  # noqa: E402
from fixedfontocr.preprocess import normalize, preprocess  # noqa: E402
from fixedfontocr.types import default_profile  # noqa: E402

BUNDLED_FONT = defaults.resolve_font(defaults.FONT_PATH)

ASCII_CHARSET = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "+-×÷%.,:;!?()[]"
)
CJK_CHARSET = "获得金币消耗数量提示确定取消返回攻击防御生命法力0123456789+-×÷%.,:;!?()[]{}「」"
E2E_LINES = [
    ("template-ascii", BUNDLED_FONT, "HELLO WORLD 1234567890"),
    ("template-cjk", BUNDLED_FONT, "获得金币1000 数量提示"),
    ("tinycnn-digits", BUNDLED_FONT, "1234567890 4242"),
    ("tinycnn-cjk", BUNDLED_FONT, "获得金币1000"),
    ("tinycnn-game-cn", BUNDLED_FONT, "俾斯麦提尔比茨大型单装炮"),
]
FIXTURES = {
    "tinycnn-digits": ("tests/fixtures/cnn_digits", BUNDLED_FONT),
    "tinycnn-cjk": ("tests/fixtures/cnn_cjk", BUNDLED_FONT),
    "tinycnn-game-cn": (str(defaults.MODEL_PATH), BUNDLED_FONT),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def render_text(
    text: str, font_path: Path | str, font_size: int = 32
) -> np.ndarray:
    """Render text on black; return an RGB uint8 array."""
    font = ImageFont.truetype(str(font_path), font_size)
    bbox = font.getbbox(text)
    pad = 8
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGB", (max(1, w), max(1, h)), (0, 0, 0))
    ImageDraw.Draw(img).text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(255, 255, 255))
    return np.asarray(img, dtype=np.uint8)


def glyph_mask(font_path: Path | str, char: str, size: int = 28) -> np.ndarray:
    """Tight boolean ink mask for a single character."""
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
    """Return the median per-call wall time (seconds) of ``fn()``."""
    fn()  # warm-up: BLAS / allocator / JIT
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


class _CountingClassifier:
    """Replicates the classifier's normalize step but skips matching, so the
    measured time is segmentation + normalization only."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, mask: np.ndarray, profile) -> tuple[str, float]:
        self.calls += 1
        normalize(mask, profile.target_size)
        return "?", 0.0


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------


def bench_model_load(repeat: int) -> None:
    print("== model loading ==")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, font, charset in (
            ("template ascii", BUNDLED_FONT, ASCII_CHARSET),
            ("template cjk", BUNDLED_FONT, CJK_CHARSET),
        ):
            chars, templates = build_templates(font, list(charset), render_size=32)
            write_model(
                tmp / name.replace(" ", "_"),
                chars,
                templates,
                font_path=BUNDLED_FONT,
            )
            t = median_time(
                lambda p=tmp / name.replace(" ", "_"): load_model(p), repeat, 5
            )
            print(f"  {name:18s} ({len(chars):3d} classes)  {t * 1e3:7.2f} ms")
    for name, (dir, _) in FIXTURES.items():
        p = Path(dir)
        model = load_model(p)
        t = median_time(lambda p=p: load_model(p), repeat, 10)
        print(f"  {name:18s} ({len(model.charset):3d} classes)  {t * 1e3:7.2f} ms")


def bench_engine_construction(repeat: int) -> None:
    print("\n== engine construction (FixedFontOCR(...)) ==")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        chars, templates = build_templates(BUNDLED_FONT, list(ASCII_CHARSET), render_size=32)
        write_model(tmp / "ascii", chars, templates, font_path=BUNDLED_FONT)
        t = median_time(
            lambda: FixedFontOCR(model_path=tmp / "ascii", backend="cpu"), repeat, 5
        )
        print(f"  template ({len(chars):3d} classes)          {t * 1e3:7.2f} ms")
        chars, templates = build_templates(BUNDLED_FONT, list(CJK_CHARSET), render_size=32)
        write_model(tmp / "cjk", chars, templates, font_path=BUNDLED_FONT)
        t = median_time(
            lambda: FixedFontOCR(model_path=tmp / "cjk", backend="cpu"), repeat, 5
        )
        print(f"  template ({len(chars):3d} classes)          {t * 1e3:7.2f} ms")
    for name, (dir, _) in FIXTURES.items():
        p = Path(dir)
        t = median_time(
            lambda p=p: FixedFontOCR(model_path=p, backend="cpu"), repeat, 10
        )
        print(f"  {name}  {t * 1e3:7.2f} ms")


def bench_classifiers(repeat: int, iters: int) -> None:
    print("\n== per-glyph classification ==")
    print(f"  {'scenario':24s} {'latency':>12s} {'throughput':>12s}")
    profile = default_profile()

    def cycle(clf, masks):
        state = {"i": 0}
        n = len(masks)

        def one() -> None:
            clf(masks[state["i"] % n], profile)
            state["i"] += 1

        return one

    # Template baseline: ASCII and CJK charsets.
    for name, font, charset in (
        ("template ascii", BUNDLED_FONT, ASCII_CHARSET),
        ("template cjk", BUNDLED_FONT, CJK_CHARSET),
    ):
        chars, templates = build_templates(font, list(charset), render_size=32)
        clf = TemplateClassifier(templates, chars)
        masks = [glyph_mask(font, c) for c in chars]
        t = median_time(cycle(clf, masks), repeat, iters)
        print(f"  {name:24s} {fmt_us(t):>12s} {fmt_rate(1 / t):>12s}")

    # TinyCNN: single-shot and batched forward.
    for name, (dir, font) in FIXTURES.items():
        model = load_model(Path(dir))
        clf = TinyCNNClassifier(
            weights=model.weights, charset=model.charset, input_size=model.input_size
        )
        masks = [glyph_mask(font, c) for c in model.charset]
        t = median_time(cycle(clf, masks), repeat, iters)
        print(f"  {name:24s} {fmt_us(t):>12s} {fmt_rate(1 / t):>12s}")

    # Batched CNN forward: same glyphs in one tensor.
    print("\n== TinyCNN batched forward (vectorization gain) ==")
    print(f"  {'batch':>8s} {'latency':>12s} {'throughput':>12s} {'gain vs N=1':>12s}")
    model = load_model(Path(FIXTURES["tinycnn-digits"][0]))
    glyphs = [
        normalize(glyph_mask(BUNDLED_FONT, c), 24).astype(np.float32) / 255.0
        for c in model.charset
    ]
    x1 = np.stack([glyphs[0]])[:, None]
    t1 = median_time(lambda: forward(x1, model.weights), repeat, iters)
    print(f"  {1:>8d} {fmt_us(t1):>12s} {fmt_rate(1 / t1):>12s} {'1.0x':>12s}")
    for batch in (8, 16, 64, 256):
        xb = np.stack([glyphs[i % len(glyphs)] for i in range(batch)])[:, None]
        t = median_time(lambda xb=xb: forward(xb, model.weights), repeat, max(1, iters // 4))
        # throughput gain vs N=1: (batch/t) / (1/t1) = batch * t1 / t
        print(
            f"  {batch:>8d} {fmt_us(t):>12s} {fmt_rate(batch / t):>12s} "
            f"{batch * t1 / t:>11.1f}x"
        )


def bench_end_to_end(repeat: int, iters: int) -> None:
    print("\n== end-to-end recognize() ==")
    print(f"  {'scenario':20s} {'latency':>12s} {'throughput':>12s} "
          f"{'segmentation':>12s} {'classification':>14s}")
    profile = default_profile()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Template models (ascii + cjk), keyed by scenario name.
        model_dirs: dict[str, Path] = {}
        for scenario, font, charset in (
            ("template-ascii", BUNDLED_FONT, ASCII_CHARSET),
            ("template-cjk", BUNDLED_FONT, CJK_CHARSET),
        ):
            chars, templates = build_templates(font, list(charset), render_size=32)
            model_dirs[scenario] = tmp / scenario
            write_model(
                model_dirs[scenario],
                chars,
                templates,
                font_path=BUNDLED_FONT,
            )

        for scenario, font, text in E2E_LINES:
            if scenario.startswith("tinycnn"):
                dir, font = FIXTURES[scenario]
                ocr = FixedFontOCR(model_path=dir, backend="cpu")
            else:
                ocr = FixedFontOCR(model_path=model_dirs[scenario], backend="cpu")
            image = render_text(text, font)
            result = ocr.recognize(image)
            n_chars = len(result.text)
            t = median_time(lambda: ocr.recognize(image), repeat, max(1, iters // 2))

            # Segmentation (+normalize) vs classification split.
            counter = _CountingClassifier()
            t_seg = median_time(
                lambda: preprocess(image, profile, counter), repeat, max(1, iters // 2)
            )
            t_clf = max(0.0, t - t_seg)
            print(
                f"  {scenario:20s} {fmt_us(t):>12s} {fmt_rate(n_chars / t):>12s} "
                f"{fmt_us(t_seg):>12s} {fmt_us(t_clf):>14s}"
                f"   (text: {result.text!r})"
            )


def bench_wgpu_backends(repeat: int, iters: int) -> None:
    """Compare the numpy and WGPU TinyCNN backends (batched classify)."""
    print("\n== TinyCNN backend: CPU vs WGPU (classify) ==")
    model = load_model(Path(FIXTURES["tinycnn-digits"][0]))
    cpu = CPUBackend(model.weights, model.input_size)
    try:
        gpu = WGPUBackend(model.weights, model.input_size)
    except Exception as exc:
        print(f"  WGPU unavailable: {exc}")
        return
    glyphs = [
        normalize(glyph_mask(BUNDLED_FONT, c), model.input_size)
        for c in model.charset
    ]
    print(
        f"  {'batch':>8s} {'cpu latency':>12s} {'gpu latency':>12s} "
        f"{'cpu rate':>12s} {'gpu rate':>12s} {'speedup':>8s}"
    )
    for batch in (1, 8, 16, 64, 256):
        g = np.stack([glyphs[i % len(glyphs)] for i in range(batch)])
        tc = median_time(lambda: cpu.classify(g), repeat, max(1, iters // 4))
        tg = median_time(lambda: gpu.classify(g), repeat, max(1, iters // 4))
        print(
            f"  {batch:>8d} {fmt_us(tc):>12s} {fmt_us(tg):>12s} "
            f"{fmt_rate(batch / tc):>12s} {fmt_rate(batch / tg):>12s} "
            f"{tc / tg:>7.1f}x"
        )


def bench_wgpu_end_to_end(repeat: int, iters: int) -> None:
    """End-to-end recognize() with backend='cpu' vs backend='wgpu'."""
    print("\n== end-to-end recognize(): cpu vs wgpu ==")
    print(
        f"  {'scenario':20s} {'cpu latency':>12s} {'gpu latency':>12s} "
        f"{'cpu rate':>12s} {'gpu rate':>12s} {'speedup':>8s}"
    )
    for scenario, font, text in E2E_LINES:
        if not scenario.startswith("tinycnn"):
            continue
        model_dir, font = FIXTURES[scenario]
        ocr_cpu = FixedFontOCR(model_path=model_dir, backend="cpu")
        try:
            ocr_gpu = FixedFontOCR(model_path=model_dir, backend="wgpu")
        except Exception as exc:
            print(f"  {scenario:20s} WGPU unavailable: {exc}")
            continue
        image = render_text(text, font)
        result = ocr_cpu.recognize(image)
        n_chars = len(result.text)
        t_cpu = median_time(lambda: ocr_cpu.recognize(image), repeat, max(1, iters // 2))
        t_gpu = median_time(lambda: ocr_gpu.recognize(image), repeat, max(1, iters // 2))
        print(
            f"  {scenario:20s} {fmt_us(t_cpu):>12s} {fmt_us(t_gpu):>12s} "
            f"{fmt_rate(n_chars / t_cpu):>12s} {fmt_rate(n_chars / t_gpu):>12s} "
            f"{t_cpu / t_gpu:>7.1f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=7, help="timed runs per measurement")
    parser.add_argument("--iters", type=int, default=100, help="calls per timed run")
    args = parser.parse_args()

    print(f"machine: {platform.processor() or platform.machine()} | "
          f"cpu_count={os.cpu_count()} | numpy={np.__version__}")

    bench_model_load(args.repeat)
    bench_engine_construction(args.repeat)
    bench_classifiers(args.repeat, args.iters)
    bench_end_to_end(args.repeat, args.iters)
    bench_wgpu_backends(args.repeat, args.iters)
    bench_wgpu_end_to_end(args.repeat, args.iters)


if __name__ == "__main__":
    main()
