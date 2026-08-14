#!/usr/bin/env python3
"""CPU benchmark suite for the frozen NumPy OCR reference (P7).

Records, per stage, per CNN batch size and per charset size:

* OCR pipeline stages: color mask, line detection, connected components,
  candidate generation, normalize, template match, TinyCNN, visual DP and
  end-to-end recognize();
* TinyCNN batch classify() for N = 1/8/16/32/64/128;
* template + CNN classification over digit (~10), small (~100), CJK (~3000)
  and CJK (~7000) charsets (the large sets are sampled from the registered
  font's Unicode coverage, not committed as files);
* the optimized stride-2 forward vs the full-then-slice reference
  (max abs error and argmax identity).

Every timing is reported as median and p95 over many calls, never a single
run.

Usage:
    python tools/benchmark/cpu_benchmark.py [--output cpu_benchmark.json]
    python tools/benchmark/cpu_benchmark.py --fast   # smoke run for tests
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from fixedfontocr import defaults  # noqa: E402
from fixedfontocr.backends import CPUBackend  # noqa: E402
from fixedfontocr.classifier import TemplateClassifier  # noqa: E402
from fixedfontocr.fontgen import build_templates, render_glyph  # noqa: E402
from fixedfontocr.frontend import extract_frontend  # noqa: E402
from fixedfontocr.model import load_model  # noqa: E402
from fixedfontocr.reference_cnn import reference_forward  # noqa: E402
from fixedfontocr.preprocess import (  # noqa: E402
    Component,
    Segment,
    _connected_components,
    compute_normalize_spec,
    find_lines,
    glyph_normalize_geometry,
    normalize,
)
from fixedfontocr.scorer import SegmentScorer  # noqa: E402
from fixedfontocr.segmentation import (  # noqa: E402
    build_candidates,
    connected_components,
    decode,
    geometry_score,
    segment_line,
)
from fixedfontocr.types import default_profile  # noqa: E402


def median_p95(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    ordered = sorted(samples)
    med = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return med, p95


def timed_samples(fn, calls: int) -> tuple[float, float]:
    """Return (median, p95) per-call wall time in seconds."""
    fn()  # warm-up (allocators, BLAS, template filter caches)
    samples: list[float] = []
    for _ in range(calls):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e9)
    return median_p95(samples)


def _normalize_template_glyph(
    mask: np.ndarray,
    spec,
) -> np.ndarray:
    """Goal 3 frame: same normalization templates use."""
    h, w = mask.shape
    cand = Component(mask=mask, x=0, y=0, w=w, h=h)
    baseline_offset, scale = glyph_normalize_geometry(cand, spec, 24)
    return normalize(
        mask,
        24,
        baseline_offset=baseline_offset,
        scale=scale,
        baseline_row=spec.baseline_row,
    )


def render_line(
    text: str,
    font_path: Path,
    size: int = 32,
) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), size)
    bbox = font.getbbox(text)
    pad = 8
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGB", (max(1, w), max(1, h)), (0, 0, 0))
    ImageDraw.Draw(img).text(
        (pad - bbox[0], pad - bbox[1]), text, font=font, fill=(255, 255, 255)
    )
    return np.asarray(img, dtype=np.uint8)


def cjk_charset(
    font_path: Path,
    size: int,
    seed: int = 0,
) -> list[str]:
    """Sample ``size`` CJK ideographs the registered font can render."""
    from fontTools.ttLib import TTFont

    font = TTFont(font_path)
    try:
        cmap = set(font.getBestCmap() or {})
    finally:
        font.close()
    cjk = sorted(
        cp
        for cp in cmap
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
    )
    if len(cjk) < size:
        raise RuntimeError(
            f"font covers {len(cjk)} CJK ideographs, need {size}"
        )
    rng = np.random.default_rng(seed)
    picked = rng.choice(cjk, size=size, replace=False)
    return [chr(int(cp)) for cp in sorted(picked)]


def make_cnn_weights(num_classes: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Real TinyCNN conv weights + a random top layer of ``num_classes``."""
    fixture = Path(ROOT) / "tests" / "fixtures" / "cnn_digits"
    if (fixture / "config.json").exists():
        base = load_model(fixture).weights
    else:
        base = {
            "conv1.weight": np.zeros((8, 1, 3, 3), dtype=np.float32),
            "conv1.bias": np.zeros(8, dtype=np.float32),
            "dw1.weight": np.zeros((8, 3, 3), dtype=np.float32),
            "dw1.bias": np.zeros(8, dtype=np.float32),
            "pw1.weight": np.zeros((16, 8), dtype=np.float32),
            "pw1.bias": np.zeros(16, dtype=np.float32),
            "dw2.weight": np.zeros((16, 3, 3), dtype=np.float32),
            "dw2.bias": np.zeros(16, dtype=np.float32),
            "pw2.weight": np.zeros((32, 16), dtype=np.float32),
            "pw2.bias": np.zeros(32, dtype=np.float32),
        }
    rng = np.random.default_rng(seed)
    weights = dict(base)
    weights["fc.weight"] = rng.standard_normal(
        (num_classes, 32), dtype=np.float32
    )
    weights["fc.bias"] = rng.standard_normal(num_classes, dtype=np.float32)
    return weights


def bench_ocr_stages(
    ocr,
    image: np.ndarray,
    profile,
    calls: int,
) -> dict[str, dict[str, float]]:
    frontend = extract_frontend(image, profile)
    mask = frontend.binary_mask
    lines = find_lines(mask, profile)
    line = lines[0]
    comps = connected_components(line)
    cands = build_candidates(comps, profile)
    cand_segments = [c.segment for c in cands]
    scorer: SegmentScorer = ocr._scorer
    scores = scorer.score(cand_segments, soft=frontend.soft_foreground)
    for cand, score in zip(cands, scores):
        cand.score = score
        cand.geometry = geometry_score(cand, comps, profile)

    stages = {
        "frontend": (lambda: extract_frontend(image, profile), 200),
        "mask": (lambda: profile.color_mask(image), 200),
        "soft_foreground": (lambda: profile.soft_foreground(image), 200),
        "line_detection": (lambda: find_lines(mask, profile), 200),
        "connected_components": (
            lambda: _connected_components(line.mask),
            200,
        ),
        "candidate_generation": (
            lambda: build_candidates(comps, profile),
            200,
        ),
        "normalize": (
            lambda: np.stack([normalize(s.mask, 24) for s in cand_segments]),
            200,
        ),
        "normalize_soft": (
            lambda: frontend.soft_glyph_batch(cand_segments, 24),
            200,
        ),
        "classifier": (
            lambda: scorer.score(cand_segments, soft=frontend.soft_foreground),
            50,
        ),
        "decoder": (lambda: decode(cands, len(comps)), 200),
        "recognize_total": (lambda: ocr.recognize(image), 30),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (fn, n) in stages.items():
        med, p95 = timed_samples(fn, min(calls, n))
        out[name] = {"median": med, "p95": p95}
    return out


def bench_cnn_batches(weights, calls: int) -> dict[str, dict[str, float]]:
    backend = CPUBackend(weights, input_size=24)
    rng = np.random.default_rng(7)
    out: dict[str, dict[str, float]] = {}
    for n in (1, 8, 16, 32, 64, 128):
        glyphs = (rng.random((n, 24, 24)) > 0.5).astype(np.uint8) * 255
        med, p95 = timed_samples(lambda g=glyphs: backend.classify(g), calls)
        out[str(n)] = {"median": med, "p95": p95}
    return out


def bench_charsets(
    font_path: Path,
    sizes: list[int],
    calls: int,
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for size in sizes:
        if size <= 200:
            charset = list("0123456789")[: min(10, size)]
            alphabet = (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                "+-×÷%.,:;!?()[]{}「」"
            )
            while len(charset) < size:
                charset.append(alphabet[len(charset) % len(alphabet)])
        else:
            charset = cjk_charset(font_path, size)
        chars, templates = build_templates(
            font_path, charset, render_size=32
        )
        spec = compute_normalize_spec(font_path, charset, 24, 32)
        clf = TemplateClassifier(
            templates, charset, candidate_filter=False, normalize_spec=spec
        )
        glyph_batch = np.stack(
            [
                _normalize_template_glyph(
                    render_glyph(font_path, ch, 32), spec
                )
                for ch in charset[: min(64, len(charset))]
            ]
        )
        med_t, p95_t = timed_samples(
            lambda b=glyph_batch: clf.match_batch(b), calls
        )
        weights = make_cnn_weights(size)
        med_c, p95_c = timed_samples(
            lambda: CPUBackend(weights, input_size=24).classify(glyph_batch),
            max(1, calls // 2),
        )
        out[str(size)] = {
            "template_batch": {"median": med_t, "p95": p95_t},
            "cnn_batch": {"median": med_c, "p95": p95_c},
        }
    return out


def bench_optimized_vs_reference() -> dict[str, float | bool]:
    """Re-check the P2 acceptance: max error and argmax identity."""
    rng = np.random.default_rng(17)
    weights = {
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
        "fc.weight": rng.standard_normal((64, 32), dtype=np.float32),
        "fc.bias": rng.standard_normal(64, dtype=np.float32),
    }
    x = np.random.default_rng(23).random((4, 1, 24, 24), dtype=np.float32)
    from fixedfontocr.cnn import forward

    ref = reference_forward(x, weights)
    got = forward(x, weights)
    return {
        "max_error": float(np.abs(ref - got).max()),
        "argmax_match": bool(np.array_equal(ref.argmax(1), got.argmax(1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write results as JSON")
    parser.add_argument(
        "--calls",
        type=int,
        default=50,
        help="timed calls per measurement (smoke tests should use ~5)",
    )
    parser.add_argument(
        "--max-charset",
        type=int,
        default=7000,
        help="largest CJK charset to benchmark",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="smoke-run with tiny charsets and few calls",
    )
    args = parser.parse_args()

    calls = 5 if args.fast else args.calls
    sizes = [10, 100] if args.fast else [10, 100, 3000, args.max_charset]
    sizes = sorted(set(sizes))

    font_path = defaults.resolve_font(defaults.FONT_PATH)
    profile = default_profile()
    ocr = __import__(
        "fixedfontocr", fromlist=["FixedFontOCR"]
    ).FixedFontOCR(model_path=defaults.MODEL_PATH, backend="cpu")
    image = render_line("获得金币1000数量提示", font_path)

    results: dict = {
        "machine": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "font_sha256": defaults.compute_font_sha256(font_path),
        "ocr_stages": bench_ocr_stages(ocr, image, profile, calls),
        "cnn_batch": bench_cnn_batches(make_cnn_weights(64), calls),
        "charsets": bench_charsets(font_path, sizes, calls),
        "optimized_vs_reference": bench_optimized_vs_reference(),
    }

    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
