#!/usr/bin/env python3
"""Tune Goal 10 visual-scoring weights + calibration on a real test set.

Usage:
    python tools/train/tune_visual.py \
        --samples data/cn/real_game.npz \
        --model model/game_cn \
        --out model/game_cn/config.json

The script grids over ``visual_weights`` with the pure Goal 10 fusion
(``score_fused``), keeps the weights with the best real-set top-1 accuracy,
then fits a smoothed monotone piecewise-linear calibration from unified
visual scores to observed correctness. Results are written back into the
model config so runtime decoding and public confidence use the same values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fixedfontocr.model import load_model  # noqa: E402
from fixedfontocr.preprocess import (  # noqa: E402
    Component,
    Segment,
    glyph_normalize_geometry,
)
from fixedfontocr.scorer import (  # noqa: E402
    SegmentScorer,
    VisualWeights,
)
from fixedfontocr.segmentation import geometry_score  # noqa: E402
from fixedfontocr.types import VisualCandidate, default_profile  # noqa: E402


def _load_samples(path: Path):
    data = np.load(path)
    chars = [str(c) for c in data["chars"]]
    return data["x"], data["y"], chars


def _geometry_for(
    seg: Segment,
    true_id: int,
    scorer: SegmentScorer,
) -> float:
    cand = VisualCandidate(
        start=0,
        end=1,
        components=(0,),
        atoms=(0, 1),
        segment=seg,
    )
    geom = None
    if scorer.normalize_spec is not None:
        geom = glyph_normalize_geometry(
            seg, scorer.normalize_spec, scorer.input_size
        )
    return geometry_score(
        cand,
        [seg],
        default_profile(),
        geometry=scorer.geometry,
        char_id=true_id,
        normalize_geometry=geom,
    )


def _bin_calibration(
    visual: np.ndarray,
    correct: np.ndarray,
    bins: int,
) -> list[list[float]]:
    """Smoothed monotone piecewise-linear calibration from real correctness."""

    order = np.argsort(visual, kind="stable")
    visual_sorted = visual[order]
    correct_sorted = correct[order].astype(np.float64)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    n = len(visual_sorted)
    if n == 0:
        return [[0.0, 0.0], [1.0, 1.0]]
    edges = np.linspace(0, n, bins + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        block = slice(a, b)
        x = float(np.mean(visual_sorted[block]))
        hits = int(correct_sorted[block].sum())
        count = b - a
        # Laplace-smoothed observed accuracy so a perfect tiny test set does
        # not map every score to 1.0.
        p = (hits + 1.0) / (count + 2.0)
        points.append((x, p))
    points.append((1.0, 1.0))
    # Enforce monotonicity: observed bins can jitter, but the runtime
    # calibration must never lower confidence as visual score increases.
    mono: list[tuple[float, float]] = []
    for x, y in points:
        if not mono or y >= mono[-1][1]:
            mono.append((x, y))
        else:
            mono.append((x, mono[-1][1]))
    return [[float(x), float(y)] for x, y in mono]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=ROOT / "data" / "cn" / "real_game.npz",
        help="real-screenshot npz (x/y/chars) collected by collect_real_samples.py",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "model" / "game_cn",
        help="hybrid model directory to tune",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="config.json to write (default: <model>/config.json)",
    )
    parser.add_argument("--calibration-bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = load_model(args.model)
    if model.classifier != "hybrid":
        raise SystemExit("Goal 10 tuning requires a hybrid model")
    scorer = SegmentScorer(model)

    x, y, chars = _load_samples(args.samples)
    model_ids: list[int] = []
    segments: list[Segment] = []
    geometry_scores: list[float] = []
    for i in range(len(x)):
        ch = chars[int(y[i])]
        if ch not in model.charset:
            print(f"skip sample {i}: {ch!r} not in model charset")
            continue
        seg = Segment(mask=x[i] > 0, x=0, y=0, w=24, h=24)
        segments.append(seg)
        true_id = model.charset.index(ch)
        model_ids.append(true_id)
        geometry_scores.append(_geometry_for(seg, true_id, scorer))
    if not segments:
        raise SystemExit("no usable samples")

    allowed = set(model_ids)
    cnn_grid = (0.30, 0.45, 0.60)
    template_grid = (0.60, 0.45, 0.30)
    geometry_grid = (0.05, 0.10, 0.20)
    best: tuple[float, VisualWeights, list] | None = None
    for cnn_w in cnn_grid:
        for template_w in template_grid:
            for geometry_w in geometry_grid:
                if cnn_w + template_w > 0.9 + 1e-9:
                    continue
                weights = VisualWeights(
                    cnn=cnn_w,
                    template=template_w,
                    geometry=geometry_w,
                )
                scores = scorer.score_fused(
                    segments,
                    allowed,
                    visual_weights=weights,
                )
                pred = np.asarray([s.char_id for s in scores], dtype=np.int64)
                truth = np.asarray(model_ids, dtype=np.int64)
                correct = pred == truth
                acc = float(correct.mean())
                visual = np.asarray(
                    [
                        float(
                            np.clip(
                                s.visual_score
                                + weights.geometry * geometry_scores[k],
                                0.0,
                                1.0,
                            )
                        )
                        for k, s in enumerate(scores)
                    ],
                    dtype=np.float64,
                )
                # Prefer weights that put more confidence on correct samples.
                margin = float(visual[correct].mean()) if correct.any() else 0.0
                key = (acc, margin, cnn_w, template_w, geometry_w)
                if best is None or key > best[0]:
                    best = (key, weights, scores)

    assert best is not None
    _, weights, scores = best
    pred = np.asarray([s.char_id for s in scores], dtype=np.int64)
    truth = np.asarray(model_ids, dtype=np.int64)
    correct = pred == truth
    visual = np.asarray(
        [
            float(
                np.clip(
                    s.visual_score
                    + weights.geometry * geometry_scores[k],
                    0.0,
                    1.0,
                )
            )
            for k, s in enumerate(scores)
        ],
        dtype=np.float64,
    )
    calibration = _bin_calibration(
        visual, correct, max(2, args.calibration_bins)
    )

    out = args.out or (args.model / "config.json")
    config = json.loads(out.read_text(encoding="utf-8"))
    config["visual_weights"] = {
        "cnn": weights.cnn,
        "template": weights.template,
        "geometry": weights.geometry,
    }
    config["visual_calibration"] = calibration
    out.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    model_json = args.model / "model.json"
    if model_json.exists():
        model_json.write_text(
            json.dumps(config, indent=4) + "\n", encoding="utf-8"
        )
    print(
        f"tuned on {len(scores)} real samples: "
        f"accuracy {correct.mean():.3f}, weights {weights}, "
        f"calibration {len(calibration)} points -> {out}"
    )


if __name__ == "__main__":
    main()
