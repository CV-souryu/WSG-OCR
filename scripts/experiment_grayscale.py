#!/usr/bin/env python3
"""P1 experiment: binary vs grayscale TinyCNN input.

Segmentation and Template keep the binary mask; this script tests whether
feeding the TinyCNN soft/grayscale glyphs (anti-aliasing, edge strength,
sub-pixel scaling) materially improves accuracy on confusable characters:

    日/曰  土/士  未/末  甲/申  0/O  1/I/l

Two identical TinyCNNs are trained from the same render pipeline, one on
``0/255`` masks and one on ``0..255`` grayscale glyphs, and evaluated on a
held-out set. The conclusion rule (from the plan): keep grayscale only if it
is materially better (>= 1 pt overall and >= 2 pt on the confusable pairs);
otherwise keep binary.

Usage:
    python scripts/experiment_grayscale.py [--epochs 20] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from fixedfontocr import defaults  # noqa: E402
from fixedfontocr.preprocess import normalize, normalize_grayscale  # noqa: E402
from tools.train import tinycnn_arch  # noqa: E402


CHARSET = list("日曰土士未末甲申0O1Il人入已己天夫")
CONFUSABLE_PAIRS = [("日", "曰"), ("土", "士"), ("未", "末"), ("甲", "申"), ("0", "O"), ("1", "I"), ("1", "l")]


def render_sample(
    font_path: Path,
    char: str,
    rng: np.random.Generator,
    _attempts: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one char and return (binary glyph, grayscale glyph)."""

    render_size = int(rng.integers(20, 31))
    font = ImageFont.truetype(str(font_path), render_size)
    bbox = font.getbbox(char)
    pad = 10
    w = bbox[2] - bbox[0] + pad * 2 + 4
    h = bbox[3] - bbox[1] + pad * 2 + 4
    bg = int(rng.integers(0, 71))
    img = Image.new("L", (max(1, w), max(1, h)), bg)
    draw = ImageDraw.Draw(img)
    fill = int(rng.integers(190, 256))
    dx, dy = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
    if rng.random() < 0.4:
        sh = int(rng.integers(1, 3))
        draw.text(
            (pad - bbox[0] + dx + sh, pad - bbox[1] + dy + sh),
            char,
            font=font,
            fill=int(rng.integers(30, 100)),
        )
    stroke = int(rng.integers(0, 3))
    draw.text(
        (pad - bbox[0] + dx, pad - bbox[1] + dy),
        char,
        font=font,
        fill=fill,
        stroke_width=stroke,
        stroke_fill=int(rng.integers(80, 180)) if stroke else None,
    )
    if rng.random() < 0.5:
        radius = float(rng.uniform(0.0, 1.6))
        if radius >= 0.05:
            img = img.filter(ImageFilter.GaussianBlur(radius))
    arr = np.asarray(img, dtype=np.uint8)
    threshold = int(rng.integers(100, 181))
    mask = arr >= threshold
    ys, xs = np.where(mask)
    if ys.size == 0:
        if _attempts < 3:
            return render_sample(font_path, char, rng, _attempts + 1)
        raise RuntimeError(f"no ink rendered for {char!r}")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    tight_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    tight_gray = arr[y0 : y1 + 1, x0 : x1 + 1]
    return normalize(tight_mask, 24), normalize_grayscale(tight_gray, 24)


def build_data(
    font_path: Path,
    samples_per_char: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xb: list[np.ndarray] = []
    xg: list[np.ndarray] = []
    y: list[int] = []
    for label, char in enumerate(CHARSET):
        for _ in range(samples_per_char):
            b, g = render_sample(font_path, char, rng)
            xb.append(b)
            xg.append(g)
            y.append(label)
    return (
        np.stack(xb),
        np.stack(xg),
        np.asarray(y, dtype=np.int64),
        np.asarray(CHARSET, dtype="<U8"),
    )


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[tinycnn_arch.TinyCNNBN, float]:
    torch.manual_seed(seed)
    model = tinycnn_arch.TinyCNNBN(len(CHARSET)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    n = x.shape[0]
    steps = max(1, n // batch_size)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for step in range(steps):
            idx = np.arange(step * batch_size, min(n, (step + 1) * batch_size))
            xb = torch.from_numpy(x[idx].astype(np.float32)[:, None] / 255.0).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach())
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(x.astype(np.float32)[:, None] / 255.0).to(device)
        ).argmax(1)
        acc = float((pred.cpu().numpy() == y).mean())
    return model, acc


def evaluate(
    model: tinycnn_arch.TinyCNNBN,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    with torch.no_grad():
        logits = model(
            torch.from_numpy(x.astype(np.float32)[:, None] / 255.0).to(device)
        )
        pred = logits.argmax(1).cpu().numpy()
    overall = float((pred == y).mean())
    per_pair: dict[str, float] = {}
    for a, b in CONFUSABLE_PAIRS:
        ids = {i for i, ch in enumerate(CHARSET) if ch in (a, b)}
        sel = np.array([i in ids for i in y])
        if sel.sum() == 0:
            continue
        per_pair[f"{a}/{b}"] = float((pred[sel] == y[sel]).mean())
    return overall, per_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-char", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="optional JSON results")
    args = parser.parse_args()

    font_path = defaults.resolve_font(defaults.FONT_PATH)
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"device: {device} | classes: {len(CHARSET)}")
    xb, xg, y, chars = build_data(font_path, args.samples_per_char, args.seed)
    n = y.size
    perm = np.random.default_rng(999).permutation(n)
    n_val = max(1, int(n * 0.15))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    print("training binary-input CNN ...")
    model_b, _ = train_model(
        xb[train_idx], y[train_idx], device, args.epochs, args.batch_size, args.seed
    )
    print("training grayscale-input CNN ...")
    model_g, _ = train_model(
        xg[train_idx], y[train_idx], device, args.epochs, args.batch_size, args.seed + 1
    )

    acc_b, pairs_b = evaluate(model_b, xb[val_idx], y[val_idx], device)
    acc_g, pairs_g = evaluate(model_g, xg[val_idx], y[val_idx], device)
    results = {
        "charset": "".join(CHARSET),
        "classes": len(CHARSET),
        "val_samples": int(n_val),
        "binary": {"overall": acc_b, "confusable_pairs": pairs_b},
        "grayscale": {"overall": acc_g, "confusable_pairs": pairs_g},
        "decision": None,
    }
    pair_gain = float(
        np.mean([pairs_g[k] - pairs_b[k] for k in pairs_b if k in pairs_g])
        if pairs_b
        else 0.0
    )
    material = acc_g >= acc_b + 0.01 and pair_gain >= 0.02
    results["decision"] = "keep grayscale" if material else "keep binary"

    print("\n== results ==")
    print(f"  binary    overall={acc_b:.4f}")
    for k, v in pairs_b.items():
        print(f"    {k}: {v:.4f}")
    print(f"  grayscale overall={acc_g:.4f}")
    for k, v in pairs_g.items():
        print(f"    {k}: {v:.4f}")
    print(f"  mean pair gain: {pair_gain:+.4f}")
    print(f"  decision: {results['decision']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
