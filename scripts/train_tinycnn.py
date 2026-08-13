#!/usr/bin/env python3
"""Train the TinyCNN classifier on synthetic glyphs from a TTF/OTF font.

The trained model is written in the same model directory format used by
``FixedFontOCR`` (config.json + charset.txt + weights.bin), so the numpy CPU
forward and a future WGPU backend can consume it directly.

Usage:
    python scripts/train_tinycnn.py font.ttf model_dir --charset charset.txt

Requires PyTorch (training only; runtime inference is pure numpy).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    sys.exit("PyTorch is required for training: pip install torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fixedfontocr.model import write_cnn_model  # noqa: E402
from fixedfontocr.preprocess import normalize  # noqa: E402
from fixedfontocr.types import Profile  # noqa: E402


def read_charset(path: str | None, font: str, default: str) -> list[str]:
    if path:
        chars: list[str] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for ch in line:
                if not ch.isspace():
                    chars.append(ch)
        if not chars:
            raise SystemExit(f"charset file {path} contains no characters")
        return chars
    from PIL import ImageFont

    font_obj = ImageFont.truetype(font, 24)
    return [ch for ch in default if font_obj.getlength(ch) > 0]


def make_dataset(
    font: str,
    charset: list[str],
    samples_per_char: int,
    rng: np.random.Generator,
    size_min: int,
    size_max: int,
    threshold_min: int = 100,
    threshold_max: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """Render ``charset`` at random sizes/thresholds into 24x24 float tensors."""

    from PIL import Image, ImageDraw, ImageFont

    profile = Profile(name="train", use_grayscale=True, grayscale_threshold=140)
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    for label, char in enumerate(charset):
        for _ in range(samples_per_char):
            size = int(rng.integers(size_min, size_max + 1))
            font_obj = ImageFont.truetype(font, size)
            bbox = font_obj.getbbox(char)
            pad = 8
            w = bbox[2] - bbox[0] + pad * 2
            h = bbox[3] - bbox[1] + pad * 2
            img = Image.new("RGB", (max(1, w), max(1, h)), (0, 0, 0))
            ImageDraw.Draw(img).text(
                (pad - bbox[0], pad - bbox[1]), char, font=font_obj, fill=(255, 255, 255)
            )
            gray = np.asarray(img).astype(np.float32) @ np.array(
                [0.299, 0.587, 0.114], dtype=np.float32
            )
            threshold = float(rng.integers(threshold_min, threshold_max + 1))
            mask = gray >= threshold
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            tight = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            glyph = normalize(tight, 24).astype(np.float32) / 255.0
            # Random sub-pixel segmentation jitter: shift by -1..1 px.
            dx = int(rng.integers(-1, 2))
            dy = int(rng.integers(-1, 2))
            if dx or dy:
                shifted = np.zeros_like(glyph)
                y0, y1 = max(0, dy), min(24, 24 + dy)
                x0, x1 = max(0, dx), min(24, 24 + dx)
                sy0, sy1 = max(0, -dy), min(24, 24 - dy)
                sx0, sx1 = max(0, -dx), min(24, 24 - dx)
                shifted[y0:y1, x0:x1] = glyph[sy0:sy1, sx0:sx1]
                glyph = shifted
            x_list.append(glyph[None, :, :])
            y_list.append(label)
    x = np.stack(x_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return x, y


class TinyCNN(nn.Module):
    """PyTorch mirror of the numpy forward in fixedfontocr/cnn.py."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.dw1 = nn.Conv2d(8, 8, 3, padding=1, groups=8)
        self.pw1 = nn.Conv2d(8, 16, 1, stride=2)
        self.dw2 = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.pw2 = nn.Conv2d(16, 32, 1, stride=2)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.dw1(x))
        x = torch.relu(self.pw1(x))
        x = torch.relu(self.dw2(x))
        x = torch.relu(self.pw2(x))
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def export_weights(model: nn.Module) -> dict[str, np.ndarray]:
    def arr(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    return {
        "conv1.weight": arr(model.conv1.weight),
        "conv1.bias": arr(model.conv1.bias),
        "dw1.weight": arr(model.dw1.weight).reshape(8, 3, 3),
        "dw1.bias": arr(model.dw1.bias),
        "pw1.weight": arr(model.pw1.weight).reshape(16, 8),
        "pw1.bias": arr(model.pw1.bias),
        "dw2.weight": arr(model.dw2.weight).reshape(16, 3, 3),
        "dw2.bias": arr(model.dw2.bias),
        "pw2.weight": arr(model.pw2.weight).reshape(32, 16),
        "pw2.bias": arr(model.pw2.bias),
        "fc.weight": arr(model.fc.weight),
        "fc.bias": arr(model.fc.bias),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("font", help="path to a .ttf/.otf/.ttc font")
    parser.add_argument("output", help="output model directory")
    parser.add_argument("--charset", help="text file listing characters")
    parser.add_argument("--samples-per-char", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--render-size-min", type=int, default=24)
    parser.add_argument("--render-size-max", type=int, default=36)
    parser.add_argument("--threshold-min", type=int, default=100)
    parser.add_argument("--threshold-max", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    default_charset = (
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "获得金币消耗数量提示确定取消返回攻击防御生命法力"
        "+-×÷%.,:;!?()[]{}「」"
    )
    charset = read_charset(args.charset, args.font, default_charset)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x, y = make_dataset(
        args.font,
        charset,
        args.samples_per_char,
        rng,
        args.render_size_min,
        args.render_size_max,
        args.threshold_min,
        args.threshold_max,
    )
    n = x.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_split))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    x_val, y_val = x[val_idx], y[val_idx]
    x_tr, y_tr = x[train_idx], y[train_idx]
    print(
        f"data: {n} samples, {len(charset)} classes "
        f"(train {len(train_idx)}, val {len(val_idx)})"
    )

    torch.set_num_threads(max(1, min(8, __import__("os").cpu_count() or 4)))
    model = TinyCNN(len(charset))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    steps = max(1, len(train_idx) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for step in range(steps):
            idx = train_idx[step * args.batch_size : (step + 1) * args.batch_size]
            xb = torch.from_numpy(x[idx])
            yb = torch.from_numpy(y[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x_val)).argmax(1).numpy()
        acc = float(np.mean(pred == y_val))
        print(f"epoch {epoch + 1:>2}: loss {total / steps:.4f}  val_acc {acc:.3f}")

    out = Path(args.output)
    write_cnn_model(out, charset, export_weights(model))
    print(f"wrote TinyCNN model ({len(charset)} classes) to {out}")


if __name__ == "__main__":
    main()
