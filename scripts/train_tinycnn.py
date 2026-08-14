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
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    sys.exit("PyTorch is required for training: pip install torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fixedfontocr import defaults  # noqa: E402
from fixedfontocr.defaults import resolve_font  # noqa: E402
from fixedfontocr.model import write_cnn_model  # noqa: E402
from fixedfontocr.preprocess import compute_normalize_spec  # noqa: E402
from tools.dataset.generate_font_dataset import generate_dataset  # noqa: E402


def read_charset(path: str | None, default: str) -> list[str]:
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
    # Missing glyphs are reported by make_dataset instead of being silently
    # filtered out here.
    return list(default)


def make_dataset(
    font: str,
    charset: list[str],
    samples_per_char: int,
    rng: np.random.Generator,
    size_min: int,
    size_max: int,
    threshold_min: int = 100,
    threshold_max: int = 180,
    soft: bool = False,
    supersample_min: int = 2,
    supersample_max: int = 3,
    downsample: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    """Render ``charset`` through the Goal 7 low-res domain into float tensors.

    Delegates to :func:`tools.dataset.generate_font_dataset.generate_dataset`
    so this legacy training entry point gets the same 10-18 px supersampled
    render, bilinear/area-like downsample, sub-pixel offset, alpha,
    brightness, background blend, blur, outline and scale augmentation as
    the primary pipeline.
    """

    with tempfile.TemporaryDirectory() as td:
        npz_path = Path(td) / "synthetic.npz"
        generate_dataset(
            font,
            charset,
            samples_per_char,
            npz_path,
            seed=int(rng.integers(0, 2**31 - 1)),
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            soft=soft,
            render_size_min=size_min,
            render_size_max=size_max,
            supersample_min=supersample_min,
            supersample_max=supersample_max,
            downsample=downsample,
        )
        data = np.load(npz_path)
        x = data["x"].astype(np.float32)[:, None, :, :] / 255.0
        y = data["y"]
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
    parser.add_argument(
        "font",
        nargs="?",
        default=str(defaults.FONT_PATH),
        help="path to a registered .ttf/.otf font under fonts/ "
        "(default: bundled game font)",
    )
    parser.add_argument("output", help="output model directory")
    parser.add_argument(
        "--charset",
        default=str(defaults.CHARSET_PATH),
        help="text file listing characters (default: charsets/sets/combined.txt)",
    )
    parser.add_argument("--samples-per-char", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--render-size-min", type=int, default=10)
    parser.add_argument("--render-size-max", type=int, default=18)
    parser.add_argument("--supersample-min", type=int, default=2)
    parser.add_argument("--supersample-max", type=int, default=3)
    parser.add_argument(
        "--downsample",
        choices=("bilinear", "area", "random"),
        default="random",
        help="degradation used when resizing to the final screenshot size",
    )
    parser.add_argument("--threshold-min", type=int, default=100)
    parser.add_argument("--threshold-max", type=int, default=180)
    parser.add_argument(
        "--soft",
        action="store_true",
        help="train on 0..255 soft foreground glyphs (Goal 2 Visual Frontend)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    default_charset = defaults.read_charset()
    font = resolve_font(args.font)
    charset = read_charset(args.charset, default_charset)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x, y = make_dataset(
        font,
        charset,
        args.samples_per_char,
        rng,
        args.render_size_min,
        args.render_size_max,
        args.threshold_min,
        args.threshold_max,
        soft=args.soft,
        supersample_min=args.supersample_min,
        supersample_max=args.supersample_max,
        downsample=args.downsample,
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
    normalize_spec = compute_normalize_spec(font, charset, 24, 32).to_dict()
    write_cnn_model(
        out,
        charset,
        export_weights(model),
        font_path=font,
        input_mode="soft" if args.soft else "binary",
        normalize_spec=normalize_spec,
    )
    print(f"wrote TinyCNN model ({len(charset)} classes) to {out}")


if __name__ == "__main__":
    main()
