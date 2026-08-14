#!/usr/bin/env python3
"""Train the TinyCNN classifier on synthetic + real glyph data.

The model uses Conv + BatchNorm + ReLU blocks. BatchNorm is training-only:
``tools/train/export_model.py`` folds it into the convolution weights so the
runtime (numpy / WGSL) stays Conv + ReLU.

Usage:
    python tools/dataset/generate_font_dataset.py font.ttf synth.npz ...
    python tools/dataset/collect_real_samples.py real/ real.npz
    python tools/train/train.py --synthetic synth.npz --real real.npz \
        --output model.pth --device auto
    python tools/train/export_model.py model.pth --output runtime_model/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from tools.train import tinycnn_arch  # noqa: E402


def load_npz(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str], str | None, str]:
    data = np.load(path)
    chars = [str(c) for c in data["chars"]]
    font_sha256 = None
    if "font_sha256" in data.files:
        arr = data["font_sha256"]
        font_sha256 = str(arr.item() if arr.ndim == 0 else arr[0])
    input_mode = "binary"
    if "input_mode" in data.files:
        arr = data["input_mode"]
        input_mode = str(arr.item() if arr.ndim == 0 else arr[0])
    return data["x"], data["y"], chars, font_sha256, input_mode


def build_dataset(
    synthetic: str | Path, real: str | Path | None
) -> tuple[np.ndarray, np.ndarray, list[str], str, str]:
    """Merge synthetic + real npz files into a common charset.

    All datasets that record a font must come from the same registered font;
    a synthetic dataset without ``font_sha256`` is rejected because it cannot
    prove that training used only ``fonts/``.
    """

    datasets: list[
        tuple[np.ndarray, np.ndarray, list[str], str | None, str]
    ] = []
    syn = load_npz(synthetic)
    if syn[3] is None:
        raise SystemExit(
            f"{synthetic} has no font_sha256; regenerate it with "
            f"tools/dataset/generate_font_dataset.py"
        )
    datasets.append(syn)
    if real is not None:
        datasets.append(load_npz(real))

    font_shas = {sha for _, _, _, sha, _ in datasets if sha is not None}
    if len(font_shas) > 1:
        raise SystemExit(
            "training data mixes multiple fonts; font-family augmentation "
            f"is forbidden (got {sorted(font_shas)})"
        )

    input_modes = {mode for _, _, _, _, mode in datasets}
    if len(input_modes) > 1:
        raise SystemExit(
            "training data mixes binary and soft glyphs; regenerate datasets "
            f"with one input_mode (got {sorted(input_modes)})"
        )

    charset: list[str] = []
    char_id: dict[str, int] = {}
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x, y, chars, _, _ in datasets:
        if x.shape[1:] != (24, 24):
            raise SystemExit(f"expected 24x24 glyphs, got {x.shape[1:]}")
        for ch in chars:
            if ch not in char_id:
                char_id[ch] = len(charset)
                charset.append(ch)
        mapping = np.array([char_id[ch] for ch in chars])
        xs.append(x)
        ys.append(mapping[y])
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        charset,
        datasets[0][3],
        datasets[0][4],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", required=True, help="npz from generate_font_dataset.py")
    parser.add_argument("--real", help="npz from collect_real_samples.py")
    parser.add_argument("--output", default="model.pth", help="checkpoint path")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="existing checkpoint to fine-tune from (same architecture/classes)",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device: cpu, mps, cuda, or auto (default: auto)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    print(f"device: {device}")
    x, y, charset, font_sha256, input_mode = build_dataset(
        args.synthetic, args.real
    )
    n = x.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_split))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    print(
        f"data: {n} samples, {len(charset)} classes "
        f"(train {len(train_idx)}, val {len(val_idx)})"
    )

    model = tinycnn_arch.TinyCNNBN(len(charset)).to(device)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt.get("architecture") != "tinycnn_bn":
            raise SystemExit(
                f"unsupported resume architecture: {ckpt.get('architecture')}"
            )
        ck_charset = [str(c) for c in ckpt["charset"]]
        if ck_charset != charset:
            raise SystemExit(
                "resume checkpoint charset differs from the merged dataset "
                f"({len(ck_charset)} vs {len(charset)} classes)"
            )
        model.load_state_dict(ckpt["state_dict"])
        print(f"resumed from {args.resume}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    steps = max(1, len(train_idx) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for step in range(steps):
            idx = train_idx[step * args.batch_size : (step + 1) * args.batch_size]
            xb = torch.from_numpy(
                x[idx].astype(np.float32)[:, None] / 255.0
            ).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            pred = model(
                torch.from_numpy(
                    x_val.astype(np.float32)[:, None] / 255.0
                ).to(device)
            ).argmax(1)
            acc = float((pred.cpu().numpy() == y_val).mean())
        print(f"epoch {epoch + 1:>2}: loss {total / steps:.4f}  val_acc {acc:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "state_dict": state_dict,
            "charset": charset,
            "architecture": "tinycnn_bn",
            "input_size": 24,
            "version": 1,
            "font_sha256": font_sha256,
            "input_mode": input_mode,
        },
        out,
    )
    print(f"wrote checkpoint ({len(charset)} classes) to {out}")


if __name__ == "__main__":
    main()
