#!/usr/bin/env python3
"""CPU experiment: CNN + BiGRU + CTC for the ship word lexicon.

This is deliberately a separate experiment model.  The frozen 24x24
character model and its NumPy/WGPU format are not changed here.  The input is
the complete crop (two channels: soft foreground and binary mask), while CTC
emits the visible character sequence.  A circular lexicon matcher then maps
``suffix + prefix`` scroll windows back to one of the registered ship terms.

The synthetic renderer uses only a registered font under ``fonts/``.  It
produces three domains: full words, ordinary substring crops, and horizontal
scroll windows that cross the end/start boundary of a word.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from fixedfontocr.defaults import compute_font_sha256, resolve_font  # noqa: E402
from fixedfontocr.types import default_profile  # noqa: E402


WIDTH = 128
HEIGHT = 30
SCROLL_GAP = 4


def load_terms(path: str | Path) -> list[str]:
    terms = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not terms:
        raise ValueError(f"no terms in {path}")
    if len(set(terms)) != len(terms):
        raise ValueError(f"duplicate terms in {path}")
    return terms


def validate_font_glyphs(font_path: Path, terms: Iterable[str]) -> None:
    """Fail explicitly if the registered font has a missing dictionary glyph."""

    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:  # pragma: no cover - project environments include it
        raise RuntimeError("fontTools is required for font glyph validation") from exc
    font = TTFont(str(font_path), lazy=False)
    cmap = {cp for table in font["cmap"].tables for cp in table.cmap}
    missing = sorted({ch for term in terms for ch in term if ord(ch) not in cmap})
    if missing:
        raise ValueError(
            f"registered font {font_path} is missing dictionary glyphs: {missing}"
        )


def vocabulary(terms: list[str]) -> tuple[list[str], dict[str, int]]:
    chars: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for ch in term:
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
    return chars, {ch: i for i, ch in enumerate(chars)}


def _visible_sample(term: str, rng: np.random.Generator) -> tuple[str, str]:
    """Return ``(render_text, ctc_text)`` for one word-domain sample."""

    n = len(term)
    mode = str(rng.choice(("full", "full", "partial", "wrap")))
    if n < 3:
        mode = "full"
    if mode == "full":
        return term, term
    if mode == "partial":
        start = int(rng.integers(0, max(1, n - 1)))
        end = int(rng.integers(start + 2, n + 1))
        text = term[start:end]
        return text, text

    # A horizontal marquee can show the tail followed by the head.  The
    # blank gap is part of the image geometry but not part of the CTC label.
    left_len = int(rng.integers(2, min(6, n) + 1))
    right_len = int(rng.integers(2, min(6, n) + 1))
    left = term[-left_len:]
    right = term[:right_len]
    return left + (" " * SCROLL_GAP) + right, left + right


def render_word(
    text: str,
    font_path: Path,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render one crop as ``uint8 [2, 30, 128]`` soft+binary channels."""

    supersample = int(rng.integers(2, 4))
    font_px = int(rng.integers(12, 19))
    bg = int(rng.integers(55, 116))
    fill = int(rng.integers(220, 256))
    stroke = int(rng.integers(0, 2))

    # Fit long dictionary entries without changing the fixed crop geometry.
    while True:
        font_obj = ImageFont.truetype(str(font_path), font_px * supersample)
        bbox = font_obj.getbbox(text, stroke_width=stroke * supersample)
        text_w = bbox[2] - bbox[0]
        if text_w <= (WIDTH - 8) * supersample or font_px <= 8:
            break
        font_px -= 1

    canvas = Image.new(
        "L", (WIDTH * supersample, HEIGHT * supersample), color=bg
    )
    draw = ImageDraw.Draw(canvas)
    text_h = bbox[3] - bbox[1]
    jitter_x = int(rng.integers(-2, 3)) * supersample
    jitter_y = int(rng.integers(-1, 2)) * supersample
    x = (WIDTH * supersample - text_w) // 2 - bbox[0] + jitter_x
    y = (HEIGHT * supersample - text_h) // 2 - bbox[1] + jitter_y
    draw.text(
        (x, y),
        text,
        font=font_obj,
        fill=fill,
        stroke_width=stroke * supersample,
        stroke_fill=max(100, fill - int(rng.integers(20, 80))),
    )

    if rng.random() < 0.35:
        canvas = canvas.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.0, 1.1))))
    resample = Image.Resampling.BOX if rng.random() < 0.5 else Image.Resampling.BILINEAR
    image = canvas.resize((WIDTH, HEIGHT), resample)
    arr = np.asarray(image, dtype=np.uint8)
    threshold = int(rng.integers(115, 176))
    binary = (arr >= threshold).astype(np.uint8) * 255
    # Normalize the soft channel against the local background.  This keeps
    # brightness useful without making the model learn a particular blue UI
    # background value.
    soft = np.clip(
        (arr.astype(np.float32) - float(bg)) * 255.0 / max(1.0, 255.0 - bg),
        0.0,
        255.0,
    ).astype(np.uint8)
    return np.stack([soft, binary], axis=0)


class SyntheticShipWords(Dataset):
    def __init__(
        self,
        terms: list[str],
        font_path: Path,
        samples_per_term: int,
        seed: int,
    ) -> None:
        self.terms = terms
        self.font_path = font_path
        self.samples_per_term = int(samples_per_term)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.terms) * self.samples_per_term

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        term_id = index // self.samples_per_term
        sample_id = index % self.samples_per_term
        rng = np.random.default_rng(self.seed + term_id * 100003 + sample_id)
        _render, label = _visible_sample(self.terms[term_id], rng)
        x = render_word(_render, self.font_path, rng)
        return x, np.asarray([ord(ch) for ch in label], dtype=np.int64)


def collate_ctc(batch: list[tuple[np.ndarray, np.ndarray]]):
    xs, labels = zip(*batch)
    x = torch.from_numpy(np.stack(xs).astype(np.float32) / 255.0)
    lengths = torch.tensor([len(y) for y in labels], dtype=torch.long)
    target = torch.from_numpy(np.concatenate(labels).astype(np.int64))
    return x, target, lengths


class ShipCRNN(nn.Module):
    """Small rectangular CNN + BiGRU CTC recognizer."""

    def __init__(self, num_chars: int, hidden: int = 96):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Keep width at 32 time steps so the longest ship term fits CTC.
            nn.Conv2d(32, 64, 3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=64,
            hidden_size=hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Linear(hidden * 2, num_chars + 1)
        # With a large character alphabet and very short CTC targets, the
        # zero-initialized blank logit can dominate the first epochs.  A small
        # negative blank prior keeps the experiment from getting stuck in the
        # all-blank solution while CTC learns the visual sequence.
        with torch.no_grad():
            self.head.bias[num_chars] = -1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).mean(dim=2).transpose(1, 2)
        sequence, _ = self.rnn(features)
        return self.head(sequence)


def _ctc_to_text(logits: torch.Tensor, chars: list[str], blank: int) -> list[str]:
    # The model emits batch-first logits: [B, T, C].  CTC decoding must keep
    # the batch dimension first; transposing here would decode time steps as
    # independent samples and silently invalidate validation results.
    ids = logits.argmax(dim=-1).cpu().numpy()
    out: list[str] = []
    for row in ids:
        prev = -1
        text: list[str] = []
        for cid in row:
            cid = int(cid)
            if cid != blank and cid != prev and 0 <= cid < len(chars):
                text.append(chars[cid])
            prev = cid
        out.append("".join(text))
    return out


def circular_candidates(visible: str, terms: list[str]) -> list[tuple[str, float, tuple[tuple[int, int], ...]]]:
    """Return terms containing ``visible`` as a linear or circular window."""

    compact = visible.replace(" ", "")
    if not compact:
        return []
    matches: list[tuple[str, float, tuple[tuple[int, int], ...]]] = []
    for term in terms:
        if len(compact) > len(term):
            continue
        doubled = term + term
        start = doubled.find(compact)
        if start < 0 or start >= len(term):
            continue
        end = start + len(compact)
        if end <= len(term):
            spans = ((start, end),)
        else:
            spans = ((start, len(term)), (0, end - len(term)))
        score = len(compact) / max(1, len(term))
        matches.append((term, score, spans))
    matches.sort(key=lambda row: (-row[1], len(row[0]), row[0]))
    return matches


def map_ctc_to_term(text: str, terms: list[str]) -> tuple[str | None, tuple[tuple[int, int], ...] | None]:
    candidates = circular_candidates(text, terms)
    if not candidates:
        return None, None
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][1] >= best[1] - 0.03:
        return None, None
    return best[0], best[2]


def make_channels(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != (HEIGHT, WIDTH):
        rgb = np.asarray(image.convert("RGB").resize((WIDTH, HEIGHT)), dtype=np.uint8)
    profile = default_profile()
    # Match the synthetic renderer: remove the local UI background before
    # feeding the soft channel.  The project's generic soft_foreground keeps
    # grayscale background brightness, which is useful for the character
    # model but creates a large train/inference domain shift here.
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    border = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
    )
    background = float(np.median(border))
    soft = np.clip(
        (gray - background) * 255.0 / max(1.0, 255.0 - background),
        0.0,
        255.0,
    ).astype(np.uint8)
    binary = profile.color_mask(rgb).astype(np.uint8) * 255
    return np.stack([soft, binary], axis=0)


@dataclass
class EvalResult:
    total: int
    decoded_nonempty: int
    term_correct: int
    misses: list[tuple[str, str, str, str | None]]


def evaluate_real(
    model: ShipCRNN,
    chars: list[str],
    terms: list[str],
    csv_path: Path,
    crops_dir: Path,
    device: torch.device,
) -> EvalResult:
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("预期值")]
    rows = [row for row in rows if (crops_dir / row["文件名"]).is_file()]
    inputs: list[np.ndarray] = []
    names: list[str] = []
    expected: list[str] = []
    for row in rows:
        inputs.append(make_channels(Image.open(crops_dir / row["文件名"])))
        names.append(row["文件名"])
        expected.append(row["预期值"])
    model.eval()
    predicted: list[str] = []
    with torch.no_grad():
        for start in range(0, len(inputs), 32):
            xb = torch.from_numpy(np.stack(inputs[start : start + 32])).float().to(device) / 255.0
            predicted.extend(_ctc_to_text(model(xb), chars, len(chars)))
    misses: list[tuple[str, str, str, str | None]] = []
    correct = 0
    nonempty = 0
    for name, want, text in zip(names, expected, predicted):
        term, _spans = map_ctc_to_term(text, terms)
        nonempty += bool(text)
        correct += term == want
        if term != want:
            misses.append((name, want, text, term))
    return EvalResult(len(rows), nonempty, correct, misses)


def train(args: argparse.Namespace) -> None:
    font = resolve_font(args.font)
    terms = load_terms(args.terms)
    if args.max_terms > 0:
        terms = terms[: args.max_terms]
    validate_font_glyphs(font, terms)
    chars, char_to_id = vocabulary(terms)

    # Dataset labels are stored as Unicode code points so the collate path is
    # independent of the order in which the vocabulary was constructed.
    class EncodedDataset(Dataset):
        def __init__(self, base: SyntheticShipWords):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, codepoints = self.base[i]
            ids = np.asarray([char_to_id[chr(int(c))] for c in codepoints], dtype=np.int64)
            return x, ids

    class CachedDataset(Dataset):
        """Materialize synthetic samples once to make CPU experiments repeatable."""

        def __init__(self, base: Dataset):
            self.samples = [base[i] for i in range(len(base))]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            return self.samples[i]

    train_ds: Dataset = EncodedDataset(
        SyntheticShipWords(terms, font, args.samples_per_term, args.seed)
    )
    val_ds: Dataset = EncodedDataset(
        SyntheticShipWords(terms, font, args.val_samples_per_term, args.seed + 9000001)
    )
    if args.cache:
        train_ds = CachedDataset(train_ds)
        val_ds = CachedDataset(val_ds)
        cache_mb = sum(x.nbytes for x, _ in train_ds.samples + val_ds.samples) / (1024 * 1024)
        print(f"cached synthetic inputs={cache_mb:.1f} MiB")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_ctc,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ctc,
    )

    device = torch.device(args.device)
    model = ShipCRNN(len(chars), hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CTCLoss(blank=len(chars), zero_infinity=True)
    print(
        f"device={device} terms={len(terms)} chars={len(chars)} "
        f"train={len(train_ds)} val={len(val_ds)} params="
        f"{sum(p.numel() for p in model.parameters())}"
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for xb, target, target_lengths in train_loader:
            xb = xb.to(device)
            target = target.to(device)
            target_lengths = target_lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            input_lengths = torch.full(
                (xb.shape[0],), log_probs.shape[0], dtype=torch.long, device=device
            )
            loss = loss_fn(log_probs, target, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        scheduler.step()

        model.eval()
        exact = 0
        total = 0
        with torch.no_grad():
            for xb, _target, target_lengths in val_loader:
                xb = xb.to(device)
                pred = _ctc_to_text(model(xb), chars, len(chars))
                # Recreate labels from the deterministic validation dataset;
                # this avoids storing a second copy of the rendered images.
                start = total
                for i, text in enumerate(pred):
                    term_id = (start + i) // args.val_samples_per_term
                    sample_id = (start + i) % args.val_samples_per_term
                    rng = np.random.default_rng(
                        args.seed + 9000001 + term_id * 100003 + sample_id
                    )
                    _render, label = _visible_sample(terms[term_id], rng)
                    exact += text == label
                total += len(pred)
        print(f"epoch {epoch + 1:>2}: loss={total_loss / max(1, batches):.4f} val_exact={exact / max(1, total):.3f}")
        if args.save_each_epoch:
            epoch_path = Path(args.output).with_name(
                f"{Path(args.output).stem}.epoch{epoch + 1:02d}{Path(args.output).suffix}"
            )
            epoch_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": "ship_crnn_ctc_v1",
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "charset": chars,
                    "terms": terms,
                    "font_sha256": compute_font_sha256(font),
                    "input_height": HEIGHT,
                    "input_width": WIDTH,
                    "input_channels": 2,
                    "hidden": args.hidden,
                    "blank": len(chars),
                    "epoch": epoch + 1,
                    "val_exact": exact / max(1, total),
                },
                epoch_path,
            )
            print(f"wrote {epoch_path} ({epoch_path.stat().st_size} bytes)")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "ship_crnn_ctc_v1",
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "charset": chars,
            "terms": terms,
            "font_sha256": compute_font_sha256(font),
            "input_height": HEIGHT,
            "input_width": WIDTH,
            "input_channels": 2,
            "hidden": args.hidden,
            "blank": len(chars),
        },
        output,
    )
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    if args.eval_csv:
        result = evaluate_real(
            model,
            chars,
            terms,
            Path(args.eval_csv),
            Path(args.crops_dir),
            device,
        )
        print(
            f"real LOCKCSV: {result.term_correct}/{result.total} term_correct "
            f"decoded_nonempty={result.decoded_nonempty}/{result.total}"
        )
        for miss in result.misses[:20]:
            print("MISS", miss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=ROOT / "fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf")
    parser.add_argument("--terms", type=Path, default=ROOT / "charsets/words/ship_names.txt")
    parser.add_argument("--max-terms", type=int, default=0, help="limit the lexicon for a small training sanity check")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ship_crnn.pth")
    parser.add_argument("--samples-per-term", type=int, default=16)
    parser.add_argument("--val-samples-per-term", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--cache", action="store_true", help="cache rendered synthetic samples in RAM")
    parser.add_argument("--save-each-epoch", action="store_true")
    parser.add_argument("--eval-csv", type=Path)
    parser.add_argument("--crops-dir", type=Path, default=ROOT / "fonts/SourceHanSansSC/crops/crops_items")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
