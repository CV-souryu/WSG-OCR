"""Real-glyph prototype bank + NCC arbitration (pure numpy).

Built by ``tools/lexicon/build_real_glyph_bank.py`` from a labeled crops
corpus: each prototype is a 24x24 soft glyph extracted from a *real game
crop*, so NCC matching compares game renders with game renders (no
font-render domain gap). The bank records the registered font's SHA256
(Font Policy) because the OCR pipeline that produced the crops runs on
that font only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

BANK_HEIGHT = 24
NCC_MEAN_FLOOR = 0.55  # matched-position mean floor for NCC evidence


@dataclass(frozen=True)
class RealGlyphBank:
    """char -> prototype stack of soft glyph patches [N, 24, 24]."""

    path: Path
    font_sha256: str
    prototypes: dict[str, np.ndarray]

    def best_evidence(self, glyph: np.ndarray, char: str) -> float | None:
        """Best NCC (clamped >= 0) of one glyph against a character."""

        stack = self.prototypes.get(char)
        if stack is None:
            return None
        best = -1.0
        for proto in stack:
            score = ncc(glyph, proto)
            if score > best:
                best = score
        return max(0.0, best)


@lru_cache(maxsize=4)
def load_real_glyphs(path: Path) -> RealGlyphBank | None:
    """Load a ``<csv>.realglyphs.npz`` bank; ``None`` when missing."""

    p = Path(path)
    if not p.is_file():
        return None
    z = np.load(p)
    labels = z["labels"]
    data = z["data"]
    prototypes: dict[str, list[np.ndarray]] = {}
    for label, patch in zip(labels, data):
        prototypes.setdefault(str(label), []).append(patch)
    font_sha256 = str(z["font_sha256"]) if "font_sha256" in z else ""
    return RealGlyphBank(
        path=p,
        font_sha256=font_sha256,
        prototypes={c: np.stack(v) for c, v in prototypes.items()},
    )


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of two same-shape arrays."""

    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx == 0.0 or ny == 0.0:
        return 0.0
    return float(x @ y / (nx * ny))


def resize24(mask: np.ndarray) -> np.ndarray:
    """Bilinear-resize any 2-D patch to (24, 24) (pure numpy)."""

    h, w = mask.shape
    if h == BANK_HEIGHT and w == BANK_HEIGHT:
        return mask.astype(np.float32)
    ys = np.linspace(0, h - 1, BANK_HEIGHT)
    xs = np.linspace(0, w - 1, BANK_HEIGHT)
    y0 = np.floor(ys).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = np.floor(xs).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    fy = (ys - y0)[:, None]
    fx = xs - x0
    m = mask.astype(np.float32)
    top = m[y0][:, x0] * (1 - fx) + m[y0][:, x1] * fx
    bot = m[y1][:, x0] * (1 - fx) + m[y1][:, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)
