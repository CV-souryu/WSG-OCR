"""Classifier interface plus the CPU template-matching baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .preprocess import normalize
from .postprocess import pick
from .types import Profile


class Classifier(ABC):
    """Maps a normalized glyph bitmap to a character ID and confidence."""

    @abstractmethod
    def __call__(
        self,
        mask: NDArray[np.bool_],
        profile: Profile,
        allowed_ids: set[int] | None = None,
    ) -> tuple[str, float]:
        ...


class TemplateClassifier(Classifier):
    """Template baseline: XOR + popcount over bitset templates.

    With ``candidate_filter=True`` (default) a coarse pre-pass compares
    ink pixel count, glyph bbox and margins between the normalized input
    and every template, so the expensive XOR + popcount only runs over a
    small candidate subset. For a 3000-char fixed font this is the
    "width filter -> bitmap coarse feature -> exact match" cascade from the
    design plan; tolerances are deliberately generous so the exact-match
    winner is preserved.
    """

    def __init__(
        self,
        templates: NDArray[np.uint8],
        charset: list[str],
        input_size: int = 24,
        candidate_filter: bool = True,
    ):
        if len(charset) != templates.shape[0]:
            raise ValueError("charset and templates must have the same length")
        if templates.shape[1] != (input_size * input_size + 7) // 8:
            raise ValueError("template byte width does not match input_size")
        self.templates = templates
        self.charset = list(charset)
        self.input_size = input_size
        self.candidate_filter = bool(candidate_filter)
        self.last_candidates: int | None = None
        self._template_bits = np.ascontiguousarray(
            templates.reshape(len(charset), -1)
        ).view(np.uint64)
        self._features = self._build_features()
        # numpy >= 2.0 has a vectorized bit_count; keep the 16-bit table as
        # a fallback for older numpy instead of always reserving 64 KiB.
        self._popcount_table = (
            None if hasattr(np, "bitwise_count") else np.zeros(1 << 16, dtype=np.uint8)
        )
        if self._popcount_table is not None:
            for i in range(1, 1 << 16):
                self._popcount_table[i] = (
                    self._popcount_table[i >> 1] + (i & 1)
                )

    def __call__(
        self,
        mask: NDArray[np.bool_],
        profile: Profile,
        allowed_ids: set[int] | None = None,
    ) -> tuple[str, float]:
        return self.match(mask, profile, allowed_ids)

    def match(
        self,
        mask: NDArray[np.bool_],
        profile: Profile,
        allowed_ids: set[int] | None = None,
    ) -> tuple[str, float]:
        """Best template (optionally restricted to ``allowed_ids``)."""
        if allowed_ids is not None and not allowed_ids:
            return "?", 0.0
        glyph = normalize(mask, self.input_size)
        bits = np.packbits(glyph.reshape(-1), bitorder="little").view(np.uint64)
        candidates = self._allowed_candidates(allowed_ids)
        if self.candidate_filter and candidates.size:
            best_idx, best_dist = self._match_with_filter(
                glyph.astype(np.bool_), bits, candidates
            )
        else:
            best_idx, best_dist = self._scan(bits, candidates)
        total = self.input_size * self.input_size
        confidence = 1.0 - best_dist / total
        return self.charset[best_idx], confidence

    def _allowed_candidates(self, allowed_ids: set[int] | None) -> NDArray[np.int64]:
        all_idx = np.arange(len(self.charset), dtype=np.int64)
        if allowed_ids is not None:
            mask = np.zeros(len(self.charset), dtype=bool)
            mask[list(allowed_ids)] = True
            all_idx = all_idx[mask]
        return all_idx

    def _scan(
        self, bits: NDArray[np.uint64], candidates: NDArray[np.int64]
    ) -> tuple[int, int]:
        """Exact XOR+popcount scan over ``candidates``; returns (id, dist)."""
        template_bits = self._template_bits[candidates]
        diff = np.asarray(bits).reshape(1, -1) ^ template_bits
        dist = self._popcount16(diff).sum(axis=1)
        best = int(np.argmin(dist))
        return int(candidates[best]), int(dist[best])

    def _glyph_features(
        self, glyph: NDArray[np.bool_]
    ) -> tuple[dict[str, int], int]:
        rows = glyph.any(axis=1)
        cols = glyph.any(axis=0)
        ys = np.where(rows)[0]
        xs = np.where(cols)[0]
        ink = int(glyph.sum())
        if ys.size == 0 or xs.size == 0:
            return {"ink": ink}, int(0.20 * ink) + 3
        feats = {
            "ink": ink,
            "h": int(ys[-1] - ys[0] + 1),
            "w": int(xs[-1] - xs[0] + 1),
            "top": int(ys[0]),
            "left": int(xs[0]),
            "bottom": int(self.input_size - 1 - ys[-1]),
            "right": int(self.input_size - 1 - xs[-1]),
        }
        return feats, int(0.20 * ink) + 3

    def _match_with_filter(
        self,
        glyph: NDArray[np.bool_],
        bits: NDArray[np.uint64],
        candidates: NDArray[np.int64],
    ) -> tuple[int, int]:
        """Coarse filter + exact fallback scan.

        The coarse pass only computes XOR+popcount on candidates whose
        features are close. It is safe because any candidate that could
        still win must have ``|ink_input - ink_template| <= best_dist``
        (Hamming distance is always >= the ink-count difference), so a
        second scan over that set makes the result identical to a full scan.
        """

        feats, ink_tol = self._glyph_features(glyph)
        f = self._features
        # Compact dtypes (uint8/uint16) underflow on subtraction; compare in
        # a signed dtype so e.g. ink 79 vs input 87 is -8, not 65528.
        ink = f["ink"][candidates].astype(np.int32)
        h = f["h"][candidates].astype(np.int16)
        w = f["w"][candidates].astype(np.int16)
        top = f["top"][candidates].astype(np.int16)
        left = f["left"][candidates].astype(np.int16)
        bottom = f["bottom"][candidates].astype(np.int16)
        right = f["right"][candidates].astype(np.int16)
        ok = (
            (np.abs(ink - feats["ink"]) <= ink_tol)
            & (np.abs(h - feats["h"]) <= 1)
            & (np.abs(w - feats["w"]) <= 1)
            & (np.abs(top - feats["top"]) <= 2)
            & (np.abs(left - feats["left"]) <= 2)
            & (np.abs(bottom - feats["bottom"]) <= 2)
            & (np.abs(right - feats["right"]) <= 2)
        )
        tight = candidates[ok] if np.any(ok) else candidates
        self.last_candidates = int(tight.size)
        best_idx, best_dist = self._scan(bits, tight)

        # Exact fallback: candidates with an ink-count delta no larger than
        # the current best cannot be excluded from the race.
        possible = candidates[np.abs(ink - feats["ink"]) <= best_dist]
        if possible.size > tight.size:
            best_idx, best_dist = self._scan(bits, possible)
        return best_idx, best_dist

    def _build_features(self) -> dict[str, NDArray]:
        """Coarse per-template features from the normalized bitmaps.

        Features are stored in the smallest dtypes that fit them: ink count
        (<= 24*24 = 576) is uint16, bbox/margins (<= 24) are uint8. That is
        8 bytes per character instead of 28 for the int32 version.
        """
        bits = (
            np.unpackbits(self.templates, axis=1, bitorder="little")[
                :, : self.input_size * self.input_size
            ]
            .reshape(-1, self.input_size, self.input_size)
            .astype(np.bool_)
        )
        ink = bits.sum(axis=(1, 2))
        rows = bits.any(axis=2)
        cols = bits.any(axis=1)
        ys = np.argmax(rows, axis=1)
        xs = np.argmax(cols, axis=1)
        # Bottom/right are measured from the other edge: max row/col index.
        bottom = np.full(len(bits), self.input_size - 1, dtype=np.int32)
        right = np.full(len(bits), self.input_size - 1, dtype=np.int32)
        for i in range(len(bits)):
            bottom[i] = int(np.where(rows[i])[0][-1])
            right[i] = int(np.where(cols[i])[0][-1])
        return {
            "ink": ink.astype(np.uint16),
            "h": (bottom - ys + 1).astype(np.uint8),
            "w": (right - xs + 1).astype(np.uint8),
            "top": ys.astype(np.uint8),
            "left": xs.astype(np.uint8),
            "bottom": (self.input_size - 1 - bottom).astype(np.uint8),
            "right": (self.input_size - 1 - right).astype(np.uint8),
        }

    def _popcount16(self, words: NDArray[np.uint64]) -> NDArray[np.int32]:
        if self._popcount_table is None:
            return np.bitwise_count(words).astype(np.int32)
        lo = (words & 0xFFFF).astype(np.uint64)
        mid = ((words >> 16) & 0xFFFF).astype(np.uint64)
        hi = ((words >> 32) & 0xFFFF).astype(np.uint64)
        top = ((words >> 48) & 0xFFFF).astype(np.uint64)
        return (
            self._popcount_table[lo]
            + self._popcount_table[mid]
            + self._popcount_table[hi]
            + self._popcount_table[top]
        ).astype(np.int32)
