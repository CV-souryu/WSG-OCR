"""Classifier interface plus the CPU template-matching baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .preprocess import normalize
from .types import Profile


class Classifier(ABC):
    """Maps a normalized glyph bitmap to a character ID and confidence."""

    @abstractmethod
    def __call__(self, mask: NDArray[np.bool_], profile: Profile) -> tuple[str, float]:
        ...


class TemplateClassifier(Classifier):
    """Template baseline: XOR + popcount over 72-byte bitset templates."""

    def __init__(
        self,
        templates: NDArray[np.uint8],
        charset: list[str],
        input_size: int = 24,
    ):
        if len(charset) != templates.shape[0]:
            raise ValueError("charset and templates must have the same length")
        if templates.shape[1] != (input_size * input_size + 7) // 8:
            raise ValueError("template byte width does not match input_size")
        self.templates = templates
        self.charset = list(charset)
        self.input_size = input_size
        self._template_bits = np.ascontiguousarray(
            templates.reshape(len(charset), -1)
        ).view(np.uint64)
        self._popcount_table = np.zeros(1 << 16, dtype=np.uint8)
        for i in range(1, 1 << 16):
            self._popcount_table[i] = (
                self._popcount_table[i >> 1] + (i & 1)
            )

    def __call__(self, mask: NDArray[np.bool_], profile: Profile) -> tuple[str, float]:
        glyph = normalize(mask, self.input_size)
        bits = np.packbits(glyph.reshape(-1), bitorder="little").view(np.uint64)
        diff = np.asarray(bits).reshape(1, -1) ^ self._template_bits
        dist = self._popcount16(diff).sum(axis=1)
        best = int(np.argmin(dist))
        best_dist = int(dist[best])
        total = self.input_size * self.input_size
        confidence = 1.0 - best_dist / total
        return self.charset[best], confidence

    def _popcount16(self, words: NDArray[np.uint64]) -> NDArray[np.int32]:
        # x86-64 POPCNT semantics via the 16-bit table; portable and vectorized.
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
