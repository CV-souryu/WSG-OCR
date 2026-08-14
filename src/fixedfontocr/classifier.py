"""Classifier interface plus the CPU template-matching baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .preprocess import Component, NormalizeSpec, glyph_normalize_geometry, normalize
from .postprocess import pick
from .types import Profile

# Prototype downsample mode encoding used by Template V2 (Goal 9):
# 0=bilinear downsample, 1=area-like downsample, 2=clean high-res render.
DOWNSAMPLE_BILINEAR = 0
DOWNSAMPLE_AREA = 1
DOWNSAMPLE_CLEAN = 2


@dataclass(frozen=True)
class TemplateBatch:
    """Top-2 template match for a batch of normalized glyphs.

    ``scores`` are ``1 - dist / area`` (0..1); ``margins`` are the
    normalized top-1/top-2 Hamming-distance gap in the same 0..1 scale.
    ``dists`` keep the raw distances for reporting and ``second_ids`` the
    second-ranked character id (``-1`` when the charset has one class).

    Template V2 (Goal 9) additionally fills ``top_k_ids`` /
    ``top_k_scores`` (``K`` characters by their best prototype) and the
    winning prototype's metadata (``best_prototypes`` + per-prototype
    render size / sub-pixel phase / downsample mode). V1 matching leaves
    those fields as ``None``.
    """

    ids: np.ndarray  # int32 [N]
    scores: np.ndarray  # f32 [N]
    second_scores: np.ndarray  # f32 [N]
    margins: np.ndarray  # f32 [N]
    dists: np.ndarray  # int32 [N]
    second_dists: np.ndarray  # int32 [N]
    second_ids: np.ndarray | None = None  # int32 [N]
    top_k_ids: np.ndarray | None = None  # int32 [N, K]
    top_k_scores: np.ndarray | None = None  # f32 [N, K]
    best_prototypes: np.ndarray | None = None  # int32 [N] prototype index
    prototype_render_sizes: np.ndarray | None = None  # int32 [N] final px
    prototype_dx: np.ndarray | None = None  # int32 [N] sub-pixel phase, 1/8 px
    prototype_dy: np.ndarray | None = None  # int32 [N] sub-pixel phase, 1/8 px
    prototype_downsample_modes: np.ndarray | None = None  # int32 [N] 0/1


@dataclass(frozen=True)
class TemplateV2Data:
    """Goal 9 multi-prototype template set.

    One character owns ``P`` prototypes (``P = prototypes_per_char``)
    covering the Goal 9 low-resolution grid: render sizes 11..16 px times
    sub-pixel phases (stored in eighths of a final pixel, ``0..8``) times
    bilinear/area-like downsample modes. ``bits`` is the packed bitset
    payload in the same little-endian bit order as V1 templates.
    """

    bits: np.ndarray  # uint8 [C, P, B]
    render_sizes: np.ndarray  # uint8 [C, P] final pixel size (11..16)
    dx: np.ndarray  # uint8 [C, P] sub-pixel phase in 1/8 final px
    dy: np.ndarray  # uint8 [C, P] sub-pixel phase in 1/8 final px
    downsample_modes: np.ndarray  # uint8 [C, P] 0=bilinear, 1=area-like

    @property
    def num_classes(self) -> int:
        return int(self.bits.shape[0])

    @property
    def prototypes_per_char(self) -> int:
        return int(self.bits.shape[1])

    @property
    def bytes_per_template(self) -> int:
        return int(self.bits.shape[2])

    def __post_init__(self) -> None:
        if self.bits.ndim != 3:
            raise ValueError(
                f"TemplateV2Data.bits must be [C, P, B], got {self.bits.shape}"
            )
        for name, arr in (
            ("render_sizes", self.render_sizes),
            ("dx", self.dx),
            ("dy", self.dy),
            ("downsample_modes", self.downsample_modes),
        ):
            if arr.shape != self.bits.shape[:2]:
                raise ValueError(
                    f"TemplateV2Data.{name} has shape {arr.shape}, "
                    f"expected {self.bits.shape[:2]}"
                )
            if arr.dtype != np.uint8:
                raise ValueError(f"TemplateV2Data.{name} must be uint8")
        if self.bits.dtype != np.uint8:
            raise ValueError("TemplateV2Data.bits must be uint8")
        if np.any(np.all(self.bits == 0, axis=2)):
            raise ValueError(
                "TemplateV2Data contains an all-zero prototype "
                "(the font could not render this glyph in that prototype)"
            )


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
        normalize_spec: NormalizeSpec | None = None,
    ):
        if len(charset) != templates.shape[0]:
            raise ValueError("charset and templates must have the same length")
        if templates.shape[1] != (input_size * input_size + 7) // 8:
            raise ValueError("template byte width does not match input_size")
        self.templates = templates
        self.charset = list(charset)
        self.input_size = input_size
        self.candidate_filter = bool(candidate_filter)
        self.normalize_spec = normalize_spec
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
        if self.normalize_spec is not None:
            h, w = mask.shape
            candidate = Component(mask=mask, x=0, y=0, w=w, h=h)
            baseline_offset, scale = glyph_normalize_geometry(
                candidate, self.normalize_spec, self.input_size
            )
            glyph = normalize(
                mask,
                self.input_size,
                baseline_offset=baseline_offset,
                scale=scale,
                baseline_row=self.normalize_spec.baseline_row,
            )
        else:
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

    # ------------------------------------------------------------------
    # Batched matching (used by the segmentation candidate lattice)
    # ------------------------------------------------------------------

    def match_batch(
        self,
        glyphs: NDArray[np.uint8],
        allowed_ids: set[int] | None = None,
    ) -> TemplateBatch:
        """Top-2 template match for an ``uint8 [N, H, W]`` glyph batch.

        Unlike :meth:`match` this runs a full vectorized scan over every
        candidate, which is fast enough for the small candidate sets the
        segmentation lattice produces and avoids per-glyph Python overhead.
        """

        glyphs = np.asarray(glyphs, dtype=np.uint8)
        if glyphs.ndim != 3:
            raise ValueError(f"glyphs must be [N, H, W], got {glyphs.shape}")
        n = glyphs.shape[0]
        if n == 0:
            return TemplateBatch(
                ids=np.empty(0, dtype=np.int32),
                scores=np.empty(0, dtype=np.float32),
                second_scores=np.empty(0, dtype=np.float32),
                margins=np.empty(0, dtype=np.float32),
                dists=np.empty(0, dtype=np.int32),
                second_dists=np.empty(0, dtype=np.int32),
                second_ids=np.empty(0, dtype=np.int32),
            )
        if allowed_ids is not None and not allowed_ids:
            return TemplateBatch(
                ids=np.full(n, -1, dtype=np.int32),
                scores=np.zeros(n, dtype=np.float32),
                second_scores=np.zeros(n, dtype=np.float32),
                margins=np.zeros(n, dtype=np.float32),
                dists=np.full(n, self.input_size * self.input_size, dtype=np.int32),
                second_dists=np.full(
                    n, self.input_size * self.input_size, dtype=np.int32
                ),
                second_ids=np.full(n, -1, dtype=np.int32),
            )
        if glyphs.shape[1:] != (self.input_size, self.input_size):
            raise ValueError(
                f"expected {self.input_size}x{self.input_size} glyphs, "
                f"got {glyphs.shape[1:]}"
            )

        bits = np.packbits(glyphs.reshape(n, -1), axis=1, bitorder="little")
        words = bits.view(np.uint64).reshape(n, -1)
        candidates = self._allowed_candidates(allowed_ids)
        template_bits = self._template_bits[candidates]
        diff = words[:, None, :] ^ template_bits[None, :, :]
        dists = self._popcount16(diff).sum(axis=-1)  # [N, C]

        if dists.shape[1] == 1:
            best = np.zeros(n, dtype=np.int64)
            second = np.full(n, self.input_size * self.input_size, dtype=np.int64)
            second_ids_raw = np.full(n, -1, dtype=np.int64)
        else:
            idx = np.argpartition(dists, 1, axis=-1)[:, :2]
            v = dists[np.arange(n)[:, None], idx]
            a, b = v[:, 0], v[:, 1]
            best = np.minimum(a, b)
            second = np.maximum(a, b)
            # Recover the ids that produced the two distances.
            best_ids = np.where(a <= b, idx[:, 0], idx[:, 1])
            second_ids_raw = np.where(a <= b, idx[:, 1], idx[:, 0])
        if dists.shape[1] == 1:
            best_ids = np.zeros(n, dtype=np.int64)
        ids = candidates[best_ids].astype(np.int32)
        second_ids = candidates[second_ids_raw].astype(np.int32)
        area = self.input_size * self.input_size
        scores = (1.0 - best.astype(np.float32) / area).astype(np.float32)
        second_scores = (1.0 - second.astype(np.float32) / area).astype(np.float32)
        margins = ((second - best).astype(np.float32) / area).astype(np.float32)
        return TemplateBatch(
            ids=ids,
            scores=scores,
            second_scores=second_scores,
            margins=margins,
            dists=best.astype(np.int32),
            second_dists=second.astype(np.int32),
            second_ids=second_ids,
        )


class TemplateV2Classifier(Classifier):
    """Goal 9 template matcher over multi-prototype template sets.

    Each character owns several prototypes (render sizes 11..16 px ×
    sub-pixel phases × bilinear/area-like downsample). Matching keeps the
    V1 cascade -- coarse geometry prefilter, then XOR + popcount -- but
    runs it per prototype and aggregates by character:

    * ``best`` = the character whose best prototype has the smallest
      Hamming distance (its score is that prototype's ``1 - dist / area``);
    * ``second`` = the second-ranked character (by its own best prototype);
    * ``margin`` = normalized best/second character distance gap;
    * ``top_k`` = the top ``K`` characters with per-prototype metadata for
      the winner.

    The coarse pass is *exact for the returned Top-K*: after scanning the
    geometrically close prototypes, any prototype whose ink-count delta is
    no larger than the current K-th character distance is also scanned
    (Hamming distance >= |ink delta|), so the final ranking cannot miss a
    contender.
    """

    def __init__(
        self,
        data: TemplateV2Data,
        charset: list[str],
        input_size: int = 24,
        candidate_filter: bool = True,
        normalize_spec: NormalizeSpec | None = None,
        default_top_k: int = 5,
    ):
        if len(charset) != data.num_classes:
            raise ValueError("charset and TemplateV2Data must have the same length")
        expected_bytes = (input_size * input_size + 7) // 8
        if data.bytes_per_template != expected_bytes:
            raise ValueError(
                "TemplateV2Data bytes_per_template does not match input_size"
            )
        self.data = data
        self.charset = list(charset)
        self.input_size = input_size
        self.candidate_filter = bool(candidate_filter)
        self.normalize_spec = normalize_spec
        self.default_top_k = max(2, int(default_top_k))
        self.last_candidates: int | None = None
        total = data.num_classes * data.prototypes_per_char
        self._template_bits = np.ascontiguousarray(
            data.bits.reshape(total, -1)
        ).view(np.uint64)
        self._proto_char = np.repeat(
            np.arange(data.num_classes, dtype=np.int64),
            data.prototypes_per_char,
        )
        self._features = self._build_features()
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
        """Best character (optionally restricted to ``allowed_ids``)."""
        if allowed_ids is not None and not allowed_ids:
            return "?", 0.0
        if self.normalize_spec is not None:
            h, w = mask.shape
            candidate = Component(mask=mask, x=0, y=0, w=w, h=h)
            baseline_offset, scale = glyph_normalize_geometry(
                candidate, self.normalize_spec, self.input_size
            )
            glyph = normalize(
                mask,
                self.input_size,
                baseline_offset=baseline_offset,
                scale=scale,
                baseline_row=self.normalize_spec.baseline_row,
            )
        else:
            glyph = normalize(mask, self.input_size)
        tb = self.match_batch(glyph[None, :, :], allowed_ids)
        cid = int(tb.ids[0])
        if cid < 0:
            return "?", 0.0
        return self.charset[cid], float(tb.scores[0])

    def _allowed_prototypes(self, allowed_ids: set[int] | None) -> NDArray[np.int64]:
        """Global prototype indices for every (optionally restricted) char."""
        if allowed_ids is None:
            return np.arange(self._proto_char.size, dtype=np.int64)
        chars = np.fromiter(sorted(allowed_ids), dtype=np.int64)
        if chars.size == 0:
            return np.empty(0, dtype=np.int64)
        p = self.data.prototypes_per_char
        base = np.repeat(chars * p, p)
        off = np.tile(np.arange(p, dtype=np.int64), chars.size)
        return base + off

    def _build_features(self) -> dict[str, NDArray]:
        """Coarse per-prototype features (same compact dtypes as V1)."""
        bits = (
            np.unpackbits(self.data.bits, axis=1, bitorder="little")[
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

    def _glyph_features_impl(
        self,
        glyph: NDArray[np.bool_],
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

    def _distances(
        self, bits_row: NDArray[np.uint64], prototypes: NDArray[np.int64]
    ) -> NDArray[np.int32]:
        """Exact XOR + popcount distances for one glyph vs prototype subset."""
        template_bits = self._template_bits[prototypes]
        diff = np.asarray(bits_row).reshape(1, -1) ^ template_bits
        return self._popcount16(diff).sum(axis=1).astype(np.int32)

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

    def _filter_mask(
        self,
        feats: dict[str, int],
        ink_tol: int,
        prototypes: NDArray[np.int64],
    ) -> NDArray[np.bool_]:
        f = self._features
        ink = f["ink"][prototypes].astype(np.int32)
        h = f["h"][prototypes].astype(np.int16)
        w = f["w"][prototypes].astype(np.int16)
        top = f["top"][prototypes].astype(np.int16)
        left = f["left"][prototypes].astype(np.int16)
        bottom = f["bottom"][prototypes].astype(np.int16)
        right = f["right"][prototypes].astype(np.int16)
        return (
            (np.abs(ink - feats["ink"]) <= ink_tol)
            & (np.abs(h - feats["h"]) <= 1)
            & (np.abs(w - feats["w"]) <= 1)
            & (np.abs(top - feats["top"]) <= 2)
            & (np.abs(left - feats["left"]) <= 2)
            & (np.abs(bottom - feats["bottom"]) <= 2)
            & (np.abs(right - feats["right"]) <= 2)
        )

    @staticmethod
    def _char_min(
        dists: NDArray[np.int32],
        proto_ids: NDArray[np.int64],
        proto_char: NDArray[np.int64],
        num_classes: int,
        fill: int,
    ) -> NDArray[np.int32]:
        per_char = np.full(num_classes, fill, dtype=np.int32)
        np.minimum.at(per_char, proto_char[proto_ids], dists)
        return per_char

    def match_batch(
        self,
        glyphs: NDArray[np.uint8],
        allowed_ids: set[int] | None = None,
        top_k: int | None = None,
    ) -> TemplateBatch:
        """Top-K template match for a ``uint8 [N, H, W]`` glyph batch.

        Every glyph is matched against its character's prototypes and the
        batch returns Top-K characters (default :attr:`default_top_k`),
        the best/second character scores, the normalized margin, and the
        winning prototype's metadata (Goal 9). The geometry prefilter keeps
        the XOR/popcount pass small while the ink-bound fallback keeps the
        returned Top-K exact.
        """

        glyphs = np.asarray(glyphs, dtype=np.uint8)
        if glyphs.ndim != 3:
            raise ValueError(f"glyphs must be [N, H, W], got {glyphs.shape}")
        n = glyphs.shape[0]
        area = self.input_size * self.input_size
        k = self.default_top_k if top_k is None else int(top_k)
        k = max(1, min(k, self.data.num_classes))
        if n == 0:
            return TemplateBatch(
                ids=np.empty(0, dtype=np.int32),
                scores=np.empty(0, dtype=np.float32),
                second_scores=np.empty(0, dtype=np.float32),
                margins=np.empty(0, dtype=np.float32),
                dists=np.empty(0, dtype=np.int32),
                second_dists=np.empty(0, dtype=np.int32),
                second_ids=np.empty(0, dtype=np.int32),
                top_k_ids=np.empty((0, k), dtype=np.int32),
                top_k_scores=np.empty((0, k), dtype=np.float32),
                best_prototypes=np.empty(0, dtype=np.int32),
                prototype_render_sizes=np.empty(0, dtype=np.int32),
                prototype_dx=np.empty(0, dtype=np.int32),
                prototype_dy=np.empty(0, dtype=np.int32),
                prototype_downsample_modes=np.empty(0, dtype=np.int32),
            )
        if glyphs.shape[1:] != (self.input_size, self.input_size):
            raise ValueError(
                f"expected {self.input_size}x{self.input_size} glyphs, "
                f"got {glyphs.shape[1:]}"
            )
        proto_ids = self._allowed_prototypes(allowed_ids)
        if proto_ids.size == 0:
            return TemplateBatch(
                ids=np.full(n, -1, dtype=np.int32),
                scores=np.zeros(n, dtype=np.float32),
                second_scores=np.zeros(n, dtype=np.float32),
                margins=np.zeros(n, dtype=np.float32),
                dists=np.full(n, area, dtype=np.int32),
                second_dists=np.full(n, area, dtype=np.int32),
                second_ids=np.full(n, -1, dtype=np.int32),
                top_k_ids=np.full((n, k), -1, dtype=np.int32),
                top_k_scores=np.zeros((n, k), dtype=np.float32),
                best_prototypes=np.full(n, -1, dtype=np.int32),
                prototype_render_sizes=np.full(n, -1, dtype=np.int32),
                prototype_dx=np.full(n, -1, dtype=np.int32),
                prototype_dy=np.full(n, -1, dtype=np.int32),
                prototype_downsample_modes=np.full(n, -1, dtype=np.int32),
            )

        bits = np.packbits(glyphs.reshape(n, -1), axis=1, bitorder="little")
        words = bits.view(np.uint64).reshape(n, -1)
        chars_allowed = np.sort(np.unique(self._proto_char[proto_ids]))
        fill = area + 1
        proto_ink = self._features["ink"][proto_ids].astype(np.int32)
        glyph_bool = glyphs > 0

        out_ids = np.empty((n, k), dtype=np.int32)
        out_scores = np.empty((n, k), dtype=np.float32)
        out_dists = np.empty(n, dtype=np.int32)
        out_second_dists = np.empty(n, dtype=np.int32)
        out_second_ids = np.full(n, -1, dtype=np.int32)
        out_best_proto = np.full(n, -1, dtype=np.int32)
        out_rs = np.full(n, -1, dtype=np.int32)
        out_dx = np.full(n, -1, dtype=np.int32)
        out_dy = np.full(n, -1, dtype=np.int32)
        out_mode = np.full(n, -1, dtype=np.int32)

        for i in range(n):
            feats, ink_tol = self._glyph_features_impl(glyph_bool[i])
            dists = np.full(proto_ids.size, fill, dtype=np.int32)
            if self.candidate_filter:
                ok = self._filter_mask(feats, ink_tol, proto_ids)
                tight = proto_ids[ok]
                if tight.size:
                    dists[ok] = self._distances(words[i], tight)
                per_char = self._char_min(
                    dists, proto_ids, self._proto_char, self.data.num_classes, fill
                )
                # K-th character distance bounds the exact-fallback scan:
                # any prototype with |ink delta| > bound has distance > bound
                # and therefore cannot enter the returned Top-K.
                vals = per_char[chars_allowed]
                kth = int(np.sort(vals)[:k][-1]) if vals.size else fill
                need = (np.abs(proto_ink - feats["ink"]) <= kth) & (~ok)
                if np.any(need):
                    idx = np.flatnonzero(need)
                    dists[idx] = self._distances(words[i], proto_ids[idx])
                self.last_candidates = int(tight.size)
            else:
                dists = self._distances(words[i], proto_ids)
                kth = fill

            per_char = self._char_min(
                dists, proto_ids, self._proto_char, self.data.num_classes, fill
            )
            char_vals = per_char[chars_allowed]
            kk = min(k, char_vals.size)
            if kk == 0:
                continue
            top = np.argpartition(char_vals, kk - 1)[:kk]
            order = np.lexsort((chars_allowed[top], char_vals[top]))
            top_chars = chars_allowed[top[order]].astype(np.int32)
            top_dists = char_vals[top[order]].astype(np.int32)
            out_ids[i, :kk] = top_chars
            out_scores[i, :kk] = (1.0 - top_dists.astype(np.float32) / area)
            if kk < k:
                out_ids[i, k:] = -1
                out_scores[i, k:] = 0.0
            out_dists[i] = top_dists[0]
            out_second_dists[i] = top_dists[1] if kk > 1 else area
            if kk > 1:
                out_second_ids[i] = top_chars[1]
            # Winning prototype inside the best character.
            best_char = top_chars[0]
            mask = self._proto_char[proto_ids] == best_char
            if np.any(mask):
                j = int(np.argmin(np.where(mask, dists, fill)))
                proto = proto_ids[j]
                out_best_proto[i] = int(proto)
                out_rs[i] = int(self.data.render_sizes.reshape(-1)[proto])
                out_dx[i] = int(self.data.dx.reshape(-1)[proto])
                out_dy[i] = int(self.data.dy.reshape(-1)[proto])
                out_mode[i] = int(self.data.downsample_modes.reshape(-1)[proto])

        scores = (1.0 - out_dists.astype(np.float32) / area).astype(np.float32)
        second_scores = (
            1.0 - out_second_dists.astype(np.float32) / area
        ).astype(np.float32)
        margins = (
            (out_second_dists - out_dists).astype(np.float32) / area
        ).astype(np.float32)
        return TemplateBatch(
            ids=out_ids[:, 0].copy(),
            scores=scores,
            second_scores=second_scores,
            margins=margins,
            dists=out_dists,
            second_dists=out_second_dists,
            second_ids=out_second_ids,
            top_k_ids=out_ids,
            top_k_scores=out_scores,
            best_prototypes=out_best_proto,
            prototype_render_sizes=out_rs,
            prototype_dx=out_dx,
            prototype_dy=out_dy,
            prototype_downsample_modes=out_mode,
        )
