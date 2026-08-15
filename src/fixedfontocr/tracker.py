"""Goal 16: cross-frame tracking, decoupled from single-frame OCR.

The single-frame API stays a pure function::

    result = ocr.recognize(image)

For an online/repeated-crop use case a separate :class:`FrameTracker`
keeps the temporal state::

    tracker = ocr.tracker()
    result = tracker.update(frame1)
    result = tracker.update(frame2)

The tracker adds four pieces of state that never touch ``recognize``:

* ROI change detection -- a lightweight text-region signature decides
  whether the same crop is being seen again;
* result caching -- an unchanged ROI returns the cached OCR result
  instead of re-running segmentation/classification;
* multi-frame logits fusion -- per-position Top-K evidence from recent
  frames is summed so a single bad frame cannot flip a stable character;
* stable text voting -- once enough frames agree, the agreed text is
  returned and the flickering frame is suppressed.

Nothing in this module mutates ``FixedFontOCR`` and the baseline OCR tests
never construct a tracker, so temporal state cannot leak into them.
"""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .lexicon import apply_lexicon, is_lexicon_ref
from .types import CharResult, DecodePath, OCRResult, VisualCandidate, VisualScores


@dataclass(frozen=True)
class TrackerConfig:
    """Tuning knobs for :class:`FrameTracker`.

    ``window`` is how many recent frames participate in voting/fusion.
    ``min_frames`` is the minimum number of observations before stable
    text voting may fire; ``vote_ratio`` is the fraction of the window that
    must agree. ``fusion_min_frames`` is the (smaller) threshold after
    which per-character Top-K evidence starts being fused.
    """

    window: int = 8
    min_frames: int = 3
    vote_ratio: float = 0.6
    fusion_min_frames: int = 2
    cache_size: int = 32
    cache_enabled: bool = True
    fusion_enabled: bool = True
    stable_voting_enabled: bool = True
    bright_text: bool = True
    roi_threshold: int = 128
    roi_quantize: int = 2
    signature_width: int = 24
    signature_height: int = 8
    align_px: float = 8.0

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.min_frames < 1:
            raise ValueError("min_frames must be >= 1")
        if not 0.0 < self.vote_ratio <= 1.0:
            raise ValueError("vote_ratio must be in (0, 1]")
        if self.fusion_min_frames < 1:
            raise ValueError("fusion_min_frames must be >= 1")
        if self.cache_size < 1:
            raise ValueError("cache_size must be >= 1")
        if self.signature_width < 1 or self.signature_height < 1:
            raise ValueError("signature grid must be at least 1x1")


@dataclass
class _Position:
    """Per-character evidence extracted from one OCRResult."""

    x: float
    y: float
    w: float
    h: float
    char: str
    confidence: float
    logits: dict[str, float] = field(default_factory=dict)


@dataclass
class _FrameEvidence:
    """One frame's per-character Top-K evidence."""

    text: str
    positions: list[_Position] = field(default_factory=list)


@dataclass
class _HistoryItem:
    """A recognized frame plus its extracted evidence."""

    result: OCRResult
    evidence: _FrameEvidence


@dataclass
class _FusedPosition:
    """Aligned multi-frame position with summed Top-K logits."""

    x: float
    y: float
    w: float
    h: float
    logits: dict[str, float] = field(default_factory=dict)
    votes: Counter[str] = field(default_factory=Counter)
    confidences: list[float] = field(default_factory=list)
    latest_char: str = ""
    char: str = ""
    confidence: float = 0.0

    def add(self, pos: _Position) -> None:
        for ch, value in pos.logits.items():
            self.logits[ch] = self.logits.get(ch, 0.0) + float(value)
        self.votes[pos.char] += 1
        self.confidences.append(float(pos.confidence))
        self.latest_char = pos.char
        self.x = pos.x
        self.y = pos.y
        self.w = pos.w
        self.h = pos.h


@dataclass
class _Slot:
    """Temporal history for one ROI + call configuration."""

    key: tuple[Any, ...]
    items: list[_HistoryItem] = field(default_factory=list)
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


def _grid_edges(size: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-empty, integer-aligned block edges for downsampling ``size``."""

    if size <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = np.floor(np.linspace(0, size, count + 1)[:-1]).astype(np.int64)
    ends = np.maximum(
        np.ceil(np.linspace(0, size, count + 1)[1:]).astype(np.int64),
        starts + 1,
    )
    starts = np.clip(starts, 0, size - 1)
    ends = np.clip(ends, 1, size)
    ends = np.maximum(ends, starts + 1)
    return starts, ends


class FrameTracker:
    """Cross-frame tracker layered over any ``ocr.recognize(image, ...)``.

    Parameters mirror :class:`TrackerConfig`; a complete config object can
    also be passed as ``config=``.
    """

    def __init__(
        self,
        ocr: Any,
        *,
        config: TrackerConfig | None = None,
        window: int | None = None,
        min_frames: int | None = None,
        vote_ratio: float | None = None,
        fusion_min_frames: int | None = None,
        cache_size: int | None = None,
        cache_enabled: bool | None = None,
        fusion_enabled: bool | None = None,
        stable_voting_enabled: bool | None = None,
        bright_text: bool | None = None,
        roi_threshold: int | None = None,
        roi_quantize: int | None = None,
        signature_width: int | None = None,
        signature_height: int | None = None,
        align_px: float | None = None,
    ):
        if callable(ocr):
            self._recognize = ocr
        elif callable(getattr(ocr, "recognize", None)):
            self._recognize = ocr.recognize
        else:
            raise TypeError("FrameTracker needs a recognize() callable")
        base = config or TrackerConfig()
        self.config = TrackerConfig(
            window=window if window is not None else base.window,
            min_frames=min_frames if min_frames is not None else base.min_frames,
            vote_ratio=vote_ratio if vote_ratio is not None else base.vote_ratio,
            fusion_min_frames=(
                fusion_min_frames
                if fusion_min_frames is not None
                else base.fusion_min_frames
            ),
            cache_size=cache_size if cache_size is not None else base.cache_size,
            cache_enabled=(
                cache_enabled if cache_enabled is not None else base.cache_enabled
            ),
            fusion_enabled=(
                fusion_enabled if fusion_enabled is not None else base.fusion_enabled
            ),
            stable_voting_enabled=(
                stable_voting_enabled
                if stable_voting_enabled is not None
                else base.stable_voting_enabled
            ),
            bright_text=(
                bright_text if bright_text is not None else base.bright_text
            ),
            roi_threshold=(
                roi_threshold if roi_threshold is not None else base.roi_threshold
            ),
            roi_quantize=(
                roi_quantize if roi_quantize is not None else base.roi_quantize
            ),
            signature_width=(
                signature_width
                if signature_width is not None
                else base.signature_width
            ),
            signature_height=(
                signature_height
                if signature_height is not None
                else base.signature_height
            ),
            align_px=align_px if align_px is not None else base.align_px,
        )
        self._ocr = ocr
        self._cache: OrderedDict[tuple[Any, ...], OCRResult] = OrderedDict()
        self._slots: dict[tuple[Any, ...], _Slot] = {}
        self._last_roi_box: tuple[int, int, int, int] | None = None
        self._last_signature: bytes | None = None
        self._last_changed = True
        self._last_result: OCRResult | None = None
        self._stable_text = ""
        self._fused_text = ""
        self.frames = 0
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        image: np.ndarray,
        *args: Any,
        roi: tuple[int, int, int, int] | None = None,
        region: tuple[int, int, int, int] | None = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Track one frame and return the stabilized OCR result.

        ``image`` is validated exactly like ``recognize`` (uint8 RGB).
        ``roi`` optionally restricts ROI change detection to ``(x, y, w,
        h)``; the full image is still recognized unless the OCR engine is
        given a crop by the caller.
        """

        image = np.asarray(image)
        if roi is None:
            roi = region
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"expected RGB image with shape (H, W, 3), got {image.shape}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")

        call_key = self._call_key(args, kwargs)
        roi_box = self._roi_box(image, roi)
        signature = self._signature(image, roi_box)

        changed = (
            self._last_roi_box is not None
            and (roi_box != self._last_roi_box or signature != self._last_signature)
        )
        self._last_roi_box = roi_box
        self._last_signature = signature
        self._last_changed = bool(changed)

        slot_key = (self._quantize(roi_box), call_key)
        cache_key = (roi_box, signature, call_key)
        self.frames += 1

        if self.config.cache_enabled and cache_key in self._cache:
            result = self._cache[cache_key]
            self._cache.move_to_end(cache_key)
            self.hits += 1
        else:
            result = self._recognize(image, *args, **kwargs)
            self.misses += 1
            if self.config.cache_enabled:
                self._store_cache(cache_key, result)

        slot = self._slots.get(slot_key)
        if slot is None:
            slot = _Slot(key=slot_key)
            self._slots[slot_key] = slot
        slot.args = args
        slot.kwargs = dict(kwargs)
        slot.items.append(_HistoryItem(result=result, evidence=self._evidence(result)))
        del slot.items[: -self.config.window]

        self._last_result = self._select(slot)
        return self._last_result

    def recognize(
        self,
        image: np.ndarray,
        *args: Any,
        roi: tuple[int, int, int, int] | None = None,
        region: tuple[int, int, int, int] | None = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Alias for :meth:`update`."""
        return self.update(image, *args, roi=roi, region=region, **kwargs)

    def reset(self) -> None:
        """Clear all temporal state (cache, histories, counters)."""
        self._cache.clear()
        self._slots.clear()
        self._last_roi_box = None
        self._last_signature = None
        self._last_changed = True
        self._last_result = None
        self._stable_text = ""
        self._fused_text = ""
        self.frames = 0
        self.hits = 0
        self.misses = 0

    @property
    def last_result(self) -> OCRResult | None:
        """Most recent stabilized result."""
        return self._last_result

    @property
    def result(self) -> OCRResult | None:
        """Alias for :attr:`last_result`."""
        return self._last_result

    @property
    def last_roi_box(self) -> tuple[int, int, int, int] | None:
        """Most recently observed text ROI ``(x, y, w, h)``."""
        return self._last_roi_box

    @property
    def roi_changed(self) -> bool:
        """Whether the previous update saw a different ROI/signature."""
        return self._last_changed

    @property
    def stable_text(self) -> str:
        """Current majority-voted text (empty until enough frames agree)."""
        return self._stable_text

    @property
    def stable(self) -> bool:
        """Whether stable text voting is currently active."""
        return bool(self._stable_text)

    @property
    def fused_text(self) -> str:
        """Current logits-fused text (empty until enough frames exist)."""
        return self._fused_text

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def cache(self) -> dict[tuple[Any, ...], OCRResult]:
        """Current result cache (ROI+signature -> result)."""
        return dict(self._cache)

    @property
    def history(self) -> tuple[OCRResult, ...]:
        """Recent results of the active ROI slot, oldest first."""
        if not self._slots:
            return ()
        slot = next(reversed(self._slots.values()))
        return tuple(item.result for item in slot.items)

    # ------------------------------------------------------------------
    # ROI / signature
    # ------------------------------------------------------------------

    def _roi_box(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int]:
        h, w = image.shape[:2]
        if roi is not None:
            if len(roi) != 4:
                raise ValueError("roi must be (x, y, w, h)")
            x, y, rw, rh = (int(v) for v in roi)
            if x < 0 or y < 0 or rw < 0 or rh < 0 or x + rw > w or y + rh > h:
                raise ValueError(f"roi {roi!r} is outside image shape {(h, w)}")
            return (x, y, rw, rh)
        gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        if self.config.bright_text:
            mask = gray >= self.config.roi_threshold
        else:
            mask = gray <= self.config.roi_threshold
        if not mask.any():
            return (0, 0, 0, 0)
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        pad = max(1, min(4, (x1 - x0) // 4, (y1 - y0) // 4))
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(h, y1 + pad)
        x1 = min(w, x1 + pad)
        return (x0, y0, x1 - x0, y1 - y0)

    def _signature(
        self,
        image: np.ndarray,
        roi_box: tuple[int, int, int, int],
    ) -> bytes:
        x, y, w, h = roi_box
        if w <= 0 or h <= 0:
            return b"empty"
        crop = image[y : y + h, x : x + w]
        row_starts, row_ends = _grid_edges(h, self.config.signature_height)
        col_starts, col_ends = _grid_edges(w, self.config.signature_width)
        cells: list[np.ndarray] = []
        for i, (r0, r1) in enumerate(zip(row_starts, row_ends)):
            for j, (c0, c1) in enumerate(zip(col_starts, col_ends)):
                cells.append(
                    crop[r0:r1, c0:c1].mean(axis=(0, 1))
                )
        sig = np.stack(cells).astype(np.uint8)
        return hashlib.sha256(sig.tobytes()).digest()

    def _quantize(self, roi_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        q = max(1, int(self.config.roi_quantize))
        return tuple(v // q for v in roi_box)  # type: ignore[return-value]

    @staticmethod
    def _call_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        def norm(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return ("ndarray", value.shape, value.dtype.str, value.tobytes())
            if isinstance(value, Path):
                return str(value)
            try:
                hash(value)
            except TypeError:
                return repr(value)
            return value

        return (
            tuple(norm(a) for a in args),
            tuple(sorted((k, norm(v)) for k, v in kwargs.items())),
        )

    # ------------------------------------------------------------------
    # Evidence extraction / fusion / voting
    # ------------------------------------------------------------------

    def _charset(self) -> list[str] | None:
        charset = getattr(self._ocr, "charset", None)
        if charset is None:
            model = getattr(self._ocr, "model", None)
            charset = getattr(model, "charset", None)
        if charset is None:
            return None
        return list(charset)

    def _evidence(self, result: OCRResult) -> _FrameEvidence:
        charset = self._charset()
        positions: list[_Position] = []
        path = result.path
        candidates: tuple[VisualCandidate, ...] = ()
        if path is not None:
            candidates = path.candidates
        for i, char_result in enumerate(result.chars):
            logits: dict[str, float] = {}
            cand = candidates[i] if i < len(candidates) else None
            if cand is not None and cand.scores is not None:
                scores: VisualScores = cand.scores
                if charset is not None:
                    for cid, value in zip(scores.char_ids, scores.logits):
                        cid = int(cid)
                        if 0 <= cid < len(charset):
                            logits[charset[cid]] = float(value)
                elif scores.logits:
                    # Without a charset mapping keep only the frame's chosen
                    # character, using its confidence as its score.
                    logits[char_result.char] = float(char_result.confidence)
            if not logits:
                logits[char_result.char] = float(char_result.confidence)
            positions.append(
                _Position(
                    x=float(char_result.x),
                    y=float(char_result.y),
                    w=float(char_result.w),
                    h=float(char_result.h),
                    char=char_result.char,
                    confidence=float(char_result.confidence),
                    logits=logits,
                )
            )
        return _FrameEvidence(text=result.text, positions=positions)

    def _select(self, slot: _Slot) -> OCRResult:
        latest = slot.items[-1].result
        stable = self._stable(slot) if self.config.stable_voting_enabled else None
        if stable is not None:
            self._stable_text = stable.text
            self._fused_text = self._fuse_text(slot)
            return stable

        self._stable_text = ""
        fused = self._fused(slot) if self.config.fusion_enabled else None
        if fused is not None:
            self._fused_text = fused.text
            return fused
        self._fused_text = ""
        return latest

    def _stable(self, slot: _Slot) -> OCRResult | None:
        n = len(slot.items)
        if n < self.config.min_frames:
            return None
        counts = Counter(item.result.text for item in slot.items)
        text, count = counts.most_common(1)[0]
        if count / n < self.config.vote_ratio:
            return None
        for item in reversed(slot.items):
            if item.result.text == text:
                return item.result
        return None

    def _fuse_text(self, slot: _Slot) -> str:
        if not slot.items:
            return ""
        fused = _fuse_positions(
            [item.evidence for item in slot.items],
            align_px=self.config.align_px,
        )
        return fused.text if fused is not None else ""

    def _fused(self, slot: _Slot) -> OCRResult | None:
        if len(slot.items) < self.config.fusion_min_frames:
            return None
        fused = _fuse_positions(
            [item.evidence for item in slot.items],
            align_px=self.config.align_px,
        )
        if fused is None:
            return None
        latest = slot.items[-1].result
        if fused.text == latest.text:
            return None
        return self._build_fused_result(slot, latest, fused)

    def _build_fused_result(
        self,
        slot: _Slot,
        latest: OCRResult,
        fused: _FusedEvidence,
    ) -> OCRResult:
        chars = tuple(
            CharResult(
                char=pos.char,
                x=round(pos.x),
                y=round(pos.y),
                w=round(max(pos.w, 1)),
                h=round(max(pos.h, 1)),
                confidence=pos.confidence,
            )
            for pos in fused.positions
        )
        alternatives: list[str] = []
        for item in reversed(slot.items):
            if item.result.text and item.result.text not in alternatives:
                alternatives.append(item.result.text)
        for alt in latest.alternatives:
            if alt and alt not in alternatives:
                alternatives.append(alt)

        path = latest.path
        if path is not None:
            path = replace(path, text=fused.text)
        out = replace(
            latest,
            text=fused.text,
            confidence=fused.confidence,
            chars=chars,
            alternatives=tuple(alternatives[:8]),
            matched_term=None,
            matched_span=None,
            lexicon_match=None,
            path=path,
        )
        return self._apply_lexicon(out, slot)

    def _apply_lexicon(self, result: OCRResult, slot: _Slot) -> OCRResult:
        lexicon, mode = self._slot_lexicon(slot)
        if lexicon is None or mode is None:
            return result
        try:
            return apply_lexicon(
                result,
                lexicon,
                mode,
                charset=self._charset(),
            )
        except Exception:
            # The underlying OCR engine already applied the lexicon; if the
            # fused result cannot be re-annotated, keep the visible text.
            return result

    def _slot_lexicon(
        self,
        slot: _Slot,
    ) -> tuple[Any, str | None]:
        kwargs = slot.kwargs
        lexicon = kwargs.get("lexicon")
        mode = kwargs.get("lexicon_mode")
        args = slot.args
        if lexicon is None and args:
            if is_lexicon_ref(args[0]):
                lexicon = args[0]
                if len(args) > 1:
                    mode = args[1]
            elif len(args) >= 2 and is_lexicon_ref(args[1]):
                lexicon = args[1]
                if len(args) > 2:
                    mode = args[2]
        if mode is None and lexicon is not None:
            mode = "prefer"
        return lexicon, mode

    def _store_cache(
        self,
        cache_key: tuple[Any, ...],
        result: OCRResult,
    ) -> None:
        self._cache[cache_key] = result
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)


@dataclass
class _FusedEvidence:
    text: str
    confidence: float
    positions: list[_FusedPosition] = field(default_factory=list)


def _fuse_positions(
    evidences: list[_FrameEvidence],
    *,
    align_px: float,
) -> _FusedEvidence | None:
    """Align per-character evidence across frames and sum Top-K logits."""

    if not evidences:
        return None
    slots: list[_FusedPosition] = []
    for evidence in evidences:
        matched: set[int] = set()
        for pos in evidence.positions:
            best_idx: int | None = None
            best_dist = float("inf")
            for i, slot in enumerate(slots):
                if i in matched:
                    continue
                dist = abs(slot.x - pos.x)
                limit = max(slot.w, pos.w, align_px) * 1.5
                if dist <= limit and dist < best_dist:
                    best_idx = i
                    best_dist = dist
            if best_idx is not None:
                matched.add(best_idx)
                slots[best_idx].add(pos)
            else:
                slot = _FusedPosition(
                    x=pos.x,
                    y=pos.y,
                    w=pos.w,
                    h=pos.h,
                )
                slot.add(pos)
                slots.append(slot)
    if not slots:
        return None

    slots.sort(key=lambda s: s.x)
    confidences: list[float] = []
    for slot in slots:
        if not slot.logits:
            slot.char = slot.latest_char
            slot.confidence = float(np.mean(slot.confidences)) if slot.confidences else 0.0
            confidences.append(slot.confidence)
            continue
        best_char = max(
            slot.logits,
            key=lambda ch: (
                slot.logits[ch],
                slot.votes[ch],
                1 if ch == slot.latest_char else 0,
            ),
        )
        slot.char = best_char
        slot.confidence = float(np.mean(slot.confidences)) if slot.confidences else 0.0
        confidences.append(slot.confidence)

    text = "".join(slot.char for slot in slots)
    return _FusedEvidence(
        text=text,
        confidence=float(np.mean(confidences)) if confidences else 0.0,
        positions=slots,
    )
