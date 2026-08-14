"""Confidence scoring and UI-limited candidate restriction.

Fixed-font OCR can be told which characters a UI element is allowed to
contain (coins are 0-9, a level field is ``Lv.`` + digits, a username is
Chinese + ASCII, ...). Restricting the classifier to that subset removes
the confusable classes that would otherwise compete at the argmax, which is
exactly where fixed-font OCR gains accuracy for free.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def top2(
    logits: NDArray[np.float32],
) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Top-1/top-2 ids and scores from an ``[N, C]`` logit matrix.

    Uses ``argmax`` plus a single ``argpartition`` over the first two
    positions, so the cost is O(N*C) instead of a full O(N*C*log C) sort.
    ``argmax`` keeps the same tie-breaking (first occurrence) as the
    previous stable ``argsort`` reference.
    """

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    n, c = logits.shape
    if n == 0:
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float32)
        return empty_i, empty_f, empty_f, empty_f

    top1 = np.argmax(logits, axis=-1)
    best = logits[np.arange(n), top1]
    if c == 1:
        second = np.full(n, -np.inf, dtype=np.float32)
    else:
        # argpartition puts the two largest values at positions 0 and 1
        # (in either order); the "second" value is the smaller of the two.
        idx = np.argpartition(-logits, 1, axis=-1)[:, :2]
        v = logits[np.arange(n)[:, None], idx]
        second = np.minimum(v[:, 0], v[:, 1])
    margins = best - second
    return (
        np.asarray(top1, dtype=np.int32),
        np.asarray(best, dtype=np.float32),
        np.asarray(second, dtype=np.float32),
        np.asarray(margins, dtype=np.float32),
    )


def topk(
    logits: NDArray[np.float32],
    k: int,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Ranked Top-K ids and values from an ``[N, C]`` logit matrix.

    ``argpartition`` selects the top ``k`` columns in O(C) per row; only the
    selected ``k`` elements are then ordered (by descending logit, then by
    original column index for ties), so no full ``argsort`` over the charset
    is ever performed. Returns ``(ids, values)`` as ``[N, k]`` arrays.
    """

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    n, c = logits.shape
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, c)
    if n == 0:
        return (
            np.empty((0, k), dtype=np.int32),
            np.empty((0, k), dtype=np.float32),
        )
    if k == 1:
        ids = np.argmax(logits, axis=-1)
        values = logits[np.arange(n), ids]
        return (
            np.asarray(ids[:, None], dtype=np.int32),
            np.asarray(values[:, None], dtype=np.float32),
        )
    idx = np.argpartition(-logits, k - 1, axis=-1)[:, :k]
    values = logits[np.arange(n)[:, None], idx]
    cols = np.broadcast_to(np.arange(c), logits.shape)
    selected_cols = np.take_along_axis(cols, idx, axis=-1)
    order = np.lexsort((selected_cols, -values), axis=-1)
    ids = np.take_along_axis(idx, order, axis=-1)
    values = np.take_along_axis(values, order, axis=-1)
    return np.asarray(ids, dtype=np.int32), np.asarray(values, dtype=np.float32)


def second_ids(
    logits: NDArray[np.float32],
    ids: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Return the second-ranked character id per row (``-1`` if absent)."""
    logits = np.asarray(logits, dtype=np.float32)
    n, c = logits.shape
    if n == 0 or c <= 1:
        return np.full(n, -1, dtype=np.int32)
    best = np.asarray(ids, dtype=np.int64)
    # argpartition puts the two largest columns at positions 0 and 1; the
    # second id is whichever of those is not the argmax.
    idx = np.argpartition(-logits, 1, axis=-1)[:, :2]
    second_idx = np.where(idx[:, 0] != best, idx[:, 0], idx[:, 1])
    row = np.arange(n)
    second_logits = logits[row, second_idx]
    out = np.where(np.isneginf(second_logits), -1, second_idx)
    return np.asarray(out, dtype=np.int32)


def allowed_ids(charset: list[str], allowed_chars: str | None) -> set[int] | None:
    """Map an ``allowed_chars`` string to charset indices (``None`` = all)."""
    if allowed_chars is None:
        return None
    return {i for i, ch in enumerate(charset) if ch in allowed_chars}


def pick(
    logits: NDArray[np.float32],
    charset: list[str],
    allowed: set[int] | None = None,
) -> tuple[str, float]:
    """Top-1 character and confidence from one ``(C,)`` logit vector.

    Without ``allowed`` the confidence is the top1-top2 logit margin (the
    same value the backends return). With ``allowed`` the top-1/top-2 are
    computed inside the allowed subset; a single allowed class gets a
    confidence of 1.0.
    """

    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    c = logits.shape[0]
    if c != len(charset):
        raise ValueError(f"logits has {c} classes but charset has {len(charset)}")
    if allowed is not None and not allowed:
        return "?", 0.0

    if allowed is not None:
        idx = np.fromiter(sorted(allowed), dtype=np.int64)
        masked = np.full(c, -np.inf, dtype=np.float32)
        masked[idx] = logits[idx]
        ids, top1, top2, margin = top2(masked)
        best = int(ids[0])
        if len(idx) == 1:
            return charset[best], 1.0
        return charset[best], float(margin[0])

    ids, top1, top2, margin = top2(logits)
    best = int(ids[0])
    if c >= 2:
        return charset[best], float(margin[0])
    else:
        return charset[best], 1.0
