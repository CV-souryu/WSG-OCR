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
        order = np.argsort(-masked, kind="stable")
        best = int(order[0])
        if len(idx) == 1:
            return charset[best], 1.0
        second = int(order[1])
        margin = float(logits[best] - logits[second])
        return charset[best], margin

    order = np.argsort(-logits, kind="stable")
    best = int(order[0])
    if c >= 2:
        second = int(order[1])
        margin = float(logits[best] - logits[second])
    else:
        margin = 1.0
    return charset[best], margin
