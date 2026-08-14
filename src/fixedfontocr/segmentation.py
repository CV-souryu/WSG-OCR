"""Candidate-lattice segmentation with a visual DP decoder (P0).

The old pipeline merged connected components irreversibly before
classification, so fragment-heavy glyphs such as ``鲃`` or ``小`` could never
be recovered. The lattice instead keeps every original component and
generates all plausible consecutive candidates (single components plus
merges of up to ``max_merge_components``), scores them with the classifier
scorer, then decodes the best path with dynamic programming.

Path score is the *mean* visual score of the candidates on the path, with
small geometry penalties. Using the mean (instead of the sum) is essential:
summing would reward splitting one glyph into many fragments because each
fragment contributes positive score.

    components  C0  C1  C2
    candidates  C0, C0+C1, C0+C1+C2, C1, C1+C2, C2, ...
    edges       0->1, 0->2, 0->3, 1->2, 1->3, 2->3
    DP          best-scoring path from component 0 to component N
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .preprocess import (
    Segment,
    _bbox_segment,
    _connected_components,
    _union,
    normalize,
)
from .scorer import SegmentScore, SegmentScorer
from .types import Profile


@dataclass
class Candidate:
    """One candidate segment covering ``components[start:end]``."""

    segment: Segment
    start: int
    end: int  # exclusive
    components: tuple[int, ...]
    score: SegmentScore | None = None
    geometry: float = 0.0

    @property
    def total_score(self) -> float:
        if self.score is None:
            return self.geometry
        return self.score.visual_score + self.geometry


@dataclass(frozen=True)
class DecodedPath:
    """Result of decoding one line: the chosen candidates."""

    candidates: tuple[Candidate, ...] = field(default_factory=tuple)
    mean_score: float = 0.0


def connected_components(line: Segment) -> list[Segment]:
    """Original 4-connected components of a line, left to right."""

    comps = [_bbox_segment(c, line.y) for c in _connected_components(line.mask)]
    comps.sort(key=lambda s: (s.x, s.y))
    return comps


def build_candidates(
    comps: list[Segment],
    profile: Profile,
    max_merge_components: int = 3,
) -> list[Candidate]:
    """Generate all consecutive candidates up to ``max_merge_components``.

    Only consecutive component ranges are considered (``C0+C2`` without
    ``C1`` is not a character candidate). Obviously impossible combinations
    are filtered by size and internal-gap bounds before any classification
    work happens.
    """

    if max_merge_components < 1:
        raise ValueError("max_merge_components must be >= 1")
    n = len(comps)
    cands: list[Candidate] = []
    for i in range(n):
        for j in range(i + 1, min(n, i + max_merge_components) + 1):
            parts = tuple(range(i, j))
            if not _passes_filters([comps[k] for k in parts], profile):
                continue
            if j - i == 1:
                seg = comps[i]
            else:
                acc = comps[i]
                for k in range(i + 1, j):
                    acc = _union(acc, comps[k])
                seg = acc
            cands.append(Candidate(segment=seg, start=i, end=j, components=parts))
    return cands


def _passes_filters(parts: list[Segment], profile: Profile) -> bool:
    """Reject candidate shapes that cannot be a single character."""

    if len(parts) == 1:
        seg = parts[0]
        return seg.w <= profile.char_width_max and seg.h <= profile.char_height_max

    x0 = min(p.x for p in parts)
    y0 = min(p.y for p in parts)
    x1 = max(p.x + p.w for p in parts)
    y1 = max(p.y + p.h for p in parts)
    w, h = x1 - x0, y1 - y0
    if w > profile.char_width_max or h > profile.char_height_max:
        return False
    if h < max(2, profile.char_height_min):
        return False
    # Conservative internal-gap bound: a merge across a wide blank is a
    # different character, not a fragmented glyph. The bound is generous
    # because strokes of one glyph (e.g. the right dots of 亲) can be a
    # dozen pixels apart; the visual DP is what rejects actual
    # character-to-character merges.
    max_gap = max(
        parts[k + 1].x - (parts[k].x + parts[k].w) for k in range(len(parts) - 1)
    )
    gap_limit = max(2, profile.char_height_min) + 2
    return max_gap <= gap_limit


def geometry_score(
    candidate: Candidate,
    comps: list[Segment],
    profile: Profile,
) -> float:
    """Small geometric adjustments on top of the classifier visual score.

    Penalties are deliberately small (<= 0.06) so the classifier remains the
    dominant signal, but they break ties between a fragmented glyph path and
    its merged candidate and keep clearly-adjacent characters apart.
    """

    score = 0.0
    seg = candidate.segment

    # Vertical alignment: character centers should sit in one band.
    centers = [c.y + c.h / 2 for c in comps]
    median_center = float(np.median(centers))
    offset = abs(seg.y + seg.h / 2 - median_center)
    if offset > max(1.0, profile.char_height_min * 0.15):
        score -= min(0.04, offset * 0.004)

    # Internal gaps inside a merged candidate.
    if len(candidate.components) > 1:
        gaps = [
            comps[k + 1].x - (comps[k].x + comps[k].w)
            for k in range(candidate.components[0], candidate.components[-1])
        ]
        excess = max(0, max(gaps) - max(1, profile.char_spacing))
        score -= min(0.04, excess * 0.01)

        # Merged candidates much wider than the line's typical glyph are
        # almost always two adjacent characters.
        heights = [c.h for c in comps]
        median_h = float(np.median(heights))
        typical = [
            c.w for c in comps if c.h >= max(2, 0.5 * median_h)
        ]
        expected = float(np.median(typical)) if typical else float(median_h)
        if seg.w > expected * 1.9:
            score -= 0.05

    return score


def decode(
    candidates: list[Candidate],
    n_components: int,
) -> DecodedPath:
    """Visual DP over the candidate lattice (max mean score path).

    ``dp[end]`` stores the best (sum, count) covering components
    ``0..end-1``; ties are broken toward fewer candidates (i.e. prefer the
    merged glyph over a tie of its fragments).
    """

    by_end: dict[int, list[Candidate]] = {}
    for c in candidates:
        by_end.setdefault(c.end, []).append(c)

    NEG = -1e18
    dp_sum = [NEG] * (n_components + 1)
    dp_cnt = [0] * (n_components + 1)
    dp_prev: list[list | None] = [None] * (n_components + 1)
    dp_sum[0] = 0.0

    for end in range(1, n_components + 1):
        best_sum: float | None = None
        best_cnt = 0
        best_prev = None
        best_cand: Candidate | None = None
        for c in by_end.get(end, []):
            if dp_sum[c.start] <= NEG / 2:
                continue
            s = dp_sum[c.start] + c.total_score
            cnt = dp_cnt[c.start] + 1
            mean = s / cnt
            if best_sum is None or mean > best_sum / best_cnt + 1e-9 or (
                abs(mean - best_sum / best_cnt) <= 1e-9 and cnt < best_cnt
            ):
                best_sum, best_cnt, best_prev, best_cand = s, cnt, c.start, c
        if best_cand is not None:
            dp_sum[end] = best_sum
            dp_cnt[end] = best_cnt
            dp_prev[end] = [best_prev, best_cand]

    if dp_sum[n_components] <= NEG / 2:
        return DecodedPath(candidates=(), mean_score=0.0)

    chosen: list[Candidate] = []
    pos = n_components
    while pos > 0:
        prev, cand = dp_prev[pos]  # type: ignore[misc]
        chosen.append(cand)
        pos = prev
    chosen.reverse()
    return DecodedPath(
        candidates=tuple(chosen),
        mean_score=float(dp_sum[n_components] / dp_cnt[n_components]),
    )


def segment_line(
    line: Segment,
    profile: Profile,
    scorer: SegmentScorer,
    allowed_ids: set[int] | None = None,
    max_merge_components: int = 4,
) -> DecodedPath:
    """Segment one line with the candidate lattice + visual DP."""

    comps = connected_components(line)
    if not comps:
        return DecodedPath(candidates=(), mean_score=0.0)
    candidates = build_candidates(comps, profile, max_merge_components)
    if not candidates:
        return DecodedPath(candidates=(), mean_score=0.0)

    segments = [c.segment for c in candidates]
    scores = scorer.score(segments, allowed_ids)
    for cand, score in zip(candidates, scores):
        cand.score = score
        cand.geometry = geometry_score(cand, comps, profile)
    candidates = _drop_weak_merges(candidates, scorer, comps, profile)
    if not candidates:
        return DecodedPath(candidates=(), mean_score=0.0)
    return decode(candidates, len(comps))


def _drop_weak_merges(
    candidates: list[Candidate],
    scorer: SegmentScorer,
    comps: list[Segment],
    profile: Profile,
) -> list[Candidate]:
    """Drop low-confidence multi-component candidates (CNN-only models).

    The pure visual DP can merge adjacent characters when the CNN's margin
    for a combined blob is only slightly better than its margin for each
    part (e.g. three ``0``s merged into one ``0``). Template/hybrid models
    get a confident template score for real glyphs, so they do not need this
    gate. CNN-only models only allow a merge when:

    * the merged candidate looks like a real character
      (``cnn_merge_threshold``);
    * it scores at least as well as every single component inside it; and
    * the components are in glyph-internal proximity (the old merger's
      overlap rule), so unrelated characters such as ``1000`` cannot merge.
    """

    threshold = getattr(scorer, "cnn_merge_threshold", 0.0)
    if threshold <= 0.0 or not candidates:
        return candidates
    single_visual: dict[int, float] = {}
    for cand in candidates:
        if len(cand.components) == 1 and cand.score is not None:
            i = cand.components[0]
            single_visual[i] = max(
                single_visual.get(i, -1.0), cand.score.visual_score
            )
    out = []
    for cand in candidates:
        if len(cand.components) > 1 and cand.score is not None:
            if cand.score.visual_score < threshold:
                continue
            if not _proximity_merge_ok(cand, comps, profile):
                continue
            parts = [
                single_visual.get(i, threshold)
                for i in range(cand.components[0], cand.components[-1] + 1)
            ]
            if any(p > cand.score.visual_score + 0.02 for p in parts):
                continue
        out.append(cand)
    return out


def _proximity_merge_ok(
    cand: Candidate,
    comps: list[Segment],
    profile: Profile,
) -> bool:
    """Reject merges of unrelated characters using the old merger's rule.

    Two components belong to the same glyph when they overlap in x with a
    small horizontal gap, or overlap in y with a small vertical gap. The
    candidate's components must form one proximity-connected cluster (the
    old merger was transitive: 亲's right dot touches the main body even
    though it is 10px away from the middle stroke). This is exactly the
    proximity rule the pre-lattice pipeline used, applied here as a
    CNN-only safety net.
    """

    x_gap_tol = max(1, profile.stroke_width)
    y_gap_tol = max(3, profile.char_height_min // 2)
    idx = list(cand.components)
    parent = list(range(len(idx)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = comps[idx[i]], comps[idx[j]]
            gap_x = max(a.x, b.x) - min(a.x + a.w, b.x + b.w)
            gap_y = max(a.y, b.y) - min(a.y + a.h, b.y + b.h)
            overlap_x = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
            overlap_y = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
            if (gap_x <= x_gap_tol and overlap_y >= 1) or (
                gap_y <= y_gap_tol and overlap_x >= 1
            ):
                union(i, j)
    roots = {find(i) for i in range(len(idx))}
    return len(roots) == 1
