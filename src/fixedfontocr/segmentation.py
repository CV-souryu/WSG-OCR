"""Candidate-lattice segmentation with a visual DP decoder (P0).

The old pipeline merged connected components irreversibly before
classification, so fragment-heavy glyphs such as ``鲃`` or ``小`` could never
be recovered. The lattice instead keeps every original component and
generates all plausible consecutive candidates (single components plus
merges of up to ``max_merge_components`` plus ``split(Cx)`` atoms for wide
components), scores them with the classifier scorer, then decodes the best
path with dynamic programming.

Path score is the *mean* visual score of the candidates on the path, with
small geometry penalties. Using the mean (instead of the sum) is essential:
summing would reward splitting one glyph into many fragments because each
fragment contributes positive score.

    components  C0  C1  C2
    atoms       C0a C0b C1 C2   (C0 split into two pieces)
    candidates  C0a, C0b, C0a+C0b(=C0), C0a+C0b+C1, ..., C1, C1+C2, C2, ...
    edges       best-scoring path from atom 0 to atom N
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .decoder import DecoderConfig, decode_beam, decode_dp
from .geometry import FontGeometryDatabase, char_geometry
from .lexicon import Lexicon, load_lexicon
from .preprocess import (
    Segment,
    _bbox_segment,
    _connected_components,
    _union,
    glyph_normalize_geometry,
    normalize,
)
from .scorer import SegmentScore, SegmentScorer
from .types import Component, DecodePath, Profile, VisualCandidate, VisualLattice

# Goal 1 names: ``Candidate``/``DecodedPath`` remain as compatibility aliases
# for the previous internal names, but the pipeline now speaks the new core
# vocabulary (VisualCandidate / DecodePath / VisualLattice).
Candidate = VisualCandidate
DecodedPath = DecodePath


@dataclass(frozen=True)
class _Atom:
    """One atomic slice of a connected component.

    Most components have exactly one atom (the component itself). A wide
    component that provably contains several glyphs is split at vertical
    valleys into multiple atoms; the original component is then available to
    the decoder as the merge of those atoms, while each atom is also its own
    ``split(Cx)`` candidate. The lattice therefore never destroys the
    original components -- it only adds finer-grained alternatives.
    """

    component_index: int
    segment: Component


def connected_components(line: Segment) -> list[Component]:
    """Original 4-connected components of a line, left to right.

    ``_bbox_segment`` returns coordinates relative to the line mask; the
    line's own image-space offset is added so every component's ``x/y`` is
    absolute (matching the :class:`Component` contract and letting the
    Visual Frontend crop soft ROIs straight from the source image).
    """

    comps: list[Component] = []
    for c in _connected_components(line.mask):
        seg = _bbox_segment(c, line.y)
        seg.x += line.x
        comps.append(seg)
    comps.sort(key=lambda s: (s.x, s.y))
    return comps


def build_candidates(
    comps: list[Component],
    profile: Profile,
    max_merge_components: int = 3,
    split_wide: bool = True,
    geometry: FontGeometryDatabase | None = None,
) -> list[VisualCandidate]:
    """Generate all consecutive candidates up to ``max_merge_components``.

    Only consecutive ranges are considered (``C0+C2`` without ``C1`` is not
    a character candidate). Wide components are additionally split at
    vertical valleys into ``split(Cx)`` atoms, so a connected two-glyph blob
    is represented by its whole-component candidate and by each split piece.
    Obviously impossible combinations are filtered by size, vertical
    proximity, internal-gap and ink-area bounds before classification.
    """

    if max_merge_components < 1:
        raise ValueError("max_merge_components must be >= 1")
    atoms, expected_width = expand_atoms(
        comps,
        profile,
        max_merge_components,
        split_wide=split_wide,
        geometry=geometry,
    )
    n = len(atoms)
    cands: list[Candidate] = []
    for i in range(n):
        for j in range(i + 1, min(n, i + max_merge_components) + 1):
            parts = atoms[i:j]
            if not _passes_filters(
                parts, profile, expected_width, geometry=geometry
            ):
                continue
            if j - i == 1:
                seg = parts[0].segment
            else:
                acc = parts[0].segment
                for part in parts[1:]:
                    acc = _union(acc, part.segment)
                seg = acc
            comp_indices = tuple(sorted({p.component_index for p in parts}))
            cands.append(
                Candidate(
                    segment=seg,
                    start=comp_indices[0],
                    end=comp_indices[-1] + 1,
                    components=comp_indices,
                    atoms=(i, j),
                )
            )
    return cands


def _estimate_expected_width(
    comps: list[Component],
    profile: Profile,
    geometry: FontGeometryDatabase | None = None,
) -> float:
    """Estimate the line's typical glyph width from font geometry + ink height.

    Goal 8 replaces the old "0.8 * median height" stand-in with the
    registered font's real ratios: narrow glyphs (``1 I l i !``) use the
    narrow bbox ratio and full-width CJK uses the full-width ratio. Without
    a database the Goal 4 heuristic remains unchanged.
    """

    if not comps:
        return max(float(profile.char_width_min), 2.0)
    max_h = max(c.h for c in comps)
    tall = [c for c in comps if c.h >= max(2, 0.6 * max_h)]
    if geometry is not None:
        widths: list[float] = []
        for c in tall:
            aspect = c.w / max(c.h, 1)
            if aspect > 1.1:
                # A very wide component is either several glued glyphs
                # (甲+申 zero-gap) or a real glyph's wide part (命's 人+一,
                # 尔's top): its width is not a glyph width.  Excluding it
                # keeps the estimate on the line's normal glyphs -- capping
                # it instead would drag the median down and make the
                # component itself look splittable.  When every tall
                # component is wide (a single glued blob line), fall back
                # to the height heuristic below.
                continue
            if aspect < 0.8:
                em = max(
                    c.h / max(geometry.narrow_height_ratio, 1e-6),
                    1.0,
                )
                widths.append(em * geometry.narrow_width_ratio)
            else:
                em = max(
                    c.h / max(geometry.full_height_ratio, 1e-6),
                    1.0,
                )
                widths.append(em * geometry.full_width_ratio)
        if widths:
            expected = float(np.median(widths))
            return max(
                float(profile.char_width_min),
                min(float(profile.char_width_max), expected),
            )
        # All tall components are wide: the height heuristic is the only
        # glyph-width evidence (single glued-blob line).
        return max(
            float(profile.char_width_min),
            min(float(profile.char_width_max), max_h * 0.8),
        )
    median_h = float(np.median([c.h for c in tall]) if tall else max_h)
    return max(
        float(profile.char_width_min),
        min(float(profile.char_width_max), median_h * 0.8),
    )


def _valley_cuts(
    mask: np.ndarray,
    expected_width: float,
    profile: Profile,
    max_splits: int,
) -> list[int]:
    """Return x positions where a wide component should be split.

    A cut is the middle of a vertical valley: columns whose ink count is at
    most ``~18%`` of the component height. That catches a low-resolution
    connector between two glyphs (and the weak valley in touching real-game
    crops) while avoiding interior whitespace of closed CJK glyphs (口/日/中
    have top/bottom strokes keeping the valley columns well above the
    threshold). Edge valleys are ignored and each resulting piece must be
    wide enough to be a glyph.

    Edge valleys are deliberately *not* cut: a trailing sparse region is
    often a real glyph's own tail (命's 人+一 component ends in a thin
    stroke), and splitting it would push the glyph's whole merge past
    ``max_merge_components``.  Short trailing glyphs (``U-``) stay covered
    by their own connected component in clean rendering.
    """

    h, w = mask.shape
    proj = mask.sum(axis=0).astype(np.int32)
    threshold = max(1, int(h * 0.18))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x in range(w):
        if proj[x] <= threshold:
            if start is None:
                start = x
        elif start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, w - 1))

    min_piece = max(profile.char_width_min, int(expected_width * 0.4))
    cuts: list[int] = []
    for a, b in runs:
        if a < min_piece or w - 1 - b < min_piece:
            continue
        cut = (a + b + 1) // 2
        if not cuts or cut - cuts[-1] >= min_piece:
            cuts.append(cut)
        if len(cuts) >= max_splits:
            break
    return cuts


def _seam_cut(
    mask: np.ndarray,
    x_ideal: float,
    tol: int = 2,
    offset_weight: float = 0.2,
) -> tuple[np.ndarray, float] | None:
    """Minimum-cost vertical seam near ``x_ideal``.

    The cut path is not forced to be a straight column: it may snake one
    column per row to dodge strokes.  The cost of a path is the number of
    ink pixels it crosses plus ``offset_weight`` per row per pixel of
    deviation from ``x_ideal`` (so a near-ideal low-ink column wins over a
    far-away empty one).  Returns the ``(h,)`` path (column per row) and
    its cost, or ``None`` when the window is degenerate.
    """

    h, w = mask.shape
    lo = max(1, int(np.floor(x_ideal)) - tol)
    hi = min(w - 2, int(np.ceil(x_ideal)) + tol)
    if hi <= lo or h < 2:
        return None
    width = hi - lo + 1
    INF = float("inf")
    dp = np.full((h, width), INF, dtype=np.float64)
    back = np.zeros((h, width), dtype=np.int32)
    for j in range(width):
        x = lo + j
        dp[0, j] = float(mask[0, x]) + offset_weight * abs(x - x_ideal)
    for y in range(1, h):
        row = mask[y]
        for j in range(width):
            best = INF
            best_j = 0
            for dj in (-1, 0, 1):
                jj = j + dj
                if 0 <= jj < width:
                    v = dp[y - 1, jj]
                    if v < best:
                        best, best_j = v, jj
            dp[y, j] = best + float(row[lo + j]) + offset_weight * abs(
                lo + j - x_ideal
            )
            back[y, j] = best_j
    j = int(np.argmin(dp[h - 1]))
    cost = float(dp[h - 1, j])
    path = np.empty(h, dtype=np.int32)
    for y in range(h - 1, -1, -1):
        path[y] = lo + j
        j = back[y, j]
    return path, cost


def _forced_seam_cuts(
    mask: np.ndarray,
    expected_width: float,
    profile: Profile,
    max_splits: int,
) -> list[tuple[np.ndarray, float]]:
    """Seam cut positions at font-advance multiples and equal parts.

    Wide components whose ink has *no* valley at the glyph boundary (fully
    touching ``LV``/``甲申`` style blobs) never get split by
    :func:`_valley_cuts`.  This adds forced cut hypotheses near every
    plausible glyph boundary:

    * ``k * expected_width`` (font advance multiples, the user's
      ``期望字符宽度的整数倍 ±1~2 px``), and
    * ``k * w / n`` for ``n = round(w / expected_width)`` equal parts (so
      a line whose glyphs are all narrower than the estimate still gets a
      cut near the true boundary).

    Each ideal position is refined by a min-cost seam in a ``±2 px``
    window.  No decision is made here: every seam becomes an extra atom,
    the original whole component stays reachable as their merge, and the
    lattice decoder picks the best path.
    """

    h, w = mask.shape
    min_piece = max(profile.char_width_min, int(expected_width * 0.4))
    if h < 2 or w < 2 * min_piece + 2:
        return []

    n_est = int(round(w / max(expected_width, 1.0)))
    ideals: list[float] = []
    # Font advance multiples: k * expected_width.  Always considered --
    # they are the user's ``期望字符宽度的整数倍 ±1~2 px``.
    for k in range(1, n_est + 1):
        x = k * expected_width
        if min_piece <= x <= w - min_piece:
            ideals.append(x)
    # Equal parts w * k / n for n = round(w / expected) glyphs.  Gated on
    # n >= 2 *naturally*: a component whose width is only ~1.4x the
    # estimate is a single wide glyph (尔's 28px top vs a 20px estimate),
    # and cutting it at w/2 would split a real glyph, pushing its whole
    # merge past max_merge_components.  Advance multiples still cover the
    # true boundary when the estimate is high (U- style).
    if n_est >= 2:
        for k in range(1, n_est):
            x = w * k / n_est
            if min_piece <= x <= w - min_piece:
                ideals.append(x)

    seams: list[tuple[np.ndarray, float]] = []
    seen: list[float] = []
    for ideal in sorted(set(ideals)):
        # Skip ideals already covered by a kept seam (within 2 * tol).
        if any(abs(ideal - s_mean) <= 4 for s_mean in seen):
            continue
        cut = _seam_cut(mask, ideal)
        if cut is None:
            continue
        path, cost = cut
        seen.append(float(np.mean(path)))
        seams.append((path, cost))

    seams.sort(key=lambda s: s[1])
    return seams[:max_splits]


def _split_atoms(
    comp: Component,
    component_index: int,
    expected_width: float,
    profile: Profile,
    max_splits: int,
) -> list[_Atom]:
    """Return one whole atom, or one atom per split piece.

    Two independent cut mechanisms feed the atom sequence:

    * vertical valleys (:func:`_valley_cuts`) -- connectors between glyphs
      and short edge glyphs (``U-``);
    * forced seams at advance multiples / equal parts
      (:func:`_forced_seam_cuts`) -- fully-touching blobs with no valley
      at the boundary (``甲申``, ``LV``).

    Every cut is only a *hypothesis*: the original component stays
    reachable as the merge of all its atoms, and the lattice decoder picks
    the best path.  ``max_splits`` bounds the atom count so the whole
    component always fits in a ``max_merge_components`` merge.
    """

    whole = [_Atom(component_index=component_index, segment=comp)]
    if (
        max_splits < 1
        or comp.h < max(2, profile.char_height_min)
        # 1.5 * expected / 1.25 * height keeps single CJK glyphs and real
        # glyphs' wide parts (命's 人+一, 尔's top) whole; wide multi-glyph
        # blobs (甲+申 zero-gap) exceed it.
        or comp.w <= max(expected_width * 1.5, comp.h * 1.25)
    ):
        return whole

    cuts: list[tuple[np.ndarray, float]] = []
    for x in _valley_cuts(comp.mask, expected_width, profile, max_splits):
        path = np.full(comp.mask.shape[0], x, dtype=np.int32)
        cuts.append((path, 0.0))
    if not cuts and comp.w >= expected_width * 2:
        # Valley-free blob at least two glyphs wide (甲+申 zero-gap): the
        # forced seams are its only chance.  The width gate keeps real
        # glyph parts (w < 2 * expected) from being seam-sliced -- their
        # interior sparse columns would otherwise read as cheap cuts and
        # push the glyph's whole merge past max_merge_components.
        cuts = _forced_seam_cuts(
            comp.mask, expected_width, profile, max_splits
        )
    if not cuts:
        return whole

    # Keep the lowest-cost cuts (valleys count as free), ordered left to
    # right, respecting the bound.
    cuts.sort(key=lambda pc: (pc[1], float(np.mean(pc[0]))))
    cuts = cuts[:max_splits]
    cuts.sort(key=lambda pc: float(np.mean(pc[0])))

    pieces: list[Component] = []
    for piece_mask in _split_along_paths(comp.mask, [p for p, _ in cuts]):
        seg = _bbox_segment(piece_mask, comp.y)
        seg.x += comp.x
        pieces.append(seg)
    if len(pieces) < 2:
        return whole
    return [_Atom(component_index=component_index, segment=p) for p in pieces]


def _split_along_paths(
    mask: np.ndarray,
    paths: list[np.ndarray],
) -> list[np.ndarray]:
    """Slice ``mask`` into ``len(paths) + 1`` pieces along cut paths.

    Each path gives the cut column per row (a vertical valley cut is a
    constant path; a seam may snake).  Piece ``k`` owns columns
    ``(paths[k-1][y], paths[k][y])`` per row -- left-exclusive,
    right-exclusive -- so the seam's crossed pixels belong to the right
    piece (a straight column cut therefore keeps the left glyph intact).
    """

    h, w = mask.shape
    n = len(paths)
    pieces: list[np.ndarray] = []
    for k in range(n + 1):
        pm = np.zeros_like(mask)
        for y in range(h):
            a = int(paths[k - 1][y]) if k > 0 else 0
            b = int(paths[k][y]) if k < n else w
            if a < b:
                pm[y, a:b] = mask[y, a:b]
        if np.any(pm):
            pieces.append(pm)
    return pieces


def expand_atoms(
    comps: list[Component],
    profile: Profile,
    max_merge_components: int,
    split_wide: bool = True,
    geometry: FontGeometryDatabase | None = None,
) -> tuple[list[_Atom], float]:
    """Build the atom sequence, keeping every original component reachable.

    Public so diagnostic tools (e.g. the oracle lattice recall metric in
    :mod:`fixedfontocr.oracle`) can inspect the exact atom sequence the
    lattice is built from. ``build_candidates`` calls this internally;
    the oracle calls it again so its atom view always matches the lattice.
    """

    expected = _estimate_expected_width(comps, profile, geometry)
    atoms: list[_Atom] = []
    for i, comp in enumerate(comps):
        if split_wide:
            atoms.extend(
                _split_atoms(
                    comp,
                    i,
                    expected,
                    profile,
                    max_splits=max_merge_components - 1,
                )
            )
        else:
            atoms.append(_Atom(component_index=i, segment=comp))
    # Sort atoms by image position so a small fragment that belongs to a
    # glyph is adjacent to its split pieces. The pre-split component order
    # could otherwise place such a fragment after a later glyph, making the
    # only correct merge non-consecutive (real-game crops such as 江原 /
    # 追赶者).
    atoms.sort(key=lambda atom: (atom.segment.x, atom.segment.y, atom.component_index))
    return atoms, expected


# Private-name compatibility alias (pre-Goal-21 name).
_expand_atoms = expand_atoms


def _passes_filters(
    parts: list[_Atom],
    profile: Profile,
    expected_width: float,
    geometry: FontGeometryDatabase | None = None,
) -> bool:
    """Reject candidate shapes that cannot be a single character.

    Pruning covers the Goal 4 list: maximum width/height, minimum height,
    component gap, vertical overlap/proximity, ink area, and the expected
    font bbox. Rules stay conservative so ambiguous low-resolution glyphs
    are still scored by the visual DP instead of being discarded.
    """

    if len(parts) == 1:
        seg = parts[0].segment
        if seg.w > profile.char_width_max or seg.h > profile.char_height_max:
            return False
        # The Goal 8 bbox envelope is estimated from the glyph *height* and
        # is unreliable for short punctuation: a 32px ``-`` is 8x3 (ratio
        # 2.5, far above the normal 1.35x envelope).  Small candidates get
        # a wider envelope (3.0x) so ``-``/``=``/``~`` reach the scorer,
        # while near-empty 1-2 px bars (ratio ~3.3+, glyph fragments like
        # 武's tail) stay rejected.  The identity-dependent char_geometry
        # decides after classification whether the shape fits its Top-K
        # pick.
        envelope = 3.0 if seg.h < max(2, profile.char_height_min) else 1.35
        if not _geometry_bbox_ok(seg, geometry, max_multiplier=envelope):
            return False
        if seg.h >= max(2, profile.char_height_min):
            ink = int(seg.mask.sum())
            if ink < max(1, int(0.02 * seg.w * seg.h)):
                return False
        return True

    segs = [p.segment for p in parts]
    x0 = min(p.x for p in segs)
    y0 = min(p.y for p in segs)
    x1 = max(p.x + p.w for p in segs)
    y1 = max(p.y + p.h for p in segs)
    w, h = x1 - x0, y1 - y0
    if w > profile.char_width_max or h > profile.char_height_max:
        return False
    if not _geometry_bbox_ok(
        Component(mask=segs[0].mask, x=x0, y=y0, w=w, h=h),
        geometry,
    ):
        return False
    if h < max(2, profile.char_height_min):
        return False
    if w > max(profile.char_width_max, expected_width * 1.8):
        return False
    # Conservative internal-gap bound: a merge across a wide blank is a
    # different character, not a fragmented glyph. The bound is generous
    # because strokes of one glyph (e.g. the right dots of 亲) can be a
    # dozen pixels apart; the visual DP is what rejects actual
    # character-to-character merges.
    max_gap = max(
        segs[k + 1].x - (segs[k].x + segs[k].w) for k in range(len(segs) - 1)
    )
    gap_limit = max(2, profile.char_height_min) + 2
    if max_gap > gap_limit:
        return False

    # Vertical overlap/proximity: at least one adjacent pair must overlap
    # vertically or be close enough to belong to one glyph (colon and i-dot
    # punctuation are deliberately allowed by the generous gap bound).
    # Real-game crops render 二 as two horizontal bars separated by more
    # than half a char height; allow up to one full char height so that
    # multi-bar glyphs remain mergeable.
    max_v_gap = max(4, profile.char_height_min)
    if not any(
        _vertical_overlap(a.segment, b.segment) >= 1
        or _vertical_gap(a.segment, b.segment) <= max_v_gap
        for a, b in zip(parts, parts[1:])
    ):
        return False

    ink = sum(int(s.mask.sum()) for s in segs)
    if ink < max(1, int(0.02 * w * h)):
        return False
    return True


def _geometry_bbox_ok(
    seg: Component,
    geometry: FontGeometryDatabase | None,
    max_multiplier: float = 1.35,
) -> bool:
    """Goal 8 bbox prior: reject shapes no charset glyph can occupy.

    The check is deliberately generous (``max_multiplier`` x the widest
    narrow/full-width glyph, 0.35x the narrowest) so low-resolution
    fragments are still scored by the visual DP.  Without a database every
    shape passes.

    Small punctuation (below the minimum char height) uses a wider
    ``max_multiplier`` (3.0 in :func:`_passes_filters`): the height-derived
    em of a flat ``-`` (8x3, ratio 2.5) exceeds the normal 1.35x envelope
    although ``-`` is a real charset glyph.  The wider bound still rejects
    near-empty 1-2 px bars (ratio ~3.3+) that are glyph fragments, not
    punctuation.
    """

    if geometry is None or seg.h < 2 or seg.w < 1:
        return True
    h = max(seg.h, 1)
    w = max(seg.w, 1)
    if w / h < 0.8:
        em = max(
            h / max(geometry.narrow_height_ratio, 1e-6),
            1.0,
        )
        max_ratio = geometry.narrow_width_max * max_multiplier
        min_ratio = geometry.narrow_width_min * 0.35
    else:
        em = max(
            h / max(geometry.full_height_ratio, 1e-6),
            1.0,
        )
        max_ratio = geometry.full_width_max * max_multiplier
        min_ratio = geometry.full_width_min * 0.35
    ratio = w / em
    return min_ratio <= ratio <= max_ratio


def _vertical_overlap(a: Component, b: Component) -> int:
    return min(a.y + a.h, b.y + b.h) - max(a.y, b.y)


def _vertical_gap(a: Component, b: Component) -> int:
    return max(a.y, b.y) - min(a.y + a.h, b.y + b.h)


def candidate_geometry(
    candidate: VisualCandidate,
    comps: list[Component],
    profile: Profile,
) -> float:
    """Identity-independent geometry evidence for one lattice candidate.

    This half of the Goal 8 geometry score is deliberately independent of
    the chosen character: vertical alignment with the line band, internal
    gaps inside a merged candidate, and a generic segmentation-width
    penalty. Character-specific evidence (bbox / aspect / ink ratio /
    component count / baseline) is computed separately by
    :func:`fixedfontocr.geometry.char_geometry` for every
    ``(candidate, char_id)`` choice made by the decoder.

    The generic merge-width penalty is recorded on the candidate
    (``segmentation_width_penalty``); ``char_geometry`` adds it back when
    the font database identifies the chosen glyph as legitimately
    multi-component.
    """

    score = 0.0
    candidate.segmentation_width_penalty = 0.0
    seg = candidate.segment
    if seg is None:
        return score

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
        # almost always two adjacent characters. The character-specific
        # exception (a font glyph that is itself multi-component) is handled
        # by char_geometry using segmentation_width_penalty.
        heights = [c.h for c in comps]
        median_h = float(np.median(heights))
        typical = [
            c.w for c in comps if c.h >= max(2, 0.5 * median_h)
        ]
        expected = float(np.median(typical)) if typical else float(median_h)
        if seg.w > expected * 1.9:
            width_penalty = -0.05
            score += width_penalty
            candidate.segmentation_width_penalty = width_penalty

    return max(score, -0.10)


def geometry_score(
    candidate: VisualCandidate,
    comps: list[Component],
    profile: Profile,
    geometry: FontGeometryDatabase | None = None,
    char_id: int | None = None,
    normalize_geometry: tuple[float, float] | None = None,
) -> float:
    """Small geometric adjustments on top of the classifier visual score.

    Compatibility wrapper around the Goal 20 geometry split:

        total = candidate_geometry(candidate, comps, profile)
              + char_geometry(candidate, char_id, geometry, normalize_geometry)

    Candidate geometry is identity-independent; the second term is evaluated
    for the chosen ``char_id``. The decoder never calls this wrapper -- it
    uses the stored candidate geometry and computes ``char_geometry`` for
    every ``(candidate, char_id)`` alternative itself.
    """

    score = candidate_geometry(candidate, comps, profile)
    if geometry is None:
        return score
    if char_id is None and candidate.score is not None:
        char_id = getattr(candidate.score, "char_id", None)
    if char_id is not None and int(char_id) >= 0:
        score += char_geometry(
            candidate,
            int(char_id),
            geometry,
            normalize_geometry=normalize_geometry,
        )
    return max(score, -0.12)


def build_lattice(
    comps: list[Component],
    candidates: list[VisualCandidate],
    line: Segment | None = None,
) -> VisualLattice:
    """Wrap components and candidates in a non-destructive visual lattice."""
    return VisualLattice(
        components=tuple(comps),
        candidates=tuple(candidates),
        width=line.w if line is not None else 0,
        height=line.h if line is not None else 0,
    )


def _top_candidate_char_id(candidate: VisualCandidate) -> int:
    if candidate.scores is not None and candidate.scores.char_ids:
        return int(candidate.scores.char_ids[0])
    if candidate.score is not None:
        return int(getattr(candidate.score, "char_id", -1))
    return -1


def decode(
    candidates: list[VisualCandidate] | VisualLattice,
    n_components: int | None = None,
) -> DecodePath:
    """Visual DP over the candidate lattice (max mean score path).

    The DP runs over the lattice's atom sequence: an atom is a whole
    connected component or one split piece of a wide component. ``dp[end]``
    stores the best (sum, count) covering atoms ``0..end-1``; ties are broken
    toward fewer candidates (i.e. prefer the merged glyph over a tie of its
    fragments).
    """

    lattice: VisualLattice | None = None
    if isinstance(candidates, VisualLattice):
        lattice = candidates
        n_components = len(lattice.components)
        candidates = list(lattice.candidates)
    if n_components is None:
        raise ValueError("decode() needs n_components when given a candidate list")

    if not candidates:
        return DecodePath(candidates=(), mean_score=0.0, lattice=lattice)

    n_atoms = max(c.atom_span[1] for c in candidates)
    by_end: dict[int, list[tuple[Candidate, int, int]]] = {}
    for c in candidates:
        start, end = c.atom_span
        by_end.setdefault(end, []).append((c, start, end))

    NEG = -1e18
    dp_sum = [NEG] * (n_atoms + 1)
    dp_cnt = [0] * (n_atoms + 1)
    dp_prev: list[list | None] = [None] * (n_atoms + 1)
    dp_sum[0] = 0.0

    for end in range(1, n_atoms + 1):
        best_sum: float | None = None
        best_cnt = 0
        best_prev = None
        best_cand: Candidate | None = None
        for c, start, _ in by_end.get(end, []):
            if dp_sum[start] <= NEG / 2:
                continue
            s = dp_sum[start] + c.total_score
            cnt = dp_cnt[start] + 1
            mean = s / cnt
            if best_sum is None or mean > best_sum / best_cnt + 1e-9 or (
                abs(mean - best_sum / best_cnt) <= 1e-9 and cnt < best_cnt
            ):
                best_sum, best_cnt, best_prev, best_cand = s, cnt, start, c
        if best_cand is not None:
            dp_sum[end] = best_sum
            dp_cnt[end] = best_cnt
            dp_prev[end] = [best_prev, best_cand]

    if dp_sum[n_atoms] <= NEG / 2:
        return DecodePath(candidates=(), mean_score=0.0, lattice=lattice)

    chosen: list[Candidate] = []
    pos = n_atoms
    while pos > 0:
        prev, cand = dp_prev[pos]  # type: ignore[misc]
        chosen.append(cand)
        pos = prev
    chosen.reverse()
    return DecodePath(
        candidates=tuple(chosen),
        mean_score=float(dp_sum[n_atoms] / dp_cnt[n_atoms]),
        char_ids=tuple(_top_candidate_char_id(c) for c in chosen),
        lattice=lattice,
    )


def segment_line(
    line: Segment,
    profile: Profile,
    scorer: SegmentScorer,
    allowed_ids: set[int] | None = None,
    max_merge_components: int = 4,
    soft: np.ndarray | None = None,
    image: np.ndarray | None = None,
    lexicon: Lexicon | str | None = None,
    decoder_config: DecoderConfig | None = None,
) -> DecodePath:
    """Segment one line with the candidate lattice + joint decoder.

    ``soft`` is the Visual Frontend's soft foreground map; when provided it
    is passed to the scorer so the TinyCNN scores soft-normalized glyphs
    while the template path keeps using binary glyphs.

    Goal 13/P0-3: with a ``lexicon`` the line is decoded by
    :func:`decode_beam` over the scored lattice, and the beam's complete
    ``score_path()`` ranking is authoritative. With ``decoder_config`` but
    no lexicon, the joint formula is evaluated by the exact
    :func:`decode_dp` (no beam-only segmentation artefacts). Without either
    argument the legacy visual DP (:func:`decode`) is kept for callers that
    only score segmentation.
    """

    comps = connected_components(line)
    if not comps:
        return DecodePath(
            candidates=(),
            mean_score=0.0,
            lattice=build_lattice([], [], line),
        )
    if allowed_ids is not None and not allowed_ids:
        # No character is allowed: every connected component is an unknown
        # glyph. Returning one candidate per component keeps the public
        # result length aligned with the visible glyph count instead of
        # letting the zero-score DP merge adjacent unknowns into one "?".
        candidates = [
            Candidate(
                segment=comp,
                start=i,
                end=i + 1,
                components=(i,),
                atoms=(i, i + 1),
            )
            for i, comp in enumerate(comps)
        ]
        unknown = SegmentScore(
            char_id=-1,
            visual_score=0.0,
            raw_score=0.0,
            score_type="template",
        )
        for cand in candidates:
            cand.score = unknown
        return decode(build_lattice(comps, candidates, line))
    candidates = build_candidates(
        comps,
        profile,
        max_merge_components,
        geometry=getattr(scorer, "geometry", None),
    )
    if not candidates:
        return DecodePath(
            candidates=(),
            mean_score=0.0,
            lattice=build_lattice(comps, [], line),
        )

    segments = [c.segment for c in candidates]
    spec = scorer.normalize_spec
    if spec is not None:
        geometries = [
            glyph_normalize_geometry(cand.segment, spec, profile.target_size)
            for cand in candidates
        ]
    else:
        geometries = [None] * len(candidates)
    scores = scorer.score(segments, allowed_ids, soft=soft, geometries=geometries, image=image)
    geometry_db = getattr(scorer, "geometry", None)
    for cand, raw_score, geom in zip(candidates, scores, geometries):
        # Goal 20 geometry split: compute the identity-independent half once
        # here, then store only that half on the candidate. The decoder asks
        # char_geometry(candidate, char_id) for every character alternative,
        # so a Top-2/Top-3 lexicon choice never reuses the Top-1 geometry.
        cand_base = candidate_geometry(cand, comps, profile)
        char_geom = char_geometry(
            cand,
            raw_score.char_id,
            geometry_db,
            normalize_geometry=geom,
        )
        geometry = max(cand_base + char_geom, -0.12)
        # An exact visual match (template distance 0) is authoritative: the
        # candidate *is* a real glyph, so merge/split geometry penalties must
        # not let a fragmented path of weaker look-alikes win the DP. This
        # compatibility override only freezes the candidate-level half (and
        # the legacy Top-1 total); the decoder still evaluates the character-
        # specific half for alternative char_ids.
        exact_match = (
            raw_score.score_type == "template"
            and raw_score.template_raw_score >= 1.0 - 1e-9
        )
        if exact_match:
            geometry = 0.0
            cand_base = 0.0
            cand.segmentation_width_penalty = 0.0
        score = scorer.finalize_score(raw_score, geometry)
        cand.candidate_geometry = cand_base
        cand.score = score
        cand.scores = score.visual_scores
        cand.geometry_score = geometry
        cand.normalize_geometry = geom
    candidates = _drop_weak_merges(candidates, scorer, comps, profile)
    if not candidates:
        return DecodePath(
            candidates=(),
            mean_score=0.0,
            lattice=build_lattice(comps, [], line),
        )
    lattice = build_lattice(comps, candidates, line)
    geometry_db = getattr(scorer, "geometry", None)
    if lexicon is not None:
        lex = (
            lexicon
            if isinstance(lexicon, Lexicon)
            else load_lexicon(lexicon)
        )
        cfg = decoder_config or DecoderConfig()
        return decode_beam(
            lattice,
            scorer.model.charset,
            lex,
            cfg,
            geometry=geometry_db,
        )
    if decoder_config is not None:
        # Explicit joint decoding without a lexicon uses the exact DP. The
        # beam is reserved for lexicon decoding, where the complete
        # score_path() lexicon term is the final authority (P0-3).
        return decode_dp(
            lattice,
            scorer.model.charset,
            None,
            decoder_config,
            geometry=geometry_db,
        )
    return decode(lattice)


def _drop_weak_merges(
    candidates: list[Candidate],
    scorer: SegmentScorer,
    comps: list[Component],
    profile: Profile,
) -> list[Candidate]:
    """Drop low-confidence multi-atom candidates (CNN-only + hybrid models).

    The pure visual DP can merge adjacent characters when the CNN's margin
    for a combined blob is only slightly better than its margin for each
    part (e.g. three ``0``s merged into one ``0``), or when a split wide
    component's whole blob scores like one character. CNN-only models only
    allow a merge when:

    * the merged candidate looks like a real character
      (``cnn_merge_threshold``);
    * it scores at least as well as every single atom inside it; and
    * the atoms are in glyph-internal proximity (the old merger's
      overlap rule), so unrelated characters such as ``1000`` cannot merge.

    Hybrid (template + CNN) models get one more gate for *fake* merges:
    two touching real-game glyphs glued into one blob that merely looks
    like some character (``Z+1`` read as ``灶``, ``4+7`` read as ``“``).
    A real glyph's merged whole is a confident template match, so a merge
    whose Top-1 pick is not template-confirmed is vetoed when

    * one single atom inside it is already a better character than the
      blob (the part beats the whole: ``1`` 0.964 > ``灶`` 0.898), or
    * the blob has no template support at all (template raw score ~0):
      it is not any glyph of the registered font, only a CNN look-alike.
    """

    threshold = getattr(scorer, "cnn_merge_threshold", 0.0)
    has_templates = getattr(scorer, "template", None) is not None
    if not candidates or (threshold <= 0.0 and not has_templates):
        return candidates
    single_visual: dict[int, float] = {}
    atom_info: dict[int, tuple[float, int, int]] = {}
    for cand in candidates:
        start, end = cand.atom_span
        if end - start == 1 and cand.score is not None:
            vis = getattr(
                cand.score, "classifier_visual_score", cand.score.visual_score
            )
            if cand.scores is not None and cand.scores.logits:
                vis = float(cand.scores.logits[0])
            prev = atom_info.get(start)
            if prev is None or vis > prev[0]:
                atom_info[start] = (vis, cand.segment.w, cand.segment.h)
            if len(cand.components) == 1:
                i = cand.components[0]
                single_visual[i] = max(single_visual.get(i, -1.0), vis)
    line_h = max((c.h for c in comps), default=1)
    out = []
    for cand in candidates:
        start, end = cand.atom_span
        if end - start > 1 and cand.score is not None:
            visual = getattr(
                cand.score, "classifier_visual_score", cand.score.visual_score
            )
            scores = cand.scores
            if scores is not None and scores.logits:
                visual = float(scores.logits[0])
            if threshold > 0.0:
                # CNN-only models: the original gate, unchanged.
                if visual < threshold:
                    continue
                if not _proximity_merge_ok(cand, comps, profile):
                    continue
                parts = [
                    single_visual.get(i, threshold)
                    for i in range(cand.components[0], cand.components[-1] + 1)
                ]
                if any(p > visual + 0.02 for p in parts):
                    continue
            elif scores is not None:
                penalty = _hybrid_fake_merge_penalty(
                    scores, cand, comps, atom_info, start, line_h
                )
                if penalty > 0.0:
                    cand.scores = replace(
                        scores,
                        logits=tuple(
                            max(0.0, float(v) - penalty) for v in scores.logits
                        ),
                    )
        out.append(cand)
    return out


HYBRID_FAKE_MERGE_PENALTY = 0.12
# A merged blob is vetoed when a single atom nearly beats it: with the
# margin, a strong atom (e.g. template-exact ``4`` at 0.941) still vetoes
# a CNN look-alike blob (``43`` read as ``“`` at 0.977) even though the
# blob's fused score edges it out by a few points. Template-confident
# blobs (>= 0.9) stay exempt, so real glyphs are never touched.
HYBRID_FAKE_ATOM_MARGIN = 0.05


def _hybrid_fake_merge_penalty(
    scores,
    cand: Candidate,
    comps: list[Component],
    atom_info: dict[int, tuple[float, int, int]],
    start: int,
    line_h: int,
) -> float:
    """Penalty for a merged blob that is not a single registered glyph.

    Hybrid models get a confident template score for real glyphs, so a
    multi-atom merge whose Top-1 pick is not template-confident is suspect.
    Two evidence patterns veto it (by down-weighting its Top-K logits, never
    by removing it, so the lattice stays coverable):

    * two or more of its atoms are already full-size, confident glyphs and
      the blob has *no* template support at all (template raw score ~0):
      the blob is not any glyph of the registered font (real-game ``Z+2``
      glued into one component read as ``灶``);
    * its components are strictly side-by-side (no x-overlap -- glyph-
      internal radicals overlap horizontally, adjacent characters do not)
      and one atom alone is already a better character than the blob
      (``1`` 0.964 beats ``灶`` 0.898 for the glued ``Z+1``).

    Both patterns require the blob to be wide enough to hold two of its own
    atoms (>= 1.6x the widest atom; >= 1.7x for the zero-template pattern),
    measured against the candidate's own atoms instead of the line-wide
    glyph width so mixed CJK+Latin lines (``Z18 Z17 岛风...``) work too.
    A template-confident whole (raw score >= 0.9 or template-chosen Top-1)
    is always trusted.
    """

    score_type = getattr(scores, "score_type", "")
    tpl_raw = float(getattr(scores, "template_raw_score", 0.0))
    if score_type == "template" or tpl_raw >= 0.9:
        return 0.0
    seg = cand.segment
    if seg is None:
        return 0.0
    end = cand.atom_span[1]
    infos = [atom_info[a] for a in range(start, end) if a in atom_info]
    if len(infos) < 2:
        return 0.0
    widths = [w for _, w, _ in infos]
    max_w = max(widths)
    if seg.w < 1.6 * max_w:
        return 0.0
    median_w = float(np.median(widths))

    def qualifying(floor: float) -> list[tuple[float, int, int]]:
        return [
            info
            for info in infos
            if info[0] >= floor
            and info[1] >= 0.7 * median_w
            and info[2] >= 0.5 * line_h
        ]

    q = qualifying(0.7)
    if not q:
        return 0.0
    visual = float(scores.logits[0]) if scores.logits else 0.0

    def penalty() -> float:
        # Push the blob's Top-1 down to at most ~0.60 so a decent split
        # wins even when the path mean is diluted by a long line prefix
        # (the mean of a 19-char line moves only ~0.005 per char), but
        # keep the flat floor for weak blobs so fragile coverage paths
        # (e.g. T-23's ``T`` + tiny ``-``) are not disturbed.
        return float(
            min(0.45, max(HYBRID_FAKE_MERGE_PENALTY, visual - 0.60))
        )

    if (
        tpl_raw <= 1e-6
        and seg.w >= 1.7 * max_w
        and len(qualifying(0.75)) >= 2
    ):
        return penalty()
    if len(cand.components) >= 2:
        idx = list(cand.components)
        if any(
            min(comps[i].x + comps[i].w, comps[j].x + comps[j].w)
            - max(comps[i].x, comps[j].x)
            > 0
            for i in idx
            for j in idx
            if i < j
        ):
            return 0.0
        if any(
            info[0] + HYBRID_FAKE_ATOM_MARGIN > visual for info in q
        ):
            return penalty()
    return 0.0


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
