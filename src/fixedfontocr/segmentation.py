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

from dataclasses import dataclass

import numpy as np

from .geometry import FontGeometryDatabase
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
    atoms, expected_width = _expand_atoms(
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
            if c.w / max(c.h, 1) < 0.8:
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
    most ``~12%`` of the component height. That catches a low-resolution
    connector between two glyphs while avoiding interior whitespace of
    closed CJK glyphs (口/日/中 have top/bottom strokes keeping the valley
    columns well above the threshold). Edge valleys are ignored and each
    resulting piece must be wide enough to be a glyph.
    """

    h, w = mask.shape
    proj = mask.sum(axis=0).astype(np.int32)
    threshold = max(1, int(h * 0.12))
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


def _split_atoms(
    comp: Component,
    component_index: int,
    expected_width: float,
    profile: Profile,
    max_splits: int,
) -> list[_Atom]:
    """Return one whole atom, or one atom per vertically-split piece."""

    whole = [_Atom(component_index=component_index, segment=comp)]
    if (
        max_splits < 1
        or comp.h < max(2, profile.char_height_min)
        or comp.w <= max(expected_width * 1.5, comp.h * 1.4)
    ):
        return whole
    cuts = _valley_cuts(comp.mask, expected_width, profile, max_splits)
    if not cuts:
        return whole

    bounds = [0, *cuts, comp.w]
    pieces: list[Component] = []
    for a, b in zip(bounds, bounds[1:]):
        piece_mask = comp.mask[:, a:b]
        if not np.any(piece_mask):
            continue
        seg = _bbox_segment(piece_mask, comp.y)
        seg.x += comp.x + a
        pieces.append(seg)
    if len(pieces) < 2:
        return whole
    return [_Atom(component_index=component_index, segment=p) for p in pieces]


def _expand_atoms(
    comps: list[Component],
    profile: Profile,
    max_merge_components: int,
    split_wide: bool,
    geometry: FontGeometryDatabase | None = None,
) -> tuple[list[_Atom], float]:
    """Build the atom sequence, keeping every original component reachable."""

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
    return atoms, expected


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
        if not _geometry_bbox_ok(seg, geometry):
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
    max_v_gap = max(4, profile.char_height_min // 2)
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
) -> bool:
    """Goal 8 bbox prior: reject shapes no charset glyph can occupy.

    The check is deliberately generous (1.35x the widest narrow/full-width
    glyph, 0.35x the narrowest) so low-resolution fragments are still scored
    by the visual DP. Without a database every shape passes.
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
        max_ratio = geometry.narrow_width_max * 1.35
        min_ratio = geometry.narrow_width_min * 0.35
    else:
        em = max(
            h / max(geometry.full_height_ratio, 1e-6),
            1.0,
        )
        max_ratio = geometry.full_width_max * 1.35
        min_ratio = geometry.full_width_min * 0.35
    ratio = w / em
    return min_ratio <= ratio <= max_ratio


def _vertical_overlap(a: Component, b: Component) -> int:
    return min(a.y + a.h, b.y + b.h) - max(a.y, b.y)


def _vertical_gap(a: Component, b: Component) -> int:
    return max(a.y, b.y) - min(a.y + a.h, b.y + b.h)


def geometry_score(
    candidate: VisualCandidate,
    comps: list[Component],
    profile: Profile,
    geometry: FontGeometryDatabase | None = None,
    char_id: int | None = None,
    normalize_geometry: tuple[float, float] | None = None,
) -> float:
    """Small geometric adjustments on top of the classifier visual score.

    Penalties are deliberately small (<= 0.06) so the classifier remains the
    dominant signal, but they break ties between a fragmented glyph path and
    its merged candidate and keep clearly-adjacent characters apart. With a
    Goal 8 database the score also compares the candidate's bbox, ink ratio,
    component count and baseline against the classifier's Top-1 character:
    a narrow ``1/I/l`` prior is never treated like a full-width CJK glyph.
    """

    score = 0.0
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
        # almost always two adjacent characters.
        heights = [c.h for c in comps]
        median_h = float(np.median(heights))
        typical = [
            c.w for c in comps if c.h >= max(2, 0.5 * median_h)
        ]
        expected = float(np.median(typical)) if typical else float(median_h)
        if seg.w > expected * 1.9:
            score -= 0.05

    if geometry is None:
        return max(score, -0.10)
    if char_id is None and candidate.score is not None:
        char_id = getattr(candidate.score, "char_id", None)
    if char_id is None or char_id < 0:
        return max(score, -0.10)
    entry = geometry.get(int(char_id))
    if entry is None:
        return max(score, -0.10)

    h = max(seg.h, 1)
    w = max(seg.w, 1)
    em = max(
        h / max(entry.bbox_height, 1e-6),
        w / max(entry.bbox_width, 1e-6),
        1.0,
    )
    w_ratio = w / em
    h_ratio = h / em
    width_err = abs(w_ratio - entry.bbox_width) / max(entry.bbox_width, 1e-3)
    height_err = abs(h_ratio - entry.bbox_height) / max(entry.bbox_height, 1e-3)
    aspect = w / h
    aspect_err = (
        abs(np.log(max(aspect / max(entry.aspect_ratio, 1e-3), 1e-3)))
        if entry.aspect_ratio > 0
        else 0.0
    )
    score -= min(0.05, width_err * 0.04)
    score -= min(0.04, height_err * 0.03)
    score -= min(0.03, aspect_err * 0.02)
    advance_err = abs(w_ratio - entry.advance) / max(entry.advance, 1e-3)
    score -= min(0.02, advance_err * 0.01)

    ink_ratio = float(seg.mask.sum()) / float(w * h)
    ink_err = abs(ink_ratio - entry.ink_ratio) / max(entry.ink_ratio, 0.05)
    score -= min(0.03, ink_err * 0.01)

    # Component-count prior: a fragmented glyph whose Top-1 needs 3
    # components is penalized; a merge whose components match the glyph is
    # not. Split whole-component alternatives (atoms > components) are
    # exempt because they still represent the original connected glyph.
    expected_cc = max(1, entry.component_count)
    actual_cc = max(1, len(candidate.components))
    start, end = candidate.atom_span
    is_split_whole = end - start > actual_cc
    if not is_split_whole:
        if actual_cc > 1 and expected_cc == 1:
            score -= min(0.04, (actual_cc - 1) * 0.02)
        elif actual_cc == 1 and expected_cc > 1:
            score -= min(0.03, (expected_cc - 1) * 0.015)

    if normalize_geometry is not None:
        baseline_offset = float(normalize_geometry[0])
        if 0.0 < baseline_offset <= h * 1.5 and 0.0 < entry.baseline_ratio <= 1.5:
            cand_ratio = baseline_offset / h
            score -= min(0.02, abs(cand_ratio - entry.baseline_ratio) * 0.02)

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
        lattice=lattice,
    )


def segment_line(
    line: Segment,
    profile: Profile,
    scorer: SegmentScorer,
    allowed_ids: set[int] | None = None,
    max_merge_components: int = 4,
    soft: np.ndarray | None = None,
) -> DecodePath:
    """Segment one line with the candidate lattice + visual DP.

    ``soft`` is the Visual Frontend's soft foreground map; when provided it
    is passed to the scorer so the TinyCNN scores soft-normalized glyphs
    while the template path keeps using binary glyphs.
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
    scores = scorer.score(segments, allowed_ids, soft=soft, geometries=geometries)
    for cand, raw_score, geom in zip(candidates, scores, geometries):
        geometry = geometry_score(
            cand,
            comps,
            profile,
            geometry=getattr(scorer, "geometry", None),
            char_id=raw_score.char_id,
            normalize_geometry=geom,
        )
        # An exact visual match (template distance 0) is authoritative: the
        # candidate *is* a real glyph, so merge/split geometry penalties must
        # not let a fragmented path of weaker look-alikes win the DP.
        if (
            raw_score.score_type == "template"
            and raw_score.template_raw_score >= 1.0 - 1e-9
        ):
            geometry = 0.0
        score = scorer.finalize_score(raw_score, geometry)
        cand.score = score
        cand.scores = score.visual_scores
        cand.geometry = geometry
        cand.normalize_geometry = geom
    candidates = _drop_weak_merges(candidates, scorer, comps, profile)
    if not candidates:
        return DecodePath(
            candidates=(),
            mean_score=0.0,
            lattice=build_lattice(comps, [], line),
        )
    return decode(build_lattice(comps, candidates, line))


def _drop_weak_merges(
    candidates: list[Candidate],
    scorer: SegmentScorer,
    comps: list[Component],
    profile: Profile,
) -> list[Candidate]:
    """Drop low-confidence multi-atom candidates (CNN-only models).

    The pure visual DP can merge adjacent characters when the CNN's margin
    for a combined blob is only slightly better than its margin for each
    part (e.g. three ``0``s merged into one ``0``), or when a split wide
    component's whole blob scores like one character. Template/hybrid models
    get a confident template score for real glyphs, so they do not need this
    gate. CNN-only models only allow a merge when:

    * the merged candidate looks like a real character
      (``cnn_merge_threshold``);
    * it scores at least as well as every single atom inside it; and
    * the atoms are in glyph-internal proximity (the old merger's
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
                single_visual.get(i, -1.0),
                getattr(
                    cand.score, "classifier_visual_score", cand.score.visual_score
                ),
            )
    out = []
    for cand in candidates:
        start, end = cand.atom_span
        if end - start > 1 and cand.score is not None:
            visual = getattr(
                cand.score, "classifier_visual_score", cand.score.visual_score
            )
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
