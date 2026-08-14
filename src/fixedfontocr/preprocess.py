"""Fixed CPU pipeline: mask -> lines -> characters -> 24x24 normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .types import CharResult, Profile


@dataclass
class Segment:
    """A connected text region (line or character) in image coordinates."""

    mask: NDArray[np.bool_]
    x: int
    y: int
    w: int
    h: int


def preprocess(
    image: NDArray[np.uint8],
    profile: Profile,
    classify,
    allowed_ids: set[int] | None = None,
) -> tuple[list[CharResult], str]:
    """Run the full CPU pipeline and return (char results, recognized text)."""

    mask = profile.color_mask(image)
    chars: list[CharResult] = []
    text_parts: list[str] = []

    for line in _find_lines(mask, profile):
        line_chars = _segment_line(line, profile)
        for seg in line_chars:
            char_id, conf = classify(seg.mask, profile, allowed_ids)
            chars.append(
                CharResult(
                    char=char_id,
                    x=seg.x,
                    y=seg.y,
                    w=seg.w,
                    h=seg.h,
                    confidence=float(conf),
                )
            )
            text_parts.append(char_id)

    return chars, "".join(text_parts)


def collect_glyphs(
    image: NDArray[np.uint8],
    profile: Profile,
) -> tuple[list[Segment], NDArray[np.uint8]]:
    """Segment one image into characters and normalize them as a batch.

    Phase-1 WGPU pipeline boundary: character segmentation, merging and the
    ``24x24`` resize/normalize all stay on the CPU. Returns the segments (for
    bounding boxes) and an ``uint8 [N, target_size, target_size]`` batch with
    0/255 values, ready for ``Backend.classify``.

    .. note::
        This legacy helper still uses the old irreversible merge for
        training-sample collection. The production OCR path
        (``FixedFontOCR.recognize``) uses the candidate lattice + visual DP
        in :mod:`fixedfontocr.segmentation`, which keeps every original
        component and decides merges after classification.
    """

    mask = profile.color_mask(image)
    segments: list[Segment] = []
    glyphs: list[NDArray[np.uint8]] = []
    for line in _find_lines(mask, profile):
        for seg in _segment_line(line, profile):
            segments.append(seg)
            glyphs.append(normalize(seg.mask, profile.target_size))
    if not glyphs:
        return segments, np.empty(
            (0, profile.target_size, profile.target_size), dtype=np.uint8
        )
    return segments, np.stack(glyphs)


def find_lines(mask: NDArray[np.bool_], profile: Profile) -> list[Segment]:
    """Public wrapper around the row-grouping line detector."""

    return _find_lines(mask, profile)


def _find_lines(mask: NDArray[np.bool_], profile: Profile) -> list[Segment]:
    """Group foreground rows into single-line segments."""

    rows = np.any(mask, axis=1)
    height = mask.shape[0]
    lines: list[Segment] = []
    start: int | None = None
    last_fg: int | None = None
    min_gap = max(1, profile.char_height_min // 3)

    def flush(end: int) -> None:
        nonlocal start
        if start is None:
            return
        y0, y1 = start, end
        if y1 - y0 >= profile.char_height_min:
            row_slice = slice(y0, y1 + 1)
            line_mask = mask[row_slice, :]
            lines.append(_bbox_segment(line_mask, y0))
        start = None

    for y in range(height):
        if rows[y]:
            if start is None or (last_fg is not None and y - last_fg > min_gap):
                flush(y - 1)
                start = y
            last_fg = y

    flush(height - 1)
    return lines


def _segment_line(line: Segment, profile: Profile) -> list[Segment]:
    """Split one line's mask into characters using connected components,
    then merge fragments of the same glyph and split overly wide glyphs.
    """

    comps = _connected_components(line.mask)
    segments = [_bbox_segment(comp, line.y) for comp in comps]
    segments = _merge_fragments(segments, profile)
    segments = _split_wide(segments, profile)
    segments.sort(key=lambda s: (s.x, s.y))
    return segments


def _connected_components(mask: NDArray[np.bool_]) -> list[NDArray[np.bool_]]:
    """Label connected components with a run-length union-find sweep.

    Each row is reduced to its foreground runs first, so the per-pixel loop
    of the old implementation is replaced by a per-run loop (text masks are
    sparse: a 1920x1080 UI line has far fewer runs than pixels). Only runs
    that overlap the previous row's runs participate in the union pass.
    """

    h, w = mask.shape
    label = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    nxt = 0
    prev_runs: list[tuple[int, int, int]] = []  # (start, end, label)
    for y in range(h):
        padded = np.concatenate(([False], mask[y], [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        # Every foreground run is a (start, end) pair in the edge list.
        curr_runs: list[tuple[int, int, int]] = []
        for k in range(0, edges.size, 2):
            a = int(edges[k])
            b = int(edges[k + 1])
            overlaps = [
                (ps, pe, pl)
                for (ps, pe, pl) in prev_runs
                if a < pe and ps < b
            ]
            if not overlaps:
                nxt += 1
                parent.append(nxt)
                lab = nxt
            else:
                lab = overlaps[0][2]
                for (_, _, other) in overlaps[1:]:
                    union(lab, other)
            label[y, a:b] = lab
            curr_runs.append((a, b, lab))
        prev_runs = curr_runs

    # Compress every label to its root, then renumber roots to 1..K.
    root_to_new: dict[int, int] = {}
    for v in np.unique(label):
        if v == 0:
            continue
        r = find(int(v))
        if r not in root_to_new:
            root_to_new[r] = len(root_to_new) + 1
        label[label == v] = root_to_new[r]

    out: list[NDArray[np.bool_]] = []
    for k in range(1, len(root_to_new) + 1):
        out.append(label == k)
    return out


def _connected_components_pixel(mask: NDArray[np.bool_]) -> list[NDArray[np.bool_]]:
    """Reference per-pixel 4-connectivity implementation (tests only)."""

    h, w = mask.shape
    label = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    nxt = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            up = int(label[y - 1, x]) if y > 0 and mask[y - 1, x] else 0
            left = int(label[y, x - 1]) if x > 0 and mask[y, x - 1] else 0
            if up == 0 and left == 0:
                nxt += 1
                parent.append(nxt)
                label[y, x] = nxt
            elif up == 0:
                label[y, x] = left
            elif left == 0:
                label[y, x] = up
            else:
                label[y, x] = up
                union(up, left)

    root_to_new: dict[int, int] = {}
    for v in np.unique(label):
        if v == 0:
            continue
        r = find(int(v))
        if r not in root_to_new:
            root_to_new[r] = len(root_to_new) + 1
        label[label == v] = root_to_new[r]

    out: list[NDArray[np.bool_]] = []
    for k in range(1, len(root_to_new) + 1):
        out.append(label == k)
    return out


def _bbox_segment(mask: NDArray[np.bool_], y_offset: int) -> Segment:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    y0, y1 = int(ys[0]), int(ys[-1])
    x0, x1 = int(xs[0]), int(xs[-1])
    return Segment(
        mask=mask[y0 : y1 + 1, x0 : x1 + 1],
        x=x0,
        y=y0 + y_offset,
        w=x1 - x0 + 1,
        h=y1 - y0 + 1,
    )


def _merge_fragments(segments: list[Segment], profile: Profile) -> list[Segment]:
    """Merge fragments of one glyph into a single segment.

    Components of the same glyph are near each other: they overlap in x or y,
    or are separated by a gap of at most one pixel (x) / a few pixels (y).
    Components of adjacent characters are separated by a wider horizontal
    gap. A union-find pass over these proximity rules groups the fragments.
    """

    if not segments:
        return segments
    segs = sorted(segments, key=lambda s: s.x)
    n = len(segs)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    x_gap_tol = max(1, profile.stroke_width)
    y_gap_tol = max(3, profile.char_height_min // 2)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = segs[i], segs[j]
            overlap_x = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
            overlap_y = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
            gap_x = max(a.x, b.x) - min(a.x + a.w, b.x + b.w)
            gap_y = max(a.y, b.y) - min(a.y + a.h, b.y + b.h)
            if (gap_x <= x_gap_tol and overlap_y >= 1) or (
                gap_y <= y_gap_tol and overlap_x >= 1
            ):
                union(i, j)

    groups: dict[int, list[Segment]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(segs[i])
    merged = []
    for group in groups.values():
        acc = group[0]
        for seg in group[1:]:
            acc = _union(acc, seg)
        merged.append(acc)
    merged.sort(key=lambda s: (s.x, s.y))
    return merged


def _union(a: Segment, b: Segment) -> Segment:
    x0 = min(a.x, b.x)
    y0 = min(a.y, b.y)
    x1 = max(a.x + a.w, b.x + b.w)
    y1 = max(a.y + a.h, b.y + b.h)
    combined = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    combined[
        a.y - y0 : a.y - y0 + a.h, a.x - x0 : a.x - x0 + a.w
    ] = a.mask
    combined[
        b.y - y0 : b.y - y0 + b.h, b.x - x0 : b.x - x0 + b.w
    ] |= b.mask
    seg = _bbox_segment(combined, y0)
    return Segment(
        mask=seg.mask,
        x=x0 + seg.x,
        y=seg.y,
        w=seg.w,
        h=seg.h,
    )


def _split_wide(segments: list[Segment], profile: Profile) -> list[Segment]:
    """Split merged components whose width exceeds the expected glyph width."""

    out: list[Segment] = []
    for seg in segments:
        max_w = max(profile.char_width_max, profile.char_width_min + 2)
        if seg.w <= max_w:
            out.append(seg)
            continue
        # Vertical projection valleys decide where to cut.
        proj = np.any(seg.mask, axis=0).astype(np.int32)
        width = seg.w
        est = max(profile.char_width_min, profile.char_height_min)
        cuts: list[int] = []
        i = est
        while i < width - est:
            window = proj[max(0, i - profile.stroke_width) : i + profile.stroke_width + 1]
            if window.sum() == 0:
                cuts.append(i)
                i += est
            else:
                i += 1
        if not cuts:
            out.append(seg)
            continue
        prev = 0
        for cut in cuts + [width]:
            part = seg.mask[:, prev:cut]
            if np.any(part):
                out.append(_bbox_segment(part, seg.y))
            prev = cut
    return out


def normalize(mask: NDArray[np.bool_], target: int = 24) -> NDArray[np.uint8]:
    """Center a character bitmap in a ``target x target`` binary image.

    Returns uint8 0/255 so the result can be packed or fed to a CNN.
    """

    h, w = mask.shape
    scale = min(target * 0.8 / max(h, 1), target * 0.8 / max(w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Nearest-neighbor resize is deterministic and preserves ink.
    ys = (np.arange(new_h)[:, None] * h / new_h).astype(np.int32)
    xs = (np.arange(new_w)[None, :] * w / new_w).astype(np.int32)
    resized = mask[ys, xs]

    out = np.zeros((target, target), dtype=np.uint8)
    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized.astype(np.uint8) * 255
    return out


def normalize_grayscale(roi: NDArray[np.uint8], target: int = 24) -> NDArray[np.uint8]:
    """Center a tight grayscale ROI in a ``target x target`` image.

    Unlike :func:`normalize` this keeps anti-aliasing, edge intensity and
    sub-pixel scaling information: the result is ``uint8 [target, target]``
    with values in 0..255 (not just 0/255). Used for the P1 grayscale-CNN
    experiment; segmentation still consumes the binary mask.
    """

    roi = np.asarray(roi, dtype=np.uint8)
    if roi.ndim != 2:
        raise ValueError(f"expected a grayscale ROI with shape (H, W), got {roi.shape}")
    h, w = roi.shape
    if h == 0 or w == 0:
        return np.zeros((target, target), dtype=np.uint8)
    scale = min(target * 0.8 / h, target * 0.8 / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    ys = (np.arange(new_h)[:, None] * h / new_h).astype(np.int32)
    xs = (np.arange(new_w)[None, :] * w / new_w).astype(np.int32)
    resized = roi[ys, xs]
    out = np.zeros((target, target), dtype=np.uint8)
    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def collect_glyphs_grayscale(
    image: NDArray[np.uint8],
    profile: Profile,
) -> tuple[list[Segment], NDArray[np.uint8]]:
    """Segment an image and return binary segments plus grayscale glyphs.

    The P1 input split: segmentation/template keep the binary mask while
    the TinyCNN may consume ``uint8 [N, 24, 24]`` grayscale glyphs that
    preserve anti-aliasing and edge strength.
    """

    mask = profile.color_mask(image)
    gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    segments: list[Segment] = []
    glyphs: list[NDArray[np.uint8]] = []
    for line in _find_lines(mask, profile):
        for seg in _segment_line(line, profile):
            roi = gray[seg.y : seg.y + seg.h, seg.x : seg.x + seg.w]
            segments.append(seg)
            glyphs.append(normalize_grayscale(roi, profile.target_size))
    if not glyphs:
        return segments, np.empty(
            (0, profile.target_size, profile.target_size), dtype=np.uint8
        )
    return segments, np.stack(glyphs)
