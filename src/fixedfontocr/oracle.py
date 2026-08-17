"""Oracle lattice recall: is the ground-truth segmentation in the lattice?

The lattice never decides "cut vs no-cut" irreversibly -- it *offers* every
plausible candidate and lets the decoder choose. When the final text is
wrong, the first question is therefore not "which CNN weights to tune" but
"was the correct path even in the lattice?":

* oracle path exists, decoder chose another path  -> classification /
  decoding problem (the lattice is fine; the fix belongs in the CNN /
  geometry / lexicon scoring);
* oracle path missing                             -> segmentation problem
  (candidate generation or pruning destroyed the correct answer, and no
  classifier, however strong, can recover it).

The oracle ignores all scores. Ground truth is a per-character ink box
(:class:`GroundTruthBox`); :func:`oracle_lattice_recall` builds the same
candidate lattice the production pipeline uses and checks whether a path
exists in which every GT character has a candidate whose segment

* covers at least ``coverage_threshold`` (default 0.9) of the character's
  ink, with at least ``coverage_threshold`` of its own ink inside the
  character's box, and
* has a bbox within ``tolerance`` px of the character's box on all four
  sides,

while the candidates tile the atom sequence exactly, in text order.

Failure taxonomy (:attr:`OracleResult.reason`):

* ``unassigned_atom`` -- an atom's ink is spread across two character
  boxes (or belongs to no box): a connected blob was not split anywhere
  near the GT boundary. Typical for fully-touching glyphs with no
  vertical valley (``4+3``, ``U-``) that the valley heuristic cannot cut.
* ``missing_candidate`` -- every atom individually sits inside one box,
  but no candidate covers a whole GT character: the needed merge or split
  candidate was pruned or never generated (e.g. a valley cut that landed
  inside the left glyph instead of at the boundary).
* ``no_path`` -- atoms are assignable and per-character candidates exist,
  yet no consistent tiling was found (pathological / overlapping boxes).

:func:`evaluate_oracle` additionally attributes each sample to
``ok`` / ``segmentation`` / ``decoding`` and :func:`format_report` prints
the table. ``tools/oracle_recall.py`` runs the report over a rendered
corpus including deliberately glued (fully-touching) glyph pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .geometry import FontGeometryDatabase
from .preprocess import Segment
from .segmentation import build_candidates, connected_components, expand_atoms
from .types import Profile, VisualCandidate

COVERAGE_THRESHOLD = 0.9
"""Fraction of ink a candidate must share with the GT character box."""

TOLERANCE_PX = 2
"""Max per-side bbox deviation (px) between a candidate and the GT box."""

FAIL_REASONS = ("unassigned_atom", "missing_candidate", "no_path")


@dataclass(frozen=True)
class GroundTruthBox:
    """One ground-truth character: half-open ink box in image coordinates.

    The box is the *raster ink* box (not the advance/layout box), i.e. the
    tight bounding box of the character's thresholded pixels, so it is
    directly comparable with candidate segment bboxes.
    """

    char: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass(frozen=True)
class GroundTruthLine:
    """One line of text with per-character ground-truth ink boxes.

    ``mask`` is the binary line mask in line-local coordinates; ``x``/``y``
    are the line's offset in the source image and ``boxes`` are in image
    coordinates (matching the :class:`Component` contract). ``image`` is
    the optional RGB source (used by attribution decoders).
    """

    text: str
    mask: NDArray[np.bool_]
    boxes: tuple[GroundTruthBox, ...]
    image: NDArray[np.uint8] | None = None
    x: int = 0
    y: int = 0

    def __post_init__(self) -> None:
        if len(self.text) != len(self.boxes):
            raise ValueError(
                f"text has {len(self.text)} chars but {len(self.boxes)} boxes"
            )
        if self.mask.ndim != 2 or self.mask.size == 0:
            raise ValueError("ground truth mask must be a non-empty 2D array")
        for i, b in enumerate(self.boxes):
            if b.x1 <= b.x0 or b.y1 <= b.y0:
                raise ValueError(f"empty ground truth box for {b.char!r}")
            if not (0 <= b.x0 < b.x1 <= self.mask.shape[1]):
                raise ValueError(f"box for {b.char!r} outside mask width")
            if not (0 <= b.y0 < b.y1 <= self.mask.shape[0]):
                raise ValueError(f"box for {b.char!r} outside mask height")
            ink = self.mask[
                b.y0 - self.y : b.y1 - self.y, b.x0 - self.x : b.x1 - self.x
            ]
            if not np.any(ink):
                raise ValueError(f"ground truth box for {b.char!r} has no ink")
            for j, o in enumerate(self.boxes[:i]):
                if (
                    min(b.x1, o.x1) > max(b.x0, o.x0)
                    and min(b.y1, o.y1) > max(b.y0, o.y0)
                ):
                    raise ValueError(
                        f"ground truth boxes overlap: {o.char!r} and {b.char!r}"
                    )

    @property
    def line(self) -> Segment:
        """The line as a pipeline :class:`Segment` (mask + image offset)."""
        return Segment(
            mask=self.mask,
            x=self.x,
            y=self.y,
            w=self.mask.shape[1],
            h=self.mask.shape[0],
        )


def render_glyph_mask(
    char: str,
    font_path,
    font_size: int = 32,
    threshold: int = 140,
    color: tuple[int, int, int] = (255, 255, 255),
) -> NDArray[np.bool_]:
    """Tight binary ink mask of one character (used by glue/test helpers)."""

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), font_size)
    cb = font.getbbox(char)
    pad = 8
    w = cb[2] - cb[0] + pad * 2
    h = cb[3] - cb[1] + pad * 2
    img = Image.new("RGB", (max(1, w), max(1, h)), (0, 0, 0))
    ImageDraw.Draw(img).text(
        (pad - cb[0], pad - cb[1]), char, font=font, fill=color
    )
    gray = np.asarray(img, dtype=np.uint8) @ np.array(
        [0.299, 0.587, 0.114], dtype=np.float32
    )
    mask = gray >= threshold
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError(
            f"font {font_path} renders no ink for {char!r} at {font_size}px"
        )
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def render_ground_truth(
    text: str,
    font_path,
    font_size: int = 32,
    pad: int = 8,
    threshold: int = 140,
    color: tuple[int, int, int] = (255, 255, 255),
) -> GroundTruthLine:
    """Render a text line and compute exact per-character ink boxes.

    The line is drawn with a single ``draw.text`` call (the same layout the
    pipeline sees). Each character's GT box is the *rasterized* ink bbox of
    that character (from a standalone render at the same integer origin,
    so hinting is identical), shifted by the cumulative advance. This is
    pixel-exact for integer advances and within ~1 px otherwise, which the
    oracle tolerance absorbs. A box with no ink or ink outside all boxes
    (layout mismatch) raises :class:`ValueError`.
    """

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), font_size)
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGB", (max(1, w), max(1, h)), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    ox, oy = pad - bbox[0], pad - bbox[1]
    draw.text((ox, oy), text, font=font, fill=color)
    arr = np.asarray(img, dtype=np.uint8)
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mask = gray >= threshold

    boxes: list[GroundTruthBox] = []
    for i, ch in enumerate(text):
        cb = font.getbbox(ch)
        adv = int(round(font.getlength(text[:i])))
        sw = cb[2] - cb[0] + pad * 2
        sh = cb[3] - cb[1] + pad * 2
        simg = Image.new("RGB", (max(1, sw), max(1, sh)), (0, 0, 0))
        ImageDraw.Draw(simg).text(
            (pad - cb[0], pad - cb[1]), ch, font=font, fill=color
        )
        sarr = np.asarray(simg, dtype=np.uint8)
        sgray = sarr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        smask = sgray >= threshold
        ys, xs = np.where(smask)
        if ys.size == 0:
            raise ValueError(
                f"font {font_path} renders no ink for {ch!r} at {font_size}px"
            )
        sx0, sy0 = int(xs.min()), int(ys.min())
        gx0 = ox + adv + sx0 - (pad - cb[0])
        gy0 = oy + sy0 - (pad - cb[1])
        boxes.append(
            GroundTruthBox(
                ch, gx0, gy0, gx0 + int(xs.max()) - sx0 + 1, gy0 + int(ys.max()) - sy0 + 1
            )
        )

    gt = GroundTruthLine(text=text, mask=mask, boxes=tuple(boxes), image=arr)
    covered = np.zeros_like(mask)
    for b in gt.boxes:
        covered[b.y0 : b.y1, b.x0 : b.x1] = True
    if np.any(mask & ~covered):
        raise ValueError(
            f"rendered ink outside all ground truth boxes for {text!r} -- "
            "per-character layout does not match the full-string render"
        )
    return gt


def glue_ground_truth(
    left: GroundTruthLine,
    right: GroundTruthLine,
    bridge: str = "full",
) -> GroundTruthLine:
    """Join two single-character lines into one line, optionally gluing them.

    Mimics the real-game failure mode where adjacent glyphs merge into one
    connected component. Both inputs must be single-character lines with
    ``x == y == 0`` (i.e. produced by :func:`render_ground_truth`); the
    glyphs are bottom-aligned (ink bottoms on one row, like a baseline).

    ``bridge`` (the glyphs always sit with a 1 px blank column between
    them, which the bridge fills):

    * ``"full"`` -- the blank column is filled entirely: the pair is one
      connected component with *no* vertical valley at the boundary (the
      column's projection is the full height). The valley heuristic cannot
      split it.
    * ``"row"`` -- a 1 px connector at a row where both glyphs have ink in
      their outer columns: one connected component with a thin valley at
      the boundary (the classic low-res connector).
    * ``"none"`` -- the blank column stays blank: two separate components.

    The returned GT boxes are the exact placement boxes, so the oracle
    check is self-consistent: bridge ink legitimately belongs to no box.
    """

    if len(left.text) != 1 or len(right.text) != 1:
        raise ValueError("glue_ground_truth expects two single-character lines")
    if left.x or left.y or right.x or right.y:
        raise ValueError("glue_ground_truth expects line-local (x=y=0) inputs")
    if bridge not in ("full", "row", "none"):
        raise ValueError(f"unknown bridge {bridge!r}")

    # The inputs may carry render padding around their ink; tighten each to
    # its ink bbox and re-base the GT box, so the composite places the ink
    # (not the canvas) and the GT boxes stay pixel-exact. The glyphs always
    # get a 1 px blank column between them; the bridge fills it (so a
    # zero-gap layout would already be touching and no bridge could add a
    # valley).
    left_mask, left_box = _tighten(left)
    right_mask, right_box = _tighten(right)
    hl, wl = left_mask.shape
    hr, wr = right_mask.shape
    h = max(hl, hr)
    w = wl + 1 + wr
    out = np.zeros((h, w), dtype=bool)
    out[h - hl :, :wl] = left_mask
    out[h - hr :, wl + 1 :] = right_mask
    if bridge == "row":
        # A thin connector in the blank column between the two glyphs'
        # outer columns: 1 px wide, from the left glyph's right-edge ink
        # row to the right glyph's left-edge ink row (the rows are chosen
        # closest together). The connector's column projection stays at or
        # below the valley threshold (~18% of the height), so the pair is
        # one connected component with a real vertical valley at the
        # boundary -- the classic low-res connector.
        rows_l = h - hl + np.where(left_mask[:, wl - 1])[0]
        rows_r = h - hr + np.where(right_mask[:, 0])[0]
        if rows_l.size == 0 or rows_r.size == 0:
            raise ValueError(
                f"glyphs {left.text!r} and {right.text!r} have no ink in "
                "their outer columns; cannot row-bridge"
            )
        d = np.abs(rows_l[:, None] - rows_r[None, :])
        i, j = np.unravel_index(int(np.argmin(d)), d.shape)
        r_lo = int(min(rows_l[i], rows_r[j]))
        r_hi = int(max(rows_l[i], rows_r[j]))
        if r_hi - r_lo + 1 > max(1, int(h * 0.18)):
            raise ValueError(
                f"glyphs {left.text!r} and {right.text!r} are {r_hi - r_lo}px "
                "apart vertically; cannot make a thin row-bridge"
            )
        out[r_lo : r_hi + 1, wl] = True
    elif bridge == "full":
        # Fill the blank column entirely: one connected component with no
        # valley at the boundary (full-height projection).
        out[:, wl] = True

    def shift(b: GroundTruthBox, dx: int, dy: int) -> GroundTruthBox:
        return GroundTruthBox(b.char, b.x0 + dx, b.y0 + dy, b.x1 + dx, b.y1 + dy)

    lb = shift(left_box, 0, h - hl)
    rb = shift(right_box, wl + 1, h - hr)
    return GroundTruthLine(
        text=left.text + right.text,
        mask=out,
        boxes=(lb, rb),
        x=0,
        y=0,
    )


def _tighten(gt: GroundTruthLine) -> tuple[NDArray[np.bool_], GroundTruthBox]:
    """Crop a single-char line to its ink bbox and re-base its GT box."""
    ys, xs = np.where(gt.mask)
    x0, y0 = int(xs.min()), int(ys.min())
    tight = gt.mask[y0 : int(ys.max()) + 1, x0 : int(xs.max()) + 1]
    b = gt.boxes[0]
    box = GroundTruthBox(b.char, b.x0 - x0, b.y0 - y0, b.x1 - x0, b.y1 - y0)
    return tight, box


def _ink_inside(seg: Segment, box: GroundTruthBox, line: Segment) -> int:
    """Ink pixels of ``seg`` (image coords) that fall inside ``box``."""
    x0 = max(0, box.x0 - line.x - seg.x)
    x1 = min(seg.w, box.x1 - line.x - seg.x)
    y0 = max(0, box.y0 - line.y - seg.y)
    y1 = min(seg.h, box.y1 - line.y - seg.y)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(seg.mask[y0:y1, x0:x1].sum())


def _box_ink(box: GroundTruthBox, line: Segment) -> int:
    bx0 = max(0, box.x0 - line.x)
    bx1 = min(line.w, box.x1 - line.x)
    by0 = max(0, box.y0 - line.y)
    by1 = min(line.h, box.y1 - line.y)
    if bx1 <= bx0 or by1 <= by0:
        return 0
    return int(line.mask[by0:by1, bx0:bx1].sum())


def _coverage(seg: Segment, box: GroundTruthBox, line: Segment) -> float:
    """min(ink(seg) in box / ink(seg), ink(seg) in box / ink(box))."""
    inside = _ink_inside(seg, box, line)
    total = int(seg.mask.sum())
    box_ink = _box_ink(box, line)
    if total <= 0 or box_ink <= 0:
        return 0.0
    return float(min(inside / total, inside / box_ink))


def _atom_coverage(seg: Segment, box: GroundTruthBox, line: Segment) -> float:
    """Fraction of the atom's ink inside the box (atom-level check)."""
    inside = _ink_inside(seg, box, line)
    total = int(seg.mask.sum())
    if total <= 0:
        return 0.0
    return float(inside / total)


def _bbox_matches(
    seg: Segment, box: GroundTruthBox, tolerance: int
) -> bool:
    return (
        abs(seg.x - box.x0) <= tolerance
        and abs(seg.y - box.y0) <= tolerance
        and abs(seg.x + seg.w - box.x1) <= tolerance
        and abs(seg.y + seg.h - box.y1) <= tolerance
    )


@dataclass(frozen=True)
class OracleResult:
    """Outcome of the oracle check for one line.

    ``recall`` is the metric: whether the GT segmentation is present in
    the candidate lattice when scores are ignored. When ``recall`` is
    False, ``reason`` names the failure class and the diagnostic fields
    locate it:

    * ``atom_assignments`` -- per atom, the GT char index whose box holds
      the largest share of the atom's ink (``None`` when that share is
      below the coverage threshold);
    * ``straddled_boundaries`` -- ``(left_gt, right_gt, x)`` triples: an
      atom carries ink from both characters around the boundary at ``x``,
      i.e. the missing cut position;
    * ``missing_chars`` -- GT char indices with no valid candidate;
    * ``best_coverage`` -- per GT char, the best coverage any candidate
      achieves (``0.0`` when none).
    """

    recall: bool
    text: str
    reason: str = ""
    path: tuple[VisualCandidate, ...] = ()
    n_atoms: int = 0
    n_candidates: int = 0
    atom_assignments: tuple[int | None, ...] = ()
    straddled_boundaries: tuple[tuple[int, int, int], ...] = ()
    missing_chars: tuple[int, ...] = ()
    best_coverage: tuple[float, ...] = ()


def oracle_lattice_recall(
    line: Segment,
    profile: Profile,
    boxes: Sequence[GroundTruthBox],
    *,
    max_merge_components: int = 4,
    split_wide: bool = True,
    geometry: FontGeometryDatabase | None = None,
    candidates: Sequence[VisualCandidate] | None = None,
    coverage_threshold: float = COVERAGE_THRESHOLD,
    tolerance: int = TOLERANCE_PX,
) -> OracleResult:
    """Score-free check: does the GT segmentation exist in the lattice?

    Builds the same candidate lattice as the production pipeline
    (:func:`fixedfontocr.segmentation.build_candidates` with the caller's
    ``max_merge_components`` / ``split_wide`` / ``geometry``) and searches
    for a path covering the ground-truth character boxes in order.
    """

    comps = connected_components(line)
    if candidates is None:
        candidates = build_candidates(
            comps,
            profile,
            max_merge_components,
            split_wide=split_wide,
            geometry=geometry,
        )
    atoms, _expected = expand_atoms(
        comps,
        profile,
        max_merge_components,
        split_wide=split_wide,
        geometry=geometry,
    )
    n = len(atoms)
    by_span: dict[tuple[int, int], VisualCandidate] = {}
    for cand in candidates:
        by_span.setdefault(cand.atom_span, cand)

    boxes = tuple(boxes)

    def valid(cand: VisualCandidate, box: GroundTruthBox) -> bool:
        seg = cand.segment
        if seg is None:
            return False
        if not _bbox_matches(seg, box, tolerance):
            return False
        return _coverage(seg, box, line) >= coverage_threshold

    def search(i: int, p: int) -> tuple[VisualCandidate, ...] | None:
        """Cover GT chars ``i..`` with candidates over atoms ``p..n``."""
        if i == len(boxes):
            return () if p == n else None
        box = boxes[i]
        for q in range(p + 1, n + 1):
            cand = by_span.get((p, q))
            if cand is None or not valid(cand, box):
                continue
            rest = search(i + 1, q)
            if rest is not None:
                return (cand,) + rest
        return None

    path = search(0, 0)

    if path is not None:
        return OracleResult(
            recall=True,
            text="".join(b.char for b in boxes),
            path=path,
            n_atoms=n,
            n_candidates=len(candidates),
            atom_assignments=_atom_assignments(atoms, boxes, line, coverage_threshold),
        )

    # --- diagnostics for the failed case -------------------------------
    assignments = _atom_assignments(atoms, boxes, line, coverage_threshold)
    unassigned = [k for k, a in enumerate(assignments) if a is None]

    straddled: list[tuple[int, int, int]] = []
    for k in range(len(boxes) - 1):
        # Boundary between GT chars k and k+1, as the column that separates
        # their boxes.  An atom whose bbox covers that column carries ink
        # from both sides (or from the connector column between the glyphs)
        # -- the missing cut position.
        x_b = (boxes[k].x1 + boxes[k + 1].x0) // 2
        for a in atoms:
            seg = a.segment
            if seg.x <= x_b < seg.x + seg.w:
                straddled.append((k, k + 1, x_b))
                break
    straddled = list(dict.fromkeys(straddled))

    if unassigned:
        reason = "unassigned_atom"
        missing: tuple[int, ...] = ()
        best: tuple[float, ...] = ()
    else:
        best_cov = []
        for box in boxes:
            b = 0.0
            for cand in candidates:
                if cand.segment is None:
                    continue
                b = max(b, _coverage(cand.segment, box, line))
            best_cov.append(b)
        best = tuple(best_cov)
        missing = tuple(i for i, b in enumerate(best_cov) if b < coverage_threshold)
        reason = "missing_candidate" if missing else "no_path"

    return OracleResult(
        recall=False,
        text="".join(b.char for b in boxes),
        reason=reason,
        n_atoms=n,
        n_candidates=len(candidates),
        atom_assignments=tuple(assignments),
        straddled_boundaries=tuple(straddled),
        missing_chars=missing,
        best_coverage=best,
    )


def _atom_assignments(
    atoms: Sequence,
    boxes: Sequence[GroundTruthBox],
    line: Segment,
    coverage_threshold: float,
) -> tuple[int | None, ...]:
    out: list[int | None] = []
    for a in atoms:
        seg = a.segment
        best_i: int | None = None
        best_cov = 0.0
        for i, box in enumerate(boxes):
            cov = _atom_coverage(seg, box, line)
            if cov > best_cov:
                best_i, best_cov = i, cov
        out.append(best_i if best_cov >= coverage_threshold else None)
    return tuple(out)


@dataclass(frozen=True)
class OracleSample:
    """Per-sample oracle + attribution outcome."""

    text: str
    oracle: bool
    status: str  # ok | segmentation | decoding
    reason: str = ""
    decoded: str | None = None
    straddled_boundaries: tuple[tuple[int, int, int], ...] = ()
    missing_chars: tuple[int, ...] = ()
    best_coverage: tuple[float, ...] = ()


@dataclass(frozen=True)
class OracleReport:
    """Aggregate oracle recall + attribution over a corpus."""

    total: int
    oracle_recall: float
    status_counts: dict[str, int]
    samples: tuple[OracleSample, ...]


def evaluate_oracle(
    lines: Sequence[GroundTruthLine],
    profile: Profile,
    *,
    max_merge_components: int = 4,
    split_wide: bool = True,
    geometry: FontGeometryDatabase | None = None,
    decode_fn: Callable[[GroundTruthLine], str] | None = None,
    coverage_threshold: float = COVERAGE_THRESHOLD,
    tolerance: int = TOLERANCE_PX,
) -> OracleReport:
    """Oracle recall over a corpus, with error attribution.

    ``decode_fn`` (optional) runs the real recognizer on a ground truth
    line and returns its text. Attribution:

    * oracle missing            -> ``"segmentation"`` (no score can fix it);
    * oracle present, decode ok -> ``"ok"``;
    * oracle present, decode wrong -> ``"decoding"`` (classifier / decoder
      problem; the lattice was fine).
    """

    samples: list[OracleSample] = []
    for gt in lines:
        res = oracle_lattice_recall(
            gt.line,
            profile,
            gt.boxes,
            max_merge_components=max_merge_components,
            split_wide=split_wide,
            geometry=geometry,
            coverage_threshold=coverage_threshold,
            tolerance=tolerance,
        )
        if res.recall:
            if decode_fn is not None:
                decoded = decode_fn(gt)
                status = "ok" if decoded == gt.text else "decoding"
            else:
                decoded, status = None, "ok"
            samples.append(
                OracleSample(
                    text=gt.text,
                    oracle=True,
                    status=status,
                    decoded=decoded,
                )
            )
        else:
            samples.append(
                OracleSample(
                    text=gt.text,
                    oracle=False,
                    status="segmentation",
                    reason=res.reason,
                    straddled_boundaries=res.straddled_boundaries,
                    missing_chars=res.missing_chars,
                    best_coverage=res.best_coverage,
                )
            )

    counts: dict[str, int] = {}
    for s in samples:
        counts[s.status] = counts.get(s.status, 0) + 1
    recall = sum(1 for s in samples if s.oracle) / max(len(samples), 1)
    return OracleReport(
        total=len(samples),
        oracle_recall=float(recall),
        status_counts=counts,
        samples=tuple(samples),
    )


def format_report(report: OracleReport) -> str:
    """Render an :class:`OracleReport` as a text table."""

    lines = [
        f"oracle lattice recall: {report.oracle_recall:.1%} "
        f"({sum(1 for s in report.samples if s.oracle)}/{report.total})",
        f"attribution: {report.status_counts}",
        "",
        f"{'text':<12} {'oracle':<7} {'status':<13} {'reason':<18} details",
        "-" * 88,
    ]
    for s in report.samples:
        details: list[str] = []
        if s.straddled_boundaries:
            parts = [
                f"missing cut @x{x} ({s.text[l]}|{s.text[r]})"
                for (l, r, x) in s.straddled_boundaries
            ]
            details.append("; ".join(parts))
        if s.missing_chars:
            details.append(
                f"chars {s.missing_chars} best-cov "
                + ", ".join(f"{c:.2f}" for c in s.best_coverage)
            )
        if s.decoded is not None and s.status != "ok":
            details.append(f"decoded {s.decoded!r}")
        lines.append(
            f"{s.text:<12} {str(s.oracle):<7} {s.status:<13} "
            f"{s.reason:<18} {' | '.join(details)}"
        )
    return "\n".join(lines)
