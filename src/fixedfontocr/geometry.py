"""Goal 8: offline font geometry database and its NumPy runtime view.

The geometry database is generated once from a registered ``fonts/`` file:

* ``char_id`` / ``advance`` / ``bbox_width`` / ``bbox_height`` /
  ``aspect_ratio`` are taken from the font's outline metrics (normalized to
  em units so they are scale-invariant);
* ``ink_count`` / ``component_count`` / ``ink_ratio`` come from a
  deterministic rasterization of the same glyph;
* ``baseline`` / ``baseline_ratio`` record where the font baseline sits
  inside each glyph's ink bbox.

The database is stored as ``geometry.json`` next to the model's
``config.json``. At runtime it is loaded as plain JSON (no fontTools/Pillow
dependency) and feeds:

* candidate pruning (narrow vs. full-width expected bbox);
* the per-candidate geometry score (bbox/aspect/ink/component/baseline
  agreement with the classifier's Top-1 character);
* split/merge decisions (expected glyph width and component count).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .defaults import compute_font_sha256, resolve_font
from .types import VisualCandidate

FORMAT = "goal8-geometry-v1"


@dataclass(frozen=True)
class FontGeometryEntry:
    """Per-character geometry facts for one registered font."""

    char_id: int
    advance: float
    bbox_width: float
    bbox_height: float
    aspect_ratio: float
    ink_count: int
    component_count: int
    baseline: float
    baseline_ratio: float
    ink_ratio: float

    def to_dict(self) -> dict:
        return {
            "char_id": self.char_id,
            "advance": self.advance,
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "aspect_ratio": self.aspect_ratio,
            "ink_count": self.ink_count,
            "component_count": self.component_count,
            "baseline": self.baseline,
            "baseline_ratio": self.baseline_ratio,
            "ink_ratio": self.ink_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FontGeometryEntry":
        return cls(
            char_id=int(data["char_id"]),
            advance=float(data["advance"]),
            bbox_width=float(data["bbox_width"]),
            bbox_height=float(data["bbox_height"]),
            aspect_ratio=float(data["aspect_ratio"]),
            ink_count=int(data["ink_count"]),
            component_count=int(data["component_count"]),
            baseline=float(data["baseline"]),
            baseline_ratio=float(data["baseline_ratio"]),
            ink_ratio=float(data["ink_ratio"]),
        )


@dataclass(frozen=True)
class FontGeometryDatabase:
    """Immutable lookup table for every character in a model charset."""

    font_sha256: str
    render_size: int
    threshold: int
    charset: str
    entries: tuple[FontGeometryEntry, ...]

    def __post_init__(self) -> None:
        ids = [e.char_id for e in self.entries]
        if ids != list(range(len(self.entries))):
            raise ValueError("font geometry entries must be dense char_id 0..N-1")
        if len(self.charset) != len(self.entries):
            raise ValueError(
                f"font geometry charset has {len(self.charset)} chars but "
                f"{len(self.entries)} entries"
            )
        object.__setattr__(self, "_by_id", {e.char_id: e for e in self.entries})

    @property
    def by_id(self) -> dict[int, FontGeometryEntry]:
        return self._by_id

    def get(self, char_id: int) -> FontGeometryEntry | None:
        return self._by_id.get(char_id)

    @property
    def narrow_entries(self) -> list[FontGeometryEntry]:
        return [e for e in self.entries if e.aspect_ratio < 0.8]

    @property
    def full_entries(self) -> list[FontGeometryEntry]:
        return [e for e in self.entries if e.aspect_ratio >= 0.8]

    @property
    def narrow_width_ratio(self) -> float:
        values = [e.bbox_width for e in self.narrow_entries]
        return float(statistics.median(values)) if values else 0.4

    @property
    def narrow_height_ratio(self) -> float:
        values = [e.bbox_height for e in self.narrow_entries]
        return float(statistics.median(values)) if values else 0.75

    @property
    def narrow_width_max(self) -> float:
        values = [e.bbox_width for e in self.narrow_entries]
        return max(values) if values else 0.6

    @property
    def narrow_width_min(self) -> float:
        values = [e.bbox_width for e in self.narrow_entries]
        return min(values) if values else 0.15

    @property
    def narrow_advance_max(self) -> float:
        values = [e.advance for e in self.narrow_entries]
        return max(values) if values else 0.7

    @property
    def full_width_ratio(self) -> float:
        values = [e.bbox_width for e in self.full_entries]
        return float(statistics.median(values)) if values else 0.93

    @property
    def full_height_ratio(self) -> float:
        values = [e.bbox_height for e in self.full_entries]
        return float(statistics.median(values)) if values else 0.93

    @property
    def full_width_max(self) -> float:
        values = [e.bbox_width for e in self.full_entries]
        return max(values) if values else 1.0

    @property
    def full_width_min(self) -> float:
        values = [e.bbox_width for e in self.full_entries]
        return min(values) if values else 0.55

    @property
    def full_advance_max(self) -> float:
        values = [e.advance for e in self.full_entries]
        return max(values) if values else 1.0

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "font_sha256": self.font_sha256,
            "render_size": self.render_size,
            "threshold": self.threshold,
            "charset": self.charset,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FontGeometryDatabase":
        if data.get("format") != FORMAT:
            raise ValueError(f"unsupported font geometry format: {data.get('format')!r}")
        return cls(
            font_sha256=str(data["font_sha256"]),
            render_size=int(data["render_size"]),
            threshold=int(data["threshold"]),
            charset=str(data.get("charset", "")),
            entries=tuple(
                FontGeometryEntry.from_dict(e) for e in data.get("entries", ())
            ),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "FontGeometryDatabase":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def char_geometry(
    candidate: VisualCandidate,
    char_id: int,
    geometry: FontGeometryDatabase | None,
    normalize_geometry: tuple[float, float] | None = None,
) -> float:
    """Character-specific geometry agreement for one ``(candidate, char_id)``.

    This is the identity-dependent half of the Goal 8 geometry score:
    expected bbox / aspect / advance / ink ratio / component count /
    baseline are compared against the font geometry entry of the character
    actually chosen by the decoder. Candidate-level evidence (alignment,
    component gaps, segmentation width) lives in
    :func:`fixedfontocr.segmentation.candidate_geometry` and is intentionally
    absent here.
    """

    if geometry is None or char_id is None or int(char_id) < 0:
        return 0.0
    entry = geometry.get(int(char_id))
    seg = candidate.segment
    if entry is None or seg is None or seg.h < 1 or seg.w < 1:
        return 0.0
    if normalize_geometry is None:
        normalize_geometry = candidate.normalize_geometry

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
    score = 0.0
    score -= min(0.05, width_err * 0.04)
    score -= min(0.04, height_err * 0.03)
    score -= min(0.03, aspect_err * 0.02)
    advance_err = abs(w_ratio - entry.advance) / max(entry.advance, 1e-3)
    score -= min(0.02, advance_err * 0.01)

    ink_ratio = float(seg.mask.sum()) / float(w * h)
    ink_err = abs(ink_ratio - entry.ink_ratio) / max(entry.ink_ratio, 0.05)
    score -= min(0.03, ink_err * 0.01)

    # Component-count prior: a fragmented glyph whose chosen character needs
    # several components is penalized; a merge whose components match the
    # glyph is rewarded. Split whole-component alternatives (atoms >
    # components) are exempt because they still represent the original
    # connected glyph.
    expected_cc = max(1, entry.component_count)
    actual_cc = max(1, len(candidate.components))
    # If candidate_geometry penalized a merged candidate for being wider
    # than the line's typical glyph, that generic penalty is identity-
    # independent evidence. A database entry with a multi-component glyph
    # (小/鲃/潜) is the authoritative exception: add the penalty back here,
    # exactly matching the Goal 14 split between the two geometry halves.
    if len(candidate.components) > 1 and expected_cc > 1:
        score -= candidate.segmentation_width_penalty
    start, end = candidate.atom_span
    is_split_whole = end - start > actual_cc
    if not is_split_whole:
        if actual_cc > 1 and expected_cc == 1:
            score -= min(0.04, (actual_cc - 1) * 0.02)
        elif actual_cc == 1 and expected_cc > 1:
            score -= min(0.03, (expected_cc - 1) * 0.015)
        elif expected_cc == actual_cc and actual_cc > 1:
            # Goal 14 merge agreement: the chosen character's database
            # entry expects exactly the number of components this candidate
            # merged (小/鲃/获/得 at low resolution). That agreement is
            # positive evidence for a real fragmented glyph, so it offsets
            # the generic merge penalties; without it the correct
            # whole-glyph candidate keeps losing to look-alike single-char
            # fragments (小 -> fj\, 鲃 -> $8) at 12-18 px.
            score += min(0.05, (actual_cc - 1) * 0.025)

    if normalize_geometry is not None:
        baseline_offset = float(normalize_geometry[0])
        if 0.0 < baseline_offset <= h * 1.5 and 0.0 < entry.baseline_ratio <= 1.5:
            cand_ratio = baseline_offset / h
            score -= min(0.02, abs(cand_ratio - entry.baseline_ratio) * 0.02)

    return float(score)


def _connected_count(mask) -> int:
    """4-connected component count, matching runtime connected_components."""
    from .preprocess import _connected_components

    return len(list(_connected_components(mask)))


def build_geometry_database(
    font_path: str | Path,
    charset: list[str],
    render_size: int = 32,
    threshold: int = 140,
    font_sha256: str | None = None,
) -> FontGeometryDatabase:
    """Generate the Goal 8 database from a registered font (offline only).

    Outline metrics (advance, bbox, baseline) come from fontTools; ink and
    component counts come from the same deterministic rasterizer used by
    template generation. A missing glyph or an empty ink mask is an explicit
    error, per the project Font Policy.
    """

    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont

    from .fontgen import render_glyph

    font = resolve_font(font_path)
    sha = font_sha256 or compute_font_sha256(font)
    tt = TTFont(str(font))
    try:
        upem = int(tt["head"].unitsPerEm)
        cmap = tt.getBestCmap() or {}
        glyph_set = tt.getGlyphSet()
        hmtx = tt["hmtx"]
        entries: list[FontGeometryEntry] = []
        for char_id, ch in enumerate(charset):
            glyph_name = cmap.get(ord(ch))
            if glyph_name is None:
                raise ValueError(
                    f"font {font} cannot render {ch!r}: missing glyph"
                )
            pen = BoundsPen(glyph_set)
            glyph_set[glyph_name].draw(pen)
            bounds = pen.bounds
            if bounds is None:
                raise ValueError(
                    f"font {font} cannot render {ch!r}: no outline ink"
                )
            x0, y0, x1, y1 = bounds
            h = y1 - y0
            w = x1 - x0
            if h <= 0 or w <= 0:
                raise ValueError(
                    f"font {font} cannot render {ch!r}: empty outline bbox"
                )
            advance = float(hmtx[glyph_name][0]) / upem
            bbox_width = float(w) / upem
            bbox_height = float(h) / upem
            baseline = float(y1) / upem
            baseline_ratio = float(y1) / h
            mask = render_glyph(font, ch, render_size, threshold)
            if mask.size == 0:
                raise ValueError(
                    f"font {font} cannot render {ch!r}: no ink rasterized"
                )
            ink_count = int(mask.sum())
            ink_ratio = float(ink_count) / float(mask.shape[0] * mask.shape[1])
            entries.append(
                FontGeometryEntry(
                    char_id=char_id,
                    advance=advance,
                    bbox_width=bbox_width,
                    bbox_height=bbox_height,
                    aspect_ratio=float(w) / float(h),
                    ink_count=ink_count,
                    component_count=_connected_count(mask),
                    baseline=baseline,
                    baseline_ratio=baseline_ratio,
                    ink_ratio=ink_ratio,
                )
            )
    finally:
        tt.close()
    return FontGeometryDatabase(
        font_sha256=sha,
        render_size=render_size,
        threshold=threshold,
        charset="".join(charset),
        entries=tuple(entries),
    )


def write_geometry_json(
    model_dir: str | Path,
    font_path: str | Path,
    charset: list[str],
    render_size: int = 32,
    threshold: int = 140,
    font_sha256: str | None = None,
) -> Path:
    """Build and write ``geometry.json`` into a model directory."""
    db = build_geometry_database(
        font_path,
        charset,
        render_size,
        threshold,
        font_sha256=font_sha256,
    )
    path = Path(model_dir) / "geometry.json"
    db.save(path)
    return path
