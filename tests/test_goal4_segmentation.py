"""Goal 4 acceptance: the segmentation lattice (merge + split candidates).

The Goal 4 contract from ``fonts/goal``:

* original connected components are preserved -- nothing is irreversibly
  merged before scoring;
* the lattice contains single components, consecutive merges
  (``C0``, ``C0+C1``, ``C0+C1+C2``, ...) and ``split(Cx)`` candidates for
  wide components;
* candidate generation is bounded by ``max_merge_components`` and pruned
  early by width/height, component gap, vertical proximity, ink area and the
  expected font bbox;
* the acceptance strings ``鲃`` / ``小`` / ``鲃鱼。`` keep working and
  ``潜甲`` / ``潜乙`` / ``巴尔的摩`` / ``Z17`` are not broken.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from fixedfontocr import FixedFontOCR
from fixedfontocr.preprocess import Segment
from fixedfontocr.scorer import SegmentScore
from fixedfontocr.segmentation import (
    build_candidates,
    connected_components,
    decode,
)
from fixedfontocr.types import VisualCandidate, default_profile

from conftest import render_text


MODEL = Path("model/game_cn")


def _require_game_model() -> None:
    if not (MODEL / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")


def _tight_glyph(font_path: Path, char: str, size: int = 32) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), size)
    bbox = font.getbbox(char)
    pad = 8
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), char, font=font, fill=(255, 255, 255))
    arr = np.asarray(img, dtype=np.uint8)
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mask = gray >= 140
    ys, xs = np.where(mask)
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _connected_pair(
    font_path: Path, left: str, right: str, size: int = 32
) -> np.ndarray:
    """Render two glyphs and force them into one connected component."""

    m1 = _tight_glyph(font_path, left, size)
    m2 = _tight_glyph(font_path, right, size)
    gap = 1
    h = max(m1.shape[0], m2.shape[0]) + 4
    w = m1.shape[1] + gap + m2.shape[1] + 4
    out = np.zeros((h, w), dtype=bool)
    y1 = (h - m1.shape[0]) // 2
    y2 = (h - m2.shape[0]) // 2
    x2 = m1.shape[1] + gap
    out[y1 : y1 + m1.shape[0], : m1.shape[1]] = m1
    out[y2 : y2 + m2.shape[0], x2 : x2 + m2.shape[1]] = m2
    rows1 = set(np.where(np.any(m1, axis=1))[0].tolist())
    rows2 = set(np.where(np.any(m2, axis=1))[0].tolist())
    common = sorted(rows1 & rows2)
    cy = int(np.median(common))
    out[cy, m1.shape[1] - 1 : x2 + 1] = True
    return out


def _rgb(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)


def test_goal4_acceptance_regressions(font_path):
    """The five Goal 4 strings plus the not-to-break regressions."""

    _require_game_model()
    ocr = FixedFontOCR(model_path=MODEL, backend="cpu")
    for text in ("鲃", "鲃鱼。", "小", "潜甲", "潜乙", "巴尔的摩", "Z17"):
        result = ocr.recognize(render_text(text, font_path))
        assert result.text == text, f"{text!r} -> {result.text!r}"
        assert len(result.chars) == len(text)

    # A connected two-glyph blob must be recoverable through split(Cx).
    mask = _connected_pair(font_path, "甲", "申")
    result = ocr.recognize(_rgb(mask))
    assert result.text == "甲申", f"connected pair -> {result.text!r}"
    assert len(result.chars) == 2


def test_wide_component_generates_split_candidates(font_path):
    """split(Cx) turns one wide component into atom alternatives."""

    mask = _connected_pair(font_path, "甲", "申")
    line = Segment(mask=mask, x=0, y=0, w=mask.shape[1], h=mask.shape[0])
    comps = connected_components(line)
    assert len(comps) == 1

    cands = build_candidates(comps, default_profile(), max_merge_components=4)
    spans = {c.atom_span for c in cands}
    assert spans == {(0, 1), (0, 2), (1, 2)}

    whole = next(c for c in cands if c.atom_span == (0, 2))
    pieces = [c for c in cands if c.atom_span in ((0, 1), (1, 2))]
    assert whole.segment is not None
    assert all(p.segment is not None for p in pieces)
    assert whole.segment.w > pieces[0].segment.w
    assert whole.segment.w > pieces[1].segment.w
    # The original component remains reachable as a single candidate.
    assert whole.components == (0,)
    assert {p.components[0] for p in pieces} == {0}


def test_split_candidates_can_be_disabled(font_path):
    """split_wide=False keeps the legacy component-level lattice."""

    mask = _connected_pair(font_path, "甲", "申")
    line = Segment(mask=mask, x=0, y=0, w=mask.shape[1], h=mask.shape[0])
    comps = connected_components(line)
    cands = build_candidates(
        comps,
        default_profile(),
        max_merge_components=4,
        split_wide=False,
    )
    assert [c.atom_span for c in cands] == [(0, 1)]


def test_decode_uses_atom_spans_not_only_component_indices():
    """DP coverage is measured over atoms, so split pieces are decodable."""

    score = SegmentScore(
        char_id=0,
        visual_score=1.0,
        raw_score=0.0,
        score_type="template",
    )
    whole = VisualCandidate(
        start=0,
        end=1,
        components=(0,),
        atoms=(0, 2),
        score=score,
    )
    left = VisualCandidate(
        start=0,
        end=1,
        components=(0,),
        atoms=(0, 1),
        score=score,
    )
    right = VisualCandidate(
        start=0,
        end=1,
        components=(0,),
        atoms=(1, 2),
        score=score,
    )
    path = decode([left, right, whole], n_components=1)
    assert len(path.candidates) == 1
    assert path.candidates[0].atom_span == (0, 2)

    weak = SegmentScore(
        char_id=0,
        visual_score=0.1,
        raw_score=0.0,
        score_type="template",
    )
    whole.score = weak
    path = decode([left, right, whole], n_components=1)
    assert [c.atom_span for c in path.candidates] == [(0, 1), (1, 2)]


def test_gap_pruning_rejects_character_to_character_merge(font_path):
    """A blank wider than the glyph gap cannot be a single merge candidate."""

    mask = np.zeros((20, 50), dtype=bool)
    mask[5:15, 5:15] = True
    mask[5:15, 30:40] = True
    line = Segment(mask=mask, x=0, y=0, w=mask.shape[1], h=mask.shape[0])
    comps = connected_components(line)
    cands = build_candidates(comps, default_profile(), max_merge_components=4)
    assert len(comps) == 2
    assert {c.atom_span for c in cands} == {(0, 1), (1, 2)}


def test_vertical_proximity_pruning(font_path):
    """Vertically disconnected blobs are not merged into one candidate."""

    mask = np.zeros((30, 20), dtype=bool)
    mask[0:10, 5:13] = True
    mask[20:30, 5:13] = True
    line = Segment(mask=mask, x=0, y=0, w=mask.shape[1], h=mask.shape[0])
    comps = connected_components(line)
    cands = build_candidates(comps, default_profile(), max_merge_components=4)
    assert len(comps) == 2
    assert {c.atom_span for c in cands} == {(0, 1), (1, 2)}
